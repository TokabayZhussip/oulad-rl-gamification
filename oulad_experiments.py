# -*- coding: utf-8 -*-
"""
Расширенные эксперименты для статьи:
адаптивная геймификация на основе RL на данных OULAD,
надёжность и защита от отравления данных.

  E1  сравнение стратегий (10 повторов, среднее ± ст. откл., перестановочный тест)
  E2  анализ чувствительности к гипотезе об эффектах геймификации
  E3  надёжность: шум наблюдений и отказы RL-модуля (fallback)
  E4  атаки отравления данных: наивные боты и «умные» фармеры, 3 метода защиты
  E5  проверка на 7 курсах OULAD по отдельности

Запуск:
    python oulad_experiments.py          # реальные данные из папки data/
    python oulad_experiments.py --demo   # синтетические данные (проверка)
    python oulad_experiments.py --fast   # ускоренный режим (меньше повторов)
Результаты: папка results_v2/ (таблицы CSV, графики PNG, summary.txt)
"""
import argparse
import os
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

DATA_DIR = "data"
OUT_DIR = "results_v2"
H = 30                                   # недель в семестре (горизонт)
N_LEVELS = 3                             # уровни вовлечённости: 0 низкий, 1 средний, 2 высокий
N_PRESS = 3                              # счётчик давления (челленджей подряд): 0, 1, 2+
N_STATES = N_LEVELS * N_PRESS
ACTIONS = ["Нет действия", "Бейдж", "Рейтинг", "Челлендж"]
N_ACTIONS = len(ACTIONS)

# Гипотеза об эффектах (в OULAD интервенций нет): сдвиг вероятности
# перехода на уровень выше для действия a при уровне l.
BASE_EFFECT = np.array([[0.00, 0.00, 0.00],
                        [0.10, 0.05, 0.02],
                        [0.02, 0.08, 0.06],
                        [0.12, 0.10, 0.04]])
BASE_FATIGUE = 0.15
PENALTY = 0.5


def log(msg):
    print(time.strftime("[%H:%M:%S] ") + msg, flush=True)


# =============================================================== ДАННЫЕ
def make_demo_vle(seed=0, n=3000):
    rng = np.random.default_rng(seed)
    mods = ["AAA", "BBB", "CCC", "DDD", "EEE", "FFF", "GGG"]
    rows = []
    for sid in range(n):
        base = rng.gamma(2.0, 30)
        mod = mods[sid % 7]
        for w in range(H):
            c = int(max(0, rng.normal(base * (1 - w / 60), base * 0.6)))
            if c > 0:
                rows.append((mod, "2014J", sid, w * 7, c))
    return pd.DataFrame(rows, columns=["code_module", "code_presentation",
                                       "id_student", "date", "sum_click"])


def load_weekly(demo):
    if demo:
        vle = make_demo_vle()
    else:
        path = os.path.join(DATA_DIR, "studentVle.csv")
        log(f"Читаю {path} (около минуты)...")
        vle = pd.read_csv(path, usecols=["code_module", "code_presentation",
                                         "id_student", "date", "sum_click"],
                          dtype={"id_student": "int32", "date": "int16",
                                 "sum_click": "int32"})
    vle = vle[vle["date"] >= 0].copy()
    vle["week"] = vle["date"] // 7
    vle = vle[vle["week"] < H]
    w = (vle.groupby(["code_module", "code_presentation", "id_student", "week"])
            ["sum_click"].sum().unstack(fill_value=0))
    w = w.reindex(columns=range(H), fill_value=0)
    log(f"Записей студент-курс: {len(w)}, курсов: {w.index.get_level_values(0).nunique()}")
    return w


def to_levels(X):
    """Клики -> уровни вовлечённости по квантилям ненулевой активности."""
    X = np.asarray(X)
    nz = X[X > 0]
    thr = np.quantile(nz, [0.33, 0.66]) if len(nz) else np.array([1, 2])
    L = np.digitize(X, thr)
    L[X == 0] = 0
    return L


def estimate_model(L):
    """Матрица переходов T[l, l'] и начальное распределение уровней."""
    T = np.ones((N_LEVELS, N_LEVELS))                     # сглаживание Лапласа
    np.add.at(T, (L[:, :-1].ravel(), L[:, 1:].ravel()), 1)
    T /= T.sum(1, keepdims=True)
    p0 = np.bincount(L[:, 0], minlength=N_LEVELS) + 1.0
    return T, p0 / p0.sum()


def load_scores(W, demo):
    """
    Средняя оценка студента за задания курса (studentAssessment + assessments),
    выровненная по строкам W. NaN - у студента нет оценок (например, выбыл).
    """
    if demo:
        rng = np.random.default_rng(1)
        m = W.values.mean(1)
        sc = np.clip(35 + 9 * np.log1p(m) + rng.normal(0, 12, len(m)), 0, 100)
        sc[rng.random(len(m)) < 0.15] = np.nan
        return sc
    sa = pd.read_csv(os.path.join(DATA_DIR, "studentAssessment.csv"),
                     usecols=["id_assessment", "id_student", "score"])
    a = pd.read_csv(os.path.join(DATA_DIR, "assessments.csv"),
                    usecols=["code_module", "code_presentation", "id_assessment"])
    sa["score"] = pd.to_numeric(sa["score"], errors="coerce")
    m = sa.merge(a, on="id_assessment")
    g = m.groupby(["code_module", "code_presentation", "id_student"])["score"].mean()
    sc = g.reindex(W.index).values.astype(float)
    log(f"Оценки найдены для {np.isfinite(sc).mean():.0%} записей студент-курс")
    return sc


# =============================================================== MDP
def build_mdp(T, effect=BASE_EFFECT, fatigue=BASE_FATIGUE):
    """
    Состояние s = уровень*3 + давление. Возвращает:
    P[s, a, l']  - вероятность следующего уровня,
    NP[s, a]     - следующее значение давления,
    PEN[s, a]    - штраф за переутомление.
    """
    P = np.zeros((N_STATES, N_ACTIONS, N_LEVELS))
    NP = np.zeros((N_STATES, N_ACTIONS), int)
    PEN = np.zeros((N_STATES, N_ACTIONS))
    for l in range(N_LEVELS):
        for pr in range(N_PRESS):
            s = l * N_PRESS + pr
            for a in range(N_ACTIONS):
                p = T[l].copy()
                sh = effect[a, l]
                if l < N_LEVELS - 1:
                    p[l + 1] += sh
                    p[0] = max(p[0] - sh, 0.0)
                new_pr = pr + 1 if a == 3 else 0
                if new_pr >= 2:
                    p[0] += fatigue
                p /= p.sum()
                P[s, a] = p
                NP[s, a] = min(new_pr, N_PRESS - 1)
                PEN[s, a] = PENALTY if new_pr >= 3 else 0.0
    return P, NP, PEN


class VecSim:
    """Векторизованный симулятор: много студентов параллельно."""

    def __init__(self, mdp, p0, rng):
        self.P, self.NP, self.PEN = mdp
        self.cdf = self.P.cumsum(-1)
        self.p0 = p0
        self.rng = rng

    def reset(self, B):
        lv = self.rng.choice(N_LEVELS, size=B, p=self.p0)
        return lv * N_PRESS

    def step(self, s, a):
        u = self.rng.random(len(s))
        lv2 = (u[:, None] > self.cdf[s, a]).sum(1)
        lv2 = np.minimum(lv2, N_LEVELS - 1)
        s2 = lv2 * N_PRESS + self.NP[s, a]
        r = lv2 - self.PEN[s, a]
        return s2, r


# =============================================================== АГЕНТЫ
def train(mdp, p0, algo="qlearning", seed=0, batches=300, B=256, gamma=0.95):
    """
    algo: qlearning | sarsa | bandit (контекстный бандит = gamma 0).
    Q-learning/SARSA с убывающим epsilon и шагом 1/N^0.6.
    """
    rng = np.random.default_rng(seed)
    sim = VecSim(mdp, p0, rng)
    g = 0.0 if algo == "bandit" else gamma
    Q = np.zeros((N_STATES, N_ACTIONS))
    N = np.zeros_like(Q)

    def act(s, eps):
        greedy = Q[s].argmax(1)
        rand = rng.integers(N_ACTIONS, size=len(s))
        return np.where(rng.random(len(s)) < eps, rand, greedy)

    for b in range(batches):
        eps = max(0.02, 1 - b / (batches * 0.7))
        s = sim.reset(B)
        a = act(s, eps)
        for t in range(H):
            s2, r = sim.step(s, a)
            a2 = act(s2, eps)
            last = t == H - 1
            if algo == "sarsa":
                nxt = Q[s2, a2]
            else:
                nxt = Q[s2].max(1)
            target = r + (0.0 if last else g) * nxt
            idx = s * N_ACTIONS + a
            td = np.bincount(idx, target - Q[s, a], N_STATES * N_ACTIONS)
            cnt = np.bincount(idx, minlength=N_STATES * N_ACTIONS)
            m = cnt > 0
            Nf, Qf = N.ravel(), Q.ravel()
            Nf[m] += cnt[m]
            Qf[m] += (1.0 / Nf[m] ** 0.6) * td[m] / cnt[m]
            s, a = s2, a2
    return Q.argmax(1)


def optimal_value(mdp, p0):
    """Точное оптимальное значение (динамическое программирование, верхняя граница)."""
    P, NP, PEN = mdp
    V = np.zeros(N_STATES)
    for _ in range(H):
        Qv = np.zeros((N_STATES, N_ACTIONS))
        for s in range(N_STATES):
            for a in range(N_ACTIONS):
                nxt = np.arange(N_LEVELS) * N_PRESS + NP[s, a]
                Qv[s, a] = P[s, a] @ (np.arange(N_LEVELS) + V[nxt]) - PEN[s, a]
        V = Qv.max(1)
    return float(p0 @ V[np.arange(N_LEVELS) * N_PRESS])


def policy_value(mdp, p0, pol):
    """Точное ожидаемое значение фиксированной политики (без симуляции)."""
    P, NP, PEN = mdp
    V = np.zeros(N_STATES)
    for _ in range(H):
        V2 = np.zeros(N_STATES)
        for s in range(N_STATES):
            a = pol[s]
            nxt = np.arange(N_LEVELS) * N_PRESS + NP[s, a]
            V2[s] = P[s, a] @ (np.arange(N_LEVELS) + V[nxt]) - PEN[s, a]
        V = V2
    return float(p0 @ V[np.arange(N_LEVELS) * N_PRESS])


def evaluate(mdp, p0, pol, seed=123, B=4000, obs_noise=0.0, fail=0.0, fallback=1):
    """Средняя суммарная вовлечённость за семестр (симуляция B студентов)."""
    rng = np.random.default_rng(seed)
    sim = VecSim(mdp, p0, rng)
    s = sim.reset(B)
    total = np.zeros(B)
    for _ in range(H):
        obs = np.where(rng.random(B) < obs_noise, rng.integers(N_STATES, size=B), s)
        a = pol[obs] if not callable(pol) else pol(obs, rng)
        a = np.where(rng.random(B) < fail, fallback, a)
        s, r = sim.step(s, a)
        total += r
    return float(total.mean())


def fixed(a):
    return np.full(N_STATES, a)


# эвристика эксперта: пассивным - челлендж (не подряд), средним - бейдж, активным - рейтинг
HEURISTIC = np.array([3, 1, 1,   1, 1, 1,   2, 0, 0])


def random_policy(obs, rng):
    return rng.integers(N_ACTIONS, size=len(obs))


def perm_test_paired(d, n_perm=20000, seed=0):
    """Перестановочный тест (sign-flip) для парных разностей; p-значение."""
    d = np.asarray(d)
    rng = np.random.default_rng(seed)
    obs = abs(d.mean())
    signs = rng.choice([-1, 1], size=(n_perm, len(d)))
    return float((np.abs((signs * d).mean(1)) >= obs - 1e-12).mean())


# =============================================================== АТАКИ И ЗАЩИТА
def naive_bots(X, n, rng):
    """Наивные боты: стабильно высокая активность с малым разбросом."""
    hi = np.quantile(X, 0.99)
    base = rng.uniform(0.6 * hi, hi, size=(n, 1))
    return (base * rng.uniform(0.9, 1.1, size=(n, H))).astype(int)


def smart_farmers(X, n, rng):
    """«Умные» фармеры: копируют траекторию реального активного студента,
    завышая активность в 1.5-2.5 раза с человекоподобным шумом."""
    act = X[X.mean(1) > np.median(X.mean(1))]
    src = act[rng.integers(len(act), size=n)]
    k = rng.uniform(1.5, 2.5, size=(n, 1))
    noise = rng.lognormal(0, 0.3, size=(n, H))
    return (src * k * noise).astype(int)


def features(X):
    v = X.astype(float)
    mean = v.mean(1)
    cv = v.std(1) / (mean + 1)
    zero = (v == 0).mean(1)
    mx = v.max(1)
    t = np.arange(H) - (H - 1) / 2
    trend = (v * t).sum(1) / (t ** 2).sum() / (mean + 1)
    a, b = v[:, :-1] - v[:, :-1].mean(1, keepdims=True), v[:, 1:] - v[:, 1:].mean(1, keepdims=True)
    lag = (a * b).sum(1) / (np.sqrt((a ** 2).sum(1) * (b ** 2).sum(1)) + 1e-9)  # автокорреляция
    return np.c_[np.log1p(mean), -cv, np.log1p(mx), -zero, trend, np.nan_to_num(lag)]


def top_k(score, share):
    k = max(int(round(len(score) * share)), 1)
    f = np.zeros(len(score), bool)
    f[np.argsort(score)[-k:]] = True
    return f


def defend_mad(F, share, ctx=None):
    """Робастная z-оценка (медиана/MAD), отклонения «вверх»."""
    med = np.median(F, 0)
    mad = np.median(np.abs(F - med), 0) * 1.4826 + 1e-9
    return top_k(np.clip((F - med) / mad, 0, None).sum(1), share)


def defend_mahalanobis(F, share, ctx=None):
    """Расстояние Махаланобиса до центра распределения."""
    mu = F.mean(0)
    C = np.cov(F.T) + 1e-6 * np.eye(F.shape[1])
    D = F - mu
    return top_k(np.einsum("ij,jk,ik->i", D, np.linalg.inv(C), D), share)


def defend_knn(F, share, ctx=None, k=10, ref=1500, seed=0):
    """Расстояние до k-го ближайшего соседа из опорной выборки (аналог LOF)."""
    rng = np.random.default_rng(seed)
    Z = (F - F.mean(0)) / (F.std(0) + 1e-9)
    R = Z[rng.choice(len(Z), size=min(ref, len(Z)), replace=False)]
    score = np.empty(len(Z))
    for i in range(0, len(Z), 2000):
        d = ((Z[i:i + 2000, None, :] - R[None]) ** 2).sum(-1)
        score[i:i + 2000] = np.sort(d, 1)[:, k]
    return top_k(score, share)


def consistency_score(F, ctx):
    """
    Согласованность активности и успеваемости. По всем студентам с оценками
    строится регрессия: оценка ~ log(средние клики). Подозрителен тот, чья
    оценка заметно НИЖЕ ожидаемой при его активности: много кликов без знаний.
    Студенты без оценок получают нейтральный балл 0.
    """
    S, act = ctx["scores"], F[:, 0]                 # F[:,0] = log(1+mean clicks)
    ok = np.isfinite(S)
    A = np.c_[np.ones(ok.sum()), act[ok]]
    coef, *_ = np.linalg.lstsq(A, S[ok], rcond=None)
    resid = np.zeros(len(S))
    resid[ok] = S[ok] - A @ coef
    mad = np.median(np.abs(resid[ok] - np.median(resid[ok]))) * 1.4826 + 1e-9
    z = np.clip(-resid / mad, 0, None)              # только «ниже ожидаемого»
    # учитываем, насколько активность выше типичной (фармеры активны)
    za = np.clip((act - np.median(act)) / (np.median(np.abs(act - np.median(act))) * 1.4826 + 1e-9), 0, None)
    return z * (1 + za)


def defend_consistency(F, share, ctx):
    return top_k(consistency_score(F, ctx), share)


def defend_hybrid(F, share, ctx):
    """Гибрид: MAD-аномальность поведения + несогласованность с оценками."""
    med = np.median(F, 0)
    mad = np.median(np.abs(F - med), 0) * 1.4826 + 1e-9
    s1 = np.clip((F - med) / mad, 0, None).sum(1)
    s2 = consistency_score(F, ctx)
    r1 = s1.argsort().argsort() / len(s1)           # ранги, чтобы шкалы были сопоставимы
    r2 = s2.argsort().argsort() / len(s2)
    return top_k(np.maximum(r1, r2), share)


DEFENSES = {"MAD": defend_mad, "Махаланобис": defend_mahalanobis, "kNN": defend_knn,
            "Согласованность": defend_consistency, "Гибрид": defend_hybrid}
ATTACKS = {"наивные боты": naive_bots, "умные фармеры": smart_farmers}


# =============================================================== MAIN
def mstd(a):
    return f"{np.mean(a):.2f} ± {np.std(a, ddof=1):.2f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--demo", action="store_true")
    ap.add_argument("--fast", action="store_true")
    args = ap.parse_args()
    os.makedirs(OUT_DIR, exist_ok=True)
    SEEDS = 4 if args.fast else 10
    SEEDS_SMALL = 3 if args.fast else 5
    t0 = time.time()
    summary = []

    def say(line=""):
        print(line, flush=True)
        summary.append(line)

    W = load_weekly(args.demo)
    X = W.values
    SCORES = load_scores(W, args.demo)
    valid_scores = SCORES[np.isfinite(SCORES)]
    L = to_levels(X)
    n_students = len(L)

    def boot_model(seed, Lm=L):
        rng = np.random.default_rng(seed)
        idx = rng.integers(len(Lm), size=len(Lm))           # бутстрап студентов
        return estimate_model(Lm[idx])

    T_full, p0_full = estimate_model(L)
    say("=" * 70)
    say(f"Данные: {n_students} записей студент-курс, горизонт {H} недель")
    say("Матрица переходов уровней вовлечённости (по данным):")
    say(str(np.round(T_full, 3)))
    say(f"Начальное распределение уровней: {np.round(p0_full, 3)}")

    # ------------------------------------------------------------- E1
    log("E1: сравнение стратегий...")
    names = ["Без геймификации", "Случайная", "Статичная (бейдж)",
             "Эвристика эксперта", "Контекстный бандит", "SARSA", "Q-learning",
             "Оптимум (DP)"]
    res = {n: [] for n in names}
    policies_q = []
    for sd in range(SEEDS):
        T, p0 = boot_model(sd)
        mdp = build_mdp(T)
        res["Без геймификации"].append(evaluate(mdp, p0, fixed(0), seed=1000 + sd))
        res["Случайная"].append(evaluate(mdp, p0, random_policy, seed=1000 + sd))
        res["Статичная (бейдж)"].append(evaluate(mdp, p0, fixed(1), seed=1000 + sd))
        res["Эвристика эксперта"].append(evaluate(mdp, p0, HEURISTIC, seed=1000 + sd))
        for algo, nm in [("bandit", "Контекстный бандит"), ("sarsa", "SARSA"),
                         ("qlearning", "Q-learning")]:
            pol = train(mdp, p0, algo, seed=sd)
            res[nm].append(evaluate(mdp, p0, pol, seed=1000 + sd))
            if algo == "qlearning":
                policies_q.append(pol)
        res["Оптимум (DP)"].append(optimal_value(mdp, p0))
    df1 = pd.DataFrame({n: res[n] for n in names})
    df1.to_csv(os.path.join(OUT_DIR, "E1_policies_by_seed.csv"), index_label="seed")
    say()
    say(f"[E1] Средняя суммарная вовлечённость за семестр ({SEEDS} повторов, среднее ± ст.откл.)")
    for n in names:
        say(f"  {n:22s} {mstd(res[n])}")
    best_base = max(["Без геймификации", "Случайная", "Статичная (бейдж)",
                     "Эвристика эксперта"], key=lambda n: np.mean(res[n]))
    d = np.array(res["Q-learning"]) - np.array(res[best_base])
    p = perm_test_paired(d)
    gain_none = (np.mean(res["Q-learning"]) / np.mean(res["Без геймификации"]) - 1) * 100
    gain_best = (np.mean(res["Q-learning"]) / np.mean(res[best_base]) - 1) * 100
    opt_share = np.mean(res["Q-learning"]) / np.mean(res["Оптимум (DP)"]) * 100
    say(f"  Q-learning vs «{best_base}»: +{gain_best:.1f}% (перестановочный тест p = {p:.4f})")
    say(f"  Q-learning vs «Без геймификации»: +{gain_none:.1f}%")
    say(f"  Q-learning достигает {opt_share:.1f}% от теоретического оптимума")
    # самая частая политика
    pol_mode = np.array([np.bincount(np.array(policies_q)[:, s], minlength=N_ACTIONS).argmax()
                         for s in range(N_STATES)])
    say("  Политика Q-learning (уровень / давление -> действие):")
    for l, ln in enumerate(["низкий", "средний", "высокий"]):
        say("    " + ln + ": " + ", ".join(f"давл.{pr}->{ACTIONS[pol_mode[l * 3 + pr]]}"
                                           for pr in range(N_PRESS)))

    # ------------------------------------------------------------- E2
    log("E2: анализ чувствительности...")
    scales = [0.25, 0.5, 0.75, 1.0, 1.5, 2.0]
    fatigues = [0.05, 0.15, 0.30]
    rows2 = []
    for sc in scales:
        for fat in fatigues:
            for sd in range(SEEDS_SMALL):
                T, p0 = boot_model(100 + sd)
                mdp = build_mdp(T, BASE_EFFECT * sc, fat)
                pol = train(mdp, p0, "qlearning", seed=sd)
                q = evaluate(mdp, p0, pol, seed=2000 + sd)
                b = evaluate(mdp, p0, fixed(1), seed=2000 + sd)
                n0 = evaluate(mdp, p0, fixed(0), seed=2000 + sd)
                h = evaluate(mdp, p0, HEURISTIC, seed=2000 + sd)
                rows2.append((sc, fat, sd, q, b, h, n0))
    df2 = pd.DataFrame(rows2, columns=["effect_scale", "fatigue", "seed",
                                       "qlearning", "badge", "heuristic", "none"])
    df2.to_csv(os.path.join(OUT_DIR, "E2_sensitivity.csv"), index=False)
    agg2 = df2.groupby(["effect_scale", "fatigue"]).mean(numeric_only=True)
    agg2["gain_vs_best_static_%"] = (agg2["qlearning"] /
                                     agg2[["badge", "heuristic", "none"]].max(axis=1) - 1) * 100
    wins = (agg2["gain_vs_best_static_%"] > 0).sum()
    say()
    say("[E2] Чувствительность: выигрыш Q-learning над лучшей статичной стратегией, %")
    say("  (строки - масштаб эффектов геймификации, столбцы - сила утомления)")
    say(agg2["gain_vs_best_static_%"].unstack().round(1).to_string())
    say(f"  RL лучше статичных стратегий в {wins} из {len(agg2)} сценариев")

    # ------------------------------------------------------------- E3
    log("E3: надёжность...")
    noise = [0, 0.1, 0.2, 0.3, 0.5]
    fails = [0, 0.1, 0.3, 0.5, 1.0]
    r_noise = {n: [] for n in noise}
    r_fail = {f: [] for f in fails}
    r_fail_nofb = {f: [] for f in fails}
    for sd in range(SEEDS):
        T, p0 = boot_model(sd)
        mdp = build_mdp(T)
        pol = policies_q[sd]
        for n in noise:
            r_noise[n].append(evaluate(mdp, p0, pol, seed=3000 + sd, obs_noise=n))
        for f in fails:
            r_fail[f].append(evaluate(mdp, p0, pol, seed=3000 + sd, fail=f, fallback=1))
            r_fail_nofb[f].append(evaluate(mdp, p0, pol, seed=3000 + sd, fail=f, fallback=0))
    pd.DataFrame({"noise": noise, "mean": [np.mean(r_noise[n]) for n in noise],
                  "std": [np.std(r_noise[n], ddof=1) for n in noise]}) \
        .to_csv(os.path.join(OUT_DIR, "E3_noise.csv"), index=False)
    pd.DataFrame({"fail_rate": fails,
                  "fallback_mean": [np.mean(r_fail[f]) for f in fails],
                  "fallback_std": [np.std(r_fail[f], ddof=1) for f in fails],
                  "no_fallback_mean": [np.mean(r_fail_nofb[f]) for f in fails],
                  "no_fallback_std": [np.std(r_fail_nofb[f], ddof=1) for f in fails]}) \
        .to_csv(os.path.join(OUT_DIR, "E3_failures.csv"), index=False)
    say()
    say("[E3] Надёжность (среднее ± ст.откл.)")
    say("  Шум наблюдений: " + "; ".join(f"{int(n*100)}% -> {mstd(r_noise[n])}" for n in noise))
    say("  Отказы RL с fallback (бейдж): " +
        "; ".join(f"{int(f*100)}% -> {mstd(r_fail[f])}" for f in fails))
    say("  Отказы RL без fallback (ничего): " +
        "; ".join(f"{int(f*100)}% -> {mstd(r_fail_nofb[f])}" for f in fails))
    drop = (1 - np.mean(r_noise[0.5]) / np.mean(r_noise[0])) * 100
    say(f"  При 50% шума качество снижается на {drop:.1f}%")

    # ------------------------------------------------------------- E4
    log("E4: атаки и защита (самый долгий этап)...")
    shares = [0.05, 0.10, 0.20]
    rows4 = []
    pol_ref = pol_mode
    for att_name, att in ATTACKS.items():
        for sh in shares:
            for sd in range(SEEDS_SMALL):
                rng = np.random.default_rng(500 + sd)
                n_att = int(len(X) * sh)
                Xp = np.vstack([X, att(X, n_att, rng)])
                lab = np.r_[np.zeros(len(X), bool), np.ones(n_att, bool)]
                # фармер набирает активность, но не знания: его оценка - как у
                # случайного студента курса, независимо от числа кликов
                Sp = np.r_[SCORES, rng.choice(valid_scores, size=n_att)]
                Tp, p0p = estimate_model(to_levels(Xp))
                dist0 = np.linalg.norm(Tp - T_full)
                # завышение прогноза вовлечённости: модель на отравленных данных
                # предсказывает ценность политики выше, чем в действительности
                v_true = policy_value(build_mdp(T_full), p0_full, pol_ref)
                v_pois = policy_value(build_mdp(Tp), p0p, pol_ref)
                rows4.append((att_name, sh, sd, "без защиты", dist0,
                              (v_pois / v_true - 1) * 100, np.nan, np.nan))
                F = features(Xp)
                for dn, dfun in DEFENSES.items():
                    fl = dfun(F, sh, {"scores": Sp})
                    prec = (fl & lab).sum() / max(fl.sum(), 1)
                    rec = (fl & lab).sum() / lab.sum()
                    Tc, p0c = estimate_model(to_levels(Xp[~fl]))
                    v_c = policy_value(build_mdp(Tc), p0c, pol_ref)
                    rows4.append((att_name, sh, sd, dn, np.linalg.norm(Tc - T_full),
                                  (v_c / v_true - 1) * 100, prec, rec))
    df4 = pd.DataFrame(rows4, columns=["attack", "share", "seed", "defense",
                                       "distortion", "forecast_bias_%",
                                       "precision", "recall"])
    df4.to_csv(os.path.join(OUT_DIR, "E4_attacks.csv"), index=False)
    agg4 = df4.groupby(["attack", "share", "defense"]).mean(numeric_only=True) \
              .drop(columns="seed")
    say()
    say("[E4] Атаки отравления данных (среднее по повторам)")
    say("  distortion - искажение матрицы переходов ||T_атака - T||,")
    say("  forecast_bias_% - завышение прогноза вовлечённости моделью")
    say(agg4.round(3).to_string())
    for att_name in ATTACKS:
        sub = agg4.xs(att_name, level="attack")
        for sh in shares:
            base = sub.loc[(sh, "без защиты"), "distortion"]
            best = sub.loc[sh].drop("без защиты")["distortion"].idxmin()
            bv = sub.loc[(sh, best), "distortion"]
            say(f"  {att_name}, {int(sh*100)}%: лучшая защита - {best} "
                f"(искажение {base:.3f} -> {bv:.3f}, в {base / max(bv, 1e-9):.1f} раза)")

    # ------------------------------------------------------------- E5
    log("E5: проверка по курсам...")
    rows5 = []
    mods = W.index.get_level_values(0)
    for m in sorted(mods.unique()):
        Xm = X[mods == m]
        Lm = to_levels(Xm)
        for sd in range(SEEDS_SMALL):
            T, p0 = boot_model(700 + sd, Lm)
            mdp = build_mdp(T)
            pol = train(mdp, p0, "qlearning", seed=sd)
            rows5.append((m, len(Xm), sd,
                          evaluate(mdp, p0, pol, seed=4000 + sd),
                          evaluate(mdp, p0, fixed(1), seed=4000 + sd),
                          evaluate(mdp, p0, HEURISTIC, seed=4000 + sd),
                          evaluate(mdp, p0, fixed(0), seed=4000 + sd)))
    df5 = pd.DataFrame(rows5, columns=["module", "n", "seed", "qlearning",
                                       "badge", "heuristic", "none"])
    df5.to_csv(os.path.join(OUT_DIR, "E5_courses.csv"), index=False)
    agg5 = df5.groupby("module").mean(numeric_only=True).drop(columns="seed")
    agg5["gain_vs_best_static_%"] = (agg5["qlearning"] /
                                     agg5[["badge", "heuristic", "none"]].max(axis=1) - 1) * 100
    say()
    say("[E5] Результаты по курсам OULAD (среднее по повторам)")
    say(agg5.round(2).to_string())
    say(f"  RL лучше статичных стратегий на {(agg5['gain_vs_best_static_%'] > 0).sum()} "
        f"из {len(agg5)} курсов")

    # ------------------------------------------------------------- ГРАФИКИ
    log("Строю графики...")
    plt.rcParams.update({"font.size": 10})

    fig, ax = plt.subplots(figsize=(10, 4.5))
    m_ = [np.mean(res[n]) for n in names]
    s_ = [np.std(res[n], ddof=1) for n in names]
    cols = ["#999"] * 4 + ["#6a9fd8", "#3b7dc4", "#1f4e8c", "#2a9d5c"]
    ax.bar(names, m_, yerr=s_, capsize=4, color=cols)
    ax.set_ylabel("Суммарная вовлечённость за семестр")
    ax.set_title("E1. Сравнение стратегий геймификации")
    ax.tick_params(axis="x", rotation=25)
    plt.tight_layout(); plt.savefig(os.path.join(OUT_DIR, "fig1_policies.png"), dpi=200); plt.close()

    fig, ax = plt.subplots(figsize=(7, 4.5))
    for fat in fatigues:
        sub = agg2.xs(fat, level="fatigue")
        ax.plot(sub.index, sub["gain_vs_best_static_%"], "o-", label=f"утомление {fat}")
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xlabel("Масштаб эффектов геймификации (1.0 = базовая гипотеза)")
    ax.set_ylabel("Выигрыш RL над лучшей статичной, %")
    ax.set_title("E2. Анализ чувствительности"); ax.legend()
    plt.tight_layout(); plt.savefig(os.path.join(OUT_DIR, "fig2_sensitivity.png"), dpi=200); plt.close()

    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    mn = np.array([np.mean(r_noise[n]) for n in noise]); sn = np.array([np.std(r_noise[n], ddof=1) for n in noise])
    ax[0].plot(noise, mn, "o-"); ax[0].fill_between(noise, mn - sn, mn + sn, alpha=0.2)
    ax[0].set_xlabel("Доля искажённых наблюдений"); ax[0].set_ylabel("Вовлечённость")
    ax[0].set_title("E3a. Шум входных данных")
    for dct, lb in [(r_fail, "с fallback (бейдж)"), (r_fail_nofb, "без fallback")]:
        mf = np.array([np.mean(dct[f]) for f in fails]); sf = np.array([np.std(dct[f], ddof=1) for f in fails])
        ax[1].plot(fails, mf, "o-", label=lb); ax[1].fill_between(fails, mf - sf, mf + sf, alpha=0.2)
    ax[1].set_xlabel("Вероятность отказа RL-модуля"); ax[1].set_title("E3b. Отказы и резервирование")
    ax[1].legend()
    plt.tight_layout(); plt.savefig(os.path.join(OUT_DIR, "fig3_reliability.png"), dpi=200); plt.close()

    fig, ax = plt.subplots(1, 2, figsize=(14, 4.8))
    for i, att_name in enumerate(ATTACKS):
        sub = agg4.xs(att_name, level="attack")["distortion"].unstack()
        sub = sub[["без защиты"] + list(DEFENSES)]
        sub.plot(kind="bar", ax=ax[i], rot=0)
        ax[i].set_title(f"E4. {att_name}"); ax[i].set_xlabel("Доля атакующих")
        ax[i].set_ylabel("Искажение модели ||T_атака - T||")
    plt.tight_layout(); plt.savefig(os.path.join(OUT_DIR, "fig4_attacks.png"), dpi=200); plt.close()

    fig, ax = plt.subplots(figsize=(9, 4.5))
    agg5[["none", "badge", "heuristic", "qlearning"]].rename(
        columns={"none": "Без геймиф.", "badge": "Бейдж", "heuristic": "Эвристика",
                 "qlearning": "Q-learning"}).plot(kind="bar", ax=ax, rot=0)
    ax.set_ylabel("Вовлечённость"); ax.set_xlabel("Курс OULAD")
    ax.set_title("E5. Результаты по курсам")
    plt.tight_layout(); plt.savefig(os.path.join(OUT_DIR, "fig5_courses.png"), dpi=200); plt.close()

    say()
    say(f"Время выполнения: {(time.time() - t0) / 60:.1f} мин")
    with open(os.path.join(OUT_DIR, "summary.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(summary))
    log(f"Готово. Все таблицы, графики и summary.txt в папке {OUT_DIR}/")


if __name__ == "__main__":
    main()
