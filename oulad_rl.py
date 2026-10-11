# -*- coding: utf-8 -*-
"""
Адаптивная геймификация на основе RL (Q-learning) на данных OULAD
+ эксперименты по надёжности и безопасности ИС.

Запуск:
    python oulad_rl.py            # на реальных данных из папки data/
    python oulad_rl.py --demo     # на синтетических данных (проверка, что всё работает)
"""
import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

DATA_DIR = "data"
OUT_DIR = "results"
RNG = np.random.default_rng(42)

N_LEVELS = 3          # уровни вовлечённости: 0 - низкий, 1 - средний, 2 - высокий
ACTIONS = ["Нет действия", "Бейдж", "Рейтинг", "Челлендж"]
N_ACTIONS = len(ACTIONS)
EPISODE_LEN = 30      # недель в семестре


# ---------------------------------------------------------------- 1. Данные
def make_demo_data(n_students=2000):
    """Синтетические данные в формате studentVle.csv (только для проверки кода)."""
    rows = []
    for sid in range(n_students):
        base = RNG.gamma(2.0, 30)
        for week in range(EPISODE_LEN):
            clicks = max(0, int(RNG.normal(base * (1 - week / 60), base * 0.5)))
            if clicks > 0:
                rows.append(("AAA", "2013J", sid, 1, week * 7, clicks))
    return pd.DataFrame(rows, columns=["code_module", "code_presentation",
                                       "id_student", "id_site", "date", "sum_click"])


def load_weekly_clicks(demo=False):
    if demo:
        vle = make_demo_data()
    else:
        path = os.path.join(DATA_DIR, "studentVle.csv")
        print(f"Читаю {path} (файл большой, ~1 мин)...")
        vle = pd.read_csv(path, usecols=["code_module", "code_presentation",
                                         "id_student", "date", "sum_click"],
                          dtype={"id_student": "int32", "date": "int16",
                                 "sum_click": "int32"})
    vle = vle[vle["date"] >= 0].copy()
    vle["week"] = vle["date"] // 7
    vle = vle[vle["week"] < EPISODE_LEN]
    weekly = (vle.groupby(["code_module", "code_presentation", "id_student", "week"])
                 ["sum_click"].sum().unstack(fill_value=0))
    weekly = weekly.reindex(columns=range(EPISODE_LEN), fill_value=0)
    print(f"Студентов (записей на курс): {len(weekly)}")
    return weekly  # строки - студенты, столбцы - недели


def to_levels(weekly, thresholds=None):
    """Дискретизация кликов в уровни вовлечённости по квантилям."""
    if thresholds is None:
        nz = weekly.values[weekly.values > 0]
        thresholds = np.quantile(nz, [0.33, 0.66])
    levels = np.digitize(weekly.values, thresholds)
    levels[weekly.values == 0] = 0
    return levels, thresholds


def transition_matrix(levels):
    """P(уровень_{t+1} | уровень_t), оценённая по реальным траекториям."""
    T = np.ones((N_LEVELS, N_LEVELS))  # сглаживание Лапласа
    for traj in levels:
        for a, b in zip(traj[:-1], traj[1:]):
            T[a, b] += 1
    return T / T.sum(axis=1, keepdims=True)


# ---------------------------------------------------------------- 2. Симулятор
class StudentSimulator:
    """
    Базовая динамика берётся из OULAD. Эффекты геймификации в OULAD отсутствуют,
    поэтому задаются ГИПОТЕЗОЙ (обязательно указать это в статье):
    действие сдвигает вероятность перехода на уровень выше, но частое давление
    (челлендж подряд) вызывает утомление.
    """
    # сдвиг вероятности вверх для каждого действия при уровне 0/1/2
    EFFECT = np.array([[0.00, 0.00, 0.00],    # нет действия
                       [0.10, 0.05, 0.02],    # бейдж
                       [0.02, 0.08, 0.06],    # рейтинг
                       [0.12, 0.10, 0.04]])   # челлендж
    FATIGUE = 0.15

    def __init__(self, T, seed=0):
        self.T = T
        self.rng = np.random.default_rng(seed)

    def reset(self):
        self.level = self.rng.choice(N_LEVELS, p=[0.5, 0.3, 0.2])
        self.pressure = 0
        self.t = 0
        return self.state()

    def state(self):
        return self.level * 3 + min(self.pressure, 2)  # 9 состояний: уровень x давление

    def step(self, action):
        p = self.T[self.level].copy()
        shift = self.EFFECT[action, self.level]
        if self.level < N_LEVELS - 1:
            p[self.level + 1] += shift
            p[0] = max(p[0] - shift, 0.0)
        self.pressure = self.pressure + 1 if action == 3 else 0
        if self.pressure >= 2:                      # утомление от давления
            p[0] += self.FATIGUE
        p /= p.sum()
        self.level = self.rng.choice(N_LEVELS, p=p)
        self.t += 1
        reward = self.level - (0.5 if self.pressure >= 3 else 0.0)
        return self.state(), reward, self.t >= EPISODE_LEN


N_STATES = N_LEVELS * 3


# ---------------------------------------------------------------- 3. Агент
def train_q(sim, episodes=60000, gamma=0.95):
    """
    Q-learning с убывающим epsilon (больше исследования в начале)
    и скоростью обучения 1/N^0.6 (N - сколько раз пара состояние-действие
    уже встречалась). Это обеспечивает сходимость в стохастической среде.
    """
    Q = np.zeros((N_STATES, N_ACTIONS))
    N = np.zeros_like(Q)
    rng = np.random.default_rng(1)
    for ep in range(episodes):
        eps = max(0.02, 1 - ep / (episodes * 0.7))
        s, done = sim.reset(), False
        while not done:
            a = rng.integers(N_ACTIONS) if rng.random() < eps else int(Q[s].argmax())
            s2, r, done = sim.step(a)
            N[s, a] += 1
            alpha = 1.0 / N[s, a] ** 0.6
            Q[s, a] += alpha * (r + gamma * Q[s2].max() * (not done) - Q[s, a])
            s = s2
    return Q


def evaluate(sim, policy, episodes=3000, obs_noise=0.0, failure_rate=0.0):
    """
    obs_noise    - вероятность, что агент видит неверный уровень (сбой датчиков/логов)
    failure_rate - вероятность отказа RL-модуля; тогда срабатывает fallback-правило
    """
    sim.rng = np.random.default_rng(123)       # одинаковые случайные условия для всех политик
    totals = []
    for _ in range(episodes):
        s, done, total = sim.reset(), False, 0.0
        while not done:
            obs = s
            if RNG.random() < obs_noise:
                obs = RNG.integers(N_STATES)
            if RNG.random() < failure_rate:
                a = 1                                # fallback: статичный бейдж
            else:
                a = policy(obs)
            s, r, done = sim.step(a)
            total += r
        totals.append(total)
    return float(np.mean(totals))


# ---------------------------------------------------------------- 4. Безопасность
def inject_farmers(weekly, share):
    """Атака: добавляем «фармеров» - аномально много кликов без реальной учёбы."""
    n = int(len(weekly) * share)
    if n == 0:
        return weekly.copy(), np.zeros(len(weekly), bool)
    hi = np.quantile(weekly.values, 0.99)  # правдоподобный, а не экстремальный уровень
    # бот-подобное поведение: стабильно высокая активность с малым разбросом
    base = RNG.uniform(0.6 * hi, hi, size=(n, 1))
    fake = (base * RNG.uniform(0.9, 1.1, size=(n, EPISODE_LEN))).astype(int)
    fake_df = pd.DataFrame(fake, columns=weekly.columns)
    data = pd.concat([weekly.reset_index(drop=True), fake_df], ignore_index=True)
    labels = np.r_[np.zeros(len(weekly), bool), np.ones(n, bool)]
    return data, labels


def detect_anomalies(data, contamination):
    """
    Детектор аномалий на основе робастной z-оценки (медиана и MAD).
    Для каждого студента считаются признаки активности; чем сильнее
    признаки отклоняются от «типичного» студента, тем выше балл аномальности.
    Помечается доля contamination студентов с наибольшим баллом.
    (Без scikit-learn, чтобы не зависеть от scipy.)
    """
    v = data.values.astype(float)
    mean = v.mean(1)
    cv = v.std(1) / (mean + 1)                      # коэффициент вариации
    X = np.c_[mean, -cv, v.max(1), -(v == 0).mean(1)]
    med = np.median(X, axis=0)
    mad = np.median(np.abs(X - med), axis=0) * 1.4826 + 1e-9
    z = (X - med) / mad
    score = np.clip(z, 0, None).sum(1)              # учитываем отклонения «вверх»
    k = max(int(round(len(v) * contamination)), 1)
    flagged = np.zeros(len(v), bool)
    flagged[np.argsort(score)[-k:]] = True
    return flagged


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--demo", action="store_true")
    args = ap.parse_args()
    os.makedirs(OUT_DIR, exist_ok=True)

    weekly = load_weekly_clicks(args.demo)
    levels, thr = to_levels(weekly)
    T = transition_matrix(levels)
    print("Матрица переходов уровней вовлечённости:\n", np.round(T, 3))

    sim = StudentSimulator(T)
    print("Обучение Q-learning агента (1-3 мин)...")
    Q = train_q(sim)
    rl = lambda s: int(Q[s].argmax())
    policies = {
        "Без геймификации": lambda s: 0,
        "Случайная": lambda s: int(RNG.integers(N_ACTIONS)),
        "Статичная (бейдж)": lambda s: 1,
        "RL-агент": rl,
    }

    # Эксперимент 1: сравнение политик
    print("\n[1] Средняя вовлечённость за семестр:")
    res1 = {k: evaluate(sim, p) for k, p in policies.items()}
    for k, v in res1.items():
        print(f"  {k:20s} {v:6.2f}")
    print("  Политика RL по состояниям:",
          [ACTIONS[int(Q[s].argmax())] for s in range(N_STATES)])

    # Эксперимент 2: надёжность при шуме наблюдений и отказах
    noise = [0, 0.1, 0.2, 0.3, 0.5]
    r_noise = [evaluate(sim, rl, obs_noise=n) for n in noise]
    fails = [0, 0.1, 0.3, 0.5, 1.0]
    r_fail = [evaluate(sim, rl, failure_rate=f) for f in fails]
    print("\n[2] Шум наблюдений:", {n: round(v, 2) for n, v in zip(noise, r_noise)})
    print("    Отказы RL (с fallback):", {f: round(v, 2) for f, v in zip(fails, r_fail)})

    # Эксперимент 3: отравление данных «фармерами» и защита.
    # Мера ущерба: насколько искажается модель динамики вовлечённости (матрица T),
    # на которой обучается агент (норма Фробениуса разницы с чистой T).
    print("\n[3] Атака отравления данных:")
    rows = []
    for share in [0.0, 0.05, 0.1, 0.2]:
        data, labels = inject_farmers(weekly, share)
        lv, _ = to_levels(data)
        d_attack = np.linalg.norm(transition_matrix(lv) - T)
        if share > 0:
            flagged = detect_anomalies(data, contamination=share)
            recall = (flagged & labels).sum() / labels.sum()
            precision = (flagged & labels).sum() / max(flagged.sum(), 1)
            lv2, _ = to_levels(data[~flagged])
            d_def = np.linalg.norm(transition_matrix(lv2) - T)
        else:
            recall = precision = np.nan
            d_def = d_attack
        rows.append((share, d_attack, d_def, precision, recall))
        print(f"  доля атаки {share:4.0%}: искажение без защиты {d_attack:.3f}, "
              f"с защитой {d_def:.3f}, precision {precision:.2f}, recall {recall:.2f}")
    pd.DataFrame(rows, columns=["share", "distortion_no_defense", "distortion_defense",
                                "precision", "recall"]) \
      .to_csv(os.path.join(OUT_DIR, "attack_results.csv"), index=False)

    # Графики
    fig, ax = plt.subplots(1, 3, figsize=(15, 4))
    ax[0].bar(res1.keys(), res1.values()); ax[0].set_title("Сравнение политик")
    ax[0].tick_params(axis="x", rotation=20)
    ax[1].plot(noise, r_noise, "o-", label="шум наблюдений")
    ax[1].plot(fails, r_fail, "s-", label="отказ RL → fallback")
    ax[1].set_title("Надёжность агента"); ax[1].set_xlabel("вероятность"); ax[1].legend()
    sh = [r[0] for r in rows]
    ax[2].plot(sh, [r[1] for r in rows], "o-", label="без защиты")
    ax[2].plot(sh, [r[2] for r in rows], "s-", label="с защитой")
    ax[2].set_title("Искажение модели при отравлении"); ax[2].set_ylabel("||T_атака - T||"); ax[2].set_xlabel("доля фармеров"); ax[2].legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "results.png"), dpi=150)
    print(f"\nГрафики и таблицы сохранены в папку {OUT_DIR}/")


if __name__ == "__main__":
    main()
