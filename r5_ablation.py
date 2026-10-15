# -*- coding: utf-8 -*-
"""
R5: абляционный анализ критерия согласованности активности и успеваемости.

Критерий (полный вариант):  c_i = z_i^- * (1 + a_i^+), где
  z_i^- = max(0, -e_i / s)  - односторонний робастный остаток регрессии
                              «оценка ~ log(1 + средняя активность)»
                              (только оценка НИЖЕ ожидаемой),
  a_i^+ = max(0, (x_i - med x) / (1.4826 MAD x)) - робастное превышение
                              типичной активности (весовой коэффициент).

Упрощённые варианты:
  V1 без весового коэффициента активности:  z_i^-
  V2 обычный анализ остатков (двусторонний): |e_i| / s
  V3 без учёта успеваемости (только активность): a_i^+
  V4 MAD по всем признакам активности (базовый детектор статьи)

Протоколы:
  A  известная доля атакующих (как в основном эксперименте E4): каждый вариант
     исключает долю наиболее подозрительных записей, равную доле атакующих;
     все данные 2013-2014 гг., 5 повторов.
  B  без знания доли: статистика (регрессия, медианы, MAD) и пороги - по чистым
     данным 2013 г. на 5% ложных срабатываний, атака и оценка - на 2014 г.,
     5 повторов, включая сценарий без атаки.

Метрики: Precision, Recall, F1, доля ложных срабатываний, искажение ||T' - T||_F,
смещение прогноза ценности стратегии Q-learning, %.

Запуск (из корня репозитория, данные OULAD в data/):
    python r5_ablation.py
Результаты: results_revision/R5_ablation.csv, results_revision/summary_R5.txt
"""
import os
import numpy as np
import pandas as pd
import oulad_experiments as E

OUT = "results_revision"
SHARES = [0.0, 0.05, 0.10, 0.20]
SEEDS = 5
FPR = 0.05

VARIANTS = ["Полный критерий", "V1: без веса активности", "V2: двусторонний остаток",
            "V3: без успеваемости", "V4: MAD (все признаки)"]


class Scorer:
    """Все компоненты оцениваются на опорной выборке (F_ref, S_ref)."""

    def __init__(self, F_ref, S_ref):
        act = F_ref[:, 0]
        ok = np.isfinite(S_ref)
        A = np.c_[np.ones(ok.sum()), act[ok]]
        self.coef, *_ = np.linalg.lstsq(A, S_ref[ok], rcond=None)
        r = S_ref[ok] - A @ self.coef
        self.rs = np.median(np.abs(r - np.median(r))) * 1.4826 + 1e-9
        self.am = np.median(act)
        self.asd = np.median(np.abs(act - self.am)) * 1.4826 + 1e-9
        self.fm = np.median(F_ref, 0)
        self.fs = np.median(np.abs(F_ref - self.fm), 0) * 1.4826 + 1e-9

    def scores(self, F, S):
        act = F[:, 0]
        ok = np.isfinite(S)
        r = np.zeros(len(F))
        r[ok] = (S[ok] - (self.coef[0] + self.coef[1] * act[ok])) / self.rs
        a_plus = np.clip((act - self.am) / self.asd, 0, None)
        z_minus = np.clip(-r, 0, None)
        return {
            "Полный критерий": z_minus * (1 + a_plus),
            "V1: без веса активности": z_minus,
            "V2: двусторонний остаток": np.abs(r),
            "V3: без успеваемости": a_plus,
            "V4: MAD (все признаки)": np.clip((F - self.fm) / self.fs, 0, None).sum(1),
        }


def metrics(fl, lab):
    tp = (fl & lab).sum()
    prec = tp / fl.sum() if fl.sum() else np.nan
    rec = tp / lab.sum() if lab.sum() else np.nan
    f1 = 2 * prec * rec / (prec + rec) if (lab.sum() and fl.sum() and prec + rec > 0) else np.nan
    if lab.sum() and fl.sum() and tp == 0:
        f1 = 0.0
    fpr = (fl & ~lab).sum() / max((~lab).sum(), 1)
    return prec, rec, f1, fpr


def run(X, S, T_ref, p0_ref, pol, protocol, scorer=None, thresholds=None):
    valid = S[np.isfinite(S)]
    v_true = E.policy_value(E.build_mdp(T_ref), p0_ref, pol)
    rows = []
    for att_name, att in E.ATTACKS.items():
        for sh in SHARES:
            if protocol == "A" and sh == 0:
                continue
            for sd in range(SEEDS):
                rng = np.random.default_rng(500 + sd)
                n = int(len(X) * sh)
                Xp = np.vstack([X, att(X, n, rng)]) if n else X
                Sp = np.r_[S, rng.choice(valid, size=n)] if n else S
                lab = np.r_[np.zeros(len(X), bool), np.ones(n, bool)]
                Tp, p0p = E.estimate_model(E.to_levels(Xp))
                bias0 = (E.policy_value(E.build_mdp(Tp), p0p, pol) / v_true - 1) * 100
                rows.append((protocol, att_name, sh, sd, "без защиты",
                             np.nan, np.nan, np.nan, 0.0, np.linalg.norm(Tp - T_ref), bias0))
                F = E.features(Xp)
                sc = (scorer or Scorer(F, Sp)).scores(F, Sp)
                for v in VARIANTS:
                    if protocol == "A":
                        fl = E.top_k(sc[v], sh)
                    else:
                        fl = sc[v] > thresholds[v]
                    p, r, f1, fpr = metrics(fl, lab)
                    Tc, p0c = E.estimate_model(E.to_levels(Xp[~fl]))
                    bias = (E.policy_value(E.build_mdp(Tc), p0c, pol) / v_true - 1) * 100
                    rows.append((protocol, att_name, sh, sd, v, p, r, f1, fpr,
                                 np.linalg.norm(Tc - T_ref), bias))
    return rows


def main():
    os.makedirs(OUT, exist_ok=True)
    W = E.load_weekly(False)
    X = W.values
    S = E.load_scores(W, False)

    # --- протокол A: все данные, модель по полной выборке
    T_full, p0_full = E.estimate_model(E.to_levels(X))
    pols = []
    for sd in range(10):                      # как в E1: мода стратегий Q-learning
        rng = np.random.default_rng(sd)
        L = E.to_levels(X)
        Tb, p0b = E.estimate_model(L[rng.integers(len(L), size=len(L))])
        pols.append(E.train(E.build_mdp(Tb), p0b, "qlearning", seed=sd))
    pol_mode = np.array([np.bincount(np.array(pols)[:, s], minlength=E.N_ACTIONS).argmax()
                         for s in range(E.N_STATES)])
    rows = run(X, S, T_full, p0_full, pol_mode, "A")

    # --- протокол B: опорная статистика и пороги по 2013 г., проверка на 2014 г.
    pres = np.asarray(W.index.get_level_values(1).astype(str))
    cal, ev = np.char.startswith(pres.astype(str), "2013"), np.char.startswith(pres.astype(str), "2014")
    Fc = E.features(X[cal])
    scorer = Scorer(Fc, S[cal])
    sc_cal = scorer.scores(Fc, S[cal])
    thresholds = {v: np.quantile(sc_cal[v], 1 - FPR) for v in VARIANTS}
    # у записей без оценок балл критериев с успеваемостью = 0: порог не может быть 0
    for v in VARIANTS:
        if thresholds[v] <= 0:
            thresholds[v] = np.min(sc_cal[v][sc_cal[v] > 0])
    T_ev, p0_ev = E.estimate_model(E.to_levels(X[ev]))
    pol_ev = E.train(E.build_mdp(T_ev), p0_ev, "qlearning", seed=0)
    rows += run(X[ev], S[ev], T_ev, p0_ev, pol_ev, "B", scorer, thresholds)

    df = pd.DataFrame(rows, columns=["protocol", "attack", "share", "seed", "variant",
                                     "precision", "recall", "f1", "false_pos",
                                     "distortion", "bias_%"])
    df.to_csv(os.path.join(OUT, "R5_ablation.csv"), index=False)
    agg = df.groupby(["protocol", "attack", "share", "variant"], sort=False) \
            .mean(numeric_only=True).drop(columns="seed")
    lines = ["[R5] Абляция критерия согласованности (среднее по 5 повторам)",
             "A - известная доля атакующих (все данные); B - пороги на 5% ложных срабатываний",
             "по чистым данным 2013 г., проверка на 2014 г.", "",
             agg.round(3).to_string()]
    with open(os.path.join(OUT, "summary_R5.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
