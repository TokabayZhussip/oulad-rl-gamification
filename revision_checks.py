# -*- coding: utf-8 -*-
"""
Дополнительные проверки по замечаниям рецензента (запуск из папки репозитория,
рядом с oulad_experiments.py; данные OULAD в папке data/).

  R1  бандит vs Q-learning: парный тест, точные значения стратегий, объём обучения
  R2  разделение данных: модель для обучения агента и модель для проверки
      оцениваются по непересекающимся выборкам студентов (50/50) и по годам (2013 -> 2014)
  R3  детекторы без знания доли атакующих: пороги по робастной оценке баллов,
      включая сценарий без атаки (ложные срабатывания)
  R4  каскадная защита с условием включения уровня MAD

Запуск:
    python revision_checks.py          # реальные данные
    python revision_checks.py --demo   # синтетические данные (проверка кода)
Результаты: results_revision/summary_revision.txt и CSV-таблицы.
"""
import argparse
import os

import numpy as np
import pandas as pd

import oulad_experiments as E

OUT = "results_revision"
SHARES = [0.0, 0.05, 0.10, 0.20]
K_MAD = 3.5            # порог: балл > медиана + K * MAD(балла)
C_CON = 3.0            # порог критерия согласованности (в единицах робастной z-оценки остатка)
GATE_FACTOR = 2.0      # уровень MAD включается, если доля превышений порога больше
GATE_MIN = 0.01        # max(GATE_FACTOR x доли на чистых (исторических) данных, GATE_MIN)


def robust_threshold(score, k=K_MAD):
    med = np.median(score)
    mad = np.median(np.abs(score - med)) * 1.4826 + 1e-9
    return score > med + k * mad


def mad_score(F):
    med = np.median(F, 0)
    mad = np.median(np.abs(F - med), 0) * 1.4826 + 1e-9
    return np.clip((F - med) / mad, 0, None).sum(1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--demo", action="store_true")
    ap.add_argument("--seeds", type=int, default=10)
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    lines = []

    def say(s=""):
        print(s, flush=True)
        lines.append(s)

    W = E.load_weekly(args.demo)
    X = W.values
    SCORES = E.load_scores(W, args.demo)
    valid_scores = SCORES[np.isfinite(SCORES)]
    L = E.to_levels(X)
    T_full, p0_full = E.estimate_model(L)
    mdp_full = E.build_mdp(T_full)

    # ------------------------------------------------------------------ R1
    say("[R1] Бандит vs Q-learning")
    rows = []
    for sd in range(args.seeds):
        rng = np.random.default_rng(sd)
        idx = rng.integers(len(L), size=len(L))          # тот же бутстрап, что в E1
        T, p0 = E.estimate_model(L[idx])
        mdp = E.build_mdp(T)
        P, NP, PEN = mdp
        myopic = ((P * np.arange(E.N_LEVELS)).sum(-1) - PEN).argmax(1)
        opt = E.optimal_value(mdp, p0)
        rec = {"seed": sd, "DP": opt, "myopic_exact": E.policy_value(mdp, p0, myopic)}
        for algo in ["bandit", "qlearning", "sarsa"]:
            for nb in [300, 1200, 4800]:
                pol = E.train(mdp, p0, algo, seed=sd, batches=nb)
                rec[f"{algo}_{nb}_exact"] = E.policy_value(mdp, p0, pol)
                if nb == 300:
                    rec[f"{algo}_300_sim"] = E.evaluate(mdp, p0, pol, seed=1000 + sd)
        rows.append(rec)
    r1 = pd.DataFrame(rows)
    r1.to_csv(os.path.join(OUT, "R1_bandit_vs_q.csv"), index=False)
    d = r1["bandit_300_sim"] - r1["qlearning_300_sim"]
    say(f"  бандит - Q-learning (симуляция, 76 800 эпизодов): {d.mean():+.3f}, "
        f"бандит лучше в {(d > 0).sum()}, равны в {(d.abs() < 1e-9).sum()} из {len(d)}; "
        f"p = {E.perm_test_paired(d.values):.4f}")
    say(f"  близорукая стратегия = оптимум DP в {(np.abs(r1['myopic_exact'] - r1['DP']) < 1e-6).sum()} "
        f"из {len(r1)} бутстрап-моделей")
    for algo in ["bandit", "qlearning", "sarsa"]:
        say("  " + algo + ": " + "; ".join(
            f"{nb * 256} эпизодов -> {r1[f'{algo}_{nb}_exact'].mean():.3f} ± "
            f"{r1[f'{algo}_{nb}_exact'].std(ddof=1):.3f}" for nb in [300, 1200, 4800]))

    # ------------------------------------------------------------------ R2
    say()
    say("[R2] Разделение данных: обучение по одной выборке, проверка по другой")
    rows = []
    pres = W.index.get_level_values(1).astype(str)
    splits = []
    for sd in range(args.seeds):
        rng = np.random.default_rng(900 + sd)
        m = rng.random(len(L)) < 0.5
        splits.append(("студенты 50/50", sd, m, ~m))
    y13, y14 = pres.str.startswith("2013"), pres.str.startswith("2014")
    if y13.sum() > 0 and y14.sum() > 0:
        splits.append(("2013 -> 2014", 0, np.asarray(y13), np.asarray(y14)))
    for name, sd, tr, te in splits:
        Ttr, p0tr = E.estimate_model(L[tr])
        Tte, p0te = E.estimate_model(L[te])
        mtr, mte = E.build_mdp(Ttr), E.build_mdp(Tte)
        rec = {"split": name, "seed": sd,
               "DP_test": E.optimal_value(mte, p0te),
               "none": E.policy_value(mte, p0te, E.fixed(0)),
               "heuristic": E.policy_value(mte, p0te, E.HEURISTIC)}
        for algo in ["bandit", "qlearning"]:
            pol = E.train(mtr, p0tr, algo, seed=sd)
            rec[algo + "_test"] = E.policy_value(mte, p0te, pol)
            rec[algo + "_train"] = E.policy_value(mtr, p0tr, pol)
        rows.append(rec)
    r2 = pd.DataFrame(rows)
    r2.to_csv(os.path.join(OUT, "R2_split.csv"), index=False)
    for name, g in r2.groupby("split"):
        say(f"  {name} ({len(g)} разбиений): Q-learning на проверочной модели "
            f"{g['qlearning_test'].mean():.2f} (на обучающей {g['qlearning_train'].mean():.2f}), "
            f"бандит {g['bandit_test'].mean():.2f}, эвристика {g['heuristic'].mean():.2f}, "
            f"без геймификации {g['none'].mean():.2f}, оптимум {g['DP_test'].mean():.2f}; "
            f"Q-learning / оптимум = {100 * (g['qlearning_test'] / g['DP_test']).mean():.1f}%")

    # ------------------------------------------------------------------ R3, R4
    say()
    say("[R3/R4] Защита без знания доли атакующих")
    pol_ref = E.train(mdp_full, p0_full, "qlearning", seed=0)
    v_true = E.policy_value(mdp_full, p0_full, pol_ref)
    F0 = E.features(X)
    base_rate_mad = robust_threshold(mad_score(F0)).mean()     # «исторические» чистые данные
    say(f"  доля превышений порога MAD на чистых данных: {base_rate_mad:.3f}")
    rows = []
    for att_name, att in E.ATTACKS.items():
        for sh in SHARES:
            for sd in range(5):
                rng = np.random.default_rng(500 + sd)
                n_att = int(len(X) * sh)
                Xa = att(X, n_att, rng) if n_att else np.zeros((0, E.H), int)
                Xp = np.vstack([X, Xa])
                lab = np.r_[np.zeros(len(X), bool), np.ones(n_att, bool)]
                Sp = np.r_[SCORES, rng.choice(valid_scores, size=n_att)]
                F = E.features(Xp)
                s_mad = mad_score(F)
                s_con = E.consistency_score(F, {"scores": Sp})
                f_mad = robust_threshold(s_mad)
                f_con = s_con > C_CON
                gate = f_mad.mean() > max(GATE_FACTOR * base_rate_mad, GATE_MIN)
                variants = {
                    "без защиты": np.zeros(len(Xp), bool),
                    "MAD (порог)": f_mad,
                    "Согласованность (порог)": f_con,
                    "Объединение MAD или согл. (порог)": f_mad | f_con,
                    "Каскад: MAD по условию + согл.": (f_mad if gate else False) | f_con,
                    "Гибрид (известная доля, как в статье)":
                        E.defend_hybrid(F, sh, {"scores": Sp}) if sh > 0 else np.zeros(len(Xp), bool),
                }
                for vn, fl in variants.items():
                    fl = np.asarray(fl, bool)
                    Tc, p0c = E.estimate_model(E.to_levels(Xp[~fl]))
                    tp = (fl & lab).sum()
                    rows.append((att_name, sh, sd, vn, bool(gate), fl.mean(),
                                 tp / max(fl.sum(), 1) if fl.sum() else np.nan,
                                 tp / lab.sum() if lab.sum() else np.nan,
                                 (fl & ~lab).sum() / len(X),
                                 np.linalg.norm(Tc - T_full),
                                 (E.policy_value(E.build_mdp(Tc), p0c, pol_ref) / v_true - 1) * 100))
    r3 = pd.DataFrame(rows, columns=["attack", "share", "seed", "defense", "gate_on",
                                     "flagged_share", "precision", "recall",
                                     "false_pos_rate", "distortion", "forecast_bias_%"])
    r3.to_csv(os.path.join(OUT, "R3_R4_thresholds.csv"), index=False)
    agg = r3.groupby(["attack", "share", "defense"]).mean(numeric_only=True).drop(columns="seed")
    say(agg.round(3).to_string())
    with open(os.path.join(OUT, "summary_revision.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"\nГотово: {OUT}/summary_revision.txt")


if __name__ == "__main__":
    main()
