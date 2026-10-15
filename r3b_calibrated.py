# R3b: пороги, откалиброванные по чистым историческим данным (2013) на заданную долю
# ложных срабатываний; атака и оценка - на презентациях 2014 г. Доля атакующих не используется.
import numpy as np, pandas as pd, oulad_experiments as E
from revision_checks import mad_score
W = E.load_weekly(False); X = W.values; S = E.load_scores(W, False)
pres = W.index.get_level_values(1).astype(str)
cal, ev = np.asarray(pres.str.startswith("2013")), np.asarray(pres.str.startswith("2014"))
Xc, Sc, Xe, Se = X[cal], S[cal], X[ev], S[ev]
Fc = E.features(Xc)
sc_mad = mad_score(Fc); sc_con = E.consistency_score(Fc, {"scores": Sc})
T_ev, p0_ev = E.estimate_model(E.to_levels(Xe))
pol = E.train(E.build_mdp(T_ev), p0_ev, "qlearning", seed=0)
v_true = E.policy_value(E.build_mdp(T_ev), p0_ev, pol)
valid = Se[np.isfinite(Se)]
rows = []
for fpr in [0.01, 0.05]:
    th_mad, th_con = np.quantile(sc_mad, 1 - fpr), np.quantile(sc_con, 1 - fpr)
    for att_name, att in E.ATTACKS.items():
        for sh in [0.0, 0.05, 0.10, 0.20]:
            for sd in range(5):
                rng = np.random.default_rng(500 + sd)
                n = int(len(Xe) * sh)
                Xp = np.vstack([Xe, att(Xe, n, rng) if n else np.zeros((0, E.H), int)])
                lab = np.r_[np.zeros(len(Xe), bool), np.ones(n, bool)]
                Sp = np.r_[Se, rng.choice(valid, size=n)]
                F = E.features(Xp)
                f_mad = mad_score(F) > th_mad
                f_con = E.consistency_score(F, {"scores": Sp}) > th_con
                gate = f_mad.mean() > 2 * fpr
                V = {"без защиты": np.zeros(len(Xp), bool), "MAD": f_mad, "Согласованность": f_con,
                     "Объединение": f_mad | f_con, "Каскад": (f_mad if gate else np.zeros(len(Xp), bool)) | f_con}
                for vn, fl in V.items():
                    Tc, p0c = E.estimate_model(E.to_levels(Xp[~fl]))
                    tp = (fl & lab).sum()
                    rows.append((fpr, att_name, sh, sd, vn, gate, fl.mean(),
                                 tp / fl.sum() if fl.sum() else np.nan, tp / n if n else np.nan,
                                 (fl & ~lab).sum() / len(Xe), np.linalg.norm(Tc - T_ev),
                                 (E.policy_value(E.build_mdp(Tc), p0c, pol) / v_true - 1) * 100))
df = pd.DataFrame(rows, columns=["fpr_target", "attack", "share", "seed", "defense", "gate_on", "flagged",
                                 "precision", "recall", "false_pos", "distortion", "bias_%"])
df.to_csv("results_revision/R3b_calibrated.csv", index=False)
pd.set_option("display.width", 200)
agg = df.groupby(["fpr_target", "attack", "share", "defense"]).mean(numeric_only=True).drop(columns="seed")
print(f"калибровка: {cal.sum()} записей 2013 г., оценка: {ev.sum()} записей 2014 г.")
print(agg.round(3).to_string())
open("results_revision/summary_R3b.txt", "w").write(agg.round(3).to_string())
