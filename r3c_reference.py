# R3c: медиана/MAD признаков и регрессия «оценка ~ активность» берутся из чистых данных 2013 г.,
# пороги - на 1% / 5% ложных срабатываний на них же; атака и оценка на 2014 г.
import numpy as np, pandas as pd, oulad_experiments as E
W = E.load_weekly(False); X = W.values; S = E.load_scores(W, False)
pres = W.index.get_level_values(1).astype(str)
cal, ev = np.asarray(pres.str.startswith("2013")), np.asarray(pres.str.startswith("2014"))
Fc = E.features(X[cal]); Sc = S[cal]
med = np.median(Fc, 0); mad = np.median(np.abs(Fc - med), 0) * 1.4826 + 1e-9
def s_mad(F): return np.clip((F - med) / mad, 0, None).sum(1)
ok = np.isfinite(Sc); A = np.c_[np.ones(ok.sum()), Fc[ok, 0]]
coef, *_ = np.linalg.lstsq(A, Sc[ok], rcond=None)
res_c = Sc[ok] - A @ coef; rs = np.median(np.abs(res_c - np.median(res_c))) * 1.4826
am = np.median(Fc[:, 0]); asd = np.median(np.abs(Fc[:, 0] - am)) * 1.4826
def s_con(F, Sx):
    r = np.zeros(len(F)); m = np.isfinite(Sx)
    r[m] = Sx[m] - (coef[0] + coef[1] * F[m, 0])
    return np.clip(-r / rs, 0, None) * (1 + np.clip((F[:, 0] - am) / asd, 0, None))
cm, cc = s_mad(Fc), s_con(Fc, Sc)
Xe, Se = X[ev], S[ev]; valid = Se[np.isfinite(Se)]
T_ev, p0_ev = E.estimate_model(E.to_levels(Xe))
pol = E.train(E.build_mdp(T_ev), p0_ev, "qlearning", seed=0)
v_true = E.policy_value(E.build_mdp(T_ev), p0_ev, pol)
rows = []
for fpr in [0.01, 0.05]:
    tm, tcn = np.quantile(cm, 1 - fpr), np.quantile(cc, 1 - fpr)
    for att_name, att in E.ATTACKS.items():
        for sh in [0.0, 0.05, 0.10, 0.20]:
            for sd in range(5):
                rng = np.random.default_rng(500 + sd); n = int(len(Xe) * sh)
                Xp = np.vstack([Xe, att(Xe, n, rng) if n else np.zeros((0, E.H), int)])
                lab = np.r_[np.zeros(len(Xe), bool), np.ones(n, bool)]
                Sp = np.r_[Se, rng.choice(valid, size=n)]
                F = E.features(Xp)
                f_mad, f_con = s_mad(F) > tm, s_con(F, Sp) > tcn
                gate = f_mad.mean() > 2 * fpr
                V = {"без защиты": np.zeros(len(Xp), bool), "MAD": f_mad, "Согласованность": f_con,
                     "Объединение": f_mad | f_con, "Каскад": (f_mad if gate else np.zeros(len(Xp), bool)) | f_con}
                for vn, fl in V.items():
                    Tc, p0c = E.estimate_model(E.to_levels(Xp[~fl])); tp = (fl & lab).sum()
                    rows.append((fpr, att_name, sh, sd, vn, gate, fl.mean(),
                                 tp / fl.sum() if fl.sum() else np.nan, tp / n if n else np.nan,
                                 (fl & ~lab).sum() / len(Xe), np.linalg.norm(Tc - T_ev),
                                 (E.policy_value(E.build_mdp(Tc), p0c, pol) / v_true - 1) * 100))
df = pd.DataFrame(rows, columns=["fpr_target", "attack", "share", "seed", "defense", "gate_on", "flagged",
                                 "precision", "recall", "false_pos", "distortion", "bias_%"])
df.to_csv("results_revision/R3c_reference.csv", index=False)
pd.set_option("display.width", 200)
agg = df.groupby(["fpr_target", "attack", "share", "defense"]).mean(numeric_only=True).drop(columns="seed")
print(agg.round(3).to_string())
open("results_revision/summary_R3c.txt", "w").write(agg.round(3).to_string())
