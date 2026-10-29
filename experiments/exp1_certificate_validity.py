"""Experiment 1 -- Is the selective disagreement certificate valid, and what
does it cost?

For each task, teacher seed and compression operator we draw 200 random
calibration/test splits of the held-out pool. On each we fit
  * the certified threshold (LTT, alpha, delta),
  * the naive threshold (no correction),
and evaluate the *selective disagreement risk* on the test split. The
certificate claims P(risk > alpha) <= delta; the naive method claims nothing.
We also report coverage (fraction served by C) and whether a marginal
(no-abstention) Clopper-Pearson bound could certify alpha at all.

Hypothesis H1: certified violation rate <= delta (0.10) in every setting,
naive violation rate substantially above delta for small calibration sets;
cost of the guarantee = coverage loss of a few points, shrinking with n_cal.
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from common import load_task, train_teacher, compress_all, disagreement, split_pool, save_csv, fig_path
from tec.certify import SelectiveCertificate, NaiveThreshold, margin_score, selective_risk, hoeffding_marginal_bound

ALPHA, DELTA, N_SPLITS = 0.05, 0.10, 200
rows = []
for task in ["digits", "toxic"]:
    for seed in range(3):
        (Xtr, ytr, _), (Xpo, ypo, _) = load_task(task, seed)
        T = train_teacher(task, Xtr, ytr, seed)
        comps, sizes = compress_all(T, Xtr, ytr, task, seed)
        for name, C in comps.items():
            s_all = margin_score(C, Xpo)
            d_all = disagreement(T, C, Xpo)
            for n_cal in ([100, 200, 400] if task == "digits" else [100, 200, 400, 1000]):
                rng = np.random.default_rng(seed)
                for r in range(N_SPLITS):
                    ca, te = split_pool(len(Xpo), n_cal, rng)
                    cert = SelectiveCertificate(ALPHA, DELTA).fit(s_all[ca], d_all[ca])
                    naive = NaiveThreshold(ALPHA, DELTA).fit(s_all[ca], d_all[ca])
                    for mname, m in [("certified", cert), ("naive", naive)]:
                        sm = m.serve_mask(s_all[te])
                        rows.append({"task": task, "seed": seed, "method": name, "size_ratio": sizes[name],
                                     "n_cal": n_cal, "split": r, "certifier": mname, "feasible": m.feasible,
                                     "coverage": float(sm.mean()), "test_risk": selective_risk(sm, d_all[te]),
                                     "violation": float(selective_risk(sm, d_all[te]) > ALPHA),
                                     "marginal_cp_bound": hoeffding_marginal_bound(d_all[ca], DELTA),
                                     "marginal_disagreement": float(d_all[te].mean())})
            print(task, seed, name, "pool disagreement", round(d_all.mean(), 3))

res = pd.DataFrame(rows)
save_csv(res, "exp1_certificate_validity.csv")
summ = res.groupby(["task", "method", "n_cal", "certifier"]).agg(
    violation_rate=("violation", "mean"), coverage=("coverage", "mean"), test_risk=("test_risk", "mean"),
    feasible=("feasible", "mean"), marginal_cp_bound=("marginal_cp_bound", "mean"),
    marginal_disagreement=("marginal_disagreement", "mean")).reset_index()
save_csv(summ, "exp1_summary.csv")
pd.set_option("display.width", 220)
print(summ.round(3).to_string())

fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
for ax, task in zip(axes, ["digits", "toxic"]):
    s = summ[(summ.task == task)]
    for cert, c, mk in [("certified", "tab:red", "o"), ("naive", "tab:blue", "x")]:
        ss = s[s.certifier == cert].groupby("n_cal").violation_rate.mean()
        ax.plot(ss.index, ss.values, marker=mk, color=c, label=cert)
    ax.axhline(DELTA, ls="--", color="k", lw=0.8, label="delta")
    ax.set_xscale("log"); ax.set_xlabel("calibration size"); ax.set_ylabel("P(test risk > alpha)"); ax.set_title(task)
axes[0].legend()
fig.suptitle("Exp1: certificate validity (alpha=0.05, delta=0.10)")
fig.tight_layout(); fig.savefig(fig_path("exp1_validity.png"), dpi=130)
print("done")
