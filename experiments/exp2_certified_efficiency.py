"""Experiment 2 -- Certified efficiency: which compression should you deploy?

Under a certificate the real cost of a compressed model is not its size but
    certified_cost = coverage * size_C + (1 - coverage) * size_L,
because every abstention is answered by the large model. We sweep each
compression family finely, certify at alpha in {0.01, 0.02, 0.05, 0.10}
(delta = 0.1, n_cal = 400) and compare three model-selection rules:
  (a) smallest model within 1 accuracy point of the teacher,
  (b) highest in-distribution agreement,
  (c) lowest certified cost.

Hypothesis H2: (a) and (c) pick different models; aggressive compression
that looks fine on accuracy has low certified coverage and ends up costlier
than a milder compression, and quantisation dominates the certified frontier.
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from common import load_task, train_teacher, disagreement, split_pool, save_csv, fig_path
from tec.compress import magnitude_prune, quantize, low_rank, distill, size_bits
from tec.certify import SelectiveCertificate, margin_score, selective_risk, certified_cost

DELTA, N_CAL = 0.10, 400
rows = []
for task in ["digits", "toxic"]:
    for seed in range(3):
        (Xtr, ytr, _), (Xpo, ypo, _) = load_task(task, seed)
        T = train_teacher(task, Xtr, ytr, seed)
        base = size_bits(T)
        accT = float((T.predict(Xpo) == ypo).mean())
        grid = []
        for s in [0.5, 0.7, 0.8, 0.85, 0.9, 0.93, 0.95]:
            m = magnitude_prune(T, s); grid.append(("prune", s, m, size_bits(m) / base))
        for b in [8, 5, 4, 3, 2]:
            m = quantize(T, b); grid.append(("quant", b, m, size_bits(m, bits=b) / base))
        for r in [0.5, 0.3, 0.2, 0.1]:
            m = low_rank(T, r); grid.append(("lowrank", r, m, size_bits(m) / base))
        for h in [128, 64, 32, 16, 8]:
            m = distill(T, Xtr, ytr, [Xtr.shape[1], h, h, int(ytr.max()) + 1], seed=seed, epochs=25)
            grid.append(("distill", h, m, size_bits(m) / base))
        rng = np.random.default_rng(seed)
        ca, te = split_pool(len(Xpo), N_CAL, rng)
        for fam, knob, C, ratio in grid:
            s_all = margin_score(C, Xpo); d_all = disagreement(T, C, Xpo)
            accC = float((C.predict(Xpo[te]) == ypo[te]).mean())
            for alpha in [0.01, 0.02, 0.05, 0.10]:
                cert = SelectiveCertificate(alpha, DELTA).fit(s_all[ca], d_all[ca])
                sm = cert.serve_mask(s_all[te])
                cov = float(sm.mean())
                rows.append({"task": task, "seed": seed, "family": fam, "knob": knob, "size_ratio": ratio,
                             "alpha": alpha, "acc_T": accT, "acc_C": accC, "acc_gap": accT - accC,
                             "agreement": 1 - float(d_all[te].mean()), "coverage": cov,
                             "test_risk": selective_risk(sm, d_all[te]), "certified_cost": certified_cost(cov, ratio)})
        print(task, seed, "done")

res = pd.DataFrame(rows)
save_csv(res, "exp2_certified_efficiency.csv")
summ = res.groupby(["task", "family", "knob", "alpha"]).mean(numeric_only=True).drop(columns="seed").reset_index()
save_csv(summ, "exp2_summary.csv")

# model-selection rules
sel = []
for (task, alpha), g in summ.groupby(["task", "alpha"]):
    a = g[g.acc_gap <= 0.01].sort_values("size_ratio").iloc[0] if (g.acc_gap <= 0.01).any() else g.sort_values("acc_gap").iloc[0]
    b = g.sort_values("agreement", ascending=False).iloc[0]
    c = g.sort_values("certified_cost").iloc[0]
    for rule, r in [("min size | acc within 1pt", a), ("max agreement", b), ("min certified cost", c)]:
        sel.append({"task": task, "alpha": alpha, "rule": rule, "pick": f"{r.family}-{r.knob}", "size_ratio": r.size_ratio,
                    "acc_gap": r.acc_gap, "agreement": r.agreement, "coverage": r.coverage, "certified_cost": r.certified_cost})
sel = pd.DataFrame(sel); save_csv(sel, "exp2_model_selection.csv")
pd.set_option("display.width", 220)
print(sel.round(3).to_string())

fig, axes = plt.subplots(1, 2, figsize=(11, 4))
for ax, task in zip(axes, ["digits", "toxic"]):
    g = summ[(summ.task == task) & (summ.alpha == 0.05)]
    for fam, c in [("prune", "tab:blue"), ("quant", "tab:orange"), ("lowrank", "tab:green"), ("distill", "tab:red")]:
        gg = g[g.family == fam].sort_values("size_ratio")
        ax.plot(gg.size_ratio, gg.certified_cost, marker="o", ms=3, color=c, label=fam)
    ax.plot([0.005, 1], [0.005, 1], ls=":", color="gray", label="size (no certificate)")
    ax.axhline(1, color="k", lw=0.6)
    ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlabel("model size ratio"); ax.set_ylabel("certified cost (alpha=0.05)")
    ax.set_title(task)
axes[0].legend(fontsize=7)
fig.suptitle("Exp2: nominal size vs certified inference cost")
fig.tight_layout(); fig.savefig(fig_path("exp2_certified_cost.png"), dpi=130)
print("done")
