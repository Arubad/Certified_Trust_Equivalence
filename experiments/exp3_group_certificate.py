"""Experiment 3 -- Group-conditional certificates (fairness of trust).

Toxicity task with identity groups (none / A / B) and a training-time
spurious correlation between group-B terms and the toxic label. A *marginal*
certificate guarantees disagreement <= alpha on average; we ask whether it
holds for group B, and compare with a Mondrian (per-group) certificate.

Hypothesis H3: the marginal certificate is violated on group B far more often
than delta for compressions that concentrate silent flips on B (one-shot
pruning), while the Mondrian certificate holds per group at a modest
coverage cost on B.
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from common import load_task, train_teacher, disagreement, split_pool, save_csv, fig_path
from tec.compress import magnitude_prune, prune_finetune, quantize, distill
from tec.certify import SelectiveCertificate, GroupCertificate, margin_score, selective_risk

ALPHA, DELTA, N_CAL, N_SPLITS = 0.05, 0.10, 600, 100
rows = []
for spurious in [0.5, 0.8, 0.95]:
    for seed in range(3):
        (Xtr, ytr, gtr), (Xpo, ypo, gpo) = load_task("toxic", seed, spurious=spurious)
        T = train_teacher("toxic", Xtr, ytr, seed)
        comps = {"prune95": magnitude_prune(T, 0.95), "prune90": magnitude_prune(T, 0.9),
                 "pruneFT95": prune_finetune(T, 0.95, Xtr, ytr, seed=seed),
                 "quant3b": quantize(T, 3), "distill10": distill(T, Xtr, ytr, [Xtr.shape[1], 25, 25, 2], seed=seed)}
        for name, C in comps.items():
            s_all = margin_score(C, Xpo); d_all = disagreement(T, C, Xpo)
            rng = np.random.default_rng(seed)
            for r in range(N_SPLITS):
                ca, te = split_pool(len(Xpo), N_CAL, rng)
                marg = SelectiveCertificate(ALPHA, DELTA).fit(s_all[ca], d_all[ca])
                mond = GroupCertificate(ALPHA, DELTA).fit(s_all[ca], d_all[ca], gpo[ca])
                sm_m = marg.serve_mask(s_all[te]); sm_g = mond.serve_mask(s_all[te], gpo[te])
                for cname, sm in [("marginal", sm_m), ("mondrian", sm_g)]:
                    row = {"spurious": spurious, "seed": seed, "method": name, "split": r, "certifier": cname,
                           "coverage_all": float(sm.mean()), "risk_all": selective_risk(sm, d_all[te]),
                           "unc_disagreement_all": float(d_all[te].mean())}
                    for g, gn in [(0, "none"), (1, "A"), (2, "B")]:
                        mg = gpo[te] == g
                        row[f"coverage_{gn}"] = float(sm[mg].mean())
                        row[f"risk_{gn}"] = selective_risk(sm & mg, d_all[te])
                        row[f"viol_{gn}"] = float(selective_risk(sm & mg, d_all[te]) > ALPHA)
                        row[f"unc_disagreement_{gn}"] = float(d_all[te][mg].mean())
                    rows.append(row)
        print("spurious", spurious, "seed", seed, "done")

res = pd.DataFrame(rows)
save_csv(res, "exp3_group_certificate.csv")
summ = res.groupby(["spurious", "method", "certifier"]).mean(numeric_only=True).drop(columns=["seed", "split"]).reset_index()
save_csv(summ, "exp3_summary.csv")
pd.set_option("display.width", 250)
print(summ[["spurious", "method", "certifier", "unc_disagreement_A", "unc_disagreement_B", "coverage_A", "coverage_B",
            "risk_A", "risk_B", "viol_A", "viol_B"]].round(3).to_string())

fig, ax = plt.subplots(figsize=(8, 3.8))
s = summ[summ.spurious == 0.8]
x = np.arange(len(s.method.unique())); w = 0.2
for i, (cert, grp, c) in enumerate([("marginal", "A", "tab:blue"), ("marginal", "B", "tab:red"),
                                     ("mondrian", "A", "lightblue"), ("mondrian", "B", "salmon")]):
    ss = s[s.certifier == cert].set_index("method").loc[sorted(s.method.unique())]
    ax.bar(x + (i - 1.5) * w, ss[f"viol_{grp}"], w, color=c, label=f"{cert} / group {grp}")
ax.axhline(DELTA, ls="--", color="k", lw=0.8); ax.set_xticks(x); ax.set_xticklabels(sorted(s.method.unique()))
ax.set_ylabel("P(group risk > alpha)"); ax.legend(fontsize=7); ax.set_title("Exp3: certificate violation per identity group (spurious=0.8)")
fig.tight_layout(); fig.savefig(fig_path("exp3_group_violation.png"), dpi=130)
print("done")
