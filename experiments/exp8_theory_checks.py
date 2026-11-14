"""Experiment 8 -- numerical checks of the theory (docs/theory.md).

  (a) Prop 2: expected audit delay D(r) (formula) vs the simulated mean delay of
      ShadowAuditMartingale on streams with post-change disagreement rate r,
      audit budget b, coverage c = 1. Writes results/exp8_delay_check.csv (+ png).
  (b) Thm 1: validity vs n_cal on the real-model arrays is already computed in
      exp6 (results/real/exp6a_summary.csv); this script only points there.
  (c) Cor 3: simultaneous violation over groups for GroupCertificate at delta and
      delta/|G| on the real-model arrays (results/real/real_preds.npz). Skipped when
      that file is missing. Writes results/exp8_group_union.csv.
"""
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
sys.path.insert(0, ROOT)
from tec.certify import GroupCertificate, SelectiveCertificate, selective_risk  # noqa: E402
from tec.monitor import ShadowAuditMartingale, expected_audit_delay  # noqa: E402

RES = os.path.join(ROOT, "results")
REAL = os.path.join(RES, "real")
ALPHA, DELTA = 0.05, 0.10
R_GRID = [0.075, 0.10, 0.15, 0.20, 0.30]
B_GRID = [0.02, 0.05, 0.10, 0.20]
N_REP = 300
HORIZON_MULT = 20
COMPRESSED = ["int8", "q4", "q3", "p30", "p50", "p70"]
MAIN_GROUPS = ["African", "Islam", "Jewish", "Homosexual", "Women"]


def simulate_delay(r, budget, rep_seed, cap):
    """Updates until alarm for one stream; a stream that never alarms counts at cap."""
    mon = ShadowAuditMartingale(ALPHA, DELTA, budget, seed=rep_seed)
    d_rng = np.random.default_rng(10_000 + rep_seed)
    for t in range(1, cap + 1):
        if mon.update(lambda: d_rng.random() < r):
            return t
    return cap


def part_a():
    rows = []
    for r in R_GRID:
        for b in B_GRID:
            formula = expected_audit_delay(r, ALPHA, DELTA, b)
            cap = int(HORIZON_MULT * formula)
            sims = np.array([simulate_delay(r, b, k, cap) for k in range(N_REP)], dtype=float)
            n_capped = int((sims >= cap).sum())
            rows.append({"r": r, "budget": b, "formula": formula, "sim_mean": sims.mean(),
                         "sim_median": float(np.median(sims)), "ratio": sims.mean() / formula,
                         "n_capped": n_capped})
            print(f"r={r:.3f} b={b:.2f} formula={formula:10.1f} sim_mean={sims.mean():10.1f} "
                  f"ratio={sims.mean() / formula:.3f} capped={n_capped}")
    df = pd.DataFrame(rows)
    df.drop(columns="n_capped").to_csv(os.path.join(RES, "exp8_delay_check.csv"), index=False)
    print("[saved] results/exp8_delay_check.csv")
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed; skipping results/exp8_delay_check.png")
        return
    fig, ax = plt.subplots(figsize=(6, 4.5))
    for b in B_GRID:
        g = df[df.budget == b]
        ax.plot(g.r, g.formula, "-", label=f"formula b={b}")
        ax.plot(g.r, g.sim_mean, "o", label=f"simulated b={b}")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("post-change disagreement rate r"); ax.set_ylabel("expected delay (served inputs)")
    ax.set_title("Prop 2: audit-martingale delay (alpha=0.05, delta=0.10, c=1)")
    ax.legend(fontsize=7)
    fig.tight_layout(); fig.savefig(os.path.join(RES, "exp8_delay_check.png"), dpi=130)
    print("[saved] results/exp8_delay_check.png")


def first_group(g):
    for t in MAIN_GROUPS:
        if t in g.split("|"):
            return MAIN_GROUPS.index(t)
    return len(MAIN_GROUPS)  # "other / none"


def part_c():
    path = os.path.join(REAL, "real_preds.npz")
    if not os.path.exists(path):
        print("skip (c): real_preds.npz missing")
        return
    Z = np.load(path, allow_pickle=False)
    split = Z["split"]
    idx = np.where(np.isin(split, ["val", "test"]))[0]
    predL = Z["prob_L"].argmax(1)
    G = np.array([first_group(g) for g in Z["groups"]])
    n_groups = len(MAIN_GROUPS) + 1
    GNAMES = MAIN_GROUPS + ["other"]
    deltas = {"delta": DELTA, "delta/G": DELTA / n_groups}
    rows = []
    n_cal, n_split = 2000, 300
    for c in COMPRESSED:
        P = Z["prob_" + c]
        s_all = np.sort(P, 1); s_all = s_all[:, -1] - s_all[:, -2]
        d_all = (P.argmax(1) != predL).astype(int)
        s, d, g = s_all[idx], d_all[idx], G[idx]
        viol = {(cert, dn): 0 for cert in ("marginal", "group") for dn in deltas}
        rng = np.random.default_rng(1)
        # eval="population": i.i.d. draws with replacement from the pool; risks are exact over
        # the whole pool (no held-out test noise), so the union-bound check is not confounded.
        for _ in range(n_split):
            ca = rng.integers(0, len(s), n_cal)
            for dn, dl in deltas.items():
                marg = SelectiveCertificate(ALPHA, dl).fit(s[ca], d[ca])
                grp = GroupCertificate(ALPHA, dl).fit(s[ca], d[ca], g[ca])
                for cert, sm in (("marginal", marg.serve_mask(s)), ("group", grp.serve_mask(s, g))):
                    any_viol = any(selective_risk(sm[g == gi], d[g == gi]) > ALPHA
                                   for gi in range(n_groups) if (g == gi).any())
                    viol[(cert, dn)] += int(any_viol)
        for (cert, dn), k in viol.items():
            dl = deltas[dn]
            rows.append({"model": c, "certificate": cert, "delta_label": dn, "delta": dl,
                         "n_groups": n_groups, "n_splits": n_split, "eval": "population",
                         "simultaneous_violation": k / n_split,
                         "union_bound": 1 - (1 - dl) ** n_groups})
        print(c, {f"{cert}/{dn}": round(k / n_split, 3) for (cert, dn), k in viol.items()})
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(RES, "exp8_group_union.csv"), index=False)
    print("[saved] results/exp8_group_union.csv")
    print("groups:", GNAMES)


def main():
    os.makedirs(RES, exist_ok=True)
    part_a()
    print("(b) Thm 1 validity vs n_cal: see results/real/exp6a_summary.csv (computed by exp6, not recomputed here)")
    part_c()


if __name__ == "__main__":
    main()
