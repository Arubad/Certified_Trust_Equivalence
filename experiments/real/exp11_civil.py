"""Experiment 11 -- CivilComments: validity, groups, temporal drift, monitors, streaming.

Input: results/real/civil_preds.npz written by experiments/real/colab_civil.py.
Rows: civil_cal (calibration pool, population mode) then civil_stream (2017 Q1-Q4, time-ordered).
  C score  = max class probability of C (two classes: margin = 2 * maxprob - 1, same serve sets as Exp 6)
  d        = 1[C(x) != L(x)] (argmax disagreement with the large model L)

  (a) validity + cost on civil_cal, population mode, certified (same / split) and naive
  (b) marginal vs group (Mondrian) certificate per WILDS identity group (civil_group)
  (c) risk per quarter of the stream for a certificate fitted on civil_cal, with Prop 2's predicted
      audit delay per quarter (expected_audit_delay) and the first quarter whose risk exceeds alpha
  (d) monitors on the time-ordered stream and on an in-domain null stream (draws from civil_cal):
      first alarm, delay vs the change quarter, audits
  (e) streaming policy (static, audit 10%, adaptive) with recalibration (Exp 6d rule)

CLI:
  python exp11_civil.py [--npz PATH] [--out DIR] [--models a,b] [--reps 300] [--mon-reps 20]
  python exp11_civil.py --plot-only [--out DIR]     # figure from the CSVs (needs matplotlib)
"""
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", ".."))
import exp6_real_models as ex  # noqa: E402
from tec.certify import SelectiveCertificate, NaiveThreshold, GroupCertificate, selective_risk, certified_cost  # noqa: E402
from tec.monitor import (DeferralRateCUSUM, ShadowAuditMartingale, ScoreKS,  # noqa: E402
                         AdaptiveAuditMartingale, expected_audit_delay)

ALPHA = 0.05
DELTA = ex.DELTA
ALPHAS_A = [0.01, 0.02, 0.05]
NCALS_A = [200, 500, 1000]
N_CAL_B = 2000
MODELS = ["q8", "q4", "p30", "p50", "distil6", "p50ft"]
BUDGETS = {"audit2": 0.02, "audit5": 0.05, "audit10": 0.10, "audit20": 0.20}
MON_NAMES = ["audit2", "audit5", "audit10", "audit20", "adaptive", "ks", "cusum"]
POLICIES = [("static", 0.0), ("audit10", 0.10), ("adaptive", 0.0)]
# minority identities first (Review Focus 5); a comment is assigned to its first match
GROUP_ORDER = ["muslim", "black", "LGBTQ", "white", "christian", "other_religions", "female", "male"]
GROUP_NAMES = GROUP_ORDER + ["none"]
DEFAULT_NPZ = os.path.join(HERE, "..", "..", "results", "real", "civil_preds.npz")
DEFAULT_OUT = os.path.join(HERE, "..", "..", "results", "real")


# ------------------------------------------------------------------ data --
def civil_group(groups_str):
    """Group of a comment from its '|'-joined identity string (first match in GROUP_ORDER, else 'none')."""
    ids = set(groups_str.split("|")) if groups_str else set()
    for g in GROUP_ORDER:
        if g in ids:
            return g
    return "none"


def quarter_of(days):
    """Calendar quarter label ('2017Q1') of int days since 1970-01-01."""
    t = pd.DatetimeIndex(pd.to_datetime(np.asarray(days).astype(np.int64), unit="D"))
    return np.asarray(t.to_period("Q").astype(str))


def load_npz(path):
    with np.load(path, allow_pickle=False) as F:
        return {k: F[k] for k in F.files}


def prepare(Z):
    """Row indices of the calibration pool (file order) and of the stream, sorted by date (stable)."""
    cal = np.where(Z["split"] == "civil_cal")[0]
    st = np.where(Z["split"] == "civil_stream")[0]
    st = st[np.argsort(Z["date"][st], kind="stable")]
    return cal, st


def scores_and_disagreement(Z, models):
    predL = Z["prob_L"].argmax(1)
    S, D = {}, {}
    for c in models:
        P = Z["prob_" + c]
        S[c] = P.max(1).astype(float)
        D[c] = (P.argmax(1) != predL).astype(int)
    return S, D


def degenerate(cert, s_cal):
    """Infeasible, or serving under 5% of calibration (Exp 6c): its monitor rows are skipped."""
    return (not cert.feasible) or cert.coverage(s_cal) < 0.05


def _variants_a():
    return [("certified", lambda a: SelectiveCertificate(a, DELTA, mode="same")),
            ("certified_split", lambda a: SelectiveCertificate(a, DELTA, mode="split")),
            ("naive", lambda a: NaiveThreshold(a, DELTA))]


# ------------------------------------------------------------ (a) validity --
def part_a(S, D, meta, models, cal, reps):
    rows = []
    for c in models:
        s, d = S[c][cal], D[c][cal]
        for alpha in ALPHAS_A:
            for n_cal in NCALS_A:
                rng = np.random.default_rng(0)
                for _ in range(reps):
                    ca = rng.integers(0, len(s), n_cal)
                    marg_cp = ex.hoeffding_marginal_bound(d[ca], DELTA)
                    for nm, mk in _variants_a():
                        m = mk(alpha).fit(s[ca], d[ca])
                        risk, cov = ex.population_eval(m, s, d)
                        rows.append({"model": c, "alpha": alpha, "n_cal": n_cal, "certifier": nm,
                                     "coverage": cov, "risk": risk, "violation": float(risk > alpha),
                                     "feasible": m.feasible, "marginal_cp": marg_cp})
    a = pd.DataFrame(rows)
    sa = a.groupby(["model", "alpha", "n_cal", "certifier"]).agg(
        violation=("violation", "mean"), coverage=("coverage", "mean"), risk=("risk", "mean"),
        feasible=("feasible", "mean"), marginal_cp=("marginal_cp", "mean")).reset_index()
    sa["size"] = sa.model.map(lambda c: meta[c]["size"])
    sa["cost_size"] = certified_cost(sa.coverage, sa["size"])
    lat_L = meta["L"]["per_1k"]
    lat = sa.model.map(lambda c: meta[c]["per_1k"] / lat_L if c in ex.LATENCY_MODELS else np.nan)
    sa["cost_latency"] = np.where(sa.model.isin(ex.LATENCY_MODELS), certified_cost(sa.coverage, lat), np.nan)
    return sa


# ------------------------------------------------------------ (b) groups --
def part_b(S, D, models, cal, G, reps, n_cal=N_CAL_B):
    """G: group index (into GROUP_NAMES) per row. Population mode on civil_cal, n_cal draws."""
    rows = []
    gnames = GROUP_NAMES
    for c in models:
        s, d, g = S[c][cal], D[c][cal], G[cal]
        rng = np.random.default_rng(1)
        for _ in range(reps):
            ca = rng.integers(0, len(s), n_cal)
            n_draw = np.bincount(g[ca], minlength=len(gnames))
            marg = SelectiveCertificate(ALPHA, DELTA).fit(s[ca], d[ca])
            grp = GroupCertificate(ALPHA, DELTA).fit(s[ca], d[ca], g[ca])
            sm_marg, sm_grp = marg.serve_mask(s), grp.serve_mask(s, g)
            for nm, sm in [("marginal", sm_marg), ("group", sm_grp)]:
                for gi, gn in enumerate(gnames):
                    mg = g == gi
                    if not mg.any():
                        continue
                    if nm == "marginal":
                        feas = marg.feasible
                    else:
                        feas = grp.certs[gi].feasible if gi in grp.certs else False
                    rows.append({"model": c, "certificate": nm, "group": gn, "coverage": float(sm[mg].mean()),
                                 "risk": selective_risk(sm[mg], d[mg]), "feasible": feas,
                                 "violation": float(selective_risk(sm[mg], d[mg]) > ALPHA),
                                 "n_draw": int(n_draw[gi]), "n_pool": int(mg.sum()),
                                 "unc_dis": float(d[mg].mean())})
    b = pd.DataFrame(rows)
    return b.groupby(["model", "certificate", "group"]).agg(
        violation=("violation", "mean"), coverage=("coverage", "mean"), risk=("risk", "mean"),
        feasible=("feasible", "mean"), n_draw=("n_draw", "mean"), n_pool=("n_pool", "mean"),
        unc_dis=("unc_dis", "mean")).reset_index()


# ----------------------------------------------- (c) quarterly drift, delay --
def part_c(S, D, models, cal, st, quarters):
    """Certificate fitted on all of civil_cal. Returns (table, change, certs)."""
    qs = sorted(set(quarters))
    rows, change, certs = [], {}, {}
    for c in models:
        cert = SelectiveCertificate(ALPHA, DELTA).fit(S[c][cal], D[c][cal])
        certs[c] = cert
        deg = degenerate(cert, S[c][cal])
        s_st, d_st = S[c][st], D[c][st]
        sm = cert.serve_mask(s_st)
        change[c] = None
        for q in qs:
            m = quarters == q
            served = sm & m
            risk = float(d_st[served].mean()) if served.any() else np.nan
            cov = float(sm[m].mean())
            row = {"model": c, "quarter": q, "n": int(m.sum()), "coverage": cov, "risk": risk,
                   "unc_dis": float(d_st[m].mean()), "exceeds_alpha": bool(risk > ALPHA),
                   "degenerate": deg}
            for bn, b in BUDGETS.items():
                row["pred_delay_" + bn] = (expected_audit_delay(risk, ALPHA, DELTA, b, cov)
                                           if served.any() else np.nan)
            rows.append(row)
            if change[c] is None and risk > ALPHA:
                change[c] = {"quarter": q, "t": int(np.argmax(m)), "risk": risk, "coverage": cov}
    return pd.DataFrame(rows), change, certs


# -------------------------------------------------------- (d) monitors --
def make_monitor(name, cert, s_cal, phi0, seed, n_st):
    if name.startswith("audit"):
        return ShadowAuditMartingale(ALPHA, DELTA, BUDGETS[name], seed=seed)
    if name == "adaptive":
        return AdaptiveAuditMartingale(ALPHA, DELTA, s_cal=s_cal, lam=cert.lam, seed=seed)
    if name == "cusum":
        return DeferralRateCUSUM(phi0, eps=0.05, h=5.0)
    return ScoreKS(s_cal, window=200, delta=DELTA, n_windows_expected=max(1, n_st // 200))


def part_d(S, D, models, cal, st, certs, change, reps):
    rows = []
    for c in models:
        cert = certs[c]
        if degenerate(cert, S[c][cal]):
            continue
        sc, dc = S[c][cal], D[c][cal]
        s_st, d_st = S[c][st], D[c][st]
        phi0 = 1 - cert.coverage(sc)
        ch = change[c]
        t_change = ch["t"] if ch else len(st)
        for r in range(reps):
            rng = np.random.default_rng(1000 + r)
            ix = rng.integers(0, len(sc), len(st))          # in-domain null stream
            streams = {"real": (s_st, d_st, t_change), "null": (sc[ix], dc[ix], len(st))}
            for sname, (s_, d_, tc) in streams.items():
                serve = cert.serve_mask(s_)
                for mn in MON_NAMES:
                    M = make_monitor(mn, cert, sc, phi0, r, len(st))
                    t_al = ex.first_alarm(mn, M, serve, s_, d_)
                    alarm = t_al is not None
                    rows.append({"model": c, "monitor": mn, "stream": sname, "rep": r,
                                 "alarm": alarm, "alarm_t": float(t_al) if alarm else np.nan,
                                 "pre": alarm and t_al < tc,
                                 "post": alarm and t_al >= tc and ch is not None,
                                 "delay": (t_al - tc) if (alarm and t_al >= tc and ch is not None) else np.nan,
                                 "audits": getattr(M, "n_audits", 0)})
    runs = pd.DataFrame(rows, columns=["model", "monitor", "stream", "rep", "alarm", "alarm_t", "pre",
                                       "post", "delay", "audits"])
    summ = []
    for (c, mn), g in runs.groupby(["model", "monitor"]):
        real, null = g[g.stream == "real"], g[g.stream == "null"]
        ch = change[c]
        pred = np.nan
        if ch and mn in BUDGETS:
            pred = expected_audit_delay(ch["risk"], ALPHA, DELTA, BUDGETS[mn], ch["coverage"])
        summ.append({"model": c, "monitor": mn, "change_quarter": ch["quarter"] if ch else "none",
                     "alarm_real": real.alarm.mean(), "pre_change_real": real.pre.mean(),
                     "detect": real.post.mean() if ch else np.nan,
                     "median_delay": real.delay.median(), "median_audits_real": real.audits.median(),
                     "false_alarm": null.alarm.mean(), "pred_delay": pred, "n_reps": len(real)})
    return runs, pd.DataFrame(summ, columns=["model", "monitor", "change_quarter", "alarm_real",
                                              "pre_change_real", "detect", "median_delay",
                                              "median_audits_real", "false_alarm", "pred_delay", "n_reps"])


# ------------------------------------------------------ (e) streaming --
def part_e(S, D, meta, models, cal, st, quarters, certs, reps):
    qs = sorted(set(quarters))
    rows = []
    for c in models:
        if degenerate(certs[c], S[c][cal]):
            continue
        sc, dc = S[c][cal], D[c][cal]
        s_st, d_st = S[c][st], D[c][st]
        size = meta[c]["size"]
        for pol, budget in POLICIES:
            for r in range(reps):
                servedC, Lc, nre = ex.simulate_policy(pol, s_st, d_st, sc, dc, ALPHA, budget, seed=r)
                row = {"model": c, "policy": pol, "run": r, "L_call_frac": float(Lc.mean()),
                       "C_served_frac": float(servedC.mean()),
                       "risk_served": float(d_st[servedC].mean()) if servedC.any() else np.nan,
                       "n_recal": nre}
                for q in qs:
                    sv = servedC & (quarters == q)
                    row["risk_" + q] = float(d_st[sv].mean()) if sv.any() else np.nan
                row["cost"] = row["L_call_frac"] + row["C_served_frac"] * size
                row["cost_cascade"] = size + row["L_call_frac"]
                rows.append(row)
    runs = pd.DataFrame(rows)
    if runs.empty:
        return runs, runs
    summ = runs.groupby(["model", "policy"]).mean(numeric_only=True).drop(columns="run").reset_index()
    return runs, summ


# ---------------------------------------------------------------- plot --
def plot_from_csv(out):
    """Three-panel figure from the CSVs. Returns False when matplotlib is not installed."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return False
    sa = pd.read_csv(os.path.join(out, "exp11a_summary.csv"))
    qc = pd.read_csv(os.path.join(out, "exp11c_quarterly.csv"))
    md = pd.read_csv(os.path.join(out, "exp11d_monitors.csv"))
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    ax = axes[0]
    for c in sa.model.unique():
        g = sa[(sa.model == c) & (sa.alpha == ALPHA)]
        for nm, ls in [("certified", "-"), ("naive", ":")]:
            gg = g[g.certifier == nm]
            ax.plot(gg.n_cal, gg.violation, ls, marker="o", ms=3, label=f"{c} {nm}" if nm == "certified" else None)
    ax.axhline(DELTA, color="k", ls="--", lw=0.8)
    ax.set_xlabel("calibration size"); ax.set_ylabel("P(population risk > alpha)")
    ax.set_title("(a) validity on civil_cal, alpha=0.05 (dotted = naive)"); ax.legend(fontsize=7)
    ax = axes[1]
    for c in qc.model.unique():
        gg = qc[qc.model == c]
        ax.plot(range(len(gg)), gg.risk, marker="o", label=c)
    ax.axhline(ALPHA, color="k", ls="--", lw=0.8)
    ax.set_xticks(range(len(gg))); ax.set_xticklabels(gg.quarter)
    ax.set_ylabel("served disagreement risk"); ax.set_title("(c) per-quarter risk, certificate fit on civil_cal")
    ax.legend(fontsize=7)
    ax = axes[2]
    agg = md.groupby("monitor")[["alarm_real", "false_alarm"]].mean().reindex(MON_NAMES)
    xx = np.arange(len(agg))
    ax.bar(xx - 0.2, agg.alarm_real, 0.4, label="alarm on real stream")
    ax.bar(xx + 0.2, agg.false_alarm, 0.4, label="alarm on null stream")
    ax.set_xticks(xx); ax.set_xticklabels(agg.index, rotation=30)
    ax.set_title("(d) monitor alarm rates (mean over models)"); ax.legend(fontsize=7)
    fig.tight_layout()
    path = os.path.join(out, "exp11_civil.png")
    fig.savefig(path, dpi=130)
    print("[saved]", path)
    return True


# ----------------------------------------------------------------- main --
def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Exp 11 CivilComments analysis.")
    p.add_argument("--npz", default=DEFAULT_NPZ, help="civil_preds.npz from colab_civil.py")
    p.add_argument("--out", default=DEFAULT_OUT, help="output directory (default results/real)")
    p.add_argument("--models", default=",".join(MODELS), help="comma list of compressed models")
    p.add_argument("--reps", type=int, default=300, help="(a) and (b) calibration draws")
    p.add_argument("--mon-reps", type=int, default=20, help="(d) and (e) random seeds (audit coins)")
    p.add_argument("--plot-only", action="store_true", help="only build the figure from --out CSVs")
    a = p.parse_args(argv)
    a.models = [m for m in a.models.split(",") if m]
    return a


def save(df, out, name):
    df.to_csv(os.path.join(out, name), index=False)
    print("[saved]", os.path.join(out, name))


def main(argv=None):
    args = parse_args(argv)
    if args.plot_only:
        if not plot_from_csv(args.out):
            print("matplotlib not installed; skipping figure")
        return
    out, models = args.out, args.models
    os.makedirs(out, exist_ok=True)
    Z = load_npz(args.npz)
    for k in ["y", "split", "groups", "date", "prob_L", "meta"]:
        if k not in Z:
            raise SystemExit(f"{args.npz} has no {k!r}")
    for c in models:
        if "prob_" + c not in Z:
            raise SystemExit(f"model {c!r} has no prob_{c} in {args.npz}")
    meta = json.loads(str(Z["meta"]))
    cal, st = prepare(Z)
    quarters = quarter_of(Z["date"][st])
    gname = np.array([civil_group(g) for g in Z["groups"]])
    G = np.array([GROUP_NAMES.index(g) for g in gname])

    # ---------------------------------------------------------- meta / overview --
    ids_n = np.array([len(g.split("|")) if g else 0 for g in Z["groups"]])
    meta_rows = [{"key": "n_cal", "value": len(cal)}, {"key": "n_stream", "value": len(st)},
                 {"key": "multi_identity_share_cal", "value": float((ids_n[cal] >= 2).mean())},
                 {"key": "multi_identity_share_stream", "value": float((ids_n[st] >= 2).mean())}]
    for nm, idx in [("cal", cal), ("stream", st)]:
        for gn in GROUP_NAMES:
            meta_rows.append({"key": f"group_share_{nm}_{gn}", "value": float((gname[idx] == gn).mean())})
    for q in sorted(set(quarters)):
        meta_rows.append({"key": f"n_stream_{q}", "value": int((quarters == q).sum())})
    save(pd.DataFrame(meta_rows), out, "exp11_meta.csv")
    print(f"[meta] multi-identity share: cal {(ids_n[cal] >= 2).mean():.4f}, "
          f"stream {(ids_n[st] >= 2).mean():.4f}")

    S, D = scores_and_disagreement(Z, models)
    ov = []
    for c in ["L"] + models:
        p = Z["prob_" + c].argmax(1)
        predL = Z["prob_L"].argmax(1)
        ov.append({"model": c, "size": meta[c]["size"],
                   "latency_rel": meta[c]["per_1k"] / meta["L"]["per_1k"] if c in ex.LATENCY_MODELS else np.nan,
                   "acc_cal": float((p[cal] == Z["y"][cal]).mean()), "acc_stream": float((p[st] == Z["y"][st]).mean()),
                   "agree_cal": float((p[cal] == predL[cal]).mean()), "agree_stream": float((p[st] == predL[st]).mean())})
    save(pd.DataFrame(ov), out, "exp11_overview.csv")

    # ------------------------------------------------------------ (a)..(e) --
    sa = part_a(S, D, meta, models, cal, args.reps)
    save(sa, out, "exp11a_summary.csv")
    print(sa[sa.n_cal == 1000].round(3).to_string())

    sb = part_b(S, D, models, cal, G, args.reps)
    save(sb, out, "exp11b_summary.csv")
    print(sb.round(3).to_string())

    qc, change, certs = part_c(S, D, models, cal, st, quarters)
    save(qc, out, "exp11c_quarterly.csv")
    print(qc.round(3).to_string())
    print("[11c] change quarter (first quarter with risk > alpha):", {c: (v["quarter"] if v else None) for c, v in change.items()})

    runs_d, sd = part_d(S, D, models, cal, st, certs, change, args.mon_reps)
    save(runs_d, out, "exp11d_runs.csv")
    save(sd, out, "exp11d_monitors.csv")
    print(sd.round(3).to_string())

    runs_e, se = part_e(S, D, meta, models, cal, st, quarters, certs, args.mon_reps)
    if len(se):
        save(se, out, "exp11e_streaming.csv")
        print(se.round(3).to_string())
    skipped = sorted(c for c in models if degenerate(certs[c], S[c][cal]))
    print("[11d/11e] degenerate certificates (monitor/streaming rows skipped):", skipped)

    if plot_from_csv(out) is False:
        print("matplotlib not installed; skipping figure (run --plot-only on a machine with matplotlib)")
    print("done")


if __name__ == "__main__":
    main()
