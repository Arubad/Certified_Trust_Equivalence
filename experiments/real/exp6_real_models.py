"""Experiment 6 -- Exp 1-5 on a real transformer and real compressions.

Input: results/real/real_preds.npz written by experiments/real/colab_infer.py
(BERT-base HateXplain classifier L and six training-free compressions of it).
Everything here is the same certifier / monitor code as Exp 1-5; it only
consumes C's margin score and the disagreement bit 1[C(x) != L(x)].

  6a  validity + coverage + certified cost on HateXplain val+test
      (alpha in {0.01, 0.02, 0.05}, n_cal in {200, 500, 1000}, 300 splits)
  6b  marginal vs group-conditional (Mondrian) certificate over target
      communities (African, Islam, Jewish, Homosexual, Women, other)
  6c  real drift: certificate fitted on HateXplain, deployed on Davidson tweets;
      monitor false-alarm / detection / delay
  6d  streaming policy: 1 500 HateXplain posts then 3 000 Davidson tweets

CLI (defaults reproduce Exp 6 exactly):
  python exp6_real_models.py [--npz PATH] [--models a,b,c] [--tag TAG] [--ref PATH]
  Exp 10: --npz results/real/real_preds_ft.npz --models distil6,p50ft,p70ft,q4ft --tag exp10
  --ref is the CPU real_preds.npz used for the L check (Review Focus 4 / ruling R1). It
  defaults to real_preds.npz next to --npz whenever --npz is not real_preds.npz itself.
  Outputs are written to $TEC_REAL_OUT (default results/real) as TAG_* files.
"""
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", ".."))
from tec.certify import (SelectiveCertificate, NaiveThreshold, GroupCertificate, selective_risk,  # noqa: E402
                         hoeffding_marginal_bound, certified_cost)
from tec.monitor import DeferralRateCUSUM, ShadowAuditMartingale, ScoreKS  # noqa: E402

OUT = os.environ.get("TEC_REAL_OUT", os.path.join(HERE, "..", "..", "results", "real"))
DELTA = 0.10
COMPRESSED = ["int8", "q4", "q3", "p30", "p50", "p70"]
MAIN_GROUPS = ["African", "Islam", "Jewish", "Homosexual", "Women"]
AGREE_TOL = 0.995        # ruling R1: GPU prob_L is used when its HateXplain argmax agreement with CPU L >= this


def first_group(g):
    """Index of the first main target group named in a '|'-separated groups string; len(MAIN_GROUPS) = other."""
    for t in MAIN_GROUPS:
        if t in g.split("|"):
            return MAIN_GROUPS.index(t)
    return len(MAIN_GROUPS)                     # "other / none"


def save(df, name):
    df.to_csv(os.path.join(OUT, name), index=False)
    print("[saved]", os.path.relpath(os.path.join(OUT, name)))


def margin(P):
    s = np.sort(P, 1)
    return s[:, -1] - s[:, -2]


def population_eval(cert, s_pool, d_pool):
    """Exact selective risk and coverage of `cert` on the whole pool, treated as
    the population. Returns (risk, coverage); risk is 0 when nothing is served."""
    sm = cert.serve_mask(s_pool)
    return selective_risk(sm, d_pool), float(sm.mean())


def check_L(Z, R, tol=AGREE_TOL):
    """Ruling R1 / Review Focus 4. Z: the run whose prob_L is a candidate (GPU ft npz),
    R: the CPU reference (real_preds.npz). Rows must be the same (y and split equal).
    Returns (info, prob_L): prob_L is Z's when HateXplain argmax agreement >= tol, else R's."""
    if not (np.array_equal(Z["y"], R["y"]) and np.array_equal(Z["split"], R["split"])):
        raise ValueError("row order differs: y/split of the two npz files are not identical")
    hx = np.isin(Z["split"], ["val", "test"])
    dv = Z["split"] == "davidson"
    ag = lambda m: float((Z["prob_L"][m].argmax(1) == R["prob_L"][m].argmax(1)).mean())  # noqa: E731
    agree_hx = ag(hx)
    use_gpu = agree_hx >= tol
    info = {"L_source": "gpu" if use_gpu else "cpu", "agree_hx": agree_hx,
            "agree_dv": ag(dv) if dv.any() else np.nan, "threshold": tol, "n_hx": int(hx.sum())}
    return info, (Z["prob_L"] if use_gpu else R["prob_L"])


def overview(Z, meta, models=COMPRESSED):
    """Per-model accuracy and agreement with L on HateXplain (hx) and Davidson (dv).

    acc_* uses labelled rows only (y >= 0); agree_* uses every row of the domain.
    latency_rel is computed only when int8 is among `models` (the exp6 setting, where
    every timing comes from one CPU run; int8 is the one model with a real CPU speed-up).
    Otherwise (e.g. the T4 fp16 run in real_preds_ft.npz) it is NaN, so timings from
    different hardware are never compared.
    """
    y, split = Z["y"], Z["split"]
    hx, dv = np.isin(split, ["val", "test"]), split == "davidson"
    predL = Z["prob_L"].argmax(1)
    lat_L = meta["L"]["per_1k"]
    timed = "int8" in models
    ov = []
    for c in ["L"] + [c for c in models if "prob_" + c in Z]:
        p = Z["prob_" + c].argmax(1)
        lat_rel = meta[c]["per_1k"] / lat_L if (timed and c in ("L", "int8")) else np.nan
        r = {"model": c, "size": meta[c]["size"], "latency_rel": lat_rel}
        for nm, m in [("hx", hx), ("dv", dv)]:
            lab = m & (y >= 0)
            r[f"acc_{nm}"] = float((p[lab] == y[lab]).mean())
            r[f"agree_{nm}"] = float((p[m] == predL[m]).mean())
        ov.append(r)
    return pd.DataFrame(ov)


def plot_figure(sa, sc, models, path):
    """Three-panel figure from the 6a and 6c summary tables (DataFrames or CSVs read back).
    Returns False when matplotlib is not installed."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return False
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    ax = axes[0]
    for c in models:
        g = sa[(sa.model == c) & (sa.alpha == 0.05)]
        for cert, ls in [("certified", "-"), ("naive", ":")]:
            gg = g[g.certifier == cert]
            ax.plot(gg.n_cal, gg.violation, ls, marker="o", label=f"{c} {cert}" if cert == "certified" else None)
    ax.axhline(DELTA, color="k", ls="--", lw=0.8)
    ax.set_xlabel("calibration size"); ax.set_ylabel("P(population risk > alpha)"); ax.set_title("6a validity (alpha=0.05; dotted = naive)")
    ax.legend(fontsize=7)
    ax = axes[1]
    g = sa[(sa.certifier == "certified") & (sa.n_cal == 1000)]
    for c in models:
        gg = g[g.model == c]
        ax.plot(gg.alpha, gg.cost_size, marker="o", label=c)
    ax.set_xlabel("alpha"); ax.set_ylabel("certified cost (size)"); ax.set_title("6a certified cost, n_cal=1000"); ax.legend(fontsize=7)
    ax = axes[2]
    g = sc[sc.alpha == 0.05].pivot(index="model", columns="domain", values="risk").loc[models]
    xx = np.arange(len(models))
    ax.bar(xx - 0.2, g["hatexplain"], 0.4, label="HateXplain (calibration domain)")
    ax.bar(xx + 0.2, g["davidson"], 0.4, label="Davidson (shifted)")
    ax.axhline(0.05, color="k", ls="--", lw=0.8); ax.set_xticks(xx); ax.set_xticklabels(models)
    ax.set_ylabel("selective disagreement risk"); ax.set_title("6c risk on calibration vs shifted domain"); ax.legend(fontsize=7)
    fig.tight_layout(); fig.savefig(path, dpi=130)
    return True


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Exp 6 analysis (defaults reproduce Exp 6).")
    p.add_argument("--npz", default=None, help="prediction npz (default: OUT/real_preds.npz)")
    p.add_argument("--ref", default=None, help="CPU reference npz for the prob_L check (default: see module doc)")
    p.add_argument("--models", default=",".join(COMPRESSED), help="comma list of compressed models")
    p.add_argument("--tag", default="exp6", help="output file prefix (default exp6)")
    a = p.parse_args(argv)
    a.models = [m for m in a.models.split(",") if m]
    return a


def main(argv=None):
    args = parse_args(argv)
    tag, models = args.tag, args.models
    os.makedirs(OUT, exist_ok=True)
    npz = args.npz or os.path.join(OUT, "real_preds.npz")
    with np.load(npz, allow_pickle=False) as F:
        Z = {k: F[k] for k in F.files}
    meta = json.loads(str(Z["meta"]))
    for c in models:
        if "prob_" + c not in Z:
            raise SystemExit(f"model {c!r} has no prob_{c} in {npz}")

    # ------------------------------------------------ prob_L check (ruling R1) --
    ref = args.ref
    if ref is None and os.path.basename(npz) != "real_preds.npz":
        ref = os.path.join(os.path.dirname(npz), "real_preds.npz")
    if ref is not None:
        with np.load(ref, allow_pickle=False) as F:
            R = {k: F[k] for k in ["y", "split", "prob_L"]}
        info, Z["prob_L"] = check_L(Z, R)
        info.update(tag=tag, npz=os.path.basename(npz), ref=os.path.basename(ref),
                    timing=("same run" if "int8" in models else "not compared: different hardware"))
        print(f"[L check] HateXplain argmax agreement GPU vs CPU = {info['agree_hx']:.4f} "
              f"(davidson {info['agree_dv']:.4f}); threshold {AGREE_TOL}; using L = {info['L_source']}")
        save(pd.DataFrame([info]), f"{tag}_L_check.csv")

    hx, dv = np.isin(Z["split"], ["val", "test"]), Z["split"] == "davidson"
    predL = Z["prob_L"].argmax(1)
    S = {c: margin(Z["prob_" + c]) for c in models}
    D = {c: (Z["prob_" + c].argmax(1) != predL).astype(int) for c in models}
    timed = "int8" in models
    lat_L = meta["L"]["per_1k"]

    pd.set_option("display.width", 240)
    ov = overview(Z, meta, models=models); save(ov, f"{tag}_overview.csv")
    print(ov.round(3).to_string())

    # ---------------------------------------------------- 6a validity + cost --
    rows = []
    idx_hx = np.where(hx)[0]
    # eval="population": calibration draws are i.i.d. WITH replacement from the hx pool,
    # and risk/coverage are exact over the whole pool (the pool is the population).
    # This removes held-out test noise; the guarantee holds exactly for this setting.
    for c in models:
        s, d = S[c][idx_hx], D[c][idx_hx]
        for alpha in [0.01, 0.02, 0.05]:
            for n_cal in [200, 500, 1000]:
                rng = np.random.default_rng(0)
                for r in range(300):
                    ca = rng.integers(0, len(s), n_cal)
                    for nm, M in [("certified", SelectiveCertificate), ("naive", NaiveThreshold)]:
                        m = M(alpha, DELTA).fit(s[ca], d[ca])
                        risk, cov = population_eval(m, s, d)
                        rows.append({"model": c, "alpha": alpha, "n_cal": n_cal, "certifier": nm,
                                     "eval": "population", "coverage": cov, "risk": risk,
                                     "violation": float(risk > alpha),
                                     "feasible": m.feasible, "marginal_cp": hoeffding_marginal_bound(d[ca], DELTA)})
    a = pd.DataFrame(rows)
    sa = a.groupby(["model", "alpha", "n_cal", "certifier", "eval"]).agg(
        violation=("violation", "mean"), coverage=("coverage", "mean"), risk=("risk", "mean"),
        feasible=("feasible", "mean"), marginal_cp=("marginal_cp", "mean")).reset_index()
    sa["size"] = sa.model.map(lambda c: meta[c]["size"])
    # cost_size / cost_latency: free-deferral assumption (a deferred input costs only L).
    sa["cost_size"] = certified_cost(sa.coverage, sa["size"])
    if timed:
        lat = sa.model.map(lambda c: meta[c]["per_1k"] / lat_L if c in ("L", "int8") else np.nan)
        sa["cost_latency"] = np.where(sa.model.isin(["L", "int8"]),
                                      certified_cost(sa.coverage, lat), np.nan)
    else:
        sa["cost_latency"] = np.nan
    # cost_cascade: C always runs to produce its margin, then L on deferred inputs.
    sa["cost_cascade"] = sa["size"] + (1 - sa.coverage)
    save(sa, f"{tag}a_summary.csv")
    print(sa[sa.n_cal == 1000].round(3).to_string())

    # --------------------------------------------------- 6b group certificate --
    gstr = Z["groups"]

    G = np.array([first_group(g) for g in gstr])
    GNAMES = MAIN_GROUPS + ["other"]
    rows = []
    alpha, n_cal = 0.05, 2000
    for c in models:
        s, d, g = S[c][idx_hx], D[c][idx_hx], G[idx_hx]
        rng = np.random.default_rng(1)
        # eval="population": i.i.d. draws with replacement from the pool; group risks and
        # coverages are exact over the whole pool (no held-out test noise).
        for r in range(300):
            ca = rng.integers(0, len(s), n_cal)
            marg = SelectiveCertificate(alpha, DELTA).fit(s[ca], d[ca])
            grp = GroupCertificate(alpha, DELTA).fit(s[ca], d[ca], g[ca])
            for nm, sm in [("marginal", marg.serve_mask(s)), ("group", grp.serve_mask(s, g))]:
                for gi, gn in enumerate(GNAMES):
                    mg = g == gi
                    risk = selective_risk(sm[mg], d[mg])
                    rows.append({"model": c, "certificate": nm, "group": gn, "split": r, "eval": "population",
                                 "coverage": float(sm[mg].mean()), "risk": risk, "violation": float(risk > alpha),
                                 "unc_dis": float(d[mg].mean()), "n_pool": int(mg.sum())})
    b = pd.DataFrame(rows)
    sb = b.groupby(["model", "certificate", "group", "eval"]).agg(
        violation=("violation", "mean"), coverage=("coverage", "mean"), risk=("risk", "mean"),
        unc_dis=("unc_dis", "mean"), n_pool=("n_pool", "mean")).reset_index()
    save(sb, f"{tag}b_summary.csv")
    print(sb.round(3).to_string())

    # ------------------------------------------------- 6c drift to Davidson --
    idx_dv = np.where(dv)[0]
    rows, mon_rows = [], []
    skipped_monitor_models = set()
    for c in models:
        for alpha in [0.02, 0.05]:
            rng = np.random.default_rng(2)
            for r in range(50):
                perm = rng.permutation(idx_hx); ca, te = perm[:1000], perm[1000:]
                cert = SelectiveCertificate(alpha, DELTA).fit(S[c][ca], D[c][ca])
                for nm, ix in [("hatexplain", te), ("davidson", idx_dv)]:
                    sm = cert.serve_mask(S[c][ix])
                    rows.append({"model": c, "alpha": alpha, "split": r, "domain": nm, "coverage": float(sm.mean()),
                                 "risk": selective_risk(sm, D[c][ix])})
                if alpha != 0.05:
                    continue
                # monitors on a stream: 1500 in-domain then 1500 Davidson (and an all-in-domain null stream)
                # degenerate certificates (infeasible, or coverage < 0.05 on calibration) make the
                # deferral-rate monitors artifacts (CUSUM's phi0 clipping): skip their monitor rows.
                # The rng draws below still run so the random stream is unchanged by the skip.
                degenerate = (not cert.feasible) or cert.coverage(S[c][ca]) < 0.05
                for null in [True, False]:
                    st = np.concatenate([rng.choice(te, 1500), rng.choice(te if null else idx_dv, 1500)])
                    if degenerate:
                        skipped_monitor_models.add(c)
                        continue
                    s_st, d_st = S[c][st], D[c][st]
                    serve = cert.serve_mask(s_st)
                    mons = {f"audit{int(100 * bb)}": ShadowAuditMartingale(alpha, DELTA, bb, seed=r) for bb in [0.02, 0.05, 0.10]}
                    mons["cusum"] = DeferralRateCUSUM(1 - cert.coverage(S[c][ca]), eps=0.05, h=5.0)
                    mons["ks"] = ScoreKS(S[c][ca], window=200, delta=DELTA, n_windows_expected=15)
                    for mn, M in mons.items():
                        t_alarm = None
                        for t in range(len(st)):
                            if mn.startswith("audit"):
                                al = M.update(int(d_st[t])) if serve[t] else False
                            elif mn == "cusum":
                                al = M.update(not serve[t])
                            else:
                                al = M.update(s_st[t])
                            if al:
                                t_alarm = t; break
                        mon_rows.append({"model": c, "split": r, "null_stream": null, "monitor": mn,
                                         "alarm_pre": t_alarm is not None and t_alarm < 1500,
                                         "alarm_post": t_alarm is not None and t_alarm >= 1500,
                                         "delay": (t_alarm - 1500) if (t_alarm is not None and t_alarm >= 1500) else np.nan,
                                         "post_risk": selective_risk(serve[1500:], d_st[1500:])})
    cd = pd.DataFrame(rows)
    sc = cd.groupby(["model", "alpha", "domain"]).agg(coverage=("coverage", "mean"), risk=("risk", "mean"),
                                                      violation=("risk", lambda x: 0)).reset_index()
    sc["violation"] = [float((cd[(cd.model == m) & (cd.alpha == a_) & (cd.domain == dm)].risk > a_).mean())
                       for m, a_, dm in zip(sc.model, sc.alpha, sc.domain)]
    save(sc, f"{tag}c_drift_risk.csv")
    print(sc.round(3).to_string())
    mo = pd.DataFrame(mon_rows)
    # false alarm = any alarm on the null stream (3000 in-domain inputs)
    fa = mo[mo.null_stream].assign(fa=lambda x: x.alarm_pre | x.alarm_post).groupby(["model", "monitor"]).fa.mean()
    det = mo[~mo.null_stream].groupby(["model", "monitor"]).agg(
        detect=("alarm_post", "mean"), early=("alarm_pre", "mean"), median_delay=("delay", "median"),
        post_risk=("post_risk", "mean"))
    sm_ = det.join(fa.rename("false_alarm")).reset_index()
    save(sm_, f"{tag}c_monitors.csv")
    print(sm_.round(3).to_string())
    print("[6c] monitor rows skipped for degenerate certificates:", sorted(skipped_monitor_models))

    # ------------------------------------------------------ 6d streaming policy --
    RECAL = 300

    def run_policy(policy, s, d, s_ca, d_ca, alpha, budget, seed):
        n = len(s); servedC = np.zeros(n, bool); Lc = np.zeros(n, bool)
        cert = SelectiveCertificate(alpha, DELTA).fit(s_ca, d_ca)
        mon = ShadowAuditMartingale(alpha, DELTA, budget, seed=seed) if policy.startswith("audit") else None
        buf_s, buf_d, left, nre = [], [], 0, 0
        for t in range(n):
            if policy == "always-C":
                servedC[t] = True; continue
            if left > 0:
                Lc[t] = True; buf_s.append(s[t]); buf_d.append(d[t]); left -= 1
                if left == 0:
                    cert = SelectiveCertificate(alpha, DELTA).fit(np.array(buf_s), np.array(buf_d))
                    mon.reset(); buf_s, buf_d = [], []
                continue
            serve = s[t] >= cert.lam
            servedC[t], Lc[t] = serve, not serve
            if mon is not None and serve:
                na = mon.n_audits
                if mon.update(int(d[t])):
                    left = RECAL; nre += 1
                if mon.n_audits > na:
                    Lc[t] = True
        seg = np.arange(n) >= 1500
        rk = lambda m: float(d[m].mean()) if m.any() else np.nan  # noqa: E731  (NaN: nothing served)
        return {"policy": policy, "L_call_frac": float(Lc.mean()), "C_served_frac": float(servedC.mean()),
                "risk_in": rk(servedC & ~seg), "risk_shift": rk(servedC & seg),
                "risk_shift_last1000": rk(servedC & (np.arange(n) >= n - 1000)), "n_recal": nre,
                "overall_dis": float(d[servedC].sum() / n)}


    rows = []
    alpha = 0.05
    for c in models:
        for r in range(20):
            rng = np.random.default_rng(100 + r)
            perm = rng.permutation(idx_hx); ca, te = perm[:1000], perm[1000:]
            st = np.concatenate([rng.choice(te, 1500), rng.choice(idx_dv, 3000, replace=False)])
            for pol, bb in [("always-C", 0), ("static", 0), ("audit5", 0.05), ("audit10", 0.10), ("audit20", 0.20)]:
                o = run_policy(pol, S[c][st], D[c][st], S[c][ca], D[c][ca], alpha, bb, seed=r)
                # cost: free-deferral assumption (a deferred input costs only L);
                # cost_cascade: C always runs to produce its margin, then L on deferred inputs.
                o["cost"] = o["L_call_frac"] + o["C_served_frac"] * meta[c]["size"]
                o["cost_cascade"] = meta[c]["size"] + o["L_call_frac"]
                rows.append({"model": c, "run": r, **o})
    e = pd.DataFrame(rows)
    se = e.groupby(["model", "policy"]).mean(numeric_only=True).drop(columns="run").reset_index()
    save(se, f"{tag}d_streaming.csv")
    print(se.round(3).to_string())

    # ------------------------------------------------------------------ plots --
    if plot_figure(sa, sc, models, os.path.join(OUT, f"{tag}_real_models.png")):
        print("[saved]", os.path.relpath(os.path.join(OUT, f"{tag}_real_models.png")))
    else:
        print("matplotlib not installed; skipping figure")
    print("done")


if __name__ == "__main__":
    main()
