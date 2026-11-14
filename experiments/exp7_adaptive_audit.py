"""Experiment 7 -- predictable adaptive audit rate (KS-triggered) vs fixed budgets.

Shadow auditing costs one call to L per audited served input. A fixed budget b
either pays for auditing all the time (large b) or detects drift slowly (small
b). AdaptiveAuditMartingale audits at b_lo = 2% while the served scores look
like calibration, and at b_hi = 20% once a KS test on the recent served scores
flags a shift. Its audit rate is predictable (decided before the score is
seen), so Proposition 4 (docs/theory.md) keeps the false-alarm bound at delta.

Compared monitors: fixed budgets {2%, 5%, 10%, 20%} (ShadowAuditMartingale)
and the adaptive monitor (2% -> 20%). Settings:
  (a) digits: prune90, pruneFT90, quant3b, distill25 under rotate / noise /
      occlude drift at severity 0.3 (same construction as Exp 4), 5 runs x 3 seeds.
  (b) real: six compressions of the HateXplain BERT classifier, certificate on
      1000 HateXplain cal points, stream = 1500 held-out HateXplain + 1500
      Davidson, 20 runs. Needs results/real/real_preds.npz (Exp 6 format);
      skipped if the file is missing.

Per (setting, model, monitor), over runs:
  false_alarm   alarm on a null stream of the same length, all in-domain
                (digits: 3000 clean inputs; real: 3000 held-out HateXplain)
  detect        alarm at or after the change point (stream index 1500)
  median_delay  median of (alarm_t - 1500) over detected runs
  audits_per_stream  mean number of audits on the drifted stream
"""
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, ".."))
from common import load_task, train_teacher, compress_all, disagreement, split_pool, save_csv, fig_path  # noqa: E402
from tec.data import drift_digits  # noqa: E402
from tec.certify import SelectiveCertificate, margin_score  # noqa: E402
from tec.monitor import AdaptiveAuditMartingale, ShadowAuditMartingale  # noqa: E402

ALPHA, DELTA, N_CAL = 0.05, 0.10, 400
T_CLEAN, T_DRIFT, T_NULL = 1500, 1500, 3000
SEV = 0.3
FIXED_BUDGETS = [0.02, 0.05, 0.10, 0.20]
DIGITS_MODELS = ["prune90", "pruneFT90", "quant3b", "distill25"]
DIGITS_KINDS = ["rotate", "noise", "occlude"]
REAL_MODELS = ["int8", "q4", "q3", "p30", "p50", "p70"]
REAL_NPZ = os.path.join(HERE, "..", "results", "real", "real_preds.npz")


def make_monitor(name, s_cal, seed):
    if name == "adaptive":
        return AdaptiveAuditMartingale(ALPHA, DELTA, b_lo=0.02, b_hi=0.20, s_cal=s_cal,
                                       window=100, ks_p=0.01, seed=seed)
    return ShadowAuditMartingale(ALPHA, DELTA, budget=float(name[len("audit"):]), seed=seed)


MONITOR_NAMES = [f"audit{b:.2f}" for b in FIXED_BUDGETS] + ["adaptive"]


def run_stream(s, d, lam, monitor):
    """Serve x_t iff s_t >= lam. Only served inputs are fed to the monitor
    (the audit coin is drawn by the monitor). Stops at the first alarm.

    coverage = served / processed (processed = stream up to alarm, or full),
    risk     = mean disagreement over served inputs (0 if none served).
    """
    s = np.asarray(s, dtype=float)
    d = np.asarray(d)
    adaptive = isinstance(monitor, AdaptiveAuditMartingale)
    n_served, n_bad, alarm_t, processed = 0, 0, None, len(s)
    for t in range(len(s)):
        if not s[t] >= lam:
            continue
        n_served += 1
        n_bad += int(d[t])
        alarm = monitor.update(float(s[t]), int(d[t])) if adaptive else monitor.update(int(d[t]))
        if alarm:
            alarm_t, processed = t, t + 1
            break
    return {
        "coverage": n_served / processed if processed else 0.0,
        "risk": n_bad / n_served if n_served else 0.0,
        "alarm_t": alarm_t,
        "n_audits": monitor.n_audits,
    }


def _rows(setting, model, seed, run, s_ca, lam, s, d, s_null, d_null):
    """One row per monitor for a drifted stream and the matched null stream."""
    out = []
    for name in MONITOR_NAMES:
        drift = run_stream(s, d, lam, make_monitor(name, s_ca, seed=run))
        null = run_stream(s_null, d_null, lam, make_monitor(name, s_ca, seed=run))
        alarm = drift["alarm_t"]
        detected = alarm is not None and alarm >= T_CLEAN
        out.append({"setting": setting, "model": model, "monitor": name, "seed": seed, "run": run,
                    "alarm_t": np.nan if alarm is None else alarm,
                    "detected": float(detected),
                    "delay": (alarm - T_CLEAN) if detected else np.nan,
                    "null_false_alarm": float(null["alarm_t"] is not None),
                    "n_audits": drift["n_audits"]})
    return out


def digits_rows(seeds=range(3), runs=5):
    rows = []
    for seed in seeds:
        (Xtr, ytr, _), (Xpo, ypo, _) = load_task("digits", seed)
        T = train_teacher("digits", Xtr, ytr, seed)
        comps, _ = compress_all(T, Xtr, ytr, "digits", seed)
        rng = np.random.default_rng(seed)
        ca, te = split_pool(len(Xpo), N_CAL, rng)
        for name in DIGITS_MODELS:
            C = comps[name]
            s_ca = margin_score(C, Xpo[ca]); d_ca = disagreement(T, C, Xpo[ca])
            cert = SelectiveCertificate(ALPHA, DELTA).fit(s_ca, d_ca)
            for run in range(runs):
                r2 = np.random.default_rng(1000 * seed + run)
                i_clean = r2.choice(te, T_CLEAN); i_drift = r2.choice(te, T_DRIFT)
                i_null = np.random.default_rng(7000 + 1000 * seed + run).choice(te, T_NULL)
                Xnull = Xpo[i_null]
                s_null = margin_score(C, Xnull); d_null = disagreement(T, C, Xnull)
                for kind in DIGITS_KINDS:
                    Xd = drift_digits(Xpo[i_drift], kind, SEV, seed=run)
                    Xs = np.vstack([Xpo[i_clean], Xd])
                    s = margin_score(C, Xs); d = disagreement(T, C, Xs)
                    rows += _rows(f"digits-{kind}-{SEV}", name, seed, run, s_ca, cert.lam,
                                  s, d, s_null, d_null)
            print("digits seed", seed, name, "done", flush=True)
    return rows


def _margin(P):
    s = np.sort(P, 1)
    return s[:, -1] - s[:, -2]


def real_rows(path, runs=20):
    Z = np.load(path, allow_pickle=False)
    split = Z["split"]
    hx, dv = np.isin(split, ["val", "test"]), split == "davidson"
    predL = Z["prob_L"].argmax(1)
    idx_hx, idx_dv = np.where(hx)[0], np.where(dv)[0]
    rows = []
    for c in REAL_MODELS:
        S = _margin(Z["prob_" + c]); D = (Z["prob_" + c].argmax(1) != predL).astype(int)
        rng = np.random.default_rng(2)
        for run in range(runs):
            perm = rng.permutation(idx_hx); ca, te = perm[:1000], perm[1000:]
            cert = SelectiveCertificate(ALPHA, DELTA).fit(S[ca], D[ca])
            st = np.concatenate([rng.choice(te, T_CLEAN), rng.choice(idx_dv, T_DRIFT)])
            null = rng.choice(te, T_NULL)
            rows += _rows("real-hatexplain-davidson", c, 0, run, S[ca], cert.lam,
                          S[st], D[st], S[null], D[null])
        print("real", c, "done", flush=True)
    return rows


def summarise(raw):
    g = raw.groupby(["setting", "model", "monitor"])
    summ = g.agg(false_alarm=("null_false_alarm", "mean"), detect=("detected", "mean"),
                 median_delay=("delay", "median"), audits_per_stream=("n_audits", "mean")).reset_index()
    return summ[["setting", "model", "monitor", "false_alarm", "detect", "median_delay", "audits_per_stream"]]


def plot(summ):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("skip plot: matplotlib not installed")
        return
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    colors = {m: c for m, c in zip(MONITOR_NAMES, ["tab:gray", "tab:blue", "tab:orange", "tab:red", "k"])}
    for m in MONITOR_NAMES:
        g = summ[(summ.monitor == m) & summ.median_delay.notna()]
        ax.scatter(g.audits_per_stream, g.median_delay, color=colors[m], label=m, alpha=0.7, s=28)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("audits per stream (3000 served inputs)"); ax.set_ylabel("median detection delay (steps)")
    ax.set_title("audit cost vs delay (each point = setting x model)")
    ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(fig_path("exp7_adaptive.png"), dpi=130)
    print("[saved] results/exp7_adaptive.png")


def main():
    raw_rows = digits_rows()
    if os.path.exists(REAL_NPZ):
        raw_rows += real_rows(REAL_NPZ)
    else:
        print("skip (b): real_preds.npz missing")
    raw = pd.DataFrame(raw_rows)
    summ = summarise(raw)
    save_csv(summ, "exp7_adaptive.csv")
    pd.set_option("display.width", 220)
    print(summ.round(3).to_string(index=False))
    plot(summ)


if __name__ == "__main__":
    main()
