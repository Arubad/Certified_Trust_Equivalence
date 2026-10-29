"""Experiment 5 -- End-to-end: certified deployment with monitoring and
recalibration on a drifting stream.

Stream of 6000 inputs: clean for the first 2000, drift kind/severity switches
at t=2000 and again at t=4000. Policies:
  * always-L           : cost 1, disagreement 0 (reference)
  * always-C           : cost size_C, no guarantee
  * static-certificate : certify once on clean calibration; never re-check
  * cert+audit(b)      : certificate + shadow-audit martingale at budget b;
                         on alarm, pause (serve everything with L) for
                         R=300 inputs, recalibrate on those, resume
  * cert+cusum         : same with the label-free deferral-rate CUSUM
  * cert+periodic(k)   : recalibrate every k inputs regardless (no monitor)
Metrics: fraction of inputs answered by L (cost proxy), realised
disagreement rate among C-served inputs (should be <= alpha), and overall
disagreement with L across the whole stream.
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from common import load_task, train_teacher, compress_all, disagreement, split_pool, save_csv, fig_path
from tec.data import drift_digits
from tec.certify import SelectiveCertificate, margin_score
from tec.monitor import DeferralRateCUSUM, ShadowAuditMartingale

ALPHA, DELTA, N_CAL, RECAL = 0.05, 0.10, 400, 300
SEGMENTS = [(None, 0.0, 2000), ("rotate", 0.3, 2000), ("noise", 0.4, 2000)]


def make_stream(Xpo, te, rng):
    X, seg = [], []
    for i, (kind, sev, n) in enumerate(SEGMENTS):
        idx = rng.choice(te, n)
        Xi = Xpo[idx] if kind is None else drift_digits(Xpo[idx], kind, sev, seed=int(rng.integers(1e6)))
        X.append(Xi); seg += [i] * n
    return np.vstack(X), np.array(seg)


def run_policy(policy, T, C, s_ca, d_ca, Xs, budget=0.05, period=1000, seed=0):
    s = margin_score(C, Xs); d = disagreement(T, C, Xs)
    n = len(Xs)
    served_by_C = np.zeros(n, bool); L_calls = np.zeros(n, bool)
    cert = SelectiveCertificate(ALPHA, DELTA).fit(s_ca, d_ca)
    phi0 = 1 - cert.coverage(s_ca)
    mon = None
    if policy.startswith("cert+audit"):
        mon = ShadowAuditMartingale(ALPHA, DELTA, budget, seed=seed)
    elif policy == "cert+cusum":
        mon = DeferralRateCUSUM(phi0, eps=0.05, h=5.0)
    recal_buf_s, recal_buf_d, recal_left = [], [], 0
    n_recal = 0
    for t in range(n):
        if policy == "always-L":
            L_calls[t] = True; continue
        if policy == "always-C":
            served_by_C[t] = True; continue
        if recal_left > 0:                      # recalibration window: L answers, collect data
            L_calls[t] = True
            recal_buf_s.append(s[t]); recal_buf_d.append(d[t]); recal_left -= 1
            if recal_left == 0:
                cert = SelectiveCertificate(ALPHA, DELTA).fit(np.array(recal_buf_s), np.array(recal_buf_d))
                phi0 = 1 - cert.coverage(np.array(recal_buf_s))
                if mon is not None:
                    mon.reset()
                    if isinstance(mon, DeferralRateCUSUM):
                        mon.phi0 = phi0
                recal_buf_s, recal_buf_d = [], []
            continue
        serve = cert.lam <= s[t]
        if serve:
            served_by_C[t] = True
        else:
            L_calls[t] = True
        alarm = False
        if isinstance(mon, ShadowAuditMartingale):
            if serve:
                audited_before = mon.n_audits
                alarm = mon.update(int(d[t]))
                if mon.n_audits > audited_before:
                    L_calls[t] = True        # the audit itself costs an L call
        elif isinstance(mon, DeferralRateCUSUM):
            alarm = mon.update(not serve)
        if policy.startswith("cert+periodic") and (t + 1) % period == 0:
            alarm = True
        if alarm:
            recal_left = RECAL; n_recal += 1
    risk_served = float(d[served_by_C].mean()) if served_by_C.any() else 0.0
    overall_dis = float(d[served_by_C].sum() / n)   # L-served inputs have zero disagreement
    return {"policy": policy, "L_call_frac": float(L_calls.mean()), "C_served_frac": float(served_by_C.mean()),
            "risk_on_served": risk_served, "overall_disagreement": overall_dis, "n_recal": n_recal,
            "risk_seg0": float(d[served_by_C & (SEG == 0)].mean()) if (served_by_C & (SEG == 0)).any() else 0.0,
            "risk_seg1": float(d[served_by_C & (SEG == 1)].mean()) if (served_by_C & (SEG == 1)).any() else 0.0,
            "risk_seg2": float(d[served_by_C & (SEG == 2)].mean()) if (served_by_C & (SEG == 2)).any() else 0.0}


rows = []
for seed in range(3):
    (Xtr, ytr, _), (Xpo, ypo, _) = load_task("digits", seed)
    T = train_teacher("digits", Xtr, ytr, seed)
    comps, sizes = compress_all(T, Xtr, ytr, "digits", seed)
    rng = np.random.default_rng(seed)
    ca, te = split_pool(len(Xpo), N_CAL, rng)
    for name in ["prune90", "pruneFT90", "quant3b", "distill25"]:
        C = comps[name]
        s_ca = margin_score(C, Xpo[ca]); d_ca = disagreement(T, C, Xpo[ca])
        for run in range(4):
            Xs, SEG = make_stream(Xpo, te, np.random.default_rng(100 * seed + run))
            for policy in ["always-L", "always-C", "static-certificate", "cert+cusum", "cert+periodic",
                           "cert+audit2", "cert+audit5", "cert+audit10"]:
                b = {"cert+audit2": 0.02, "cert+audit5": 0.05, "cert+audit10": 0.10}.get(policy, 0.05)
                r = run_policy(policy, T, C, s_ca, d_ca, Xs, budget=b, seed=run)
                r["cost"] = r["L_call_frac"] * 1.0 + r["C_served_frac"] * sizes[name]
                rows.append({"seed": seed, "method": name, "run": run, "size_ratio": sizes[name], **r})
        print("seed", seed, name, "done")

res = pd.DataFrame(rows); save_csv(res, "exp5_streaming.csv")
summ = res.groupby(["method", "policy"]).mean(numeric_only=True).drop(columns=["seed", "run"]).reset_index()
save_csv(summ, "exp5_summary.csv")
pd.set_option("display.width", 240)
print(summ[["method", "policy", "L_call_frac", "cost", "risk_on_served", "risk_seg0", "risk_seg1", "risk_seg2",
            "overall_disagreement", "n_recal"]].round(3).to_string())

fig, ax = plt.subplots(figsize=(7, 4.2))
mk = {"always-L": "*", "always-C": "s", "static-certificate": "o", "cert+cusum": "^", "cert+periodic": "v",
      "cert+audit2": "P", "cert+audit5": "X", "cert+audit10": "D"}
for name, c in [("prune90", "tab:blue"), ("quant3b", "tab:orange"), ("distill25", "tab:red")]:
    g = summ[summ.method == name]
    for _, r in g.iterrows():
        ax.scatter(r.L_call_frac, r.risk_on_served, marker=mk[r.policy], color=c, s=50,
                   label=f"{name}: {r.policy}" if name == "prune90" else None)
ax.axhline(ALPHA, ls="--", color="k", lw=0.8)
ax.set_xlabel("fraction of inputs answered by the large model (cost)"); ax.set_ylabel("disagreement on C-served inputs")
ax.set_title("Exp5: streaming policies (marker = policy, colour = compression)")
ax.legend(fontsize=6, ncol=2)
fig.tight_layout(); fig.savefig(fig_path("exp5_streaming.png"), dpi=130)
print("done")
