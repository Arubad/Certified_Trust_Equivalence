"""Experiment 4 -- Drift voids the certificate; can we detect it, and at what
cost?

A certificate fitted on clean calibration data is deployed on a stream that
drifts (digits: rotation / noise / occlusion at increasing severity). We
measure (a) the realised selective risk after drift -- is the certificate
void? -- and (b) three monitors' detection delay and false-alarm rate:
  * DeferralRateCUSUM  (label-free, free)
  * ScoreKS            (label-free, windowed two-sample test)
  * ShadowAuditMartingale with audit budget b in {2%, 5%, 10%, 20%}
    (calls L on a random b fraction of served inputs; Ville's inequality
    bounds the stream-wise false alarm rate by delta).

Hypothesis H4: moderate drift (severity 0.3) breaks the certificate (risk
2-5x alpha) while leaving accuracy-based symptoms weak; the audit martingale
detects it within a few hundred samples at 5% budget with no false alarms,
while label-free monitors either miss it or false-alarm.
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from common import load_task, train_teacher, compress_all, disagreement, split_pool, save_csv, fig_path
from tec.data import drift_digits
from tec.certify import SelectiveCertificate, margin_score, selective_risk
from tec.monitor import DeferralRateCUSUM, ShadowAuditMartingale, ScoreKS

ALPHA, DELTA, N_CAL = 0.05, 0.10, 400
T_CLEAN, T_DRIFT = 1500, 1500
rows, mon_rows = [], []
for seed in range(3):
    (Xtr, ytr, _), (Xpo, ypo, _) = load_task("digits", seed)
    T = train_teacher("digits", Xtr, ytr, seed)
    comps, sizes = compress_all(T, Xtr, ytr, "digits", seed)
    rng = np.random.default_rng(seed)
    ca, te = split_pool(len(Xpo), N_CAL, rng)
    for name in ["prune90", "pruneFT90", "quant3b", "distill25"]:
        C = comps[name]
        s_ca = margin_score(C, Xpo[ca]); d_ca = disagreement(T, C, Xpo[ca])
        cert = SelectiveCertificate(ALPHA, DELTA).fit(s_ca, d_ca)
        phi0 = 1 - cert.coverage(s_ca)
        for kind in ["rotate", "noise", "occlude"]:
            for sev in [0.0, 0.15, 0.3, 0.5]:
                for run in range(5):
                    r2 = np.random.default_rng(1000 * seed + run)
                    i1 = r2.choice(te, T_CLEAN); i2 = r2.choice(te, T_DRIFT)
                    X1 = Xpo[i1]; X2 = drift_digits(Xpo[i2], kind, sev, seed=run) if sev > 0 else Xpo[i2]
                    Xs = np.vstack([X1, X2])
                    s = margin_score(C, Xs); d = disagreement(T, C, Xs)
                    sm = cert.serve_mask(s)
                    post = np.arange(len(Xs)) >= T_CLEAN
                    rows.append({"seed": seed, "method": name, "drift": kind, "severity": sev, "run": run,
                                 "risk_pre": selective_risk(sm & ~post, d), "risk_post": selective_risk(sm & post, d),
                                 "coverage_pre": float(sm[~post].mean()), "coverage_post": float(sm[post].mean()),
                                 "acc_T_post": float((T.predict(X2) == ypo[i2]).mean()),
                                 "acc_C_post": float((C.predict(X2) == ypo[i2]).mean())})
                    # monitors
                    monitors = {"cusum": DeferralRateCUSUM(phi0, eps=0.05, h=5.0),
                                "ks": ScoreKS(s_ca, window=200, delta=DELTA, n_windows_expected=15)}
                    for b in [0.02, 0.05, 0.1, 0.2]:
                        monitors[f"audit{int(b*100)}"] = ShadowAuditMartingale(ALPHA, DELTA, b, seed=run)
                    for mname, mon in monitors.items():
                        alarm_t = None
                        for t in range(len(Xs)):
                            served = sm[t]
                            if mname == "cusum":
                                a = mon.update(not served)
                            elif mname == "ks":
                                a = mon.update(s[t])
                            else:
                                a = mon.update(int(d[t])) if served else False
                            if a:
                                alarm_t = t; break
                        mon_rows.append({"seed": seed, "method": name, "drift": kind, "severity": sev, "run": run,
                                         "monitor": mname, "alarm_t": alarm_t,
                                         "false_alarm": float(alarm_t is not None and alarm_t < T_CLEAN),
                                         "detected": float(alarm_t is not None and alarm_t >= T_CLEAN),
                                         "delay": (alarm_t - T_CLEAN) if (alarm_t is not None and alarm_t >= T_CLEAN) else np.nan,
                                         "audits_used": getattr(mon, "n_audits", 0)})
        print("seed", seed, name, "done")

res = pd.DataFrame(rows); save_csv(res, "exp4_drift_risk.csv")
mon = pd.DataFrame(mon_rows); save_csv(mon, "exp4_monitors.csv")
summ = res.groupby(["method", "drift", "severity"]).mean(numeric_only=True).drop(columns=["seed", "run"]).reset_index()
save_csv(summ, "exp4_risk_summary.csv")
msum = mon.groupby(["monitor", "severity"]).agg(false_alarm=("false_alarm", "mean"), detected=("detected", "mean"),
                                                 median_delay=("delay", "median"), audits=("audits_used", "mean")).reset_index()
save_csv(msum, "exp4_monitor_summary.csv")
pd.set_option("display.width", 220)
print(summ[["method", "drift", "severity", "risk_pre", "risk_post", "coverage_post", "acc_T_post", "acc_C_post"]].round(3).to_string())
print(msum.round(3).to_string())

fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
for drift, c in [("rotate", "tab:red"), ("noise", "tab:blue"), ("occlude", "tab:green")]:
    g = summ[(summ.drift == drift) & (summ.method == "prune90")]
    axes[0].plot(g.severity, g.risk_post, marker="o", color=c, label=drift)
axes[0].axhline(ALPHA, ls="--", color="k", lw=0.8); axes[0].set_xlabel("drift severity"); axes[0].set_ylabel("post-drift selective risk")
axes[0].set_title("certificate under drift (prune90)"); axes[0].legend(fontsize=7)
m3 = mon[mon.severity == 0.3].groupby("monitor").agg(fa=("false_alarm", "mean"), det=("detected", "mean"), delay=("delay", "median")).reset_index()
axes[1].bar(m3.monitor, m3.det, color="tab:red", label="detected (sev 0.3)")
axes[1].bar(m3.monitor, m3.fa, color="gray", alpha=0.6, label="false alarm")
axes[1].set_ylabel("rate"); axes[1].legend(fontsize=7); axes[1].set_title("monitor detection vs false alarms")
plt.setp(axes[1].get_xticklabels(), rotation=25)
fig.tight_layout(); fig.savefig(fig_path("exp4_drift_monitors.png"), dpi=130)
print("done")
