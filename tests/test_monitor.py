"""Tests for AdaptiveAuditMartingale (predictable KS-triggered audit rate)."""
import os
import sys

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "experiments"))


def test_adaptive_null_false_alarm():
    # Scores shift at t=1500 (so the KS suspicion must fire) but the disagreement
    # rate stays at alpha, i.e. H0 holds. Ville: P(alarm) <= delta = 0.10.
    from tec.monitor import AdaptiveAuditMartingale
    alpha, delta, T, shift_at, n_streams = 0.05, 0.10, 3000, 1500, 400
    s_cal = np.random.default_rng(0).beta(2, 5, size=400)
    alarms, suspicious = 0, 0
    for k in range(n_streams):
        r = np.random.default_rng(1000 + k)
        s = np.concatenate([r.beta(2, 5, shift_at), r.beta(5, 2, T - shift_at)])
        d = (r.random(T) < alpha).astype(int)
        mon = AdaptiveAuditMartingale(alpha, delta, s_cal=s_cal, seed=k)
        fired, used_hi = False, False
        for t in range(T):
            if mon.update(s[t], int(d[t])):
                fired = True
                break
            used_hi = used_hi or mon.budget_now == mon.b_hi
        alarms += fired
        suspicious += used_hi
    # sanity: the KS trigger actually fired on (almost) every stream, so the test is not vacuous
    assert suspicious / n_streams >= 0.9, f"KS suspicion fired on only {suspicious}/{n_streams} streams"
    rate = alarms / n_streams
    assert rate <= 0.12, f"false-alarm rate {rate:.3f} > 0.12 (delta = {delta})"


def _first_alarm(mon, s, d, feed_scores):
    """Index of first alarm (or len(d) if none): censored delay."""
    for t in range(len(d)):
        a = mon.update(s[t], int(d[t])) if feed_scores else mon.update(int(d[t]))
        if a:
            return t
    return len(d)


def test_adaptive_detects_faster_than_fixed_lo():
    # disagreement rate 0.05 -> 0.20 at t = 1500, together with a score shift
    from tec.monitor import AdaptiveAuditMartingale, ShadowAuditMartingale
    alpha, delta, T_CH, T, n_runs = 0.05, 0.10, 1500, 8000, 100
    s_cal = np.random.default_rng(0).beta(2, 5, size=400)
    delay_ad, delay_lo, audits_ad, audits_hi = [], [], [], []
    early_ad, early_lo = 0, 0  # pre-change alarms: counted, kept out of the medians
    for k in range(n_runs):
        r = np.random.default_rng(5000 + k)
        s = np.concatenate([r.beta(2, 5, T_CH), r.beta(5, 2, T - T_CH)])
        d = np.concatenate([r.random(T_CH) < 0.05, r.random(T - T_CH) < 0.20]).astype(int)
        ad = AdaptiveAuditMartingale(alpha, delta, s_cal=s_cal, seed=k)
        lo = ShadowAuditMartingale(alpha, delta, budget=0.02, seed=k)
        hi = ShadowAuditMartingale(alpha, delta, budget=0.20, seed=k)
        a = _first_alarm(ad, s, d, True)
        if a < T_CH:
            early_ad += 1
        else:
            delay_ad.append(a - T_CH)
        audits_ad.append(ad.n_audits)
        a = _first_alarm(lo, s, d, False)
        if a < T_CH:
            early_lo += 1
        else:
            delay_lo.append(a - T_CH)
        _first_alarm(hi, s, d, False)
        audits_hi.append(hi.n_audits)
    med_ad, med_lo = float(np.median(delay_ad)), float(np.median(delay_lo))
    assert med_ad < 0.6 * med_lo, (
        f"adaptive median delay {med_ad} not < 0.6 x fixed-lo {med_lo} "
        f"(pre-change alarms: adaptive {early_ad}, fixed-lo {early_lo})")
    assert np.mean(audits_ad) <= np.mean(audits_hi), (
        f"adaptive audits {np.mean(audits_ad):.1f} > fixed-0.20 audits {np.mean(audits_hi):.1f}")


def test_policy_infeasible_certificate():
    # lam = inf: nothing is served, so nothing is audited and nothing can alarm
    from exp7_adaptive_audit import run_stream
    from tec.monitor import AdaptiveAuditMartingale
    rng = np.random.default_rng(0)
    s = rng.beta(2, 5, size=1000)
    d = (rng.random(1000) < 0.05).astype(int)
    out = run_stream(s, d, lam=np.inf, monitor=AdaptiveAuditMartingale(0.05, s_cal=s[:200]))
    assert out["coverage"] == 0
    assert out["risk"] == 0
    assert out["alarm_t"] is None
    assert out["n_audits"] == 0
