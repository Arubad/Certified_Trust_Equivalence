"""Drift monitors that detect when a selective certificate has been voided.

After deployment only C runs; L is not available for every input. Two
monitors with different information budgets:

1. ``DeferralRateCUSUM`` (label-free, zero cost). Under the calibration
   distribution the fraction of inputs with s(x) < lambda (deferred) is
   phi0 = 1 - coverage. Covariate drift usually pushes scores down, so the
   deferral rate rises. A one-sided Page CUSUM on the Bernoulli deferral
   indicator raises an alarm when the rate exceeds phi0 + eps persistently.
   It needs no call to L, but it only sees *symptoms* of drift.

2. ``ShadowAuditMartingale`` (audit budget b). Each served input is, with
   probability b, also sent to L ("shadow audit") and the disagreement
   indicator d in {0,1} is observed. We test H0: P(d=1) <= alpha with a
   mixture-SPRT test martingale
       M_t = prod_i  [ q^d_i (1-q)^(1-d_i) ] / [ alpha^d_i (1-alpha)^(1-d_i) ]
   with alternative q > alpha (we use q = min(1, 2*alpha) and q = 4*alpha
   mixed). Under H0 M_t is a non-negative supermartingale, so by Ville's
   inequality P(sup_t M_t >= 1/delta) <= delta: the false-alarm probability
   over the *entire* stream is at most delta, with no horizon or
   multiplicity correction. Alarm => the certificate is void => recalibrate.

Both monitors expose ``update(x)`` returning True on alarm and ``reset()``.
"""
from __future__ import annotations

import numpy as np


class DeferralRateCUSUM:
    def __init__(self, phi0, eps=0.05, h=5.0):
        """phi0: deferral rate at calibration; eps: minimal shift to detect;
        h: decision threshold (log-likelihood units)."""
        self.phi0, self.eps, self.h = float(np.clip(phi0, 1e-3, 0.99)), eps, h
        self.reset()

    def reset(self):
        self.S = 0.0
        self.t = 0

    def update(self, deferred: bool):
        p0 = float(np.clip(self.phi0, 1e-3, 0.99)); p1 = min(0.999, p0 + self.eps)
        d = int(deferred)
        llr = d * np.log(p1 / p0) + (1 - d) * np.log((1 - p1) / (1 - p0))
        self.S = max(0.0, self.S + llr)
        self.t += 1
        return self.S >= self.h


class ShadowAuditMartingale:
    def __init__(self, alpha, delta=0.05, budget=0.05, seed=0):
        self.alpha, self.delta, self.budget = alpha, delta, budget
        self.rng = np.random.default_rng(seed)
        self.qs = [min(0.95, 2 * alpha), min(0.95, 4 * alpha)]
        self.reset()

    def reset(self):
        self.logM = np.zeros(len(self.qs))
        self.n_audits = 0
        self.t = 0

    def update(self, dis_if_audited):
        """dis_if_audited: a callable or value giving the disagreement
        indicator if we choose to audit this input (we only 'pay' when we do)."""
        self.t += 1
        if self.rng.random() >= self.budget:
            return False
        d = int(dis_if_audited() if callable(dis_if_audited) else dis_if_audited)
        self.n_audits += 1
        a = self.alpha
        for i, q in enumerate(self.qs):
            self.logM[i] += d * np.log(q / a) + (1 - d) * np.log((1 - q) / (1 - a))
        # mixture over alternatives (average of martingales is a martingale)
        M = np.mean(np.exp(self.logM))
        return M >= 1.0 / self.delta


class ScoreKS:
    """Sliding-window two-sample Kolmogorov-Smirnov test of the score
    distribution against the calibration scores (label-free, batch-style
    baseline, Bonferroni over windows)."""

    def __init__(self, s_cal, window=200, delta=0.05, n_windows_expected=50):
        from scipy.stats import ks_2samp
        self.s_cal = np.asarray(s_cal); self.window = window
        self.thr = delta / n_windows_expected
        self.ks = ks_2samp
        self.reset()

    def reset(self):
        self.buf = []
        self.t = 0

    def update(self, s):
        self.t += 1
        self.buf.append(s)
        if len(self.buf) < self.window:
            return False
        p = self.ks(np.asarray(self.buf), self.s_cal).pvalue
        self.buf = []
        return p < self.thr
