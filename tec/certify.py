"""Certified trust-equivalence via selective deployment.

Setting. A large model L is replaced in production by a compressed model C.
We want a *guarantee*, not a measurement, of how often C's prediction differs
from L's on the inputs C is allowed to answer. C may abstain (defer to L) on
inputs whose confidence score s(x) is below a threshold lambda; inputs with
s(x) >= lambda are *served* by C.

Define the selective disagreement risk
    R(lambda) = P( C(x) != L(x) | s(x) >= lambda ).
Goal: choose lambda from a calibration set (on which L is run once) such that
    P( R(lambda_hat) <= alpha ) >= 1 - delta
over the draw of the calibration set, while serving as many inputs as
possible (coverage phi(lambda) = P(s(x) >= lambda)).

Method (Learn-Then-Test / fixed-sequence testing, Angelopoulos et al. 2022;
the selective-guarantee idea follows Geifman & El-Yaniv 2017, applied here to
the *disagreement* loss instead of the error loss). Walk a grid of lambdas
from most conservative (highest threshold) to least; at each, test
H0: R(lambda) > alpha with an exact binomial p-value computed on the
calibration inputs accepted at that lambda; stop at the first lambda whose
p-value exceeds delta. Fixed-sequence testing needs no multiplicity
correction. The accepted count n(lambda) is random, but conditional on the
accepted set the disagreements are i.i.d. Bernoulli(R(lambda)) so the
binomial test is valid.

Group-conditional certificate (Mondrian): run the procedure separately per
group g with its own lambda_g, giving P(R_g <= alpha) >= 1 - delta for every
group -- a fairness guarantee that the compressed model does not silently
diverge from the large model on a protected subgroup.

Baselines: `naive_threshold` picks the smallest lambda with empirical risk
<= alpha on calibration (no correction) -- the common practice; and
`hoeffding_marginal` certifies the *unconditional* disagreement rate with no
abstention.
"""
from __future__ import annotations

import numpy as np
from scipy.stats import binom, beta


# ------------------------------------------------------------------ scores --
def margin_score(model, X):
    """Confidence score used for abstention: top-1 minus top-2 probability."""
    p = model.predict_proba(X)
    s = np.sort(p, 1)
    return s[:, -1] - s[:, -2]


def maxprob_score(model, X):
    return model.predict_proba(X).max(1)


# --------------------------------------------------------------- binomial --
def binom_pvalue_upper(k, n, alpha):
    """Exact p-value for H0: risk > alpha given k disagreements in n trials.
    P(Bin(n, alpha) <= k) is the probability of seeing at most k if the risk
    were exactly alpha (worst case within H0 is risk -> alpha+)."""
    if n == 0:
        return 1.0
    return float(binom.cdf(k, n, alpha))


def clopper_pearson_upper(k, n, delta):
    if n == 0:
        return 1.0
    if k == n:
        return 1.0
    return float(beta.ppf(1 - delta, k + 1, n - k))


# -------------------------------------------------------------- certifier --
class SelectiveCertificate:
    """Learn a threshold lambda with a (alpha, delta) selective-disagreement
    guarantee.

    Parameters
    ----------
    alpha : target bound on disagreement among served inputs
    delta : failure probability of the certificate
    n_grid: number of candidate thresholds (quantiles of calibration scores)
    min_accept: minimum number of accepted calibration points for a lambda to
                be considered (avoids vacuous tiny subsets)
    """

    def __init__(self, alpha=0.05, delta=0.1, n_grid=200, min_accept=20):
        self.alpha, self.delta, self.n_grid = alpha, delta, n_grid
        # the fixed sequence must start at a subset large enough that H0 can
        # be rejected even with zero observed disagreements:
        # (1-alpha)^n <= delta  <=>  n >= log(delta)/log(1-alpha)
        n_min_reject = int(np.ceil(np.log(delta) / np.log(1 - alpha)))
        self.min_accept = max(min_accept, n_min_reject)
        self.lam = np.inf          # inf -> serve nothing (always defer)
        self.feasible = False
        self.trace = []

    def fit(self, s_cal, dis_cal):
        """s_cal: scores on calibration inputs; dis_cal: 1 if C != L."""
        s_cal = np.asarray(s_cal); dis_cal = np.asarray(dis_cal).astype(int)
        grid = np.unique(np.quantile(s_cal, np.linspace(0, 1, self.n_grid)))[::-1]  # high -> low
        self.trace = []
        chosen = np.inf
        for lam in grid:
            acc = s_cal >= lam
            n, k = int(acc.sum()), int(dis_cal[acc].sum())
            if n < self.min_accept:
                continue
            p = binom_pvalue_upper(k, n, self.alpha)
            self.trace.append((lam, n, k, p))
            if p <= self.delta:
                chosen = lam          # H0 rejected: risk <= alpha certified at this lambda
            else:
                break                 # fixed-sequence: stop at first non-rejection
        self.lam = chosen
        self.feasible = np.isfinite(chosen)
        return self

    def serve_mask(self, s):
        return np.asarray(s) >= self.lam

    def coverage(self, s):
        return float(self.serve_mask(s).mean())


class NaiveThreshold(SelectiveCertificate):
    """No statistical correction: lowest lambda whose empirical calibration
    risk is <= alpha. This is what a practitioner tuning on a validation set
    does; it has no guarantee."""

    def fit(self, s_cal, dis_cal):
        s_cal = np.asarray(s_cal); dis_cal = np.asarray(dis_cal).astype(int)
        grid = np.unique(np.quantile(s_cal, np.linspace(0, 1, self.n_grid)))[::-1]
        chosen = np.inf
        for lam in grid:
            acc = s_cal >= lam
            n = int(acc.sum())
            if n < self.min_accept:
                continue
            if dis_cal[acc].mean() <= self.alpha:
                chosen = lam
            else:
                break
        self.lam = chosen; self.feasible = np.isfinite(chosen)
        return self


class GroupCertificate:
    """Mondrian / group-conditional selective certificate: one lambda per group."""

    def __init__(self, alpha=0.05, delta=0.1, **kw):
        self.alpha, self.delta, self.kw = alpha, delta, kw
        self.certs = {}

    def fit(self, s_cal, dis_cal, g_cal):
        for g in np.unique(g_cal):
            m = g_cal == g
            self.certs[int(g)] = SelectiveCertificate(self.alpha, self.delta, **self.kw).fit(s_cal[m], dis_cal[m])
        return self

    def serve_mask(self, s, g):
        out = np.zeros(len(s), dtype=bool)
        for gg, c in self.certs.items():
            m = g == gg
            out[m] = c.serve_mask(s[m])
        return out


def hoeffding_marginal_bound(dis_cal, delta):
    """Upper confidence bound on the unconditional disagreement rate (no
    abstention): Clopper-Pearson exact bound."""
    n, k = len(dis_cal), int(np.sum(dis_cal))
    return clopper_pearson_upper(k, n, delta)


# ------------------------------------------------------------- evaluation --
def selective_risk(serve_mask, dis):
    if serve_mask.sum() == 0:
        return 0.0
    return float(np.asarray(dis)[serve_mask].mean())


def certified_cost(serve_frac, size_C, size_L=1.0):
    """Expected relative inference cost of the certified hybrid."""
    return serve_frac * size_C + (1 - serve_frac) * size_L
