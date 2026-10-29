"""Trust-equivalence metrics between a reference (large) model L and a
compressed model C on the same inputs.

Performance parity:   acc(L), acc(C)
Prediction agreement: agreement rate, Cohen's kappa, flip rate on
                      correctly-classified-by-L inputs, "silent flips"
                      (C wrong where L right) vs "repairs" (C right, L wrong)
Confidence parity:    ECE of each model, mean |p_L - p_C| (TV distance),
                      Spearman correlation of max-prob confidences,
                      KL(L||C) of predictive distributions
Trust-Equivalence Score (TES): a single number in [0,1] combining
                      agreement, confidence TV distance and calibration gap:
                      TES = agreement * (1 - TV) * (1 - |ECE_C - ECE_L|)
"""
from __future__ import annotations

import numpy as np
from scipy.stats import spearmanr


def accuracy(p, y):
    return float((p.argmax(1) == y).mean())


def ece(p, y, n_bins=15):
    conf = p.max(1); pred = p.argmax(1); acc = (pred == y).astype(float)
    bins = np.linspace(0, 1, n_bins + 1)
    e = 0.0
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            e += m.mean() * abs(acc[m].mean() - conf[m].mean())
    return float(e)


def cohen_kappa(a, b):
    n = len(a)
    po = (a == b).mean()
    C = max(a.max(), b.max()) + 1
    pa = np.bincount(a, minlength=C) / n
    pb = np.bincount(b, minlength=C) / n
    pe = (pa * pb).sum()
    return float((po - pe) / (1 - pe + 1e-12))


def trust_report(pL, pC, y):
    aL, aC = pL.argmax(1), pC.argmax(1)
    agree = (aL == aC)
    L_right = aL == y
    tv = 0.5 * np.abs(pL - pC).sum(1)
    kl = (pL * (np.log(pL + 1e-12) - np.log(pC + 1e-12))).sum(1)
    rho = spearmanr(pL.max(1), pC.max(1)).correlation
    eL, eC = ece(pL, y), ece(pC, y)
    rep = {
        "acc_L": accuracy(pL, y), "acc_C": accuracy(pC, y),
        "acc_gap": accuracy(pL, y) - accuracy(pC, y),
        "agreement": float(agree.mean()), "kappa": cohen_kappa(aL, aC),
        "flip_rate": float(1 - agree.mean()),
        "silent_flip_rate": float((L_right & ~agree).mean()),   # L right, C wrong
        "repair_rate": float((~L_right & (aC == y)).mean()),     # L wrong, C right
        "tv_mean": float(tv.mean()), "kl_mean": float(kl.mean()),
        "conf_spearman": float(rho) if rho == rho else 0.0,
        "ece_L": eL, "ece_C": eC, "ece_gap": float(abs(eC - eL)),
    }
    rep["TES"] = float(rep["agreement"] * (1 - rep["tv_mean"]) * (1 - rep["ece_gap"]))
    return rep


def subgroup_report(pL, pC, y, groups):
    out = {}
    for g in np.unique(groups):
        m = groups == g
        r = trust_report(pL[m], pC[m], y[m])
        out[int(g)] = {"n": int(m.sum()), "acc_L": r["acc_L"], "acc_C": r["acc_C"],
                       "agreement": r["agreement"], "silent_flip_rate": r["silent_flip_rate"], "TES": r["TES"]}
    vals = np.array([v["agreement"] for v in out.values()])
    sf = np.array([v["silent_flip_rate"] for v in out.values()])
    return out, {"agreement_disparity": float(vals.max() - vals.min()),
                 "silent_flip_disparity": float(sf.max() - sf.min())}
