"""Experiment 9 -- split-grid certificate (theory.md Theorem 1', section 1.4).

Compares mode="same" (data-dependent grid, skip rule, as in Exp 1-6) with
mode="split" (grid and start index from a design split, no skip rule) on:

  (a) Exp 1 setting: digits and toxic, 3 teacher seeds, all compressions of
      experiments/common.py, n_cal in {200, 400, 1000}, 200 random splits of the pool.
      Digits' pool has 899 rows, so n_cal = 1000 is not possible there and is skipped.
  (b) real HateXplain pool (results/real/real_preds.npz), population mode as in Exp 6a:
      models int8/q4/p30, alpha in {0.02, 0.05}, n_cal in {500, 1000}, 300 bootstrap draws
      (with replacement from the pool); risk and coverage exact on the whole pool.
  (c) group (Mondrian) certificate on Exp 6b's partition, int8 only, alpha = 0.05,
      n_cal = 2000, 300 bootstrap draws; one row per target group.

Output: results/exp9_split_certificate.csv, one row per (setting, model, alpha, n_cal, mode),
with violation rate, mean coverage and feasible rate over draws. Group rows use
setting = "6b_group_<name>".
"""
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)                                   # common.py
sys.path.insert(0, os.path.join(HERE, "real"))             # exp6_real_models.py
sys.path.insert(0, os.path.join(HERE, ".."))
from common import load_task, train_teacher, compress_all, disagreement, split_pool, save_csv  # noqa: E402
from exp6_real_models import OUT, DELTA, COMPRESSED, margin, first_group, MAIN_GROUPS, population_eval  # noqa: E402
from tec.certify import SelectiveCertificate, GroupCertificate, selective_risk, margin_score  # noqa: E402

MODES = ["same", "split"]
rows = []


def add(setting, model, alpha, n_cal, mode, viol, cov, feas):
    rows.append({"setting": setting, "model": model, "alpha": alpha, "n_cal": n_cal, "mode": mode,
                 "violation": float(np.mean(viol)), "coverage": float(np.mean(cov)),
                 "feasible": float(np.mean(feas))})


# ------------------------------------------------------------- (a) Exp 1 --
ALPHA_A, N_SPLITS = 0.05, 200
for task in ["digits", "toxic"]:
    # SKIP: digits n_cal=1000 is not run. The digits pool has 899 rows, so drawing 1000
    # without replacement is impossible (Exp 1 also uses only 100/200/400 for digits).
    n_cals = [200, 400] if task == "digits" else [200, 400, 1000]
    # pooled over the 3 teacher seeds and 200 splits: one row per (task, model, n_cal, mode)
    acc_a = {(name, n_cal, m): {"viol": [], "cov": [], "feas": []}
             for name in ["prune80", "prune90", "pruneFT90", "quant3b", "quant4b", "lowrank25", "distill25"]
             for n_cal in n_cals for m in MODES}
    for seed in range(3):
        (Xtr, ytr, _), (Xpo, ypo, _) = load_task(task, seed)
        T = train_teacher(task, Xtr, ytr, seed)
        comps, _sizes = compress_all(T, Xtr, ytr, task, seed)
        for name, C in comps.items():
            s_all = margin_score(C, Xpo)
            d_all = disagreement(T, C, Xpo)
            for n_cal in n_cals:
                rng = np.random.default_rng(seed)          # same split stream as Exp 1
                for r in range(N_SPLITS):
                    ca, te = split_pool(len(Xpo), n_cal, rng)
                    for m in MODES:
                        cert = SelectiveCertificate(ALPHA_A, DELTA, mode=m).fit(s_all[ca], d_all[ca])
                        sm = cert.serve_mask(s_all[te])
                        a_ = acc_a[(name, n_cal, m)]
                        a_["viol"].append(selective_risk(sm, d_all[te]) > ALPHA_A)
                        a_["cov"].append(float(sm.mean()))
                        a_["feas"].append(cert.feasible)
        print(f"[a] {task} seed {seed} done", flush=True)
    for (name, n_cal, m), a_ in acc_a.items():
        add(f"exp1_{task}", name, ALPHA_A, n_cal, m, a_["viol"], a_["cov"], a_["feas"])

# ---------------------------------------------- (b) real pool, population --
Z = np.load(os.path.join(OUT, "real_preds.npz"), allow_pickle=False)
hx = np.isin(Z["split"], ["val", "test"])
idx_hx = np.where(hx)[0]
predL = Z["prob_L"].argmax(1)
S = {c: margin(Z["prob_" + c]) for c in COMPRESSED}
D = {c: (Z["prob_" + c].argmax(1) != predL).astype(int) for c in COMPRESSED}
for c in ["int8", "q4", "p30"]:
    s, d = S[c][idx_hx], D[c][idx_hx]
    for alpha in [0.02, 0.05]:
        for n_cal in [500, 1000]:
            rng = np.random.default_rng(0)                 # same draw stream as Exp 6a
            acc = {m: {"viol": [], "cov": [], "feas": []} for m in MODES}
            for r in range(300):
                ca = rng.integers(0, len(s), n_cal)
                for m in MODES:
                    cert = SelectiveCertificate(alpha, DELTA, mode=m).fit(s[ca], d[ca])
                    risk, cov = population_eval(cert, s, d)
                    acc[m]["viol"].append(risk > alpha)
                    acc[m]["cov"].append(cov)
                    acc[m]["feas"].append(cert.feasible)
            for m in MODES:
                add("real_pool_population", c, alpha, n_cal, m, acc[m]["viol"], acc[m]["cov"], acc[m]["feas"])
    print(f"[b] {c} done", flush=True)

# ------------------------------------------------ (c) group, Exp 6b partition --
G = np.array([first_group(g) for g in Z["groups"]])
GNAMES = MAIN_GROUPS + ["other"]
c, alpha, n_cal = "int8", 0.05, 2000
s, d, g = S[c][idx_hx], D[c][idx_hx], G[idx_hx]
rng = np.random.default_rng(1)                             # same draw stream as Exp 6b
acc = {(m, gi): {"viol": [], "cov": [], "feas": []} for m in MODES for gi in range(len(GNAMES))}
for r in range(300):
    ca = rng.integers(0, len(s), n_cal)
    for m in MODES:
        grp = GroupCertificate(alpha, DELTA, mode=m).fit(s[ca], d[ca], g[ca])
        sm = grp.serve_mask(s, g)
        for gi in range(len(GNAMES)):
            mg = g == gi
            risk = selective_risk(sm[mg], d[mg])
            acc[(m, gi)]["viol"].append(risk > alpha)
            acc[(m, gi)]["cov"].append(float(sm[mg].mean()))
            acc[(m, gi)]["feas"].append(gi in grp.certs and grp.certs[gi].feasible)
for m in MODES:
    for gi, gn in enumerate(GNAMES):
        a_ = acc[(m, gi)]
        add(f"6b_group_{gn}", c, alpha, n_cal, m, a_["viol"], a_["cov"], a_["feas"])
print("[c] done", flush=True)

out = pd.DataFrame(rows, columns=["setting", "model", "alpha", "n_cal", "mode", "violation", "coverage", "feasible"])
save_csv(out, "exp9_split_certificate.csv")
pd.set_option("display.width", 200)
print(out.round(3).to_string(index=False))
