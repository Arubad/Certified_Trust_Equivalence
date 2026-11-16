"""Tests for Exp 11 (CivilComments): group partition, time-ordered stream, end-to-end smoke run.

Real input is results/real/civil_preds.npz; these tests use a fake npz with the same keys.
"""
import json
import os
import shutil
import tempfile

import numpy as np

import exp11_civil as ex11

MODELS = ["q8", "p50"]
GROUP_COLS = ["male", "female", "LGBTQ", "christian", "muslim", "other_religions", "black", "white"]
DAY_2017 = 17167  # 2017-01-01 in days since 1970


def _fake_npz(path, n_cal=600, n_stream=800, seed=0):
    """Fake civil_preds.npz: civil_cal rows then civil_stream rows over 2017 (4 quarters).
    The stream rows are SHUFFLED on purpose so that sorting is exercised."""
    rng = np.random.default_rng(seed)
    n = n_cal + n_stream
    split = np.array(["civil_cal"] * n_cal + ["civil_stream"] * n_stream)
    date = np.concatenate([np.full(n_cal, DAY_2017 - 30), DAY_2017 + rng.integers(0, 365, n_stream)])
    perm = np.concatenate([np.arange(n_cal), n_cal + rng.permutation(n_stream)])
    date, split = date[perm], split[perm]
    ids = np.arange(n)
    groups = np.array(["|".join(g for g in GROUP_COLS if rng.random() < 0.15) for _ in range(n)])
    y = (rng.random(n) < 0.3).astype(int)
    p1 = np.clip(0.3 * y + 0.35 * rng.random(n), 0, 1).astype(np.float32)
    arrays = {"y": y, "split": split, "groups": groups, "date": date.astype(np.int64),
              "id": ids.astype(np.int64), "targets": np.array(["non_toxic", "toxic"])}
    arrays["prob_L"] = np.stack([1 - p1, p1], 1).astype(np.float32)
    for c in MODELS:
        q = np.clip(p1 + rng.normal(0, 0.05, n), 0, 1).astype(np.float32)
        arrays["prob_" + c] = np.stack([1 - q, q], 1).astype(np.float32)
    meta = {"L": {"size": 1.0, "per_1k": 100.0}, "q8": {"size": 0.42, "per_1k": 90.0},
            "p50": {"size": 0.61, "per_1k": 94.0}}
    np.savez_compressed(path, meta=json.dumps(meta), **arrays)
    return path


def _tmpdir():
    return tempfile.mkdtemp(prefix="exp11_test_")


def test_civil_groups_partition():
    # first match in the order muslim, black, LGBTQ, white, christian, other_religions, female, male
    assert ex11.civil_group("male|female") == "female"
    assert ex11.civil_group("male|christian|muslim") == "muslim"
    assert ex11.civil_group("black|muslim") == "muslim"
    assert ex11.civil_group("white|male") == "white"
    assert ex11.civil_group("LGBTQ|female") == "LGBTQ"
    assert ex11.civil_group("christian|other_religions") == "christian"
    assert ex11.civil_group("other_religions|male") == "other_religions"
    assert ex11.civil_group("male") == "male"
    assert ex11.civil_group("") == "none"


def test_civil_stream_sorted():
    d = _tmpdir()
    try:
        Z = ex11.load_npz(_fake_npz(os.path.join(d, "fake.npz")))
        raw = Z["date"][Z["split"] == "civil_stream"]
        assert not np.all(np.diff(raw) >= 0), "fake stream must be unsorted for this test to bite"
        cal, st = ex11.prepare(Z)
        assert len(cal) == 600 and len(st) == 800
        assert np.all(Z["split"][st] == "civil_stream") and np.all(Z["split"][cal] == "civil_cal")
        assert np.all(np.diff(Z["date"][st]) >= 0), "monitor stream not non-decreasing in date"
    finally:
        shutil.rmtree(d)


def test_civil_quarters():
    days = np.array([DAY_2017, DAY_2017 + 31 + 28, DAY_2017 + 91, DAY_2017 + 364])  # Jan 1, Mar 1, Apr 1, Dec 31
    assert list(ex11.quarter_of(days)) == ["2017Q1", "2017Q1", "2017Q2", "2017Q4"]


def test_exp11_end_to_end_smoke():
    d = _tmpdir()
    try:
        npz = _fake_npz(os.path.join(d, "fake.npz"))
        out = os.path.join(d, "out")
        ex11.main(["--npz", npz, "--out", out, "--models", ",".join(MODELS), "--reps", "3", "--mon-reps", "2"])
        for name in ["exp11_overview.csv", "exp11_meta.csv", "exp11a_summary.csv", "exp11b_summary.csv",
                     "exp11c_quarterly.csv", "exp11d_monitors.csv", "exp11e_streaming.csv"]:
            assert os.path.exists(os.path.join(out, name)), name
        import pandas as pd
        q = pd.read_csv(os.path.join(out, "exp11c_quarterly.csv"))
        assert sorted(q.quarter.unique()) == ["2017Q1", "2017Q2", "2017Q3", "2017Q4"]
        assert set(q.model) == set(MODELS)
        ok = q.risk.notna()
        assert (q.ci_lo[ok] <= q.risk[ok] + 1e-12).all() and (q.risk[ok] <= q.ci_hi[ok] + 1e-12).all()
        assert {"n_served", "k_served", "ci_lo", "ci_hi"} <= set(q.columns)
    finally:
        shutil.rmtree(d)


def test_clopper_pearson_ci():
    # k = 0 of n = 10: lower bound 0, upper bound 1 - 0.025**(1/10)  (exact two-sided 95%)
    lo, hi = ex11.clopper_pearson_ci(0, 10)
    assert lo == 0.0 and abs(hi - (1 - 0.025 ** 0.1)) < 1e-9
    lo, hi = ex11.clopper_pearson_ci(10, 10)
    assert hi == 1.0 and abs(lo - 0.025 ** 0.1) < 1e-9
    lo, hi = ex11.clopper_pearson_ci(100, 2000)
    assert lo < 0.05 < hi and abs((lo + hi) / 2 - 0.05) < 0.01


def test_blind_zone_rmin():
    # r_min = max_q-root of KL_q(r) for alpha = 0.05, q in {2 alpha, 4 alpha}: about 0.0724
    r = ex11.blind_zone_rmin()
    assert abs(r - 0.0724) < 1e-3, r
    from tec.monitor import expected_audit_delay
    assert expected_audit_delay(r - 0.002, 0.05, 0.10, 0.10, 1.0) == float("inf")
    assert np.isfinite(expected_audit_delay(r + 0.002, 0.05, 0.10, 0.10, 1.0))


def test_simulate_policy_pinned():
    # Regression pin of exp6 simulate_policy (shared with exp11e): fixed seeds, tiny stream.
    # Stream: 300 rows with no drift, then 300 rows with disagreement rate 0.25; disagreement-free calibration.
    import exp6_real_models as ex6
    rs = np.random.default_rng(0)
    n = 600
    s = rs.uniform(0, 1, n)
    d = (rs.uniform(0, 1, n) < np.where(np.arange(n) < 300, 0.0, 0.25)).astype(int)
    sc = np.random.default_rng(1).uniform(0, 1, 400)
    dc = np.zeros(400, int)
    cases = {"always-C": (600, 0, 0, 0), "static": (598, 2, 0, 280),
             "audit10": (505, 144, 1, 63109), "adaptive": (598, 14, 0, 3266)}
    for pol, (sv_n, lc_n, nre, lc_idx) in cases.items():
        servedC, Lc, nr = ex6.simulate_policy(pol, s, d, sc, dc, 0.05, 0.10 if pol == "audit10" else 0.0, seed=0)
        assert int(servedC.sum()) == sv_n, (pol, int(servedC.sum()))
        assert int(Lc.sum()) == lc_n, (pol, int(Lc.sum()))
        assert nr == nre, (pol, nr)
        assert int(np.flatnonzero(Lc).sum()) == lc_idx, (pol, int(np.flatnonzero(Lc).sum()))
