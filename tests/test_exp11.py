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
    finally:
        shutil.rmtree(d)
