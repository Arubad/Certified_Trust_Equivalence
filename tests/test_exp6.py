import json
import os

import numpy as np
import pandas as pd

import exp6_real_models as ex
from exp6_real_models import overview, population_eval
from tec.certify import SelectiveCertificate

REAL = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results", "real")


def _pair(N, flips):
    """GPU-style and CPU-style npz dicts whose HateXplain prob_L argmax differs on `flips` rows."""
    y = np.arange(N) % 3
    split = np.array(["val"] * N)
    gpu = np.eye(3)[y]
    cpu = gpu.copy()
    cpu[:flips] = np.eye(3)[(y[:flips] + 1) % 3]
    return {"y": y, "split": split, "prob_L": gpu}, {"y": y, "split": split, "prob_L": cpu}


def test_cli_defaults_reproduce_exp6():
    # defaults of the CLI must give exactly the committed Exp 6 overview (tag exp6, all six compressions)
    args = ex.parse_args([])
    assert args.tag == "exp6" and args.npz is None and args.models == ex.COMPRESSED
    with np.load(os.path.join(REAL, "real_preds.npz"), allow_pickle=False) as F:
        Z = {k: F[k] for k in F.files}
    meta = json.loads(str(Z["meta"]))
    ov = overview(Z, meta, models=args.models)
    ref = pd.read_csv(os.path.join(REAL, "exp6_overview.csv"))
    assert list(ov.columns) == list(ref.columns), list(ov.columns)
    assert list(ov.model) == list(ref.model)
    for col in ["size", "latency_rel", "acc_hx", "agree_hx", "acc_dv", "agree_dv"]:
        assert np.allclose(ov[col].to_numpy(float), ref[col].to_numpy(float), rtol=0, atol=1e-9,
                           equal_nan=True), col


def test_check_L_uses_gpu_when_agreement_high():
    Z, R = _pair(400, 1)                       # agreement 0.9975 >= 0.995
    info, P = ex.check_L(Z, R)
    assert info["L_source"] == "gpu" and np.array_equal(P, Z["prob_L"])


def test_check_L_falls_back_to_cpu():
    Z, R = _pair(400, 10)                      # agreement 0.975 < 0.995
    info, P = ex.check_L(Z, R)
    assert info["L_source"] == "cpu" and np.array_equal(P, R["prob_L"])


def test_check_L_rejects_row_mismatch():
    Z, R = _pair(10, 0)
    R["y"] = R["y"][::-1].copy()
    raised = False
    try:
        ex.check_L(Z, R)
    except ValueError:
        raised = True
    assert raised


def test_latency_only_for_speedup_models():
    # latency is taken from the npz's own timings (same run). Only L and the models with a real
    # speed-up (int8, distil6) get latency_rel; pruned / fake-quant models run dense -> NaN.
    Z = {"y": np.array([0, 1]), "split": np.array(["val", "val"]), "prob_L": np.eye(3)[[0, 1]],
         "prob_distil6": np.eye(3)[[0, 1]], "prob_p50ft": np.eye(3)[[0, 1]]}
    meta = {"L": {"size": 1.0, "per_1k": 0.5}, "distil6": {"size": 0.6, "per_1k": 0.2},
            "p50ft": {"size": 0.6, "per_1k": 0.4}}
    ov = overview(Z, meta, models=["distil6", "p50ft"]).set_index("model")
    assert np.isclose(ov.loc["L", "latency_rel"], 1.0)
    assert np.isclose(ov.loc["distil6", "latency_rel"], 0.4), ov.loc["distil6", "latency_rel"]
    assert np.isnan(ov.loc["p50ft", "latency_rel"])


def test_overview_excludes_no_majority():
    # 6 HateXplain rows; two have y = -1 (no majority label)
    split = np.array(["val", "test", "val", "test", "val", "test"])
    y = np.array([0, 1, 2, -1, -1, 0])
    prob_L = np.eye(3)[[0, 1, 2, 0, 0, 0]]
    prob_int8 = np.eye(3)[[0, 1, 2, 0, 0, 1]]
    Z = {"y": y, "split": split, "prob_L": prob_L, "prob_int8": prob_int8}
    meta = {"L": {"size": 1.0, "per_1k": 1.0}, "int8": {"size": 0.25, "per_1k": 0.5}}
    ov = overview(Z, meta).set_index("model")
    # acc uses the 4 labelled rows only: L is right on all 4, int8 on 3 of 4
    assert np.isclose(ov.loc["L", "acc_hx"], 1.0), ov.loc["L", "acc_hx"]
    assert np.isclose(ov.loc["int8", "acc_hx"], 0.75), ov.loc["int8", "acc_hx"]
    # agreement uses all 6 rows: int8 differs from L only on row 5
    assert np.isclose(ov.loc["int8", "agree_hx"], 5 / 6), ov.loc["int8", "agree_hx"]


def test_population_violation_zero_when_d_independent():
    # pool where d is independent of s with rate 0.02 < alpha = 0.05:
    # population-mode violation must be 0 in every split (i.i.d. draws with replacement)
    rng = np.random.default_rng(0)
    N = 20000
    s = rng.uniform(0, 1, N)
    d = (rng.uniform(0, 1, N) < 0.02).astype(int)
    viol = 0
    for _ in range(100):
        ca = rng.integers(0, N, 500)
        cert = SelectiveCertificate(0.05, 0.1).fit(s[ca], d[ca])
        risk, _ = population_eval(cert, s, d)
        viol += int(risk > 0.05)
    assert viol == 0, f"{viol} population violations out of 100 splits"
