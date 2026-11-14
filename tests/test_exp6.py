import numpy as np

from exp6_real_models import overview, population_eval
from tec.certify import SelectiveCertificate


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
