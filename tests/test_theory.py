import numpy as np

from tec.monitor import ShadowAuditMartingale, expected_audit_delay


def _simulate_delay(r, alpha, delta, budget, n_streams, cap, seed0=0):
    """Delays (in update() calls) of ShadowAuditMartingale on streams with d ~ Bern(r)."""
    delays = np.empty(n_streams)
    for k in range(n_streams):
        mon = ShadowAuditMartingale(alpha, delta, budget, seed=seed0 + k)
        d_rng = np.random.default_rng(10_000 + seed0 + k)
        delay = cap
        for t in range(1, cap + 1):
            if mon.update(lambda: d_rng.random() < r):
                delay = t
                break
        delays[k] = delay
    return delays


def test_delay_inf_under_null():
    # r <= alpha: max_q KL_q(r) <= 0, so the monitor never alarms in expectation
    assert expected_audit_delay(0.05, 0.05, 0.10, 0.10) == np.inf
    assert expected_audit_delay(0.03, 0.05, 0.10, 0.10) == np.inf
    # blind zone: alpha < r < r_q, where the drift is still non-positive for all q
    assert expected_audit_delay(0.06, 0.05, 0.10, 0.10) == np.inf


def test_delay_matches_simulation():
    alpha, delta, n_streams, horizon_mult = 0.05, 0.10, 500, 20
    for r in (0.10, 0.20):
        for budget in (0.05, 0.10):
            formula = expected_audit_delay(r, alpha, delta, budget)
            assert np.isfinite(formula) and formula > 0
            cap = int(horizon_mult * formula)
            delays = _simulate_delay(r, alpha, delta, budget, n_streams, cap, seed0=0)
            ratio = delays.mean() / formula
            assert 0.7 <= ratio <= 1.6, (
                f"r={r} b={budget}: simulated mean {delays.mean():.1f} vs formula {formula:.1f} "
                f"(ratio {ratio:.3f} outside [0.7, 1.6])")
