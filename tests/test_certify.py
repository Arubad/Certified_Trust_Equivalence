import numpy as np

from tec.certify import GroupCertificate, SelectiveCertificate


def test_certificate_validity_synthetic():
    alpha, delta, n, n_rep = 0.05, 0.1, 500, 300
    rng = np.random.default_rng(0)
    violations = feasible = 0
    for _ in range(n_rep):
        s = rng.uniform(0, 1, n)
        d = (rng.uniform(0, 1, n) < 0.2 * (1 - s)).astype(int)
        cert = SelectiveCertificate(alpha, delta).fit(s, d)
        if cert.feasible:
            feasible += 1
            # exact selective risk: P(d=1 | s >= lam) = 0.2 * (1 - lam)^2 / 2 / (1 - lam)
            true_risk = 0.1 * (1 - cert.lam)
        else:
            true_risk = 0.0  # serves nothing
        violations += int(true_risk > alpha)
    assert feasible > n_rep // 2, f"certificate almost never feasible: {feasible}/{n_rep}"
    rate = violations / n_rep
    assert rate <= 0.13, f"violation rate {rate:.3f} > 0.13 (delta={delta})"


def test_group_too_small_serves_nothing():
    rng = np.random.default_rng(1)
    s = rng.uniform(0, 1, 30)
    d = (rng.uniform(0, 1, 30) < 0.01).astype(int)
    g = np.zeros(30, dtype=int)
    cert = GroupCertificate(0.05, 0.1).fit(s, d, g)
    assert cert.certs[0].lam == np.inf
    assert not cert.serve_mask(s, g).any()


def _eta_nonmonotone(s):
    """Jagged disagreement curve: 0.3 on [0.95, 1], 0 on [0.6, 0.95), 0.12 below."""
    return np.where(s >= 0.95, 0.3, np.where(s >= 0.6, 0.0, 0.12))


def _exact_risk_nonmonotone(lam):
    """R(lam) = P(d=1 | s >= lam) under s ~ U(0,1) and the eta above."""
    if not np.isfinite(lam):
        return 0.0
    total = 0.0
    for a, b, e in [(0.0, 0.6, 0.12), (0.6, 0.95, 0.0), (0.95, 1.0, 0.3)]:
        lo = max(a, lam)
        if b > lo:
            total += e * (b - lo)
    return total / (1 - lam)


def test_split_mode_validity_nonmonotone():
    alpha, delta, n, n_rep = 0.05, 0.1, 600, 400
    rng = np.random.default_rng(0)
    violations = feasible = 0
    for _ in range(n_rep):
        s = rng.uniform(0, 1, n)
        d = (rng.uniform(0, 1, n) < _eta_nonmonotone(s)).astype(int)
        cert = SelectiveCertificate(alpha, delta, mode="split").fit(s, d)
        feasible += int(cert.feasible)
        violations += int(_exact_risk_nonmonotone(cert.lam) > alpha)
    # Validity only. Feasibility is not asserted: R > alpha at the top of the grid (a true
    # null), so a high-to-low fixed sequence stops at its first hypothesis by construction.
    assert violations / n_rep <= 0.13, f"violation rate {violations / n_rep:.3f} > 0.13"


def _eta_feasible(s):
    """0 on [0.6, 1], 0.4 on [0.5, 0.6), 0 on [0.3, 0.5), 0.2 below. Feasible region lam >~ 0.545."""
    return np.where(s >= 0.6, 0.0, np.where(s >= 0.5, 0.4, np.where(s >= 0.3, 0.0, 0.2)))


def _exact_risk_feasible(lam):
    if not np.isfinite(lam):
        return 0.0
    total = 0.0
    for a, b, e in [(0.0, 0.3, 0.2), (0.3, 0.5, 0.0), (0.5, 0.6, 0.4), (0.6, 1.0, 0.0)]:
        lo = max(a, lam)
        if b > lo:
            total += e * (b - lo)
    return total / (1 - lam)


def test_split_mode_feasible_nonmonotone():
    alpha, delta, n, n_rep = 0.05, 0.1, 600, 400
    rng = np.random.default_rng(0)
    violations = feasible = 0
    for _ in range(n_rep):
        s = rng.uniform(0, 1, n)
        d = (rng.uniform(0, 1, n) < _eta_feasible(s)).astype(int)
        cert = SelectiveCertificate(alpha, delta, mode="split").fit(s, d)
        feasible += int(cert.feasible)
        violations += int(_exact_risk_feasible(cert.lam) > alpha)
    assert violations / n_rep <= 0.13, f"violation rate {violations / n_rep:.3f} > 0.13"
    assert feasible / n_rep >= 0.5, f"feasible rate {feasible / n_rep:.3f} < 0.5"


def test_split_mode_small_n_infeasible():
    rng = np.random.default_rng(0)
    s = rng.uniform(0, 1, 50)
    d = (rng.uniform(0, 1, 50) < 0.01).astype(int)
    cert = SelectiveCertificate(0.05, 0.1, mode="split").fit(s, d)
    assert cert.lam == np.inf
    assert not cert.feasible
    assert cert.trace == []


def test_split_mode_start_fallback():
    # Step 4 fallback j0 = G'-1: 2*m0 <= n, but n_T < 2*m0, so no grid point can reach the
    # 2*m0 rescaled design count and the walk tests only the last grid value, min s[D].
    n, alpha, delta = 500, 0.01, 0.1
    rng = np.random.default_rng(0)
    s = rng.uniform(0, 1, n)
    cert = SelectiveCertificate(alpha, delta, mode="split")
    m0 = cert.min_accept
    assert m0 == int(np.ceil(np.log(delta) / np.log(1 - alpha))) == 230
    n_D = int(np.floor(cert.design_frac * n)); n_T = n - n_D
    assert 2 * m0 <= n and n_T < 2 * m0, (m0, n_D, n_T)          # the fallback case
    perm = np.random.default_rng(cert.seed).permutation(n)
    D, T = perm[:n_D], perm[n_D:]
    lam_last = s[D].min()
    # (i) no disagreements: one test at min s[D], with >= m0 accepted test points -> feasible there
    cert.fit(s, np.zeros(n, dtype=int))
    assert len(cert.trace) == 1, cert.trace
    assert cert.trace[0][1] >= m0
    assert cert.feasible and cert.lam == lam_last
    # (ii) one disagreement among the accepted test points: P(Bin(n_j, .01) <= 1) > delta -> infeasible
    d = np.zeros(n, dtype=int)
    d[T[np.argmax(s[T])]] = 1
    cert2 = SelectiveCertificate(alpha, delta, mode="split").fit(s, d)
    assert len(cert2.trace) == 1 and cert2.trace[0][2] == 1, cert2.trace
    assert not cert2.feasible and cert2.lam == np.inf


def test_same_mode_unchanged():
    # expected lam computed with the pre-split code (seed 3, n=1000, alpha=0.05, delta=0.1)
    rng = np.random.default_rng(3)
    s = rng.uniform(0, 1, 1000)
    d = (rng.uniform(0, 1, 1000) < 0.2 * (1 - s)).astype(int)
    cert = SelectiveCertificate(0.05, 0.1, mode="same").fit(s, d)
    assert cert.lam == 0.7934645677726786, repr(cert.lam)
