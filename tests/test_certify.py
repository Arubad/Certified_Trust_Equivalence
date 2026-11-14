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
