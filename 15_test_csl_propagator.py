"""
Script 15 of 23 -- Automated test suite for the CSL propagator (paper
Section 15, Reproducibility).

Covers: norm preservation, analytical-case agreement, master-equation
agreement, and the requested edge cases (lambda=0, n=1, near-deterministic
initial states). Run with pytest:

    pytest -v test_csl_propagator.py

or directly:

    python3 test_csl_propagator.py
"""

import numpy as np

try:
    import pytest  # noqa: F401  (optional; only needed if run via `pytest`)
except ImportError:
    pytest = None


# ---------------------------------------------------------------------------
# Shared helpers (mirroring Algorithm 1 / Listing 1 in the paper)
# ---------------------------------------------------------------------------
I2 = np.eye(2, dtype=complex)
Z = np.array([[1, 0], [0, -1]], dtype=complex)
X = np.array([[0, 1], [1, 0]], dtype=complex)


def embed(op, q, n):
    mats = [I2] * n
    mats[q] = op
    out = mats[0]
    for m in mats[1:]:
        out = np.kron(out, m)
    return out


def z_ops(n):
    return [embed(Z, q, n) for q in range(n)]


def csl_step(psi, Zops, lam, dt, rng):
    for Zq in Zops:
        exp_z = float(np.real(np.conjugate(psi) @ (Zq @ psi)))
        dW = rng.normal(0.0, np.sqrt(dt))
        diff_op = Zq - exp_z * np.eye(len(psi))
        psi = psi + np.sqrt(lam) * (diff_op @ psi) * dW - 0.5 * lam * (diff_op @ diff_op @ psi) * dt
    return psi / np.linalg.norm(psi)


def run_trajectory(psi0, lam, dt, T, seed):
    rng = np.random.default_rng(seed)
    n_qubits = int(np.log2(len(psi0)))
    Zops = z_ops(n_qubits)
    n_steps = int(round(T / dt))
    psi = psi0.copy()
    for _ in range(n_steps):
        psi = csl_step(psi, Zops, lam, dt, rng)
    return psi


def hamming(a, b):
    return bin(a ^ b).count("1")


def analytic_rho(rho0, lam, t, dim):
    rho_t = np.zeros_like(rho0)
    for a in range(dim):
        for b in range(dim):
            rho_t[a, b] = rho0[a, b] * np.exp(-2 * lam * hamming(a, b) * t)
    return rho_t


# ---------------------------------------------------------------------------
# 1. Norm preservation
# ---------------------------------------------------------------------------
def test_norm_preservation_ghz():
    """Algorithm 1's renormalization step must keep every trajectory's
    norm at exactly 1 (to floating-point precision) after every step,
    regardless of how many steps are taken."""
    dim = 8
    psi0 = np.zeros(dim, dtype=complex)
    psi0[0] = psi0[-1] = 1 / np.sqrt(2)
    rng = np.random.default_rng(1)
    Zops = z_ops(3)
    psi = psi0.copy()
    for _ in range(500):  # far more steps than any validation run in the paper
        psi = csl_step(psi, Zops, lam=0.5, dt=0.01, rng=rng)
        assert abs(np.linalg.norm(psi) - 1.0) < 1e-12


def test_norm_preservation_single_qubit():
    """Same check at n=1, the smallest possible register."""
    psi0 = np.array([1 / np.sqrt(2), 1 / np.sqrt(2)], dtype=complex)
    rng = np.random.default_rng(2)
    Zops = z_ops(1)
    psi = psi0.copy()
    for _ in range(200):
        psi = csl_step(psi, Zops, lam=0.3, dt=0.01, rng=rng)
        assert abs(np.linalg.norm(psi) - 1.0) < 1e-12


# ---------------------------------------------------------------------------
# 2. Edge case: lambda = 0 (no collapse -- must reduce to pure state evolution)
# ---------------------------------------------------------------------------
def test_lambda_zero_no_collapse():
    """With lambda=0 and H=0, the state must not change at all: every
    term in the update is proportional to lambda or sqrt(lambda)."""
    dim = 8
    psi0 = np.zeros(dim, dtype=complex)
    psi0[0] = psi0[-1] = 1 / np.sqrt(2)
    psi_final = run_trajectory(psi0, lam=0.0, dt=0.01, T=1.0, seed=3)
    assert np.allclose(psi_final, psi0, atol=1e-12)


def test_lambda_zero_fidelity_stays_one():
    """Equivalently: fidelity to the initial state must stay exactly 1
    throughout, for any initial state, when lambda=0."""
    rng = np.random.default_rng(4)
    psi0 = rng.normal(size=8) + 1j * rng.normal(size=8)
    psi0 /= np.linalg.norm(psi0)
    psi_final = run_trajectory(psi0, lam=0.0, dt=0.01, T=1.0, seed=5)
    fidelity = abs(np.vdot(psi0, psi_final)) ** 2
    assert abs(fidelity - 1.0) < 1e-10


# ---------------------------------------------------------------------------
# 3. Edge case: n=1 (smallest register) against the analytic formula
# ---------------------------------------------------------------------------
def test_n1_analytic_agreement():
    """For n=1, the single-qubit superposition (|0>+|1>)/sqrt(2) has
    Hamming distance 1 between its two basis states, so Eq. (4)/(5) in
    the paper predicts F(t) = 1/2 + 1/2*exp(-2*lambda*t) (the n=1
    special case of the general formula, distinct from the n=3 GHZ
    formula's factor of 6)."""
    psi0 = np.array([1 / np.sqrt(2), 1 / np.sqrt(2)], dtype=complex)
    lam, T = 0.3, 1.0
    M = 800
    fids = np.zeros(M)
    for k in range(M):
        psi_final = run_trajectory(psi0, lam, dt=0.01, T=T, seed=1000 + k)
        fids[k] = abs(np.vdot(psi0, psi_final)) ** 2
    F_sim = fids.mean()
    F_analytic = 0.5 + 0.5 * np.exp(-2 * lam * T)
    sem = fids.std(ddof=1) / np.sqrt(M)
    assert abs(F_sim - F_analytic) < 4 * sem  # within ~4 SEM, generous for a CI test


# ---------------------------------------------------------------------------
# 4. Edge case: computational basis state (zero coherence)
# ---------------------------------------------------------------------------
def test_computational_basis_state_unchanged():
    """A computational basis state has no coherences to decay; Eq. (4)
    predicts rho(t) = rho(0) exactly, for any lambda."""
    dim = 8
    psi0 = np.zeros(dim, dtype=complex)
    psi0[3] = 1.0  # |011>
    psi_final = run_trajectory(psi0, lam=0.5, dt=0.01, T=1.0, seed=6)
    fidelity = abs(np.vdot(psi0, psi_final)) ** 2
    assert abs(fidelity - 1.0) < 1e-10


# ---------------------------------------------------------------------------
# 5. Edge case: nearly deterministic state (amplitude ~1 on one basis
#    state, tiny epsilon elsewhere) -- must not produce NaN/inf
# ---------------------------------------------------------------------------
def test_nearly_deterministic_state_no_blowup():
    dim = 8
    eps = 1e-8
    psi0 = np.full(dim, eps, dtype=complex)
    psi0[0] = 1.0
    psi0 /= np.linalg.norm(psi0)
    psi_final = run_trajectory(psi0, lam=0.5, dt=0.01, T=1.0, seed=7)
    assert np.all(np.isfinite(psi_final))
    assert abs(np.linalg.norm(psi_final) - 1.0) < 1e-10


# ---------------------------------------------------------------------------
# 6. Master-equation agreement (lightweight regression version of the
#    full multi-state validation in the paper, Section 7.4)
# ---------------------------------------------------------------------------
def test_master_equation_agreement_product_state():
    """A product state distinct from GHZ, checked against the general
    closed-form solution -- a fast regression test, not a replacement
    for the full M=400 validation in the paper."""
    dim = 8
    psi0 = np.zeros(dim, dtype=complex)
    psi0[0] = 1 / np.sqrt(2)
    psi0[4] = 1 / np.sqrt(2)  # |000> and |100>: Hamming distance 1
    rho0 = np.outer(psi0, psi0.conj())
    lam, T = 0.2, 1.0
    M = 300
    rho_accum = np.zeros((dim, dim), dtype=complex)
    for k in range(M):
        psi_final = run_trajectory(psi0, lam, dt=0.01, T=T, seed=2000 + k)
        rho_accum += np.outer(psi_final, psi_final.conj()) / M
    rho_expected = analytic_rho(rho0, lam, T, dim)
    max_err = np.max(np.abs(rho_accum - rho_expected))
    assert max_err < 0.05  # loose bound appropriate for M=300


# ---------------------------------------------------------------------------
# 7. Very small dt behaves sanely (near-identity over one short step)
# ---------------------------------------------------------------------------
def test_small_dt_near_identity():
    dim = 8
    psi0 = np.zeros(dim, dtype=complex)
    psi0[0] = psi0[-1] = 1 / np.sqrt(2)
    rng = np.random.default_rng(8)
    Zops = z_ops(3)
    psi_final = csl_step(psi0, Zops, lam=0.2, dt=1e-6, rng=rng)
    assert abs(np.vdot(psi0, psi_final)) ** 2 > 1 - 1e-3  # barely moved
    assert abs(np.linalg.norm(psi_final) - 1.0) < 1e-12


if __name__ == "__main__":
    # allow running without pytest installed
    import sys
    import traceback

    tests = [obj for name, obj in list(globals().items()) if name.startswith("test_")]
    passed, failed = 0, 0
    for t in tests:
        try:
            t()
            print(f"PASS: {t.__name__}")
            passed += 1
        except Exception:
            print(f"FAIL: {t.__name__}")
            traceback.print_exc()
            failed += 1
    print(f"\n{passed} passed, {failed} failed")
    sys.exit(1 if failed else 0)
