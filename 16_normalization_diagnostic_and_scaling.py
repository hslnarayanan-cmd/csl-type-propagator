"""
Script 16 of 23 -- Substep-level diagnostic and single-trajectory
divergence scaling for the raw-vs-quotient expectation-value convention
(paper Section 6.4, Tables 3-4).

Part A reproduces Table 3: tracks ||psi||^2 and both the raw and
quotient (normalized) forms of mu_i through one full timestep, at two
step sizes, showing directly how far the intermediate (not-yet-fully-
renormalized) state drifts from unit norm within a single timestep, and
how much the two conventions' computed expectation values differ as a
result.

Part B reproduces Table 4: runs the SAME noise realization (identical
seed, identical dW sequence at every substep) through both conventions
across a range of step sizes, and reports the resulting divergence
between the two trajectories -- both in state-vector norm and in
final-time fidelity -- as a function of dt.

Both parts use the GHZ state, H=0, Pauli-Z collapse, lambda=0.20,
matching the configuration used throughout Section 6.4 of the paper.
"""

import numpy as np

n_qubits = 3
dim = 2 ** n_qubits
I2 = np.eye(2, dtype=complex)
Z = np.array([[1, 0], [0, -1]], dtype=complex)


def embed(op, q, n):
    mats = [I2] * n
    mats[q] = op
    out = mats[0]
    for m in mats[1:]:
        out = np.kron(out, m)
    return out


Z_ops = [embed(Z, q, n_qubits) for q in range(n_qubits)]

ghz = np.zeros(dim, dtype=complex)
ghz[0] = 1 / np.sqrt(2)
ghz[-1] = 1 / np.sqrt(2)

lam = 0.20
T = 1.0


# ---------------------------------------------------------------------------
# Part A: substep-level diagnostic (Table 3)
# ---------------------------------------------------------------------------
def diagnostic_run(psi0, lam, dt, seed):
    rng = np.random.default_rng(seed)
    psi = psi0.copy()
    print(f"\ndt={dt}")
    print(f"{'substep':>8} {'||psi||^2 before':>18} {'raw <A_i>':>12} {'quotient <A_i>':>16} {'|diff|':>10}")
    for i, Zq in enumerate(Z_ops):
        norm_sq = np.real(np.conjugate(psi) @ psi)
        exp_raw = float(np.real(np.conjugate(psi) @ (Zq @ psi)))
        exp_quotient = exp_raw / norm_sq
        print(f"{i:>8} {norm_sq:>18.6f} {exp_raw:>12.6f} {exp_quotient:>16.6f} {abs(exp_raw-exp_quotient):>10.6f}")
        dW = rng.normal(0.0, np.sqrt(dt))
        diff_op = Zq - exp_quotient * np.eye(dim)  # advance using the quotient (canonical) form
        psi = psi + np.sqrt(lam) * (diff_op @ psi) * dW - 0.5 * lam * (diff_op @ diff_op @ psi) * dt
    norm_sq_final = np.real(np.conjugate(psi) @ psi)
    print(f"{'final':>8} {norm_sq_final:>18.6f}")


print("=" * 70)
print("PART A: Substep-level diagnostic (paper Table 3)")
print("=" * 70)
diagnostic_run(ghz, lam, dt=0.01, seed=1)
diagnostic_run(ghz, lam, dt=0.1, seed=1)


# ---------------------------------------------------------------------------
# Part B: single-trajectory divergence vs dt (Table 4)
# ---------------------------------------------------------------------------
def csl_step(psi, A_ops, lam, dt, rng, quotient):
    for Aq in A_ops:
        if quotient:
            norm_sq = np.real(np.conjugate(psi) @ psi)
            mu = float(np.real(np.conjugate(psi) @ (Aq @ psi))) / norm_sq
        else:
            mu = float(np.real(np.conjugate(psi) @ (Aq @ psi)))
        dW = rng.normal(0.0, np.sqrt(dt))
        diff_op = Aq - mu * np.eye(dim)
        psi = psi + np.sqrt(lam) * (diff_op @ psi) * dW - 0.5 * lam * (diff_op @ diff_op @ psi) * dt
    return psi / np.linalg.norm(psi)


print("\n" + "=" * 70)
print("PART B: Single-trajectory divergence vs dt (paper Table 4)")
print("=" * 70)
print(f"{'dt':>8} {'max ||diff|| over path':>25} {'final fidelity diff':>22}")
for dt in [0.1, 0.05, 0.02, 0.01, 0.005, 0.0025]:
    n_steps = int(round(T / dt))
    rng_raw = np.random.default_rng(999)
    rng_quotient = np.random.default_rng(999)  # identical seed -> identical dW sequence
    psi_raw = ghz.copy()
    psi_quotient = ghz.copy()
    max_diff = 0.0
    for t in range(n_steps):
        psi_raw = csl_step(psi_raw, Z_ops, lam, dt, rng_raw, quotient=False)
        psi_quotient = csl_step(psi_quotient, Z_ops, lam, dt, rng_quotient, quotient=True)
        max_diff = max(max_diff, np.linalg.norm(psi_raw - psi_quotient))
    fid_raw = abs(np.vdot(ghz, psi_raw)) ** 2
    fid_quotient = abs(np.vdot(ghz, psi_quotient)) ** 2
    print(f"{dt:>8} {max_diff:>25.8f} {abs(fid_raw-fid_quotient):>22.8f}")

print("\nDone.")
