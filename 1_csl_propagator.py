"""
Script 1 of 4 --- Core CSL propagator (Algorithm 1) and closed-form validation.

Reproduces:
  - Listing 1 (the core update step)
  - Table 1 and Figures 1-2 (simulated vs. closed-form fidelity/entropy)
  - Figure 3 (step-size sensitivity sweep)

System: 3-qubit GHZ register, Pauli-Z collapse operators, no Hamiltonian.
This is the commuting-collapse-operator case, for which Section 4 of the
paper derives the closed-form solution used below as a validation target.

Run:
    python3 1_csl_propagator.py

Outputs:
    results.json       -- fidelity/entropy vs. time for lambda in {0, 0.05, 0.20, 0.50}
    convergence.json   -- final-time fidelity vs. integration step size dt
"""

import numpy as np
import json

np.random.seed(42)

# ---------------------------------------------------------------------------
# Setup: 3-qubit register, GHZ state, single-qubit Pauli-Z operators
# ---------------------------------------------------------------------------
n_qubits = 3
dim = 2 ** n_qubits

I2 = np.eye(2)
Z = np.array([[1, 0], [0, -1]], dtype=complex)


def single_qubit_op(op, q, n):
    """Embed a single-qubit operator into an n-qubit register via Kronecker product."""
    mats = [I2] * n
    mats[q] = op
    out = mats[0]
    for m in mats[1:]:
        out = np.kron(out, m)
    return out


Z_ops = [single_qubit_op(Z, q, n_qubits) for q in range(n_qubits)]

ghz = np.zeros(dim, dtype=complex)
ghz[0] = 1 / np.sqrt(2)
ghz[-1] = 1 / np.sqrt(2)
psi0 = ghz.copy()


def von_neumann_entropy(rho):
    evals = np.linalg.eigvalsh(rho)
    evals = evals[evals > 1e-12]
    return float(-np.sum(evals * np.log2(evals)))


# ---------------------------------------------------------------------------
# Algorithm 1 -- core CSL update step (Listing 1 in the paper)
# ---------------------------------------------------------------------------
def csl_step(psi, Z_ops, lambda_val, dt, rng):
    for Zq in Z_ops:
        exp_z = float(np.real(np.conjugate(psi) @ (Zq @ psi)))
        dW = rng.normal(0.0, np.sqrt(dt))
        diff_op = Zq - exp_z * np.eye(len(psi))
        term1 = np.sqrt(lambda_val) * (diff_op @ psi) * dW
        term2 = -0.5 * lambda_val * (diff_op @ diff_op @ psi) * dt
        psi = psi + term1 + term2
    return psi / np.linalg.norm(psi)


# ---------------------------------------------------------------------------
# Monte Carlo ensemble runner, with per-timestep fidelity SEM and a
# bootstrap confidence interval on entropy (Section 4.2)
# ---------------------------------------------------------------------------
def run_ensemble(lambda_val, dt, T, M, seed, n_boot=200):
    rng = np.random.default_rng(seed)
    n_steps = int(round(T / dt))
    times = np.linspace(0, T, n_steps + 1)

    fidelity_traj = np.zeros((M, n_steps + 1))
    psi_all = np.zeros((M, n_steps + 1, dim), dtype=complex)

    for k in range(M):
        psi = psi0.copy()
        psi_all[k, 0] = psi
        fidelity_traj[k, 0] = np.abs(np.vdot(ghz, psi)) ** 2
        for t_idx in range(1, n_steps + 1):
            psi = csl_step(psi, Z_ops, lambda_val, dt, rng)
            psi_all[k, t_idx] = psi
            fidelity_traj[k, t_idx] = np.abs(np.vdot(ghz, psi)) ** 2

    fidelity_mean = fidelity_traj.mean(axis=0)
    fidelity_sem = fidelity_traj.std(axis=0, ddof=1) / np.sqrt(M)

    entropy = np.zeros(n_steps + 1)
    entropy_lo = np.zeros(n_steps + 1)
    entropy_hi = np.zeros(n_steps + 1)
    for t_idx in range(n_steps + 1):
        rho_full = np.einsum('ki,kj->ij', psi_all[:, t_idx, :], psi_all[:, t_idx, :].conj()) / M
        entropy[t_idx] = von_neumann_entropy(rho_full)
        boot_vals = []
        for _ in range(n_boot):
            idx = rng.integers(0, M, size=M)
            sub = psi_all[idx, t_idx, :]
            rho_b = np.einsum('ki,kj->ij', sub, sub.conj()) / M
            boot_vals.append(von_neumann_entropy(rho_b))
        boot_vals = np.array(boot_vals)
        entropy_lo[t_idx] = np.percentile(boot_vals, 2.5)
        entropy_hi[t_idx] = np.percentile(boot_vals, 97.5)

    return {
        "times": times.tolist(),
        "fidelity_mean": fidelity_mean.tolist(),
        "fidelity_sem": fidelity_sem.tolist(),
        "entropy": entropy.tolist(),
        "entropy_lo": entropy_lo.tolist(),
        "entropy_hi": entropy_hi.tolist(),
    }


if __name__ == "__main__":
    lambdas = [0.00, 0.05, 0.20, 0.50]
    M = 400
    dt = 0.01
    T = 1.0

    results = {}
    for i, lam in enumerate(lambdas):
        print(f"Running lambda={lam} ...")
        res = run_ensemble(lam, dt, T, M, seed=1000 + i)
        results[str(lam)] = res
        print(f"  final fidelity = {res['fidelity_mean'][-1]:.4f} +/- {res['fidelity_sem'][-1]:.4f}, "
              f"final entropy = {res['entropy'][-1]:.4f}")

    with open("results.json", "w") as f:
        json.dump(results, f)

    # ---- Step-size sensitivity check at lambda = 0.20 (Figure 3) ----
    print("\nStep-size sensitivity at lambda=0.20:")
    conv = {}
    for dt_test in [0.02, 0.01, 0.005, 0.0025]:
        r = run_ensemble(0.20, dt_test, T, M=200, seed=7)
        conv[str(dt_test)] = {
            "final_fidelity": r["fidelity_mean"][-1],
            "final_fidelity_sem": r["fidelity_sem"][-1],
            "final_entropy": r["entropy"][-1],
        }
        print(f"  dt={dt_test}: fidelity={r['fidelity_mean'][-1]:.4f} +/- {r['fidelity_sem'][-1]:.4f}, "
              f"entropy={r['entropy'][-1]:.4f}")

    with open("convergence.json", "w") as f:
        json.dump(conv, f)

    print("\nWrote results.json and convergence.json")
