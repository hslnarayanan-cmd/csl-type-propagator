"""
Script 19 of 23 -- Non-diagonal Pauli-X configuration B, multi-seed
statistics (paper Table 11, Section IX-B).

Configuration B (H=0, X-collapse, GHZ) is checked against the
independent SciPy reference across five independent seeds, since a
single M=400 ensemble alone does not reliably distinguish a genuine
discrepancy from ordinary sampling variation. Uses the quotient
(normalized) expectation convention, matching Algorithm 1.
"""

import numpy as np
from scipy.integrate import solve_ivp

n_qubits = 3
dim = 2 ** n_qubits
I2 = np.eye(2, dtype=complex)
X = np.array([[0, 1], [1, 0]], dtype=complex)

def embed(op, q, n):
    mats = [I2] * n
    mats[q] = op
    out = mats[0]
    for m in mats[1:]:
        out = np.kron(out, m)
    return out

X_ops = [embed(X, q, n_qubits) for q in range(n_qubits)]

ghz = np.zeros(dim, dtype=complex)
ghz[0] = 1 / np.sqrt(2)
ghz[-1] = 1 / np.sqrt(2)

T = 1.0
lam = 0.20
dt = 0.01
M = 400


def csl_step_quotient(psi, A_ops, lam, dt, rng):
    for Aq in A_ops:
        norm_sq = np.real(np.conjugate(psi) @ psi)
        mu = float(np.real(np.conjugate(psi) @ (Aq @ psi))) / norm_sq
        dW = rng.normal(0.0, np.sqrt(dt))
        diff_op = Aq - mu * np.eye(dim)
        psi = psi + np.sqrt(lam) * (diff_op @ psi) * dW - 0.5 * lam * (diff_op @ diff_op @ psi) * dt
    return psi / np.linalg.norm(psi)


def run_ensemble(psi0, A_ops, lam, dt, T, M, seed):
    rng = np.random.default_rng(seed)
    n_steps = int(round(T / dt))
    fid = np.zeros(M)
    for k in range(M):
        psi = psi0.copy()
        for _ in range(n_steps):
            psi = csl_step_quotient(psi, A_ops, lam, dt, rng)
        fid[k] = np.abs(np.vdot(psi0, psi)) ** 2
    return fid.mean(), fid.std(ddof=1) / np.sqrt(M), fid


def lindblad_rhs(t, rho_flat, H, A_ops, lam):
    rho = rho_flat.reshape(dim, dim)
    drho = -1j * (H @ rho - rho @ H)
    for Aq in A_ops:
        drho += lam * (Aq @ rho @ Aq - 0.5 * (Aq @ Aq @ rho + rho @ Aq @ Aq))
    return drho.flatten()


H_zero = np.zeros((dim, dim), dtype=complex)
rho0 = np.outer(ghz, ghz.conj())
sol = solve_ivp(lindblad_rhs, [0, T], rho0.flatten(), args=(H_zero, X_ops, lam),
                 t_eval=[T], method="RK45", rtol=1e-9, atol=1e-12)
rho_ref = sol.y[:, 0].reshape(dim, dim)
F_ref = float(np.real(np.vdot(ghz, rho_ref @ ghz)))

print(f"SciPy reference: F_ref = {F_ref:.4f}\n")

# Five independent seeds, since a single M=400 ensemble is not sufficient
# to distinguish a genuine discrepancy from ordinary sampling variation
# at this configuration.
seeds = [202, 909, 1717, 3131, 4242]
results = {}
all_fid = []
for s in seeds:
    F, sem, fid = run_ensemble(ghz, X_ops, lam, dt, T, M, seed=s)
    sigma = abs(F - F_ref) / sem
    print(f"Seed {s}: F = {F:.4f} +/- {sem:.4f}, sigma = {sigma:.2f}")
    results[f"seed_{s}"] = {"F": F, "sem": sem, "sigma": sigma}
    all_fid.append(fid)

# Pooled estimate across all five seeds (M=2000 total)
F_all = np.concatenate(all_fid)
F_pooled = F_all.mean()
sem_pooled = F_all.std(ddof=1) / np.sqrt(len(F_all))
sigma_pooled = abs(F_pooled - F_ref) / sem_pooled
print(f"\nPooled (M=2000, 5 seeds): F = {F_pooled:.4f} +/- {sem_pooled:.4f}, sigma = {sigma_pooled:.2f}")

import json
results["F_ref"] = F_ref
results["pooled"] = {"F": F_pooled, "sem": sem_pooled, "sigma": sigma_pooled, "M_total": len(F_all)}
with open("caseB_second_seed_results.json", "w") as f:
    json.dump(results, f, indent=2)
print("\nWrote caseB_second_seed_results.json")
