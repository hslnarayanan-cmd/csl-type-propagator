"""
Script 23 of 23 -- Convergence-order check in the non-commuting regime
(paper Table 8, Section VIII-A).

The multi-seed, multi-grid strong-convergence methodology (script 9) is applied to the
non-commuting regime (H = omega*sum(X_i), omega=1, GHZ, lambda=0.20),
the baseline configuration used throughout Section IX, to check
whether the convergence order measured in the commuting case also
holds once the Hamiltonian and collapse operators no longer commute.
Same common-random-numbers methodology, same dt_fine=0.00125, 3
independent seeds, M=250
trajectories per seed.
"""

import time
import numpy as np
from scipy.linalg import expm
from scipy.stats import linregress

n_qubits = 3
dim = 2 ** n_qubits
I2 = np.eye(2, dtype=complex)
X = np.array([[0, 1], [1, 0]], dtype=complex)
Z = np.array([[1, 0], [0, -1]], dtype=complex)


def embed(op, q, n):
    mats = [I2] * n
    mats[q] = op
    out = mats[0]
    for m in mats[1:]:
        out = np.kron(out, m)
    return out


X_ops = [embed(X, q, n_qubits) for q in range(n_qubits)]
Z_ops = [embed(Z, q, n_qubits) for q in range(n_qubits)]

ghz = np.zeros(dim, dtype=complex)
ghz[0] = 1 / np.sqrt(2)
ghz[-1] = 1 / np.sqrt(2)
T, lam, omega = 1.0, 0.20, 1.0
H = omega * sum(X_ops)


def run_path_fine(psi0, fine_increments, dt, dt_fine, n_fine, H_prop):
    steps_per_block = int(round(dt / dt_fine))
    n_steps = n_fine // steps_per_block
    coarse = fine_increments[:n_steps * steps_per_block, :].reshape(n_steps, steps_per_block, 3).sum(axis=1)
    psi = psi0.copy()
    for t_idx in range(n_steps):
        psi = H_prop @ psi
        for i, Zq in enumerate(Z_ops):
            norm_sq = np.real(np.conjugate(psi) @ psi)
            mu = float(np.real(np.conjugate(psi) @ (Zq @ psi))) / norm_sq
            diff_op = Zq - mu * np.eye(dim)
            psi = psi + np.sqrt(lam) * (diff_op @ psi) * coarse[t_idx, i] - 0.5 * lam * (diff_op @ diff_op @ psi) * dt
        psi = psi / np.linalg.norm(psi)
    return psi


def convergence_order_one_seed(seed, dt_fine=0.00125, M=250):
    n_fine = int(round(T / dt_fine))
    dts_test = [0.04, 0.02, 0.01, 0.005, 0.0025]
    rng = np.random.default_rng(seed)
    fine_increments = rng.normal(0.0, np.sqrt(dt_fine), size=(M, n_fine, 3))

    H_prop_fine = expm(-1j * H * dt_fine)
    F_ref = np.zeros(M)
    for k in range(M):
        psi_ref = run_path_fine(ghz, fine_increments[k], dt_fine, dt_fine, n_fine, H_prop_fine)
        F_ref[k] = abs(np.vdot(ghz, psi_ref)) ** 2

    rms_errors = []
    for dt_test in dts_test:
        H_prop_test = expm(-1j * H * dt_test)
        errs = np.zeros(M)
        for k in range(M):
            psi_dt = run_path_fine(ghz, fine_increments[k], dt_test, dt_fine, n_fine, H_prop_test)
            F_dt = abs(np.vdot(ghz, psi_dt)) ** 2
            errs[k] = abs(F_dt - F_ref[k])
        rms_errors.append(np.sqrt(np.mean(errs ** 2)))

    fit = linregress(np.log(dts_test), np.log(rms_errors))
    return fit.slope, fit.stderr, rms_errors


print("Non-commuting hardened convergence test (H = omega*sum(X_i), omega=1, GHZ, lambda=0.20)")
print("dt_fine=0.00125, M=250, 3 seeds\n")
t0 = time.time()
seeds = [2026, 3037, 4048]
ps = []
for s in seeds:
    p, se, rms = convergence_order_one_seed(seed=s)
    ps.append(p)
    print(f"Seed {s}: p = {p:.3f} (regression SE {se:.3f}), elapsed {time.time()-t0:.0f}s")

ps = np.array(ps)
print(f"\nNon-commuting p_obs (3 seeds): {ps.mean():.3f} +/- {ps.std(ddof=1):.3f}")
print(f"Commuting p_obs (paper Table 8, GHZ, dt_fine=0.00125, 3 seeds): 0.635 +/- 0.029")

import json
with open("noncommuting_convergence_results.json", "w") as f:
    json.dump({"seeds": seeds, "p_values": ps.tolist(),
               "mean": float(ps.mean()), "std": float(ps.std(ddof=1))}, f, indent=2)
print("\nWrote noncommuting_convergence_results.json")
