"""
Script 17 of 23 -- Comprehensive raw-vs-quotient convention comparison
(paper Section 6.4, Table 5).

A single final-time fidelity comparison does not establish that two
numerical schemes are equivalent across all observables, all time
points, or in their fitted convergence behavior. This script checks,
for the SAME underlying comparison (identical noise realizations, so any
difference is attributable only to the convention):

  (1) time-resolved fidelity difference (max over the full trajectory,
      not just t=1);
  (2) time-resolved Frobenius distance between the raw and quotient
      ensemble density matrices;
  (3) time-resolved trace difference of each ensemble density matrix
      from 1 (a sanity check: both should be exactly 1 by construction,
      since both conventions renormalize the full state once per
      timestep -- this is NOT expected to show a meaningful raw/quotient
      difference, and confirming that is itself informative);
  (4) time-resolved entropy difference;
  (5) whether the convergence ORDER itself (not just point values)
      differs between conventions, using the same common-random-numbers
      methodology as the paper's main convergence analysis;
  (6) the non-diagonal (Pauli-X) cases A, B, C under both conventions.
"""

import numpy as np
from scipy.integrate import solve_ivp
from scipy.linalg import expm
from scipy.stats import linregress
import json

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


Z_ops = [embed(Z, q, n_qubits) for q in range(n_qubits)]
X_ops = [embed(X, q, n_qubits) for q in range(n_qubits)]

ghz = np.zeros(dim, dtype=complex)
ghz[0] = 1 / np.sqrt(2)
ghz[-1] = 1 / np.sqrt(2)
comp_000 = np.zeros(dim, dtype=complex)
comp_000[0] = 1.0

T = 1.0


def von_neumann_entropy(rho):
    evals = np.linalg.eigvalsh((rho + rho.conj().T) / 2)
    evals = evals[evals > 1e-12]
    return float(-np.sum(evals * np.log2(evals)))


def csl_step(psi, A_ops, lam, dt, rng, H_prop=None, normalized=True):
    if H_prop is not None:
        psi = H_prop @ psi
    for Aq in A_ops:
        if normalized:
            norm_sq = np.real(np.conjugate(psi) @ psi)
            mu = float(np.real(np.conjugate(psi) @ (Aq @ psi))) / norm_sq
        else:
            mu = float(np.real(np.conjugate(psi) @ (Aq @ psi)))
        dW = rng.normal(0.0, np.sqrt(dt))
        diff_op = Aq - mu * np.eye(dim)
        psi = psi + np.sqrt(lam) * (diff_op @ psi) * dW - 0.5 * lam * (diff_op @ diff_op @ psi) * dt
    return psi / np.linalg.norm(psi)


def run_ensemble_full(psi0, A_ops, lam, dt, T, M, seed, H_prop=None, normalized=True):
    """Returns per-timestep ensemble density matrices, fidelity, trace, entropy."""
    rng = np.random.default_rng(seed)
    n_steps = int(round(T / dt))
    psi_all = np.zeros((M, n_steps + 1, dim), dtype=complex)
    for k in range(M):
        psi = psi0.copy()
        psi_all[k, 0] = psi
        for t_idx in range(1, n_steps + 1):
            psi = csl_step(psi, A_ops, lam, dt, rng, H_prop, normalized)
            psi_all[k, t_idx] = psi
    rho_t = np.einsum('kti,ktj->tij', psi_all, psi_all.conj()) / M
    fid_t = np.abs(np.einsum('kti,i->kt', psi_all, psi0.conj())) ** 2
    F_mean = fid_t.mean(axis=0)
    trace_t = np.array([np.real(np.trace(rho_t[t])) for t in range(n_steps + 1)])
    entropy_t = np.array([von_neumann_entropy(rho_t[t]) for t in range(n_steps + 1)])
    return rho_t, F_mean, trace_t, entropy_t


results = {}

# ===========================================================================
# Part 1: Time-resolved comparison, non-commuting GHZ case (the paper's
# primary validation configuration), identical noise for raw vs normalized
# ===========================================================================
print("=" * 80)
print("PART 1: Time-resolved raw vs normalized, H=omega*sum(X_i), Z-collapse, GHZ")
print("=" * 80)
omega, lam, dt, M = 1.0, 0.20, 0.01, 400
H_nc = omega * sum(X_ops)
H_prop = expm(-1j * H_nc * dt)

rho_raw, F_raw, trace_raw, S_raw = run_ensemble_full(ghz, Z_ops, lam, dt, T, M, seed=2026, H_prop=H_prop, normalized=False)
rho_norm, F_norm, trace_norm, S_norm = run_ensemble_full(ghz, Z_ops, lam, dt, T, M, seed=2026, H_prop=H_prop, normalized=True)

n_steps = len(F_raw)
max_fid_diff = float(np.max(np.abs(F_raw - F_norm)))
max_frob_diff = float(np.max([np.linalg.norm(rho_raw[t] - rho_norm[t]) for t in range(n_steps)]))
max_trace_diff = float(np.max(np.abs(trace_raw - trace_norm)))
max_trace_dev_raw = float(np.max(np.abs(trace_raw - 1.0)))
max_trace_dev_norm = float(np.max(np.abs(trace_norm - 1.0)))
max_entropy_diff = float(np.max(np.abs(S_raw - S_norm)))

print(f"Max |F_raw(t) - F_norm(t)| over full trajectory: {max_fid_diff:.2e}")
print(f"Max ||rho_raw(t) - rho_norm(t)||_F over full trajectory: {max_frob_diff:.2e}")
print(f"Max |trace_raw(t) - trace_norm(t)|: {max_trace_diff:.2e}")
print(f"Max |trace_raw(t) - 1|: {max_trace_dev_raw:.2e}, max |trace_norm(t) - 1|: {max_trace_dev_norm:.2e}")
print(f"Max |S_raw(t) - S_norm(t)| over full trajectory: {max_entropy_diff:.2e}")

results["part1_timeresolved"] = {
    "max_fidelity_diff": max_fid_diff, "max_frobenius_diff": max_frob_diff,
    "max_trace_diff": max_trace_diff, "max_trace_dev_raw": max_trace_dev_raw,
    "max_trace_dev_norm": max_trace_dev_norm, "max_entropy_diff": max_entropy_diff,
    "F_raw_final": float(F_raw[-1]), "F_norm_final": float(F_norm[-1]),
}

# ===========================================================================
# Part 2: Convergence-order difference between conventions
# ===========================================================================
print("\n" + "=" * 80)
print("PART 2: Convergence order, raw vs normalized (common random numbers)")
print("=" * 80)

lam_c = 0.20


def run_path_fine(psi0, fine_increments, dt, dt_fine, n_fine, normalized):
    steps_per_block = int(round(dt / dt_fine))
    n_steps = n_fine // steps_per_block
    coarse = fine_increments[:n_steps * steps_per_block, :].reshape(n_steps, steps_per_block, 3).sum(axis=1)
    psi = psi0.copy()
    for t_idx in range(n_steps):
        for i, Zq in enumerate(Z_ops):
            if normalized:
                norm_sq = np.real(np.conjugate(psi) @ psi)
                mu = float(np.real(np.conjugate(psi) @ (Zq @ psi))) / norm_sq
            else:
                mu = float(np.real(np.conjugate(psi) @ (Zq @ psi)))
            diff_op = Zq - mu * np.eye(dim)
            psi = psi + np.sqrt(lam_c) * (diff_op @ psi) * coarse[t_idx, i] - 0.5 * lam_c * (diff_op @ diff_op @ psi) * dt
        psi = psi / np.linalg.norm(psi)
    return psi


def convergence_order_for_convention(normalized, dt_fine=0.00125, M=300, seed=2026):
    n_fine = int(round(T / dt_fine))
    dts_test = [0.04, 0.02, 0.01, 0.005, 0.0025]
    rng = np.random.default_rng(seed)
    fine_increments = rng.normal(0.0, np.sqrt(dt_fine), size=(M, n_fine, 3))

    F_ref = np.zeros(M)
    for k in range(M):
        psi_ref = run_path_fine(ghz, fine_increments[k], dt_fine, dt_fine, n_fine, normalized)
        F_ref[k] = np.abs(np.vdot(ghz, psi_ref)) ** 2

    rms_errors = []
    for dt_test in dts_test:
        errs = np.zeros(M)
        for k in range(M):
            psi_dt = run_path_fine(ghz, fine_increments[k], dt_test, dt_fine, n_fine, normalized)
            F_dt = np.abs(np.vdot(ghz, psi_dt)) ** 2
            errs[k] = abs(F_dt - F_ref[k])
        rms_errors.append(np.sqrt(np.mean(errs ** 2)))

    fit = linregress(np.log(dts_test), np.log(rms_errors))
    return fit.slope, fit.stderr, rms_errors


p_raw, se_raw, rms_raw = convergence_order_for_convention(normalized=False)
p_norm, se_norm, rms_norm = convergence_order_for_convention(normalized=True)

print(f"RAW:        p = {p_raw:.3f} +/- {se_raw:.3f} (regression SE)")
print(f"NORMALIZED: p = {p_norm:.3f} +/- {se_norm:.3f} (regression SE)")
print(f"Convergence-order difference: {abs(p_raw - p_norm):.3f}")

results["part2_convergence_order"] = {
    "p_raw": float(p_raw), "se_raw": float(se_raw),
    "p_norm": float(p_norm), "se_norm": float(se_norm),
    "order_difference": float(abs(p_raw - p_norm)),
    "rms_raw": rms_raw, "rms_norm": rms_norm,
}

# ===========================================================================
# Part 3: Non-diagonal (Pauli-X) cases A, B, C, both conventions, with
# time-resolved max fidelity difference as well as final-time values
# ===========================================================================
print("\n" + "=" * 80)
print("PART 3: Non-diagonal (X-collapse) cases, raw vs normalized")
print("=" * 80)


def lindblad_rhs(t, rho_flat, H, A_ops, lam):
    rho = rho_flat.reshape(dim, dim)
    drho = -1j * (H @ rho - rho @ H)
    for Aq in A_ops:
        drho += lam * (Aq @ rho @ Aq - 0.5 * (Aq @ Aq @ rho + rho @ Aq @ Aq))
    return drho.flatten()


def nondiagonal_case(name, psi0, H, lam=0.20, dt=0.01, M=400, seed=42):
    H_prop_local = expm(-1j * H * dt) if H is not None and not np.allclose(H, 0) else None
    rho0 = np.outer(psi0, psi0.conj())
    H_for_ref = H if H is not None else np.zeros((dim, dim), dtype=complex)
    sol = solve_ivp(lindblad_rhs, [0, T], rho0.flatten(), args=(H_for_ref, X_ops, lam),
                     t_eval=[T], method="RK45", rtol=1e-9, atol=1e-12)
    rho_ref = sol.y[:, 0].reshape(dim, dim)
    F_ref = float(np.real(np.vdot(psi0, rho_ref @ psi0)))

    _, F_raw_t, _, _ = run_ensemble_full(psi0, X_ops, lam, dt, T, M, seed, H_prop_local, normalized=False)
    _, F_norm_t, _, _ = run_ensemble_full(psi0, X_ops, lam, dt, T, M, seed, H_prop_local, normalized=True)

    max_diff_t = float(np.max(np.abs(F_raw_t - F_norm_t)))
    print(f"\n{name}: ref={F_ref:.4f}")
    print(f"  F_raw(t=1)={F_raw_t[-1]:.4f}, F_norm(t=1)={F_norm_t[-1]:.4f}, "
          f"final diff={abs(F_raw_t[-1]-F_norm_t[-1]):.2e}, max time-resolved diff={max_diff_t:.2e}")
    return {"F_ref": F_ref, "F_raw_final": float(F_raw_t[-1]), "F_norm_final": float(F_norm_t[-1]),
            "final_diff": float(abs(F_raw_t[-1] - F_norm_t[-1])), "max_timeresolved_diff": max_diff_t}


H_z = omega * sum(Z_ops)
results["case_A"] = nondiagonal_case("(A) H=0, X-collapse, |000>", comp_000, None, seed=101)
results["case_B"] = nondiagonal_case("(B) H=0, X-collapse, GHZ", ghz, None, seed=202)
results["case_C"] = nondiagonal_case("(C) H=omega*sum(Z_i), X-collapse, GHZ", ghz, H_z, seed=303)

with open("comprehensive_norm_comparison.json", "w") as f:
    json.dump(results, f, indent=2)

print("\n\nDone. Wrote comprehensive_norm_comparison.json")
