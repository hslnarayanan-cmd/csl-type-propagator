"""
Script 5 of 7 -- Convergence and ablation experiments (paper Sections 8.2,
8.3, and 6.2).

  (A) Step-size convergence study: RMS error vs dt, using multiple
      random seeds per dt to check whether discretization bias can be
      separated from Monte Carlo sampling noise this way. It cannot, at
      the M tested here -- see the paper (Section 8.1) and script 6 for
      a convergence estimate using common random numbers, which isolates
      discretization error from sampling noise directly.
  (B) Trajectory-count convergence: error vs M, checked against the
      expected E(M) ~ 1/sqrt(M) Monte Carlo scaling (paper Section 8.2).
  (C) Individual trajectory realizations plotted alongside the ensemble
      mean and the analytic solution (paper Section 8.3, Figure 5).
  (D) Normalization ablation: renormalized vs. unrenormalized
      Euler-Maruyama updates, tracking norm drift and resulting bias
      (paper Section 6.2, Figure 1).

System: 3-qubit GHZ register, Pauli-Z collapse operators, H=0 (the
commuting case with the closed-form solution used as ground truth).

Run:
    python3 5_convergence_and_ablation_experiments.py
"""

import numpy as np
import json
import time

n_qubits = 3
dim = 2 ** n_qubits
I2 = np.eye(2)
Z = np.array([[1, 0], [0, -1]], dtype=complex)


def single_qubit_op(op, q, n):
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


def analytic_fidelity(lam, t):
    return 0.5 + 0.5 * np.exp(-6 * lam * t)


def csl_step(psi, lambda_val, dt, rng, normalize=True):
    for Zq in Z_ops:
        exp_z = float(np.real(np.conjugate(psi) @ (Zq @ psi)))
        dW = rng.normal(0.0, np.sqrt(dt))
        diff_op = Zq - exp_z * np.eye(dim)
        term1 = np.sqrt(lambda_val) * (diff_op @ psi) * dW
        term2 = -0.5 * lambda_val * (diff_op @ diff_op @ psi) * dt
        psi = psi + term1 + term2
    if normalize:
        psi = psi / np.linalg.norm(psi)
    return psi


def run_trajectories(lam, dt, T, M, seed, normalize=True, keep_all=False):
    rng = np.random.default_rng(seed)
    n_steps = int(round(T / dt))
    fidelity_traj = np.zeros((M, n_steps + 1))
    norm_traj = np.zeros((M, n_steps + 1))
    all_traj = np.zeros((M, n_steps + 1)) if keep_all else None
    for k in range(M):
        psi = psi0.copy()
        fidelity_traj[k, 0] = np.abs(np.vdot(ghz, psi)) ** 2 / (np.linalg.norm(psi) ** 2)
        norm_traj[k, 0] = np.linalg.norm(psi)
        for t_idx in range(1, n_steps + 1):
            psi = csl_step(psi, lam, dt, rng, normalize=normalize)
            nrm = np.linalg.norm(psi)
            norm_traj[k, t_idx] = nrm
            fidelity_traj[k, t_idx] = np.abs(np.vdot(ghz, psi)) ** 2 / (nrm ** 2)
    times = np.linspace(0, T, n_steps + 1)
    return times, fidelity_traj, norm_traj


# ===========================================================================
# (A) Step-size convergence order: RMS error vs dt, multiple seeds
# ===========================================================================
print("=" * 70)
print("(A) Step-size convergence order study")
print("=" * 70)
lam = 0.20
T = 1.0
M_conv = 800
seeds = [11, 22, 33]
dts = [0.04, 0.02, 0.01, 0.005, 0.0025]

conv_results = {}
for dt in dts:
    errs = []
    for seed in seeds:
        times, fid_traj, _ = run_trajectories(lam, dt, T, M_conv, seed)
        fid_mean = fid_traj.mean(axis=0)
        analytic = analytic_fidelity(lam, times)
        rms = np.sqrt(np.mean((fid_mean - analytic) ** 2))
        errs.append(rms)
    conv_results[str(dt)] = {"rms_per_seed": errs, "rms_mean": float(np.mean(errs)), "rms_std": float(np.std(errs))}
    print(f"  dt={dt:<7} RMS error = {np.mean(errs):.5f} +/- {np.std(errs):.5f} (over {len(seeds)} seeds, M={M_conv})")

# estimate convergence order p from a log-log linear fit
log_dt = np.log(dts)
log_err = np.log([conv_results[str(dt)]["rms_mean"] for dt in dts])
p_fit, log_c = np.polyfit(log_dt, log_err, 1)
print(f"\n  Estimated convergence order p = {p_fit:.2f} (fit: RMS ~ dt^p)")
conv_results["_fit"] = {"p": float(p_fit), "log_intercept": float(log_c)}

with open("convergence_order.json", "w") as f:
    json.dump(conv_results, f)

# ===========================================================================
# (B) Trajectory-count convergence: error vs M
# ===========================================================================
print("\n" + "=" * 70)
print("(B) Trajectory-count convergence study")
print("=" * 70)
dt_fixed = 0.01
Ms = [50, 100, 200, 400, 800, 1600, 3200]
seed_base = 5000

mcount_results = {}
for M in Ms:
    times, fid_traj, _ = run_trajectories(lam, dt_fixed, T, M, seed_base + M)
    fid_mean = fid_traj.mean(axis=0)
    fid_sem = fid_traj.std(axis=0, ddof=1) / np.sqrt(M)
    analytic = analytic_fidelity(lam, times)
    final_err = abs(fid_mean[-1] - analytic[-1])
    rms = np.sqrt(np.mean((fid_mean - analytic) ** 2))
    mcount_results[str(M)] = {
        "final_fidelity": float(fid_mean[-1]),
        "final_sem": float(fid_sem[-1]),
        "final_abs_error": float(final_err),
        "rms_error": float(rms),
    }
    print(f"  M={M:5d}: final fidelity = {fid_mean[-1]:.4f} +/- {fid_sem[-1]:.4f}, "
          f"|error| = {final_err:.4f}, RMS = {rms:.4f}")

with open("trajectory_count_convergence.json", "w") as f:
    json.dump(mcount_results, f)

# ===========================================================================
# (C) Individual trajectory realizations
# ===========================================================================
print("\n" + "=" * 70)
print("(C) Individual trajectory realizations")
print("=" * 70)
M_indiv = 10
times, fid_traj_indiv, _ = run_trajectories(lam, 0.01, T, M_indiv, seed=999)
indiv_results = {
    "times": times.tolist(),
    "trajectories": fid_traj_indiv.tolist(),
    "mean": fid_traj_indiv.mean(axis=0).tolist(),
}
with open("individual_trajectories.json", "w") as f:
    json.dump(indiv_results, f)
print(f"  Saved {M_indiv} individual trajectories.")

# ===========================================================================
# (D) Normalization ablation
# ===========================================================================
print("\n" + "=" * 70)
print("(D) Normalization ablation: renormalized vs. unrenormalized updates")
print("=" * 70)
M_norm = 400
dt_norm = 0.01

times_n, fid_normed, norm_normed = run_trajectories(lam, dt_norm, T, M_norm, seed=42, normalize=True)
times_u, fid_unnormed, norm_unnormed = run_trajectories(lam, dt_norm, T, M_norm, seed=42, normalize=False)

analytic = analytic_fidelity(lam, times_n)
fid_normed_mean = fid_normed.mean(axis=0)
fid_unnormed_mean = fid_unnormed.mean(axis=0)
norm_normed_mean = norm_normed.mean(axis=0)
norm_unnormed_mean = norm_unnormed.mean(axis=0)
norm_unnormed_std = norm_unnormed.std(axis=0)

final_bias_normed = fid_normed_mean[-1] - analytic[-1]
final_bias_unnormed = fid_unnormed_mean[-1] - analytic[-1]

print(f"  Renormalized:   final fidelity = {fid_normed_mean[-1]:.4f}, "
      f"bias vs analytic = {final_bias_normed:+.4f}, mean final norm = {norm_normed_mean[-1]:.6f}")
print(f"  Unrenormalized: final fidelity = {fid_unnormed_mean[-1]:.4f}, "
      f"bias vs analytic = {final_bias_unnormed:+.4f}, mean final norm = {norm_unnormed_mean[-1]:.6f} "
      f"+/- {norm_unnormed_std[-1]:.6f}")

norm_ablation = {
    "times": times_n.tolist(),
    "analytic": analytic.tolist(),
    "fid_normed_mean": fid_normed_mean.tolist(),
    "fid_unnormed_mean": fid_unnormed_mean.tolist(),
    "norm_normed_mean": norm_normed_mean.tolist(),
    "norm_unnormed_mean": norm_unnormed_mean.tolist(),
    "norm_unnormed_std": norm_unnormed_std.tolist(),
}
with open("normalization_ablation.json", "w") as f:
    json.dump(norm_ablation, f)

print("\nAll experiments complete. JSON files written.")
