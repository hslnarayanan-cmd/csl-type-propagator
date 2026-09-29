"""
Script 11 of 11 -- Expanded non-commuting-regime validation (paper
Sections 9.2-9.5, Tables 9-10).

Strengthens the single-configuration, single-observable, final-time-only
test in the original non-commuting validation along every axis requested:

  - Multiple (omega, lambda) configurations: a baseline, one with a
    different omega, one with a different lambda.
  - A second initial state (a product state) alongside GHZ.
  - Time-resolved absolute and relative error, not just a final-time
    number and an unqualified "max deviation".
  - Multiple observables per configuration: fidelity to the initial
    state, purity Tr(rho^2), a single-qubit observable <Z_0>, a
    two-qubit correlator <X_0 X_1>, and the Frobenius distance between
    the full simulated and reference density matrices (a state-level,
    not just scalar-observable, error measure).
  - Reference-solver sanity checks: trace preservation and Hermiticity
    error of the ODE-integrated reference rho_ref(t), tracked over time.
  - Positivity of the simulated ensemble density matrix: minimum
    eigenvalue of rho_sim(t), tracked over time.
  - A timestep refinement for the propagator at the baseline
    configuration (dt in {0.02, 0.01, 0.005}).
  - Confidence intervals from REPEATED trajectory ensembles (5
    independent M=400 ensembles at the baseline configuration), reported
    as mean +/- std across ensembles, distinct from the within-ensemble
    SEM already reported elsewhere.

"Maximum deviation", wherever reported below, is defined precisely as
max_t |F_sim(t) - F_ref(t)| over the 101-point time grid t in [0,1],
where F is fidelity to the run's initial state; this is stated explicitly
to avoid any ambiguity in how the term is used.
"""

import numpy as np
from scipy.integrate import solve_ivp
import json
import time

n_qubits = 3
dim = 2 ** n_qubits
I2 = np.eye(2, dtype=complex)
X = np.array([[0, 1], [1, 0]], dtype=complex)
Z = np.array([[1, 0], [0, -1]], dtype=complex)


def single_qubit_op(op, q, n):
    mats = [I2] * n
    mats[q] = op
    out = mats[0]
    for m in mats[1:]:
        out = np.kron(out, m)
    return out


Z_ops = [single_qubit_op(Z, q, n_qubits) for q in range(n_qubits)]
X_ops = [single_qubit_op(X, q, n_qubits) for q in range(n_qubits)]
Z0 = Z_ops[0]
X0X1 = X_ops[0] @ X_ops[1]

ghz = np.zeros(dim, dtype=complex)
ghz[0] = 1 / np.sqrt(2)
ghz[-1] = 1 / np.sqrt(2)
plus3 = np.ones(dim, dtype=complex) / np.sqrt(dim)

T = 1.0
n_eval = 101
tlist = np.linspace(0, T, n_eval)


def lindblad_rhs(t, rho_flat, H, lam):
    rho = rho_flat.reshape(dim, dim)
    drho = -1j * (H @ rho - rho @ H)
    for Zq in Z_ops:
        drho += lam * (Zq @ rho @ Zq - 0.5 * (Zq @ Zq @ rho + rho @ Zq @ Zq))
    return drho.flatten()


def run_reference(psi0, H, lam, t_eval):
    rho0 = np.outer(psi0, psi0.conj())
    sol = solve_ivp(lindblad_rhs, [0, t_eval[-1]], rho0.flatten(), args=(H, lam),
                     t_eval=t_eval, method="RK45", rtol=1e-9, atol=1e-12)
    rhos = [sol.y[:, k].reshape(dim, dim) for k in range(len(t_eval))]
    return rhos


def csl_step(psi, H, lam, dt, rng, U_cache):
    key = round(dt, 10)
    if key not in U_cache:
        from scipy.linalg import expm
        U_cache[key] = expm(-1j * H * dt)
    psi = U_cache[key] @ psi
    for Zq in Z_ops:
        exp_z = float(np.real(np.conjugate(psi) @ (Zq @ psi)))
        dW = rng.normal(0.0, np.sqrt(dt))
        diff_op = Zq - exp_z * np.eye(dim)
        psi = psi + np.sqrt(lam) * (diff_op @ psi) * dW - 0.5 * lam * (diff_op @ diff_op @ psi) * dt
    return psi / np.linalg.norm(psi)


def run_mc_ensemble(psi0, H, lam, dt, M, seed):
    """Records the state at EVERY native simulation step (dense, no
    subsampling/snapshot approximation) to avoid any grid-alignment
    artifacts; the reference solver is then evaluated at this exact
    same time grid, so the two curves are compared at identical times
    with no interpolation or snapshotting error of any kind."""
    rng = np.random.default_rng(seed)
    U_cache = {}
    n_steps = int(round(T / dt))
    t_grid = np.linspace(0, T, n_steps + 1)
    psi_all = np.zeros((M, n_steps + 1, dim), dtype=complex)
    for k in range(M):
        psi = psi0.copy()
        psi_all[k, 0] = psi
        for t_idx in range(1, n_steps + 1):
            psi = csl_step(psi, H, lam, dt, rng, U_cache)
            psi_all[k, t_idx] = psi
    rho_sim = np.einsum('kti,ktj->tij', psi_all, psi_all.conj()) / M
    fid_traj = np.abs(np.einsum('kti,i->kt', psi_all, psi0.conj())) ** 2
    return rho_sim, fid_traj, t_grid


def analyze_config(name, psi0, omega, lam, dt=0.01, M=400, seed=1):
    H = omega * sum(X_ops)
    rho_sim, fid_traj, t_grid = run_mc_ensemble(psi0, H, lam, dt, M, seed)
    n_pts = len(t_grid)
    rho_ref = run_reference(psi0, H, lam, t_grid)

    F_ref = np.array([np.real(np.vdot(psi0, rho_ref[t] @ psi0)) for t in range(n_pts)])
    F_sim = fid_traj.mean(axis=0)
    F_sem = fid_traj.std(axis=0, ddof=1) / np.sqrt(M)

    abs_err = np.abs(F_sim - F_ref)
    rel_err = abs_err / np.clip(F_ref, 1e-6, None)

    purity_ref = np.array([np.real(np.trace(rho_ref[t] @ rho_ref[t])) for t in range(n_pts)])
    purity_sim = np.array([np.real(np.trace(rho_sim[t] @ rho_sim[t])) for t in range(n_pts)])

    z0_ref = np.array([np.real(np.trace(rho_ref[t] @ Z0)) for t in range(n_pts)])
    z0_sim = np.array([np.real(np.trace(rho_sim[t] @ Z0)) for t in range(n_pts)])

    xx_ref = np.array([np.real(np.trace(rho_ref[t] @ X0X1)) for t in range(n_pts)])
    xx_sim = np.array([np.real(np.trace(rho_sim[t] @ X0X1)) for t in range(n_pts)])

    frob_dist = np.array([np.linalg.norm(rho_sim[t] - rho_ref[t]) for t in range(n_pts)])

    trace_ref = np.array([np.real(np.trace(rho_ref[t])) for t in range(n_pts)])
    herm_err_ref = np.array([np.max(np.abs(rho_ref[t] - rho_ref[t].conj().T)) for t in range(n_pts)])

    min_eig_sim = np.array([np.min(np.linalg.eigvalsh((rho_sim[t] + rho_sim[t].conj().T) / 2))
                             for t in range(n_pts)])

    result = {
        "omega": omega, "lam": lam, "dt": dt, "M": M, "seed": seed,
        "F_sim_final": float(F_sim[-1]), "F_sem_final": float(F_sem[-1]), "F_ref_final": float(F_ref[-1]),
        "max_abs_err": float(np.max(abs_err)), "max_abs_err_t": float(t_grid[np.argmax(abs_err)]),
        "max_rel_err": float(np.max(rel_err)), "rms_abs_err": float(np.sqrt(np.mean(abs_err ** 2))),
        "max_purity_err": float(np.max(np.abs(purity_sim - purity_ref))),
        "max_z0_err": float(np.max(np.abs(z0_sim - z0_ref))),
        "max_xx_err": float(np.max(np.abs(xx_sim - xx_ref))),
        "max_frob_dist": float(np.max(frob_dist)),
        "ref_trace_max_dev": float(np.max(np.abs(trace_ref - 1.0))),
        "ref_herm_err_max": float(np.max(herm_err_ref)),
        "sim_min_eig_overall": float(np.min(min_eig_sim)),
    }
    print(f"\n{name} (omega={omega}, lambda={lam}):")
    print(f"  Final fidelity: sim={F_sim[-1]:.4f}+/-{F_sem[-1]:.4f}, ref={F_ref[-1]:.4f}, "
          f"diff/SEM={abs(F_sim[-1]-F_ref[-1])/F_sem[-1]:.2f}")
    print(f"  max|F_sim-F_ref| over t = {result['max_abs_err']:.4f} (at t={result['max_abs_err_t']:.2f}), "
          f"max relative err = {result['max_rel_err']:.4f}")
    print(f"  max purity err = {result['max_purity_err']:.4f}, max <Z0> err = {result['max_z0_err']:.4f}, "
          f"max <X0X1> err = {result['max_xx_err']:.4f}")
    print(f"  max Frobenius(rho_sim, rho_ref) = {result['max_frob_dist']:.4f}")
    print(f"  reference: max|trace-1| = {result['ref_trace_max_dev']:.2e}, "
          f"max Hermiticity err = {result['ref_herm_err_max']:.2e}")
    print(f"  simulated ensemble: min eigenvalue overall = {result['sim_min_eig_overall']:.6f}")
    return result


if __name__ == "__main__":
    t0 = time.time()
    all_results = {}

    configs = [
        ("baseline_GHZ", ghz, 1.0, 0.20),
        ("vary_omega_GHZ", ghz, 2.0, 0.20),
        ("vary_lambda_GHZ", ghz, 1.0, 0.50),
        ("vary_initial_state_plus3", plus3, 1.0, 0.20),
    ]
    for name, psi0, omega, lam in configs:
        all_results[name] = analyze_config(name, psi0, omega, lam, dt=0.01, M=400, seed=2026)

    with open("noncommuting_expanded.json", "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nMain configs elapsed: {time.time()-t0:.1f}s")

    # ---- Timestep refinement, baseline config ----
    print("\n" + "=" * 70)
    print("Timestep refinement, baseline config (omega=1.0, lambda=0.20, GHZ)")
    print("=" * 70)
    dt_results = {}
    for dt in [0.02, 0.01, 0.005]:
        r = analyze_config(f"dt={dt}", ghz, 1.0, 0.20, dt=dt, M=400, seed=2026)
        dt_results[str(dt)] = {"F_sim_final": r["F_sim_final"], "F_sem_final": r["F_sem_final"],
                                "F_ref_final": r["F_ref_final"], "max_abs_err": r["max_abs_err"]}
    with open("noncommuting_dt_refinement.json", "w") as f:
        json.dump(dt_results, f, indent=2)

    # ---- Repeated-ensemble confidence intervals, baseline config ----
    print("\n" + "=" * 70)
    print("Repeated-ensemble CI, baseline config (5 independent M=400 ensembles)")
    print("=" * 70)
    ensemble_finals = []
    for seed in [2026, 3037, 4048, 5059, 6070]:
        H = 1.0 * sum(X_ops)
        _, fid_traj, _ = run_mc_ensemble(ghz, H, 0.20, 0.01, 400, seed)
        F_final = fid_traj.mean(axis=0)[-1]
        ensemble_finals.append(F_final)
        print(f"  seed={seed}: F_final = {F_final:.4f}")
    ens_mean, ens_std = float(np.mean(ensemble_finals)), float(np.std(ensemble_finals, ddof=1))
    print(f"  --> across-ensemble: F_final = {ens_mean:.4f} +/- {ens_std:.4f} (std across 5 ensembles)")
    with open("noncommuting_ensemble_ci.json", "w") as f:
        json.dump({"ensemble_finals": ensemble_finals, "mean": ens_mean, "std": ens_std}, f, indent=2)

    print(f"\nTotal elapsed: {time.time()-t0:.1f}s")
