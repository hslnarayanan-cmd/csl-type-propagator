"""
Script 9 of 9 -- Multi-seed, multi-grid strong-convergence estimate
(paper Section 8.1, Table 6).

Extends the single-seed, single-fine-grid, single-initial-state estimate
in script 6 along every axis needed for a robust estimate:

  - Two independent fine-grid resolutions (dt_fine in {0.00125, 0.000625}),
    to check whether the fitted convergence order is sensitive to the
    reference solution's own (unremoved) discretization bias.
  - Three independent master seeds per configuration, so the fitted order
    p is reported as mean +/- std across seeds, not a single point estimate.
  - Two initial states: the GHZ state used in script 6, and a product state
    |+>^{ox 3} that does NOT stay confined to a 2-dimensional subspace
    under Z-dephasing (unlike GHZ), giving a structurally different test.
  - Regression diagnostics: slope, standard error of the slope, R^2, via
    scipy.stats.linregress, not just a bare polyfit slope.
  - Two distinct error metrics per configuration:
      * "observable error": |F_dt(target) - F_ref(target)|, the same
        kind of scalar fidelity error used in script 6;
      * "state-vector error": 1 - |<psi_dt|psi_ref>|^2, the infidelity
        between the FULL numerical state produced at step size dt and
        the fine-grid reference state, at the same final time -- a
        stricter, more complete error notion than a single observable.

We do NOT claim the resulting p is a general strong order for the
algorithm; it is reported per (initial state, fine-grid resolution) as an
empirical estimate at the tested parameters, consistent with the caution
the paper's Section 8.1 states, backed here by evidence across multiple
seeds and reference resolutions rather than a single seed and reference
resolution.

Run:
    python3 convergence_hardened.py
"""

import numpy as np
from scipy.stats import linregress
import json
import time

n_qubits = 3
dim = 2 ** n_qubits
I2 = np.eye(2, dtype=complex)
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

# |+>^{ox 3}: equal superposition product state, all 8 amplitudes 1/sqrt(8)
plus3 = np.ones(dim, dtype=complex) / np.sqrt(dim)

lam = 0.20
T = 1.0


def csl_step_with_noise(psi, dt, dW3):
    for Zq, dW in zip(Z_ops, dW3):
        exp_z = float(np.real(np.conjugate(psi) @ (Zq @ psi)))
        diff_op = Zq - exp_z * np.eye(dim)
        term1 = np.sqrt(lam) * (diff_op @ psi) * dW
        term2 = -0.5 * lam * (diff_op @ diff_op @ psi) * dt
        psi = psi + term1 + term2
    return psi / np.linalg.norm(psi)


def run_path(psi0, fine_increments_k, dt, dt_fine, n_fine):
    steps_per_block = int(round(dt / dt_fine))
    n_steps = n_fine // steps_per_block
    coarse = fine_increments_k[:n_steps * steps_per_block, :].reshape(n_steps, steps_per_block, 3).sum(axis=1)
    psi = psi0.copy()
    for t_idx in range(n_steps):
        psi = csl_step_with_noise(psi, dt, coarse[t_idx, :])
    return psi


def run_experiment(psi0, target, dt_fine, seed, M, dts_test):
    n_fine = int(round(T / dt_fine))
    rng = np.random.default_rng(seed)
    fine_increments = rng.normal(0.0, np.sqrt(dt_fine), size=(M, n_fine, 3))

    # fine-grid reference: full state and target-fidelity, per path
    psi_ref = np.zeros((M, dim), dtype=complex)
    for k in range(M):
        psi_ref[k] = run_path(psi0, fine_increments[k], dt_fine, dt_fine, n_fine)
    F_ref = np.abs(psi_ref.conj() @ target) ** 2

    results = {}
    for dt in dts_test:
        obs_errors = np.zeros(M)
        state_errors = np.zeros(M)
        for k in range(M):
            psi_dt = run_path(psi0, fine_increments[k], dt, dt_fine, n_fine)
            F_dt = np.abs(np.vdot(target, psi_dt)) ** 2
            obs_errors[k] = abs(F_dt - F_ref[k])
            overlap = np.vdot(psi_ref[k], psi_dt)
            state_errors[k] = 1.0 - np.abs(overlap) ** 2
        results[str(dt)] = {
            "obs_rms": float(np.sqrt(np.mean(obs_errors ** 2))),
            "state_rms": float(np.sqrt(np.mean(state_errors ** 2))),
        }
    return results


def fit_order(dts_test, results, key):
    dts_arr = np.array(dts_test)
    err_arr = np.array([results[str(dt)][key] for dt in dts_test])
    fit = linregress(np.log(dts_arr), np.log(err_arr))
    return {"p": fit.slope, "p_stderr": fit.stderr, "r_squared": fit.rvalue ** 2, "intercept": fit.intercept}


if __name__ == "__main__":
    dts_test = [0.04, 0.02, 0.01, 0.005, 0.0025]
    M = 250
    seeds = [2026, 3037, 4048]
    dt_fines = [0.00125, 0.000625]

    initial_states = {
        "GHZ": (ghz, ghz),
        "plus3": (plus3, plus3),
    }

    all_results = {}
    import os
    if os.path.exists("convergence_hardened.json"):
        with open("convergence_hardened.json") as f:
            all_results = json.load(f)
        print(f"Resuming: {len(all_results)} configuration(s) already complete: {list(all_results.keys())}")

    t_start = time.time()
    for state_name, (psi0, target) in initial_states.items():
        for dt_fine in dt_fines:
            key = f"{state_name}_dtfine{dt_fine}"
            if key in all_results:
                print(f"Skipping {key} (already complete)")
                continue
            print(f"\n{'='*70}\n{key}\n{'='*70}")
            per_seed_fits_obs = []
            per_seed_fits_state = []
            raw_runs = {}
            for seed in seeds:
                res = run_experiment(psi0, target, dt_fine, seed, M, dts_test)
                raw_runs[str(seed)] = res
                fit_obs = fit_order(dts_test, res, "obs_rms")
                fit_state = fit_order(dts_test, res, "state_rms")
                per_seed_fits_obs.append(fit_obs["p"])
                per_seed_fits_state.append(fit_state["p"])
                print(f"  seed={seed}: p_obs={fit_obs['p']:.3f} (SE={fit_obs['p_stderr']:.3f}, "
                      f"R^2={fit_obs['r_squared']:.4f}) | p_state={fit_state['p']:.3f} "
                      f"(SE={fit_state['p_stderr']:.3f}, R^2={fit_state['r_squared']:.4f})")
            p_obs_mean, p_obs_std = float(np.mean(per_seed_fits_obs)), float(np.std(per_seed_fits_obs))
            p_state_mean, p_state_std = float(np.mean(per_seed_fits_state)), float(np.std(per_seed_fits_state))
            print(f"  --> p_obs = {p_obs_mean:.3f} +/- {p_obs_std:.3f} (across {len(seeds)} seeds)")
            print(f"  --> p_state = {p_state_mean:.3f} +/- {p_state_std:.3f} (across {len(seeds)} seeds)")
            all_results[key] = {
                "raw_runs": raw_runs,
                "p_obs_mean": p_obs_mean, "p_obs_std": p_obs_std,
                "p_state_mean": p_state_mean, "p_state_std": p_state_std,
                "per_seed_p_obs": per_seed_fits_obs,
                "per_seed_p_state": per_seed_fits_state,
            }
            # write incrementally after each configuration so a timeout doesn't lose completed work
            with open("convergence_hardened.json", "w") as f:
                json.dump(all_results, f, indent=2)

    print(f"\nTotal elapsed: {time.time()-t_start:.1f}s")
    with open("convergence_hardened.json", "w") as f:
        json.dump(all_results, f, indent=2)
