"""
Script 6 of 7 -- Strong-convergence estimate for the Euler-Maruyama
CSL integrator using common random numbers, at a single configuration
(paper Section 8.1, Figure 4).

This script uses the standard "common random numbers" empirical
strong-convergence methodology (Kloeden & Platen, "Numerical Solution of
Stochastic Differential Equations", 1992): the same underlying Wiener path
is sampled at a fine resolution dt_fine, aggregated into coarser step
sizes for each tested dt, and the integrator's output at each dt is
compared pathwise against a fine-grid reference run of the SAME
(nonlinear, normalized) Euler-Maruyama scheme at dt_fine. This is the
standard approach when no closed-form solution of the physically driven
SDE is available: a closed-form solution of the *linear* (unnormalized)
representation of the SDE is only equivalent to the nonlinear equation
actually implemented *in distribution*, via a Girsanov change of measure
-- not pathwise for identical noise values -- so comparing against it
pathwise produces a constant, non-vanishing error at every dt (a
negative result discussed in the paper, Section 8.1, that motivates
using fine-grid self-convergence here instead). Script 9 extends this
single-configuration estimate to a multi-seed, multi-grid convergence
estimate; comparing independent seeds at each dt against the
closed-form ensemble solution, as in script 5, cannot on its own
isolate discretization error from Monte Carlo sampling noise.

Run:
    python3 6_rigorous_strong_convergence.py
"""

import numpy as np
import json

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

lam = 0.20
T = 1.0


def csl_step_with_noise(psi, lambda_val, dt, dW3):
    """One Euler-Maruyama step, given pre-supplied Wiener increments dW3
    (one per collapse operator) instead of drawing fresh randomness --
    this is what lets us reuse the identical noise realization across
    different step sizes dt."""
    for Zq, dW in zip(Z_ops, dW3):
        exp_z = float(np.real(np.conjugate(psi) @ (Zq @ psi)))
        diff_op = Zq - exp_z * np.eye(dim)
        term1 = np.sqrt(lambda_val) * (diff_op @ psi) * dW
        term2 = -0.5 * lambda_val * (diff_op @ diff_op @ psi) * dt
        psi = psi + term1 + term2
    return psi / np.linalg.norm(psi)


def run_path(fine_increments_k, dt, dt_fine, n_fine):
    """Run the actual nonlinear Euler-Maruyama scheme at step size dt,
    using coarse increments aggregated (summed) from the shared fine
    noise realization fine_increments_k (shape: n_fine x 3)."""
    steps_per_block = int(round(dt / dt_fine))
    n_steps = n_fine // steps_per_block
    coarse = fine_increments_k[:n_steps * steps_per_block, :].reshape(n_steps, steps_per_block, 3).sum(axis=1)
    psi = psi0.copy()
    for t_idx in range(n_steps):
        psi = csl_step_with_noise(psi, lam, dt, coarse[t_idx, :])
    return np.abs(np.vdot(ghz, psi)) ** 2


if __name__ == "__main__":
    dt_fine = 0.00125          # finest resolution: 800 steps over [0,1]; used as the reference
    n_fine = int(round(T / dt_fine))
    dts_test = [0.04, 0.02, 0.01, 0.005, 0.0025]   # all integer multiples of dt_fine, all coarser
    M = 500
    master_seed = 2026

    print(f"Fine grid: dt_fine={dt_fine}, {n_fine} steps. Reference = same scheme at dt_fine.")
    print(f"Testing dt in {dts_test}, M={M} paths, common random numbers.\n")

    rng = np.random.default_rng(master_seed)
    fine_increments = rng.normal(0.0, np.sqrt(dt_fine), size=(M, n_fine, 3))

    # Fine-grid reference: run the actual scheme at dt_fine for every path
    F_ref = np.zeros(M)
    for k in range(M):
        F_ref[k] = run_path(fine_increments[k], dt_fine, dt_fine, n_fine)

    results = {}
    for dt in dts_test:
        errors = np.zeros(M)
        for k in range(M):
            F_dt = run_path(fine_increments[k], dt, dt_fine, n_fine)
            errors[k] = abs(F_dt - F_ref[k])

        rms_err = float(np.sqrt(np.mean(errors ** 2)))
        mean_err = float(np.mean(errors))
        results[str(dt)] = {"rms_pathwise_error": rms_err, "mean_pathwise_error": mean_err}
        print(f"  dt={dt:<7} pathwise RMS error (vs. dt_fine reference, common noise) = {rms_err:.6f}  "
              f"(mean abs error = {mean_err:.6f})")

    dts_arr = np.array(dts_test)
    rms_arr = np.array([results[str(dt)]["rms_pathwise_error"] for dt in dts_test])
    p_fit, log_c = np.polyfit(np.log(dts_arr), np.log(rms_arr), 1)
    print(f"\nEstimated empirical strong-convergence order p = {p_fit:.2f} (RMS pathwise error ~ dt^p)")
    results["_fit"] = {"p": float(p_fit), "log_intercept": float(log_c), "M": M, "dt_fine": dt_fine}
    results["_note"] = ("Reference is the same nonlinear Euler-Maruyama scheme at dt_fine, "
                         "not a closed-form solution -- see module docstring for why a "
                         "closed-form linear-SDE reference is NOT valid for this comparison.")

    with open("rigorous_convergence.json", "w") as f:
        json.dump(results, f)
