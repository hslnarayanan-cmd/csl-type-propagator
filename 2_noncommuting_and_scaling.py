"""
Script 2 of 4 --- Independent reference-solver validation in a non-commuting
regime, plus the wall-clock scaling benchmark against deterministic
integration.

Reproduces:
  - Section 5.1: Figure 4, and the 0.34-sigma agreement quoted in the text
  - Section 7.1: Table 3 and Figure 6 (this produces one of the two runs;
    the second run in Table 3 was produced by re-running this same script
    on separate hardware)

System: H = omega * sum_i X_i (transverse field), Pauli-Z collapse operators.
Since [H, A_i] != 0, no closed-form solution exists here -- that is the
point of this test. The reference is an independent direct integration of
the Lindblad master equation (conceptually identical to what QuTiP's
mesolve does internally; see script 3 for the QuTiP-side comparison).

Run:
    python3 2_noncommuting_and_scaling.py

Outputs:
    noncommuting_validation.json  -- fidelity/entropy time series, MC vs. reference
    scaling_comparison.json       -- wall-clock time vs. register size, n=3..8
"""

import numpy as np
from scipy.integrate import solve_ivp
import json
import time

# ---------- Setup: 3-qubit register ----------
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

ghz = np.zeros(dim, dtype=complex)
ghz[0] = 1 / np.sqrt(2)
ghz[-1] = 1 / np.sqrt(2)
rho_ghz = np.outer(ghz, ghz.conj())
psi0 = ghz.copy()

# ---------- Non-commuting test case: H = omega * sum_i X_i, collapse operators A_i = Z_i ----------
# [H, A_i] != 0, so no simple closed-form solution exists here -- this is the point.
omega = 1.0
H = omega * sum(X_ops)

def von_neumann_entropy(rho):
    evals = np.linalg.eigvalsh(rho)
    evals = evals[evals > 1e-12]
    return float(-np.sum(evals * np.log2(evals)))

# ---------- Reference solver: direct Lindblad master-equation integration ----------
# d rho/dt = -i[H, rho] + lambda * sum_i ( A_i rho A_i - 0.5*{A_i^2, rho} )
# Since Z_i^2 = I, this simplifies to: -i[H,rho] - (lambda/2) * sum_i [A_i,[A_i,rho]]
# Implemented as direct matrix ops on rho (not a full superoperator), keeping memory at O(4^n).
def lindblad_rhs(t, rho_flat, lam):
    rho = rho_flat.reshape(dim, dim)
    drho = -1j * (H @ rho - rho @ H)
    for Zq in Z_ops:
        drho += lam * (Zq @ rho @ Zq - 0.5 * (Zq @ Zq @ rho + rho @ Zq @ Zq))
    return drho.flatten()

def run_lindblad_reference(lam, T, n_eval=101):
    t_eval = np.linspace(0, T, n_eval)
    sol = solve_ivp(lindblad_rhs, [0, T], rho_ghz.flatten(), args=(lam,),
                     t_eval=t_eval, method="RK45", rtol=1e-9, atol=1e-12)
    fidelity = np.zeros(n_eval)
    entropy = np.zeros(n_eval)
    for k in range(n_eval):
        rho_t = sol.y[:, k].reshape(dim, dim)
        fidelity[k] = np.real(np.vdot(ghz, rho_t @ ghz))
        entropy[k] = von_neumann_entropy(rho_t)
    return t_eval, fidelity, entropy

# ---------- CSL Monte Carlo propagator, including a Hamiltonian term ----------
U_cache = {}
def get_U(lam, dt):
    key = round(dt, 8)
    if key not in U_cache:
        from scipy.linalg import expm
        U_cache[key] = expm(-1j * H * dt)
    return U_cache[key]

def csl_step(psi, lam, dt):
    U = get_U(lam, dt)
    psi = U @ psi
    for Zq in Z_ops:
        exp_z = float(np.real(np.conjugate(psi) @ (Zq @ psi)))
        dW = np.random.normal(0.0, np.sqrt(dt))
        diff_op = Zq - exp_z * np.eye(dim)
        term1 = np.sqrt(lam) * (diff_op @ psi) * dW
        term2 = -0.5 * lam * (diff_op @ diff_op @ psi) * dt
        psi = psi + term1 + term2
    return psi / np.linalg.norm(psi)

def run_mc_ensemble(lam, dt, T, M, seed):
    rng = np.random.default_rng(seed)
    np.random.seed(seed)
    n_steps = int(round(T / dt))
    times = np.linspace(0, T, n_steps + 1)
    fidelity_traj = np.zeros((M, n_steps + 1))
    psi_all = np.zeros((M, n_steps + 1, dim), dtype=complex)
    for k in range(M):
        psi = psi0.copy()
        psi_all[k, 0] = psi
        fidelity_traj[k, 0] = np.abs(np.vdot(ghz, psi)) ** 2
        for t_idx in range(1, n_steps + 1):
            psi = csl_step(psi, lam, dt)
            psi_all[k, t_idx] = psi
            fidelity_traj[k, t_idx] = np.abs(np.vdot(ghz, psi)) ** 2
    fidelity_mean = fidelity_traj.mean(axis=0)
    fidelity_sem = fidelity_traj.std(axis=0, ddof=1) / np.sqrt(M)
    entropy = np.zeros(n_steps + 1)
    for t_idx in range(n_steps + 1):
        rho_full = np.einsum('ki,kj->ij', psi_all[:, t_idx, :], psi_all[:, t_idx, :].conj()) / M
        entropy[t_idx] = von_neumann_entropy(rho_full)
    return times, fidelity_mean, fidelity_sem, entropy

if __name__ == "__main__":
    lam = 0.20
    T = 1.0

    print("Running independent Lindblad ODE reference (non-commuting H, Z collapse)...")
    t_ref, F_ref, S_ref = run_lindblad_reference(lam, T)

    print("Running CSL Monte Carlo propagator (same non-commuting Hamiltonian)...")
    np.random.seed(2026)
    t_mc, F_mc, F_mc_sem, S_mc = run_mc_ensemble(lam, dt=0.01, T=T, M=400, seed=2026)

    # interpolate reference onto MC time grid for direct comparison at matching points
    F_ref_interp = np.interp(t_mc, t_ref, F_ref)
    S_ref_interp = np.interp(t_mc, t_ref, S_ref)
    max_abs_err_F = np.max(np.abs(F_mc - F_ref_interp))
    final_diff_F = F_mc[-1] - F_ref_interp[-1]
    final_sem = F_mc_sem[-1]

    print(f"Final fidelity: MC={F_mc[-1]:.4f} +/- {final_sem:.4f}, "
          f"Lindblad-ODE reference={F_ref_interp[-1]:.4f}, diff={final_diff_F:.4f} "
          f"({abs(final_diff_F)/final_sem:.2f} sigma)")
    print(f"Max |F_mc - F_ref| over trajectory: {max_abs_err_F:.4f}")

    out = {
        "t_mc": t_mc.tolist(), "F_mc": F_mc.tolist(), "F_mc_sem": F_mc_sem.tolist(), "S_mc": S_mc.tolist(),
        "t_ref": t_ref.tolist(), "F_ref": F_ref.tolist(), "S_ref": S_ref.tolist(),
        "lambda": lam, "omega": omega,
    }
    with open("noncommuting_validation.json", "w") as f:
        json.dump(out, f)

    # ---------- Runtime / scaling comparison: trajectory MC vs deterministic ODE ----------
    print("\nRuntime scaling comparison (deterministic Lindblad ODE vs MC ensemble, diagonal-optimized)...")
    scaling_results = []
    for n in [3, 4, 5, 6, 7, 8]:
        dim_n = 2 ** n
        I2_ = np.eye(2, dtype=complex)
        Zn = [np.kron(np.kron(np.eye(2**i, dtype=complex), Z), np.eye(2**(n-i-1), dtype=complex)) for i in range(n)]
        Xn = [np.kron(np.kron(np.eye(2**i, dtype=complex), X), np.eye(2**(n-i-1), dtype=complex)) for i in range(n)]
        Hn = omega * sum(Xn)
        psi0_n = np.zeros(dim_n, dtype=complex)
        psi0_n[0] = 1 / np.sqrt(2)
        psi0_n[-1] = 1 / np.sqrt(2)
        rho0_n = np.outer(psi0_n, psi0_n.conj())

        def rhs_n(t, rho_flat, lam=0.2):
            rho = rho_flat.reshape(dim_n, dim_n)
            drho = -1j * (Hn @ rho - rho @ Hn)
            for Zq in Zn:
                drho += lam * (Zq @ rho @ Zq - 0.5 * (Zq @ Zq @ rho + rho @ Zq @ Zq))
            return drho.flatten()

        t0 = time.time()
        solve_ivp(rhs_n, [0, 1.0], rho0_n.flatten(), t_eval=[1.0], method="RK45", rtol=1e-6, atol=1e-9)
        ode_time = time.time() - t0

        # diagonal-optimized MC step (Z ops are diagonal after basis rotation is NOT assumed here;
        # we use the dense update, consistent with the general-H case, for a fair comparison)
        Un = None
        from scipy.linalg import expm
        dt_n = 0.01
        Un = expm(-1j * Hn * dt_n)

        def mc_step_n(psi, lam=0.2):
            psi = Un @ psi
            for Zq in Zn:
                exp_z = float(np.real(np.conjugate(psi) @ (Zq @ psi)))
                dW = np.random.normal(0.0, np.sqrt(dt_n))
                diff_op = Zq - exp_z * np.eye(dim_n)
                psi = psi + np.sqrt(lam) * (diff_op @ psi) * dW - 0.5 * lam * (diff_op @ diff_op @ psi) * dt_n
            return psi / np.linalg.norm(psi)

        M_small = 20  # small ensemble just to time a representative run
        t0 = time.time()
        for _ in range(M_small):
            psi = psi0_n.copy()
            for _ in range(100):  # dt=0.01, T=1.0 -> 100 steps
                psi = mc_step_n(psi)
        mc_time_per_traj = (time.time() - t0) / M_small

        scaling_results.append({
            "n": n, "dim": dim_n,
            "ode_time_sec": ode_time,
            "mc_time_per_trajectory_sec": mc_time_per_traj,
        })
        print(f"  n={n:2d} (dim={dim_n:5d}): ODE reference solve = {ode_time:7.4f}s | "
              f"MC per-trajectory = {mc_time_per_traj:7.4f}s "
              f"(x{400} trajectories = {mc_time_per_traj*400:7.2f}s for M=400)")

    with open("scaling_comparison.json", "w") as f:
        json.dump(scaling_results, f)
