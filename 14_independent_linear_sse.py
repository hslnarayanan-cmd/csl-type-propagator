"""
Script 14 of 14 -- Independent linear-SSE diffusive solver cross-check
(paper Section 10.3, Table 12).

Derivation. Algorithm 1 integrates the NONLINEAR, norm-preserving SDE
(Eq. 2 in the paper):

    d|psi> = [-iH dt + sqrt(lambda) sum_i (A_i - <A_i>) dW_i
              - (lambda/2) sum_i (A_i - <A_i>)^2 dt] |psi>

with renormalization after every step. A standard alternative in quantum
trajectory theory (see e.g. Wiseman & Milburn) is the LINEAR SDE obtained
by dropping the <A_i> feedback term entirely:

    d|phi> = [-iH dt + sqrt(lambda) sum_i A_i dW_i
              - (lambda/2) sum_i A_i^2 dt] |phi>                    (*)

This equation is "blind": it never needs to compute <A_i>, and it is
NOT norm-preserving -- ||phi_t|| drifts away from 1 as the state
evolves, encoding an implicit probability weight. Applying Ito's rule to
d(|phi><phi|) and taking the expectation, EVERY term involving the
(now-absent) <A_i> feedback that appears in the nonlinear case's
derivation is simply absent, and what remains reduces directly to the
Lindblad master equation for the RAW (unnormalized, unweighted) second
moment:

    d E[|phi><phi|] / dt = -i[H, E[|phi><phi|]]
                            + lambda sum_i (A_i E[|phi><phi|] A_i
                                             - {A_i^2, E[|phi><phi|]}/2)

i.e. rho_t = E[|phi_t><phi_t|] EXACTLY, with no renormalization or
reweighting needed -- the norm decay of individual realizations already
encodes the correct statistical weighting. This gives a second,
structurally independent way to estimate rho_t: average the RAW outer
products of unnormalized trajectories, rather than Algorithm 1's
"renormalize every step, then average normalized outer products."

Because this alternative method has neither a normalization step nor
<A_i>-feedback logic, it provides a structurally independent second
implementation for validating Algorithm 1's normalization and
<A_i>-feedback logic against.

Run:
    python3 linear_sse_check.py
"""

import numpy as np
from scipy.linalg import expm
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

T = 1.0


def linear_sse_step(phi, H, lam, dt, rng):
    """One Euler-Maruyama step of the LINEAR SDE (*). No <A_i> feedback,
    no renormalization -- structurally simpler and independent of
    Algorithm 1's update rule."""
    drift = -1j * (H @ phi) * dt
    for Zq in Z_ops:
        dW = rng.normal(0.0, np.sqrt(dt))
        drift = drift + np.sqrt(lam) * (Zq @ phi) * dW - 0.5 * lam * (Zq @ Zq @ phi) * dt
    return phi + drift


def run_linear_ensemble(psi0, H, lam, dt, T, M, seed):
    rng = np.random.default_rng(seed)
    n_steps = int(round(T / dt))
    rho_accum = np.zeros((dim, dim), dtype=complex)
    norms_final = np.zeros(M)
    for k in range(M):
        phi = psi0.copy()
        for _ in range(n_steps):
            phi = linear_sse_step(phi, H, lam, dt, rng)
        rho_accum += np.outer(phi, phi.conj())  # RAW, unnormalized, unweighted
        norms_final[k] = np.linalg.norm(phi)
    rho_accum /= M  # simple arithmetic mean -- no reweighting
    return rho_accum, norms_final


def von_neumann_entropy(rho):
    evals = np.linalg.eigvalsh(rho)
    evals = evals[evals > 1e-12]
    return float(-np.sum(evals * np.log2(evals)))


if __name__ == "__main__":
    M = 2000  # larger M than Algorithm 1's validation runs, since the
              # linear method's per-trajectory variance is higher (raw,
              # unnormalized outer products have larger spread than
              # normalized ones)
    dt = 0.01
    lam = 0.20

    print("=" * 70)
    print("Case 1: H=0 (commuting), GHZ -- compare against closed form")
    print("=" * 70)
    H0 = np.zeros((dim, dim), dtype=complex)
    rho_lin, norms1 = run_linear_ensemble(ghz, H0, lam, dt, T, M, seed=1001)
    F_lin = float(np.real(np.vdot(ghz, rho_lin @ ghz)))
    S_lin = von_neumann_entropy(rho_lin)
    F_analytic = 0.5 + 0.5 * np.exp(-6 * lam * T)
    S_analytic = -F_analytic * np.log2(F_analytic) - (1 - F_analytic) * np.log2(1 - F_analytic)
    print(f"Linear-SSE method:  F={F_lin:.4f}, S={S_lin:.4f}")
    print(f"Closed form:        F={F_analytic:.4f}, S={S_analytic:.4f}")
    print(f"|diff|: F={abs(F_lin-F_analytic):.4f}, S={abs(S_lin-S_analytic):.4f}")
    print(f"Final-state norm: mean={norms1.mean():.4f}, std={norms1.std():.4f}, "
          f"min={norms1.min():.4f}, max={norms1.max():.4f}")
    print(f"trace(rho_lin) = {np.real(np.trace(rho_lin)):.6f} (should be close to 1; "
          f"this is itself a nontrivial check since nothing enforces it explicitly)")

    print("\n" + "=" * 70)
    print("Case 2: H=omega*sum(X_i), non-commuting -- compare against ODE reference")
    print("=" * 70)
    omega = 1.0
    H1 = omega * sum(X_ops)
    rho_lin2, norms2 = run_linear_ensemble(ghz, H1, lam, dt, T, M, seed=2002)
    F_lin2 = float(np.real(np.vdot(ghz, rho_lin2 @ ghz)))
    S_lin2 = von_neumann_entropy(rho_lin2)

    # independent ODE reference (same as Section 9's reference solver)
    from scipy.integrate import solve_ivp

    def lindblad_rhs(t, rho_flat, H, lam):
        rho = rho_flat.reshape(dim, dim)
        drho = -1j * (H @ rho - rho @ H)
        for Zq in Z_ops:
            drho += lam * (Zq @ rho @ Zq - 0.5 * (Zq @ Zq @ rho + rho @ Zq @ Zq))
        return drho.flatten()

    rho0 = np.outer(ghz, ghz.conj())
    sol = solve_ivp(lindblad_rhs, [0, T], rho0.flatten(), args=(H1, lam),
                     t_eval=[T], method="RK45", rtol=1e-9, atol=1e-12)
    rho_ref = sol.y[:, 0].reshape(dim, dim)
    F_ref = float(np.real(np.vdot(ghz, rho_ref @ ghz)))
    S_ref = von_neumann_entropy(rho_ref)

    print(f"Linear-SSE method:  F={F_lin2:.4f}, S={S_lin2:.4f}")
    print(f"ODE reference:      F={F_ref:.4f}, S={S_ref:.4f}")
    print(f"|diff|: F={abs(F_lin2-F_ref):.4f}, S={abs(S_lin2-S_ref):.4f}")
    print(f"Final-state norm: mean={norms2.mean():.4f}, std={norms2.std():.4f}, "
          f"min={norms2.min():.4f}, max={norms2.max():.4f}")
    print(f"trace(rho_lin2) = {np.real(np.trace(rho_lin2)):.6f}")

    max_elem_err = float(np.max(np.abs(rho_lin2 - rho_ref)))
    print(f"Max density-matrix element error vs. ODE reference: {max_elem_err:.4f}")

    out = {
        "case1_ghz_H0": {"F_lin": F_lin, "S_lin": S_lin, "F_analytic": F_analytic,
                          "S_analytic": S_analytic, "norm_mean": float(norms1.mean()),
                          "norm_std": float(norms1.std()), "trace": float(np.real(np.trace(rho_lin)))},
        "case2_ghz_noncommuting": {"F_lin": F_lin2, "S_lin": S_lin2, "F_ref": F_ref, "S_ref": S_ref,
                                    "max_elem_err": max_elem_err, "norm_mean": float(norms2.mean()),
                                    "norm_std": float(norms2.std()), "trace": float(np.real(np.trace(rho_lin2)))},
        "M": M, "dt": dt, "lam": lam,
    }
    with open("linear_sse_results.json", "w") as f:
        json.dump(out, f, indent=2)
