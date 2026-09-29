"""
Script 10 of 10 -- General closed-form derivation and multi-state
validation (paper Sections 7.1, 7.4, Table 7).

Derivation (H=0, Pauli-Z collapse operators, L_i = sqrt(lambda)*Z_i):

    d rho/dt = lambda * sum_i ( Z_i rho Z_i - rho )

In the computational basis, Z_i has eigenvalue +1 on basis states with
bit i = 0 and -1 on basis states with bit i = 1, so

    (Z_i rho Z_i)_{ab} = z_i(a) z_i(b) * rho_{ab}

where z_i(a) = +1 if bit i of a is 0, else -1. Since z_i(a)z_i(b) = +1
when bits i of a,b agree and -1 when they differ,

    d rho_{ab}/dt = -2*lambda*d_H(a,b) * rho_{ab}

where d_H(a,b) is the Hamming distance between basis strings a and b.
This integrates exactly to

    rho_{ab}(t) = rho_{ab}(0) * exp(-2*lambda*d_H(a,b)*t)      (*)

i.e. every coherence decays exponentially at a rate set by the Hamming
distance between the two basis states it connects; populations
(a=b, d_H=0) are exactly conserved. This is the general solution for
ANY initial state under H=0, Pauli-Z collapse -- the GHZ closed form used
elsewhere in the paper (F(t) = 1/2 + 1/2*exp(-6*lambda*t)) is the special
case a=|000>, b=|111>, d_H=3.

This script:
  (1) verifies (*) against an independent direct Lindblad ODE integration
      for a generic (non-symmetric) initial density matrix, confirming
      the derivation itself is correct, independent of Algorithm 1;
  (2) validates Algorithm 1's actual Monte Carlo output against (*) for
      five structurally different initial states, comparing every
      density-matrix element (not just one scalar fidelity), which
      exercises multiple simultaneous decay rates per state.
"""

import numpy as np
from scipy.integrate import solve_ivp
from scipy.linalg import expm
import json

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
lam = 0.20
T = 1.0


def hamming(a, b):
    return bin(a ^ b).count("1")


def analytic_rho(rho0, lam, t):
    rho_t = np.zeros_like(rho0)
    for a in range(dim):
        for b in range(dim):
            dH = hamming(a, b)
            rho_t[a, b] = rho0[a, b] * np.exp(-2 * lam * dH * t)
    return rho_t


# ---------------------------------------------------------------------------
# (1) Independent check: general formula vs direct Lindblad ODE integration,
#     for a generic (non-symmetric, all-elements-nonzero) initial state.
# ---------------------------------------------------------------------------
print("=" * 70)
print("(1) General formula vs. independent Lindblad ODE integration")
print("=" * 70)

rng = np.random.default_rng(7)
psi0_generic = rng.normal(size=dim) + 1j * rng.normal(size=dim)
psi0_generic /= np.linalg.norm(psi0_generic)
rho0_generic = np.outer(psi0_generic, psi0_generic.conj())


def lindblad_rhs(t, rho_flat, lam):
    rho = rho_flat.reshape(dim, dim)
    drho = np.zeros_like(rho)
    for Zq in Z_ops:
        drho += lam * (Zq @ rho @ Zq - rho)
    return drho.flatten()


sol = solve_ivp(lindblad_rhs, [0, T], rho0_generic.flatten(), args=(lam,),
                 t_eval=[T], method="RK45", rtol=1e-10, atol=1e-13)
rho_ode = sol.y[:, 0].reshape(dim, dim)
rho_formula = analytic_rho(rho0_generic, lam, T)
max_err = np.max(np.abs(rho_ode - rho_formula))
print(f"Max |rho_ODE - rho_formula| over all {dim*dim} matrix elements at t={T}: {max_err:.2e}")

# ---------------------------------------------------------------------------
# (2) Algorithm 1's actual Monte Carlo output vs. the general formula,
#     for five structurally different initial states.
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("(2) Algorithm 1 (Monte Carlo) vs. general formula, five initial states")
print("=" * 70)


def csl_step(psi, dt, rng):
    """Quotient (normalized) expectation convention, matching Algorithm 1 /
    Listing 1 in the paper: mu_i = Re<psi|A_i|psi> / <psi|psi>, using the
    current (possibly non-unit-norm) intermediate state's own squared norm.
    This convention is used consistently throughout the script and every
    other script in this package; Section 6.4 of the paper verifies that
    it agrees with the raw (unnormalized) quadratic form to within
    1e-5--1e-6 for the configurations tested."""
    for Zq in Z_ops:
        norm_sq = np.real(np.conjugate(psi) @ psi)
        mu = float(np.real(np.conjugate(psi) @ (Zq @ psi))) / norm_sq
        dW = rng.normal(0.0, np.sqrt(dt))
        diff_op = Zq - mu * np.eye(dim)
        psi = psi + np.sqrt(lam) * (diff_op @ psi) * dW - 0.5 * lam * (diff_op @ diff_op @ psi) * dt
    return psi / np.linalg.norm(psi)


def run_ensemble_rho(psi0, dt, T, M, seed):
    rng = np.random.default_rng(seed)
    n_steps = int(round(T / dt))
    psi_final = np.zeros((M, dim), dtype=complex)
    for k in range(M):
        psi = psi0.copy()
        for _ in range(n_steps):
            psi = csl_step(psi, dt, rng)
        psi_final[k] = psi
    rho = np.einsum('ki,kj->ij', psi_final, psi_final.conj()) / M
    return rho


def ket(bits):
    idx = int(bits, 2)
    v = np.zeros(dim, dtype=complex)
    v[idx] = 1.0
    return v


# five structurally different initial states
states = {}

# (a) computational basis state
states["computational_basis_|000>"] = ket("000")

# (b) single-qubit |+> on qubit 0, others |0>: (|000> + |100>)/sqrt(2)
states["single_qubit_plus_|+00>"] = (ket("000") + ket("100")) / np.sqrt(2)

# (c) product state with different local coherences on two qubits
alpha, beta = 1 / np.sqrt(2), 1 / np.sqrt(2)          # qubit 0: |+>
gamma, delta = np.sqrt(0.8), 1j * np.sqrt(0.2)         # qubit 1: unequal, complex-phase superposition
q0 = alpha * ket("000") + beta * ket("100")
psi_product = (alpha * np.array([1, 0]).astype(complex))  # placeholder, built explicitly below
# build explicitly via kron to avoid basis-ordering mistakes
qubit0 = np.array([alpha, beta], dtype=complex)
qubit1 = np.array([gamma, delta], dtype=complex)
qubit2 = np.array([1.0, 0.0], dtype=complex)  # |0>
psi_product = np.kron(np.kron(qubit0, qubit1), qubit2)
psi_product /= np.linalg.norm(psi_product)
states["product_different_local_coherences"] = psi_product

# (d) partially entangled W-like state
states["partially_entangled_W"] = (ket("001") + ket("010") + ket("100")) / np.sqrt(3)

# (e) generic state with unequal amplitudes spanning multiple Hamming distances
c = np.array([0.5, 0.3, -0.2, 0.15, 0.1, 0.25, -0.35, 0.4], dtype=complex)
c = c / np.linalg.norm(c)
states["unequal_amplitudes_generic"] = c

M = 400
dt = 0.01
results = {}

# Explicit, fixed integer seeds are used here -- NOT hash(name). Python's
# hash() is randomized per-process by default (PYTHONHASHSEED unset), so
# seeding from it would make results non-reproducible across runs. This
# convention -- explicit integer seeds throughout -- is discussed in the
# paper's Reproducibility section (7.4).
fixed_seeds = {
    "computational_basis_|000>": 3001,
    "single_qubit_plus_|+00>": 3002,
    "product_different_local_coherences": 3003,
    "partially_entangled_W": 3004,
    "unequal_amplitudes_generic": 3005,
}

for name, psi0 in states.items():
    psi0 = psi0 / np.linalg.norm(psi0)
    rho0 = np.outer(psi0, psi0.conj())
    rho_analytic = analytic_rho(rho0, lam, T)
    rho_sim = run_ensemble_rho(psi0, dt, T, M, seed=fixed_seeds[name])

    max_elem_err = float(np.max(np.abs(rho_sim - rho_analytic)))
    rms_elem_err = float(np.sqrt(np.mean(np.abs(rho_sim - rho_analytic) ** 2)))
    trace_sim = float(np.real(np.trace(rho_sim)))
    hermiticity_err = float(np.max(np.abs(rho_sim - rho_sim.conj().T)))
    min_eig = float(np.min(np.linalg.eigvalsh((rho_sim + rho_sim.conj().T) / 2)))

    # how many distinct Hamming distances (decay rates) are actually exercised
    # by this state's nonzero coherences
    nz = np.abs(rho0) > 1e-12
    dHs = set()
    for a in range(dim):
        for b in range(dim):
            if a != b and nz[a, b]:
                dHs.add(hamming(a, b))

    results[name] = {
        "max_elem_err": max_elem_err, "rms_elem_err": rms_elem_err,
        "trace_sim": trace_sim, "hermiticity_err": hermiticity_err,
        "min_eig": min_eig, "distinct_hamming_distances_tested": sorted(dHs),
    }
    print(f"\n{name}:")
    print(f"  Hamming distances exercised by nonzero coherences: {sorted(dHs)}")
    print(f"  max |rho_sim - rho_analytic| = {max_elem_err:.4f}, RMS = {rms_elem_err:.4f}")
    print(f"  trace(rho_sim) = {trace_sim:.6f}, Hermiticity error = {hermiticity_err:.2e}, "
          f"min eigenvalue = {min_eig:.6f}")

with open("general_closed_form_results.json", "w") as f:
    json.dump({"ode_check_max_err": max_err, "states": results}, f, indent=2)

print("\nDone.")
