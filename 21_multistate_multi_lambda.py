"""
Script 21 of 23 -- Multi-state closed-form validation at two lambda
values (paper Table 7, Section VII-D).

Validates the general closed-form solution against five structurally
different initial states at two collapse rates, lambda=0.20 and
lambda=0.50 (the same "strong collapse" value used in the closed-form
GHZ sweep), so the analytical check spans more than one point in
parameter space. Quotient (normalized) expectation convention
throughout, fixed explicit seeds per state and lambda value.
"""

import numpy as np
import json

n_qubits = 3
dim = 2 ** n_qubits
I2 = np.eye(2, dtype=complex)
Z = np.array([[1, 0], [0, -1]], dtype=complex)


def embed(op, q, n):
    mats = [I2] * n
    mats[q] = op
    out = mats[0]
    for m in mats[1:]:
        out = np.kron(out, m)
    return out


Z_ops = [embed(Z, q, n_qubits) for q in range(n_qubits)]


def csl_step(psi, dt, lam, rng):
    for Zq in Z_ops:
        norm_sq = np.real(np.conjugate(psi) @ psi)
        mu = float(np.real(np.conjugate(psi) @ (Zq @ psi))) / norm_sq
        dW = rng.normal(0.0, np.sqrt(dt))
        diff_op = Zq - mu * np.eye(dim)
        psi = psi + np.sqrt(lam) * (diff_op @ psi) * dW - 0.5 * lam * (diff_op @ diff_op @ psi) * dt
    return psi / np.linalg.norm(psi)


def run_ensemble_rho(psi0, dt, T, M, lam, seed):
    rng = np.random.default_rng(seed)
    n_steps = int(round(T / dt))
    psi_all = np.zeros((M, dim), dtype=complex)
    for k in range(M):
        psi = psi0.copy()
        for _ in range(n_steps):
            psi = csl_step(psi, dt, lam, rng)
        psi_all[k] = psi
    rho = np.einsum('ki,kj->ij', psi_all, psi_all.conj()) / M
    return rho


def hamming(a, b):
    return bin(a ^ b).count("1")


def analytic_rho(rho0, lam, t):
    rho_t = np.zeros_like(rho0)
    for a in range(dim):
        for b in range(dim):
            rho_t[a, b] = rho0[a, b] * np.exp(-2 * lam * hamming(a, b) * t)
    return rho_t


def ket(bits):
    idx = int(bits, 2)
    v = np.zeros(dim, dtype=complex)
    v[idx] = 1.0
    return v


states = {}
states["computational_basis_|000>"] = ket("000")
states["single_qubit_plus_|+00>"] = (ket("000") + ket("100")) / np.sqrt(2)
alpha, beta = 1 / np.sqrt(2), 1 / np.sqrt(2)
gamma, delta = np.sqrt(0.8), 1j * np.sqrt(0.2)
qubit0 = np.array([alpha, beta], dtype=complex)
qubit1 = np.array([gamma, delta], dtype=complex)
qubit2 = np.array([1.0, 0.0], dtype=complex)
psi_product = np.kron(np.kron(qubit0, qubit1), qubit2)
psi_product /= np.linalg.norm(psi_product)
states["product_different_local_coherences"] = psi_product
states["partially_entangled_W"] = (ket("001") + ket("010") + ket("100")) / np.sqrt(3)
c = np.array([0.5, 0.3, -0.2, 0.15, 0.1, 0.25, -0.35, 0.4], dtype=complex)
c = c / np.linalg.norm(c)
states["unequal_amplitudes_generic"] = c

# Fixed, explicit seeds, one per state, distinct from the lambda=0.20 run.
fixed_seeds = {
    "computational_basis_|000>": 4001,
    "single_qubit_plus_|+00>": 4002,
    "product_different_local_coherences": 4003,
    "partially_entangled_W": 4004,
    "unequal_amplitudes_generic": 4005,
}

T, M, dt = 1.0, 400, 0.01
lam = 0.50

print(f"Multi-state closed-form validation at lambda={lam}\n")
results = {}
for name, psi0 in states.items():
    psi0 = psi0 / np.linalg.norm(psi0)
    rho0 = np.outer(psi0, psi0.conj())
    rho_analytic = analytic_rho(rho0, lam, T)
    rho_sim = run_ensemble_rho(psi0, dt, T, M, lam, seed=fixed_seeds[name])
    max_err = float(np.max(np.abs(rho_sim - rho_analytic)))
    rms_err = float(np.sqrt(np.mean(np.abs(rho_sim - rho_analytic) ** 2)))
    trace_sim = float(np.real(np.trace(rho_sim)))
    herm_err = float(np.max(np.abs(rho_sim - rho_sim.conj().T)))
    min_eig = float(np.min(np.linalg.eigvalsh((rho_sim + rho_sim.conj().T) / 2)))
    results[name] = {"max_err": max_err, "rms_err": rms_err, "trace": trace_sim,
                      "herm_err": herm_err, "min_eig": min_eig, "seed": fixed_seeds[name]}
    print(f"{name} (seed={fixed_seeds[name]}): max_err={max_err:.4f}, rms_err={rms_err:.4f}, "
          f"trace={trace_sim:.6f}, herm_err={herm_err:.2e}, min_eig={min_eig:.6f}")

with open("multistate_lambda050.json", "w") as f:
    json.dump(results, f, indent=2)
print("\nWrote multistate_lambda050.json")
