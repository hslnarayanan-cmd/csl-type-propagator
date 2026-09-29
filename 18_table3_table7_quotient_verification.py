"""
Script 18 of 23 -- Quotient-convention verification for the
closed-form GHZ validation (Table 6) and the multi-state validation
(Table 7).

Runs the closed-form GHZ validation (Table 6) and the multi-state
validation (Table 7) explicitly under the quotient (normalized)
expectation-value convention, mu_i(psi) = Re<psi|A_i|psi>/<psi|psi>,
to verify -- rather than merely assert -- that the principal validation
tables hold under the convention adopted as canonical (Section 6.4).
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

ghz = np.zeros(dim, dtype=complex)
ghz[0] = 1 / np.sqrt(2)
ghz[-1] = 1 / np.sqrt(2)

T = 1.0


def csl_step_quotient(psi, dt, lam, rng):
    """Quotient (normalized) convention: mu_i = Re<psi|A_i|psi>/<psi|psi>."""
    for Zq in Z_ops:
        norm_sq = np.real(np.conjugate(psi) @ psi)
        mu = float(np.real(np.conjugate(psi) @ (Zq @ psi))) / norm_sq
        dW = rng.normal(0.0, np.sqrt(dt))
        diff_op = Zq - mu * np.eye(dim)
        psi = psi + np.sqrt(lam) * (diff_op @ psi) * dW - 0.5 * lam * (diff_op @ diff_op @ psi) * dt
    return psi / np.linalg.norm(psi)


def csl_step_raw(psi, dt, lam, rng):
    """Raw (unnormalized quadratic form) convention, for direct comparison."""
    for Zq in Z_ops:
        mu = float(np.real(np.conjugate(psi) @ (Zq @ psi)))
        dW = rng.normal(0.0, np.sqrt(dt))
        diff_op = Zq - mu * np.eye(dim)
        psi = psi + np.sqrt(lam) * (diff_op @ psi) * dW - 0.5 * lam * (diff_op @ diff_op @ psi) * dt
    return psi / np.linalg.norm(psi)


def von_neumann_entropy(rho):
    evals = np.linalg.eigvalsh((rho + rho.conj().T) / 2)
    evals = evals[evals > 1e-12]
    return float(-np.sum(evals * np.log2(evals)))


def run_ensemble(psi0, dt, T, M, lam, seed, use_raw=False):
    rng = np.random.default_rng(seed)
    n_steps = int(round(T / dt))
    step_fn = csl_step_raw if use_raw else csl_step_quotient
    psi_all = np.zeros((M, dim), dtype=complex)
    for k in range(M):
        psi = psi0.copy()
        for _ in range(n_steps):
            psi = step_fn(psi, dt, lam, rng)
        psi_all[k] = psi
    rho = np.einsum('ki,kj->ij', psi_all, psi_all.conj()) / M
    fid = np.abs(np.einsum('ki,i->k', psi_all, psi0.conj())) ** 2
    return rho, fid


# ===========================================================================
# Table 3: closed-form GHZ validation, quotient convention
# ===========================================================================
print("=" * 70)
print("Table 3 verification (quotient convention): GHZ, H=0, Z-collapse")
print("=" * 70)
dt, M = 0.01, 400
results_table3 = {}
for i, lam in enumerate([0.00, 0.05, 0.20, 0.50]):
    seed = 1000 + i
    rho_q, fid_q = run_ensemble(ghz, dt, T, M, lam, seed=seed, use_raw=False)
    rho_r, fid_r = run_ensemble(ghz, dt, T, M, lam, seed=seed, use_raw=True)
    F_sim = float(fid_q.mean())
    F_sem = float(fid_q.std(ddof=1) / np.sqrt(M))
    F_sim_raw = float(fid_r.mean())
    S_sim = von_neumann_entropy(rho_q)
    F_analytic = 0.5 + 0.5 * np.exp(-6 * lam * T)
    S_analytic = (-F_analytic * np.log2(F_analytic) - (1 - F_analytic) * np.log2(1 - F_analytic)
                  if 0 < F_analytic < 1 else 0.0)
    results_table3[str(lam)] = {"F_sim_quotient": F_sim, "F_sim_raw": F_sim_raw, "F_sem": F_sem,
                                 "S_sim": S_sim, "F_analytic": F_analytic, "S_analytic": S_analytic,
                                 "raw_vs_quotient_diff": abs(F_sim - F_sim_raw)}
    print(f"lambda={lam}: F_quotient={F_sim:.4f}+/-{F_sem:.4f}, F_raw={F_sim_raw:.4f} "
          f"(analytic {F_analytic:.4f}), raw-vs-quotient diff={abs(F_sim-F_sim_raw):.2e}, "
          f"S_sim={S_sim:.4f} (analytic {S_analytic:.4f})")

with open("table3_quotient_verification.json", "w") as f:
    json.dump(results_table3, f, indent=2)

# ===========================================================================
# Table 7: multi-state validation, quotient convention
# ===========================================================================
print("\n" + "=" * 70)
print("Table 7 verification (quotient convention): five initial states")
print("=" * 70)


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

# Explicit, fixed integer seeds are used here -- NOT hash(name). Python's
# hash() is randomized per-process by default (PYTHONHASHSEED unset), so
# seeding from it would make results non-reproducible across runs and
# would confound any raw-vs-normalized comparison with seed variation
# rather than the convention itself. Explicit integer seeds keep the
# raw-vs-quotient comparison isolated to the convention alone.
fixed_seeds = {
    "computational_basis_|000>": 3001,
    "single_qubit_plus_|+00>": 3002,
    "product_different_local_coherences": 3003,
    "partially_entangled_W": 3004,
    "unequal_amplitudes_generic": 3005,
}

lam_multi = 0.20
results_table7 = {}
for name, psi0 in states.items():
    psi0 = psi0 / np.linalg.norm(psi0)
    rho0 = np.outer(psi0, psi0.conj())
    rho_analytic = analytic_rho(rho0, lam_multi, T)
    seed = fixed_seeds[name]
    rho_sim_q, _ = run_ensemble(psi0, dt, T, M, lam_multi, seed=seed, use_raw=False)
    rho_sim_r, _ = run_ensemble(psi0, dt, T, M, lam_multi, seed=seed, use_raw=True)
    max_err_q = float(np.max(np.abs(rho_sim_q - rho_analytic)))
    max_err_r = float(np.max(np.abs(rho_sim_r - rho_analytic)))
    raw_vs_quotient = float(np.max(np.abs(rho_sim_q - rho_sim_r)))
    rms_err_q = float(np.sqrt(np.mean(np.abs(rho_sim_q - rho_analytic) ** 2)))
    trace_sim_q = float(np.real(np.trace(rho_sim_q)))
    results_table7[name] = {"max_err_quotient": max_err_q, "max_err_raw": max_err_r,
                             "raw_vs_quotient_maxdiff": raw_vs_quotient,
                             "rms_err_quotient": rms_err_q, "trace_sim_quotient": trace_sim_q,
                             "seed": seed}
    print(f"{name} (seed={seed}): max_err_quotient={max_err_q:.4f}, max_err_raw={max_err_r:.4f}, "
          f"raw-vs-quotient direct diff={raw_vs_quotient:.2e}, trace={trace_sim_q:.6f}")

with open("table7_quotient_verification.json", "w") as f:
    json.dump(results_table7, f, indent=2)

print("\nDone.")
