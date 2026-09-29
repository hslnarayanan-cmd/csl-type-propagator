"""
Script 22 of 23 -- Structurally different non-commuting Hamiltonian
(paper Section IX-C).

Non-commuting validation with a structurally different Hamiltonian:
H = omega*sum(X_i) + J*sum(Z_i Z_{i+1}), a transverse field PLUS
nearest-neighbor Ising-type ZZ coupling, rather than the uniform
transverse field alone (H = omega*sum(X_i)) used in the paper's main
non-commuting section. The coupling term makes this a genuinely
different Hamiltonian structure: it introduces two-body qubit-qubit
interactions, not just single-qubit terms, and [H, Z_i] != 0 for a
different reason than the transverse field alone (the ZZ term commutes
with each individual Z_i, but the X term does not, and the two terms
do not commute with each other either).

Uses the same independent SciPy reference solver as the paper's
main non-commuting section, and the same quotient-convention propagator.
"""

import numpy as np
from scipy.integrate import solve_ivp

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


def embed2(op1, op2, q1, q2, n):
    mats = [I2] * n
    mats[q1] = op1
    mats[q2] = op2
    out = mats[0]
    for m in mats[1:]:
        out = np.kron(out, m)
    return out


X_ops = [embed(X, q, n_qubits) for q in range(n_qubits)]
Z_ops = [embed(Z, q, n_qubits) for q in range(n_qubits)]

ghz = np.zeros(dim, dtype=complex)
ghz[0] = 1 / np.sqrt(2)
ghz[-1] = 1 / np.sqrt(2)

T, lam, dt, M = 1.0, 0.20, 0.01, 400
omega, J = 1.0, 0.5

# H = omega * sum(X_i) + J * sum_{i} Z_i Z_{i+1} (open chain, nearest-neighbor)
H = omega * sum(X_ops)
ZZ_terms = [embed2(Z, Z, i, i + 1, n_qubits) for i in range(n_qubits - 1)]
H = H + J * sum(ZZ_terms)

print(f"H = omega*sum(X_i) + J*sum(Z_i Z_i+1), omega={omega}, J={J}")
print(f"Commutator check [H, Z_0] nonzero: {not np.allclose(H @ Z_ops[0] - Z_ops[0] @ H, 0)}")
print(f"H is Hermitian: {np.allclose(H, H.conj().T)}\n")


def lindblad_rhs(t, rho_flat, H, A_ops, lam):
    rho = rho_flat.reshape(dim, dim)
    drho = -1j * (H @ rho - rho @ H)
    for Aq in A_ops:
        drho += lam * (Aq @ rho @ Aq - 0.5 * (Aq @ Aq @ rho + rho @ Aq @ Aq))
    return drho.flatten()


rho0 = np.outer(ghz, ghz.conj())
sol = solve_ivp(lindblad_rhs, [0, T], rho0.flatten(), args=(H, Z_ops, lam),
                 t_eval=np.linspace(0, T, 101), method="RK45", rtol=1e-9, atol=1e-12)
rho_ref_traj = [sol.y[:, k].reshape(dim, dim) for k in range(101)]
rho_ref_final = rho_ref_traj[-1]
F_ref = float(np.real(np.vdot(ghz, rho_ref_final @ ghz)))

# Reference solver sanity checks
trace_dev = max(abs(np.real(np.trace(r)) - 1.0) for r in rho_ref_traj)
herm_err = max(np.max(np.abs(r - r.conj().T)) for r in rho_ref_traj)
print(f"SciPy reference: F_ref(t=1) = {F_ref:.4f}")
print(f"Reference solver trace deviation: {trace_dev:.2e}, Hermiticity error: {herm_err:.2e}\n")


def csl_step_quotient(psi, H_prop, A_ops, lam, dt, rng):
    psi = H_prop @ psi
    for Aq in A_ops:
        norm_sq = np.real(np.conjugate(psi) @ psi)
        mu = float(np.real(np.conjugate(psi) @ (Aq @ psi))) / norm_sq
        dW = rng.normal(0.0, np.sqrt(dt))
        diff_op = Aq - mu * np.eye(dim)
        psi = psi + np.sqrt(lam) * (diff_op @ psi) * dW - 0.5 * lam * (diff_op @ diff_op @ psi) * dt
    return psi / np.linalg.norm(psi)


from scipy.linalg import expm
H_prop = expm(-1j * H * dt)
n_steps = int(round(T / dt))
seed = 2718
rng = np.random.default_rng(seed)
psi_all = np.zeros((M, dim), dtype=complex)
min_eig_over_time = []
for k in range(M):
    psi = ghz.copy()
    for _ in range(n_steps):
        psi = csl_step_quotient(psi, H_prop, Z_ops, lam, dt, rng)
    psi_all[k] = psi
rho_sim = np.einsum('ki,kj->ij', psi_all, psi_all.conj()) / M
fid = np.abs(np.einsum('ki,i->k', psi_all, ghz.conj())) ** 2
F_sim = fid.mean()
sem = fid.std(ddof=1) / np.sqrt(M)
sigma = abs(F_sim - F_ref) / sem
min_eig = float(np.min(np.linalg.eigvalsh((rho_sim + rho_sim.conj().T) / 2)))
trace_sim = float(np.real(np.trace(rho_sim)))
frob_dist = float(np.linalg.norm(rho_sim - rho_ref_final))

print(f"Algorithm 1 (seed={seed}, M={M}): F_sim = {F_sim:.4f} +/- {sem:.4f}, sigma = {sigma:.2f}")
print(f"Simulated ensemble: trace = {trace_sim:.6f}, min eigenvalue = {min_eig:.6f}")
print(f"Frobenius distance (sim vs ref, t=1): {frob_dist:.4f}")

import json
with open("hamiltonian_zz_coupling_results.json", "w") as f:
    json.dump({
        "omega": omega, "J": J, "F_ref": F_ref, "F_sim": F_sim, "sem": sem, "sigma": sigma,
        "trace_sim": trace_sim, "min_eig": min_eig, "frob_dist": frob_dist,
        "ref_trace_dev": trace_dev, "ref_herm_err": herm_err, "seed": seed,
    }, f, indent=2)
print("\nWrote hamiltonian_zz_coupling_results.json")
