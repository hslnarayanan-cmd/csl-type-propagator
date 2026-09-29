"""
Operator-splitting error quantification (paper Section 6, response to
Section 6.3).

Algorithm 1 is a Lie (first-order) splitting scheme: within one timestep,
it applies the full unitary sub-step U=exp(-iHdt), then processes the n
collapse operators sequentially, each using the CURRENT (already-updated)
state to compute <A_i>. Two distinct sources of ordering-dependence are
tested here:

  (A) Ordering among the n collapse operators. The A_i (Pauli-Z on
      different qubits) commute exactly as OPERATORS, but the discrete
      algorithm evaluates <A_i> from the partially-updated state after
      processing earlier operators in the sequence, so different
      processing orders are not guaranteed to agree exactly at finite dt.
      We measure how much they actually differ, and how that difference
      scales with dt.

  (B) Splitting between the unitary (Hamiltonian) sub-step and the
      collapse sub-step. Algorithm 1 applies U first, then the collapse
      operators (Lie splitting). We compare this against a symmetric
      (Strang) splitting -- half of U, then the collapse operators, then
      the other half of U -- which is the standard way to check whether
      a first-order split is contributing non-negligibly to the total
      error, since Strang splitting has a higher-order local truncation
      error and the two should converge to each other as dt -> 0 if
      splitting error is not the dominant error source.

Both tests use IDENTICAL noise realizations across the compared variants,
so any measured difference is attributable to the splitting/ordering
convention itself, not to sampling noise.
"""

import numpy as np
from scipy.linalg import expm
import json

n_qubits = 3
dim = 2 ** n_qubits
I2 = np.eye(2, dtype=complex)
Z = np.array([[1, 0], [0, -1]], dtype=complex)
X = np.array([[0, 1], [1, 0]], dtype=complex)


def single_qubit_op(op, q, n):
    mats = [I2] * n
    mats[q] = op
    out = mats[0]
    for m in mats[1:]:
        out = np.kron(out, m)
    return out


Z_ops = [single_qubit_op(Z, q, n_qubits) for q in range(n_qubits)]
X_ops = [single_qubit_op(X, q, n_qubits) for q in range(n_qubits)]

# check exact commutativity of the collapse operators, as claimed
max_comm = 0.0
for i in range(n_qubits):
    for j in range(n_qubits):
        comm = Z_ops[i] @ Z_ops[j] - Z_ops[j] @ Z_ops[i]
        max_comm = max(max_comm, np.max(np.abs(comm)))
print(f"Max |[A_i, A_j]| across all pairs (should be exactly 0): {max_comm:.2e}")

ghz = np.zeros(dim, dtype=complex)
ghz[0] = 1 / np.sqrt(2)
ghz[-1] = 1 / np.sqrt(2)

omega = 1.0
lam = 0.20
H = omega * sum(X_ops)


def collapse_substep(psi, order, dt, dWs):
    """Apply the n collapse-operator updates sequentially in the given
    order, each using <A_i> computed from the current (updated) state."""
    for i in order:
        Zq = Z_ops[i]
        exp_z = float(np.real(np.conjugate(psi) @ (Zq @ psi)))
        diff_op = Zq - exp_z * np.eye(dim)
        psi = psi + np.sqrt(lam) * (diff_op @ psi) * dWs[i] - 0.5 * lam * (diff_op @ diff_op @ psi) * dt
    return psi


def run_lie(psi0, dt, T, dW_all, order):
    """Lie splitting: full U, then collapse operators in `order`."""
    n_steps = int(round(T / dt))
    U = expm(-1j * H * dt)
    psi = psi0.copy()
    for t_idx in range(n_steps):
        psi = U @ psi
        psi = collapse_substep(psi, order, dt, dW_all[t_idx])
        psi = psi / np.linalg.norm(psi)
    return psi


def run_strang(psi0, dt, T, dW_all, order):
    """Strang splitting: half U, collapse operators, half U."""
    n_steps = int(round(T / dt))
    U_half = expm(-1j * H * dt / 2)
    psi = psi0.copy()
    for t_idx in range(n_steps):
        psi = U_half @ psi
        psi = collapse_substep(psi, order, dt, dW_all[t_idx])
        psi = U_half @ psi
        psi = psi / np.linalg.norm(psi)
    return psi


if __name__ == "__main__":
    T = 1.0
    M = 300
    master_seed = 4242

    dts = [0.04, 0.02, 0.01, 0.005, 0.0025]

    print("\n" + "=" * 70)
    print("(A) Collapse-operator ordering: forward (0,1,2) vs reversed (2,1,0)")
    print("=" * 70)
    order_results = {}
    for dt in dts:
        n_steps = int(round(T / dt))
        rng = np.random.default_rng(master_seed)
        diffs = []
        for k in range(M):
            dW_all = rng.normal(0.0, np.sqrt(dt), size=(n_steps, n_qubits))
            psi_fwd = run_lie(ghz, dt, T, dW_all, order=[0, 1, 2])
            psi_rev = run_lie(ghz, dt, T, dW_all, order=[2, 1, 0])
            F_fwd = np.abs(np.vdot(ghz, psi_fwd)) ** 2
            F_rev = np.abs(np.vdot(ghz, psi_rev)) ** 2
            diffs.append(abs(F_fwd - F_rev))
        rms = float(np.sqrt(np.mean(np.array(diffs) ** 2)))
        order_results[str(dt)] = rms
        print(f"  dt={dt:<7} RMS |F_fwd - F_rev| over {M} paths = {rms:.6f}")

    dts_arr = np.array(dts)
    rms_arr = np.array([order_results[str(dt)] for dt in dts])
    # guard against exact zeros before logging
    if np.all(rms_arr > 0):
        p_order, _ = np.polyfit(np.log(dts_arr), np.log(rms_arr), 1)
        print(f"\n  Ordering-difference scaling: RMS ~ dt^{p_order:.2f}")
    else:
        p_order = None
        print("\n  Some orderings produced zero difference at working precision.")

    print("\n" + "=" * 70)
    print("(B) Splitting scheme: Lie (current Algorithm 1) vs Strang")
    print("=" * 70)
    split_results = {}
    for dt in dts:
        n_steps = int(round(T / dt))
        rng = np.random.default_rng(master_seed + 1)
        diffs = []
        for k in range(M):
            dW_all = rng.normal(0.0, np.sqrt(dt), size=(n_steps, n_qubits))
            psi_lie = run_lie(ghz, dt, T, dW_all, order=[0, 1, 2])
            psi_strang = run_strang(ghz, dt, T, dW_all, order=[0, 1, 2])
            F_lie = np.abs(np.vdot(ghz, psi_lie)) ** 2
            F_strang = np.abs(np.vdot(ghz, psi_strang)) ** 2
            diffs.append(abs(F_lie - F_strang))
        rms = float(np.sqrt(np.mean(np.array(diffs) ** 2)))
        split_results[str(dt)] = rms
        print(f"  dt={dt:<7} RMS |F_Lie - F_Strang| over {M} paths = {rms:.6f}")

    rms_arr2 = np.array([split_results[str(dt)] for dt in dts])
    p_split, _ = np.polyfit(np.log(dts_arr), np.log(rms_arr2), 1)
    print(f"\n  Lie-vs-Strang difference scaling: RMS ~ dt^{p_split:.2f}")

    out = {
        "max_commutator": max_comm,
        "ordering_rms_vs_dt": order_results,
        "ordering_fit_p": p_order,
        "splitting_rms_vs_dt": split_results,
        "splitting_fit_p": float(p_split),
        "M": M,
        "master_seed": master_seed,
    }
    with open("splitting_error.json", "w") as f:
        json.dump(out, f, indent=2)
