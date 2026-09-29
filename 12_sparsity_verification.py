"""
Script 12 of 12 -- Sparsity verification for the complexity analysis
(paper Section 11.1).

Verifies numerically that any single-qubit Pauli operator (X, Y, or Z),
embedded into an n-qubit register via tensor product with identities,
has exactly 2^n nonzero entries -- independent of which Pauli it is or
whether it is diagonal -- and that a sum of n such operators (e.g. a
transverse-field Hamiltonian) has exactly n*2^n nonzero entries.

Run:
    python3 12_sparsity_verification.py
"""

import numpy as np

I2 = np.eye(2, dtype=complex)
X = np.array([[0, 1], [1, 0]], dtype=complex)
Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
Z = np.array([[1, 0], [0, -1]], dtype=complex)


def embed(op, q, n):
    mats = [I2] * n
    mats[q] = op
    out = mats[0]
    for m in mats[1:]:
        out = np.kron(out, m)
    return out


if __name__ == "__main__":
    print("Single-qubit Pauli operators embedded in an n-qubit register:")
    print(f"{'n':>3} {'Pauli':>6} {'nnz':>8} {'2^n':>8} {'total':>10} {'density':>10}")
    for n in [3, 4, 5, 6]:
        for name, op in [("X", X), ("Y", Y), ("Z", Z)]:
            M = embed(op, 0, n)
            nnz = np.count_nonzero(np.abs(M) > 1e-12)
            total = M.shape[0] * M.shape[1]
            print(f"{n:>3} {name:>6} {nnz:>8} {2**n:>8} {total:>10} {nnz/total:>10.6f}")

    print("\nSum of n single-qubit X operators (transverse-field Hamiltonian):")
    print(f"{'n':>3} {'nnz':>8} {'n*2^n':>8} {'dim^2':>10} {'nnz/row':>10}")
    for n in [3, 4, 5, 6, 8]:
        H = sum(embed(X, i, n) for i in range(n))
        nnz = np.count_nonzero(np.abs(H) > 1e-12)
        dim = 2 ** n
        print(f"{n:>3} {nnz:>8} {n*2**n:>8} {dim*dim:>10} {nnz/dim:>10.1f}")

    print("\nMemory cost by output mode (concrete numbers, M=400, T=100):")
    print(f"{'n':>3} {'dim':>6} {'state_vec':>12} {'ensemble_rho':>14} {'full_traj_hist':>16}")
    M, T = 400, 100
    for n in [10, 12, 13, 14]:
        dim = 2 ** n
        state_vec_bytes = dim * 16
        ensemble_bytes = dim * dim * 16
        full_traj_bytes = M * T * dim * 16

        def fmt(b):
            for unit in ["B", "KB", "MB", "GB", "TB"]:
                if b < 1024:
                    return f"{b:.2f}{unit}"
                b /= 1024
            return f"{b:.2f}PB"

        print(f"{n:>3} {dim:>6} {fmt(state_vec_bytes):>12} {fmt(ensemble_bytes):>14} "
              f"{fmt(full_traj_bytes):>16}")
