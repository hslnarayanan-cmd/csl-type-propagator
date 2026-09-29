# Supplementary Code

Code accompanying "A Finite-Dimensional Stochastic Propagator with
CSL-Type Collapse Dynamics for Multi-Qubit Registers: Algorithm,
Finite-Step Consistency, and Numerical Validation."

23 scripts, mapped to paper sections and tables/figures:

| Script | Paper section(s) | Produces |
|---|---|---|
| `1_csl_propagator.py` | Sec. VII | Table 6 (closed-form fidelity/entropy), data for Figures 2-3 |
| `2_noncommuting_and_scaling.py` | Sec. IX-A, XII | Non-commuting reference-solver check, scaling benchmark data (Table 14) |
| `3_qutip_verification.py` | Sec. X | QuTiP `mesolve`/`smesolve` comparison (requires `qutip`) |
| `4_make_figures.py` | Figs. 2, 3, 7, 9 | Plots from scripts 1 and 2's output |
| `5_convergence_and_ablation_experiments.py` | Sec. VI-B, VIII-C, VIII-D | Normalization ablation, trajectory-count convergence, individual trajectories |
| `6_rigorous_strong_convergence.py` | Sec. VIII-A | Single-configuration convergence estimate using common random numbers; script 9 extends this to a multi-seed, multi-grid estimate |
| `7_make_ablation_figures.py` | Figs. 1, 5, 6 | Plots from script 5's output |
| `8_operator_splitting_error.py` | Sec. VI-C | Table 2 (operator-splitting/ordering error) |
| `9_hardened_convergence.py` | Sec. VIII-A | Table 8 (multi-seed, multi-grid convergence order) |
| `10_general_closed_form_multistate.py` | Sec. VII-C, VII-D | Table 7 (five-state closed-form validation), quotient convention with fixed seeds |
| `11_expanded_noncommuting_validation.py` | Sec. IX-B, IX-D, IX-E | Tables 9-10 (expanded non-commuting validation, timestep refinement) |
| `12_sparsity_verification.py` | Sec. XI-A | Sparsity structure verification for tensor-product-embedded Pauli operators |
| `13_hardened_benchmark.py` | Sec. XII | Tables 15-16 (repeated-measurement benchmark, accuracy-vs-runtime) |
| `14_independent_linear_sse.py` | Sec. X-C | Table 12 (linear-SSE cross-check, preliminary) |
| `15_test_csl_propagator.py` | Sec. XV | Automated test suite (pytest-compatible, also runnable standalone) |
| `16_normalization_diagnostic_and_scaling.py` | Sec. VI-D | Tables 3-4 (substep diagnostic, single-trajectory divergence vs. dt) |
| `17_comprehensive_norm_comparison.py` | Sec. VI-D | Table 5 (six-axis raw-vs-quotient comparison) |
| `18_table3_table7_quotient_verification.py` | Sec. VI-D, VII-D | Verifies Tables 6-7 under the quotient convention |
| `19_nondiagonal_caseB_multiseed.py` | Sec. IX-B | Table 11's multi-seed statistics for the non-diagonal case-B configuration |
| `20_convergence_order_multiseed.py` | Sec. VIII-A | Table 8's 5-seed estimate (GHZ, dt_fine=0.00125) |
| `21_multistate_multi_lambda.py` | Sec. VII-D | Table 7's validation at two collapse rates, lambda=0.20 and lambda=0.50 |
| `22_hamiltonian_zz_coupling.py` | Sec. IX-C | Non-commuting validation under a structurally different Hamiltonian (transverse field plus nearest-neighbor Z-Z coupling) |
| `23_noncommuting_convergence.py` | Sec. VIII-A | Convergence-order estimate in the non-commuting regime |

## Quick start

```bash
pip install -r requirements.txt
python3 15_test_csl_propagator.py        # sanity check: 9/9 should pass
python3 1_csl_propagator.py && python3 4_make_figures.py
```

Scripts 2, 9, 11, 13, and 20 are compute-intensive (script 2's n=8 sweep
takes minutes; script 20's 5-seed convergence run takes 1-2 minutes);
everything else runs in well under a minute. Script 3 requires QuTiP
and is meant to be run independently, ideally on separate hardware, per
the paper's discussion of verification independence (Section X).

## The quotient (normalized) expectation-value convention

Every script in this package uses the quotient convention,
`mu_i(psi) = Re<psi|A_i|psi> / <psi|psi>`, at every collapse substep
(Listing 1 in the paper). This is the correct expectation value once
the state drifts slightly off unit norm between renormalization steps,
which the raw (unnormalized) form ignores. Section VI-D checks this
directly: across every configuration tested, the two forms differ by
at most 1e-5 to 1e-6 (scripts 16-18 produce that check).

## Reproducibility

Every seed in this package is an explicit, fixed integer (or simple
arithmetic on one, such as `1000+i` or `5000+M`) -- none come from
Python's `hash()` or any other process-dependent source, so
`PYTHONHASHSEED` has no effect on reproducing any result here. Table 17
in the paper's Reproducibility section cross-references every
experiment to its script, convention, and seed(s).

Two NumPy random-number APIs appear across the scripts: the modern
`numpy.random.default_rng(seed)` (used in most scripts) and the legacy
global `numpy.random.seed(seed)` state (used in a small number of
scripts and noted in-line where it occurs). These are different
algorithms and are not bit-for-bit identical under the same nominal
seed; use the RNG API named in a given script to reproduce its exact
reported figures.

No script uses multiprocessing, threading, or GPU acceleration;
trajectories within an ensemble are generated in a sequential Python
loop throughout.

## Conventions

Qubit `i` occupies tensor-factor position `i` (0-indexed) from the left
in every Kronecker product, consistently across every script. See the
paper's Reproducibility section (Section XV) for the full statement of
this and other conventions (units, basis-state indexing) used
throughout.
