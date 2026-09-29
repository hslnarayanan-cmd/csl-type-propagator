"""
Script 4 of 4 --- Generates all figures used in the paper from the JSON
outputs of scripts 1 and 2.

Run this after running:
    python3 1_csl_propagator.py             (produces results.json, convergence.json)
    python3 2_noncommuting_and_scaling.py    (produces noncommuting_validation.json,
                                               scaling_comparison.json)

Note: Figure 5 (the QuTiP verification plot) is produced directly by
script 3 (3_qutip_verification.py), which saves it as
qutip_validation_result.png. It is not regenerated here.

Run:
    python3 4_make_figures.py

Outputs:
    fig1_fidelity.png
    fig2_entropy.png
    fig3_convergence.png
    fig4_noncommuting_validation.png
    fig6_scaling.png
"""

import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ===========================================================================
# Figures 1-2: closed-form validation (Section 4)
# ===========================================================================
with open("results.json") as f:
    results = json.load(f)
with open("convergence.json") as f:
    conv = json.load(f)

lambdas = [0.00, 0.05, 0.20, 0.50]
colors = ["#4C4C4C", "#2E7D32", "#1565C0", "#C62828"]


def analytic_fidelity(lam, t):
    return 0.5 + 0.5 * np.exp(-6 * lam * t)


def analytic_entropy(F):
    F = np.clip(F, 1e-12, 1 - 1e-12)
    return -F * np.log2(F) - (1 - F) * np.log2(1 - F)


fig, ax = plt.subplots(figsize=(6.2, 4.4))
for lam, c in zip(lambdas, colors):
    r = results[str(lam)]
    t = np.array(r["times"])
    fm = np.array(r["fidelity_mean"])
    sem = np.array(r["fidelity_sem"])
    ax.plot(t, fm, color=c, lw=1.8, label=f"$\\lambda={lam:.2f}$ (simulated)")
    ax.fill_between(t, fm - sem, fm + sem, color=c, alpha=0.2, linewidth=0)
    ax.plot(t, analytic_fidelity(lam, t), color=c, lw=1.0, ls="--", alpha=0.8)
ax.plot([], [], color="black", lw=1.0, ls="--", label="Analytic (closed-form)")
ax.set_xlabel("Time $t$")
ax.set_ylabel(r"Ensemble fidelity $F(\bar{\rho}_t, |\mathrm{GHZ}\rangle)$")
ax.set_title("CSL-driven fidelity decay of a 3-qubit GHZ register")
ax.set_ylim(0.45, 1.02)
ax.legend(fontsize=8, loc="lower left")
ax.grid(alpha=0.25)
fig.tight_layout()
fig.savefig("fig1_fidelity.png", dpi=220)
plt.close(fig)

fig, ax = plt.subplots(figsize=(6.2, 4.4))
for lam, c in zip(lambdas, colors):
    r = results[str(lam)]
    t = np.array(r["times"])
    ent = np.array(r["entropy"])
    lo = np.array(r["entropy_lo"])
    hi = np.array(r["entropy_hi"])
    ax.plot(t, ent, color=c, lw=1.8, label=f"$\\lambda={lam:.2f}$ (simulated)")
    ax.fill_between(t, lo, hi, color=c, alpha=0.18, linewidth=0)
    F_analytic = analytic_fidelity(lam, t)
    ax.plot(t, analytic_entropy(F_analytic), color=c, lw=1.0, ls="--", alpha=0.8)
ax.plot([], [], color="black", lw=1.0, ls="--", label="Analytic (closed-form)")
ax.axhline(1.0, color="gray", lw=0.8, ls=":")
ax.set_xlabel("Time $t$")
ax.set_ylabel(r"von Neumann entropy $S(\bar{\rho}_t)$ [bits]")
ax.set_title("Entropy growth of a 3-qubit GHZ register under CSL collapse")
ax.set_ylim(-0.03, 1.08)
ax.legend(fontsize=8, loc="lower right")
ax.grid(alpha=0.25)
fig.tight_layout()
fig.savefig("fig2_entropy.png", dpi=220)
plt.close(fig)

# ===========================================================================
# Figure 3: step-size sensitivity (Section 4.3)
# ===========================================================================
dts = sorted([float(k) for k in conv.keys()])
fid_vals = [conv[str(dt)]["final_fidelity"] for dt in dts]
fid_sems = [conv[str(dt)]["final_fidelity_sem"] for dt in dts]
analytic_val = analytic_fidelity(0.20, 1.0)

fig, ax = plt.subplots(figsize=(6.2, 4.2))
ax.errorbar(dts, fid_vals, yerr=fid_sems, marker="o", color="#1565C0",
            capsize=4, lw=1.5, label="Simulated final fidelity")
ax.axhline(analytic_val, color="#C62828", ls="--", lw=1.3,
           label=f"Analytic value ({analytic_val:.4f})")
ax.set_xscale("log")
ax.set_xlabel("Integration step size $dt$")
ax.set_ylabel(r"Final fidelity $F(\bar{\rho}_{t=1}, |\mathrm{GHZ}\rangle)$")
ax.set_title(r"Step-size sensitivity ($\lambda=0.20$)")
ax.legend(fontsize=8)
ax.grid(alpha=0.25, which="both")
fig.tight_layout()
fig.savefig("fig3_convergence.png", dpi=220)
plt.close(fig)

# ===========================================================================
# Figure 4: independent reference-solver validation, non-commuting case
# (Section 5.1)
# ===========================================================================
with open("noncommuting_validation.json") as f:
    nc = json.load(f)

t_mc = np.array(nc["t_mc"]); F_mc = np.array(nc["F_mc"])
F_mc_sem = np.array(nc["F_mc_sem"]); S_mc = np.array(nc["S_mc"])
t_ref = np.array(nc["t_ref"]); F_ref = np.array(nc["F_ref"]); S_ref = np.array(nc["S_ref"])

fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2))
ax = axes[0]
ax.plot(t_mc, F_mc, color="#1565C0", lw=1.8, label="CSL Monte Carlo (M=400)")
ax.fill_between(t_mc, F_mc - F_mc_sem, F_mc + F_mc_sem, color="#1565C0", alpha=0.25, linewidth=0)
ax.plot(t_ref, F_ref, color="#C62828", lw=1.3, ls="--", label="Lindblad ODE reference")
ax.set_xlabel("Time $t$")
ax.set_ylabel(r"Fidelity $F(\bar{\rho}_t, |\mathrm{GHZ}\rangle)$")
ax.set_title(r"Fidelity ($H=\omega\Sigma X_i \neq 0$, $\lambda=0.20$)")
ax.legend(fontsize=8)
ax.grid(alpha=0.25)

ax = axes[1]
ax.plot(t_mc, S_mc, color="#1565C0", lw=1.8, label="CSL Monte Carlo (M=400)")
ax.plot(t_ref, S_ref, color="#C62828", lw=1.3, ls="--", label="Lindblad ODE reference")
ax.set_xlabel("Time $t$")
ax.set_ylabel(r"Entropy $S(\bar{\rho}_t)$ [bits]")
ax.set_title(r"Entropy ($H=\omega\Sigma X_i \neq 0$, $\lambda=0.20$)")
ax.legend(fontsize=8)
ax.grid(alpha=0.25)

fig.tight_layout()
fig.savefig("fig4_noncommuting_validation.png", dpi=220)
plt.close(fig)

# ===========================================================================
# Figure 6: wall-clock scaling comparison (Section 7.1)
# ===========================================================================
with open("scaling_comparison.json") as f:
    scaling = json.load(f)

ns = [r["n"] for r in scaling]
ode_t = [r["ode_time_sec"] for r in scaling]
mc_per_traj = [r["mc_time_per_trajectory_sec"] for r in scaling]
mc_ensemble_400 = [t * 400 for t in mc_per_traj]

fig, ax = plt.subplots(figsize=(6.6, 4.6))
ax.semilogy(ns, ode_t, marker="o", color="#C62828", lw=1.8, label="Deterministic Lindblad ODE (single solve)")
ax.semilogy(ns, mc_per_traj, marker="s", color="#2E7D32", lw=1.8, label="CSL Monte Carlo (single trajectory)")
ax.semilogy(ns, mc_ensemble_400, marker="^", color="#1565C0", lw=1.8, label="CSL Monte Carlo ensemble ($M{=}400$)")
ax.set_xlabel("Number of qubits $n$")
ax.set_ylabel("Wall-clock time (s, log scale)")
ax.set_title("Runtime: deterministic ODE vs. trajectory ensemble")
ax.legend(fontsize=8)
ax.grid(alpha=0.3, which="both")
fig.tight_layout()
fig.savefig("fig6_scaling.png", dpi=220)
plt.close(fig)

print("Wrote: fig1_fidelity.png, fig2_entropy.png, fig3_convergence.png,")
print("       fig4_noncommuting_validation.png, fig6_scaling.png")
print("(Figure 5, the QuTiP verification plot, is produced by script 3.)")
