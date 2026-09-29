"""
Script 7 of 23 -- Generates the convergence/ablation figures (paper Figures
1, 4, 5, 6) from the JSON output of scripts 5 and 6.

Run after:
    python3 5_convergence_and_ablation_experiments.py
    python3 6_rigorous_strong_convergence.py

Run:
    python3 7_make_ablation_figures.py
"""

import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ===========================================================================
# Figure: rigorous strong-convergence order (common random numbers)
# ===========================================================================
with open("rigorous_convergence.json") as f:
    rconv = json.load(f)

dts = sorted([float(k) for k in rconv.keys() if not k.startswith("_")])
rms = [rconv[str(dt)]["rms_pathwise_error"] for dt in dts]
p_fit = rconv["_fit"]["p"]
log_c = rconv["_fit"]["log_intercept"]

fig, ax = plt.subplots(figsize=(6.0, 4.4))
ax.loglog(dts, rms, marker="o", color="#1565C0", lw=1.8, label="Measured RMS pathwise error")
dt_line = np.array([min(dts) * 0.8, max(dts) * 1.2])
ax.loglog(dt_line, np.exp(log_c) * dt_line ** p_fit, color="#C62828", ls="--", lw=1.3,
           label=f"Fit: error $\\propto dt^{{{p_fit:.2f}}}$")
ax.set_xlabel("Integration step size $dt$")
ax.set_ylabel("RMS pathwise error vs. fine-grid reference")
ax.set_title("Empirical strong-convergence order\n(common random numbers, $M=500$)")
ax.legend(fontsize=8)
ax.grid(alpha=0.3, which="both")
fig.tight_layout()
fig.savefig("fig_convergence_order.png", dpi=220)
plt.close(fig)

# ===========================================================================
# Figure: trajectory-count convergence
# ===========================================================================
with open("trajectory_count_convergence.json") as f:
    mconv = json.load(f)

Ms = sorted([int(k) for k in mconv.keys()])
abs_err = [mconv[str(M)]["final_abs_error"] for M in Ms]
sem = [mconv[str(M)]["final_sem"] for M in Ms]

fig, ax = plt.subplots(figsize=(6.0, 4.4))
ax.loglog(Ms, sem, marker="o", color="#1565C0", lw=1.8, label="SEM of ensemble mean")
M_line = np.array([min(Ms) * 0.8, max(Ms) * 1.2])
# reference 1/sqrt(M) line anchored at the first point
ref_c = sem[0] * np.sqrt(Ms[0])
ax.loglog(M_line, ref_c / np.sqrt(M_line), color="#C62828", ls="--", lw=1.3,
           label=r"Reference: $1/\sqrt{M}$")
ax.set_xlabel("Trajectory count $M$")
ax.set_ylabel("Standard error of the mean (fidelity)")
ax.set_title(r"Monte Carlo trajectory-count convergence ($\lambda=0.20$, $dt=0.01$)")
ax.legend(fontsize=8)
ax.grid(alpha=0.3, which="both")
fig.tight_layout()
fig.savefig("fig_trajectory_count.png", dpi=220)
plt.close(fig)

# ===========================================================================
# Figure: individual trajectories vs. ensemble mean vs. analytic
# ===========================================================================
with open("individual_trajectories.json") as f:
    indiv = json.load(f)

t = np.array(indiv["times"])
trajs = np.array(indiv["trajectories"])
mean_traj = np.array(indiv["mean"])
analytic = 0.5 + 0.5 * np.exp(-6 * 0.20 * t)

fig, ax = plt.subplots(figsize=(6.6, 4.6))
for k in range(trajs.shape[0]):
    ax.plot(t, trajs[k], color="#90A4AE", lw=0.7, alpha=0.8)
ax.plot([], [], color="#90A4AE", lw=0.7, label="Individual trajectories (M=10 shown)")
ax.plot(t, mean_traj, color="#1565C0", lw=2.2, label="Ensemble mean")
ax.plot(t, analytic, color="#C62828", lw=1.5, ls="--", label="Analytic (ensemble)")
ax.set_xlabel("Time $t$")
ax.set_ylabel(r"Fidelity $F(\psi_t, |\mathrm{GHZ}\rangle)$")
ax.set_title(r"Individual trajectory realizations ($\lambda=0.20$)")
ax.legend(fontsize=8)
ax.grid(alpha=0.25)
fig.tight_layout()
fig.savefig("fig_individual_trajectories.png", dpi=220)
plt.close(fig)

# ===========================================================================
# Figure: normalization ablation
# ===========================================================================
with open("normalization_ablation.json") as f:
    norm_data = json.load(f)

t = np.array(norm_data["times"])
analytic = np.array(norm_data["analytic"])
fid_normed = np.array(norm_data["fid_normed_mean"])
fid_unnormed = np.array(norm_data["fid_unnormed_mean"])
norm_normed = np.array(norm_data["norm_normed_mean"])
norm_unnormed = np.array(norm_data["norm_unnormed_mean"])
norm_unnormed_std = np.array(norm_data["norm_unnormed_std"])

fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2))
ax = axes[0]
ax.plot(t, fid_normed, color="#1565C0", lw=1.8, label="Renormalized (Algorithm 1)")
ax.plot(t, fid_unnormed, color="#2E7D32", lw=1.8, ls=":", label="Unrenormalized")
ax.plot(t, analytic, color="#C62828", lw=1.3, ls="--", label="Analytic")
ax.set_xlabel("Time $t$")
ax.set_ylabel("Ensemble fidelity")
ax.set_title("Fidelity: renormalized vs. unrenormalized")
ax.legend(fontsize=8)
ax.grid(alpha=0.25)

ax = axes[1]
ax.plot(t, norm_normed, color="#1565C0", lw=1.8, label="Renormalized (by construction, =1)")
ax.plot(t, norm_unnormed, color="#2E7D32", lw=1.8, label="Unrenormalized (mean)")
ax.fill_between(t, norm_unnormed - norm_unnormed_std, norm_unnormed + norm_unnormed_std,
                 color="#2E7D32", alpha=0.2, linewidth=0)
ax.axhline(1.0, color="gray", lw=0.8, ls=":")
ax.set_xlabel("Time $t$")
ax.set_ylabel(r"State norm $\Vert\psi_t\Vert$")
ax.set_title("Norm drift without explicit renormalization")
ax.legend(fontsize=8)
ax.grid(alpha=0.25)

fig.tight_layout()
fig.savefig("fig_normalization_ablation.png", dpi=220)
plt.close(fig)

print("Wrote: fig_convergence_order.png, fig_trajectory_count.png,")
print("       fig_individual_trajectories.png, fig_normalization_ablation.png")
