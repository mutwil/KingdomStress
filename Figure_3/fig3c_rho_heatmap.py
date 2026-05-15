#!/usr/bin/env python3
"""
Figure 3C: Spearman rho heatmap -- correlation between phylogenetic
distance and overlap coefficient, per stress x direction.
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from scipy.stats import spearmanr

OUT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_FILE = os.path.join(OUT_DIR, "fig3c_phylo_oc_data.csv")

STRESS_LIST = ["Cold", "Drought", "Heat", "Heavy metal", "Pathogen", "Salt"]
DIRECTIONS = ["UP", "DOWN"]

# ── Load aggregated data ──
agg = pd.read_csv(DATA_FILE)

# ── Compute rho and p-value per stress x direction ──
rho_mat = np.full((len(STRESS_LIST), len(DIRECTIONS)), np.nan)
pval_mat = np.full((len(STRESS_LIST), len(DIRECTIONS)), np.nan)
n_mat = np.full((len(STRESS_LIST), len(DIRECTIONS)), 0, dtype=int)

for si, stress in enumerate(STRESS_LIST):
    for di, direction in enumerate(DIRECTIONS):
        sub = agg[(agg["stress"] == stress) & (agg["direction"] == direction)]
        if len(sub) >= 5:
            rho, pval = spearmanr(sub["divergence_Mya"], sub["mean_oc"])
            rho_mat[si, di] = rho
            pval_mat[si, di] = pval
            n_mat[si, di] = len(sub)

# ── Print table ──
print(f"{'Stress':<14} {'Dir':<6} {'n':>5} {'rho':>7} {'p-value':>12}")
print("-" * 50)
for si, stress in enumerate(STRESS_LIST):
    for di, direction in enumerate(DIRECTIONS):
        rho = rho_mat[si, di]
        pval = pval_mat[si, di]
        n = n_mat[si, di]
        if np.isnan(rho):
            print(f"{stress:<14} {direction:<6} {n:>5}     NA           NA")
        else:
            print(f"{stress:<14} {direction:<6} {n:>5} {rho:>7.3f} {pval:>12.2e}")

# Save table
rows = []
for si, stress in enumerate(STRESS_LIST):
    for di, direction in enumerate(DIRECTIONS):
        rows.append({
            "stress": stress,
            "direction": direction,
            "n": n_mat[si, di],
            "spearman_rho": rho_mat[si, di],
            "p_value": pval_mat[si, di],
        })
pd.DataFrame(rows).to_csv(os.path.join(OUT_DIR, "fig3c_rho_table.csv"), index=False)

# ── Plot heatmap ──
def sig_stars(p):
    if np.isnan(p): return ""
    if p < 0.001: return "***"
    if p < 0.01: return "**"
    if p < 0.05: return "*"
    return "ns"

fig, ax = plt.subplots(figsize=(4, 5))

# Diverging colormap centered at 0 (all values are negative here)
vmax = 0
vmin = np.nanmin(rho_mat) - 0.02
# Use a colormap from red (strong negative) to white (zero)
cmap = mcolors.LinearSegmentedColormap.from_list(
    "neg_rho", ["#b2182b", "#d6604d", "#f4a582", "#fddbc7", "#f7f7f7"]
)

# Draw cells
for si in range(len(STRESS_LIST)):
    for di in range(len(DIRECTIONS)):
        rho = rho_mat[si, di]
        pval = pval_mat[si, di]
        n = n_mat[si, di]
        if np.isnan(rho):
            ax.add_patch(plt.Rectangle(
                (di - 0.5, si - 0.5), 1, 1,
                facecolor="#e0e0e0", edgecolor="white", linewidth=1.5
            ))
            ax.text(di, si, "NA", ha="center", va="center",
                    fontsize=9, color="#999999", fontstyle="italic")
        else:
            norm_v = (rho - vmin) / (vmax - vmin)
            color = cmap(norm_v)
            ax.add_patch(plt.Rectangle(
                (di - 0.5, si - 0.5), 1, 1,
                facecolor=color, edgecolor="white", linewidth=1.5
            ))
            stars = sig_stars(pval)
            ax.text(di, si - 0.12, f"{rho:.2f}", ha="center", va="center",
                    fontsize=11, color="black")
            if stars:
                ax.text(di, si + 0.2, stars, ha="center", va="center",
                        fontsize=9, fontweight="bold", color="black")

ax.set_xlim(-0.5, len(DIRECTIONS) - 0.5)
ax.set_ylim(len(STRESS_LIST) - 0.5, -0.5)
ax.set_xticks(range(len(DIRECTIONS)))
ax.set_xticklabels(DIRECTIONS, fontsize=11, fontweight="bold")
ax.set_yticks(range(len(STRESS_LIST)))
ax.set_yticklabels(STRESS_LIST, fontsize=11)
ax.set_aspect("equal")
ax.set_title("Spearman rho\n(OC vs divergence time)", fontsize=12, fontweight="bold", pad=10)

for spine in ax.spines.values():
    spine.set_visible(False)
ax.tick_params(length=0)

# Colorbar
sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=vmin, vmax=vmax))
sm.set_array([])
cbar = fig.colorbar(sm, ax=ax, shrink=0.7, pad=0.08)
cbar.set_label("Spearman rho", fontsize=10)
cbar.ax.tick_params(labelsize=9)

# Significance legend
fig.text(0.95, 0.02, "* p<0.05  ** p<0.01  *** p<0.001",
         fontsize=7.5, ha="right", va="bottom")

plt.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "fig3c_rho_heatmap.pdf"), dpi=300, bbox_inches="tight")
fig.savefig(os.path.join(OUT_DIR, "fig3c_rho_heatmap.png"), dpi=300, bbox_inches="tight")
print("\nSaved: fig3c_rho_heatmap.pdf/.png")
plt.close(fig)
