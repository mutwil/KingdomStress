#!/usr/bin/env python3
"""
Figure 3C: Binned OC bar chart with permutation null overlay.
Side-by-side UP/DOWN bars + null 95% CI band + significance stars.
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT_DIR = os.path.dirname(os.path.abspath(__file__))

# Load observed data
agg = pd.read_csv(os.path.join(OUT_DIR, "fig3c_phylo_oc_data.csv"))
perm = pd.read_csv(os.path.join(OUT_DIR, "fig3c_permutation_results.csv"))

bin_edges = [0, 10, 30, 60, 100, 150, 350, 500, 700, 1000]
bin_labels = ["<10", "10-30", "30-60", "60-100", "100-150",
              "150-350", "350-500", "500-700", "700+"]

agg["div_bin"] = pd.cut(agg["divergence_Mya"], bins=bin_edges, labels=bin_labels, right=False)

up_color = "#d62728"
down_color = "#1f77b4"

fig, ax = plt.subplots(figsize=(11, 5.5))

n_bins = len(bin_labels)
bar_width = 0.35
x = np.arange(n_bins)

# Get null CI (averaged across UP and DOWN since they're similar)
null_up = perm[perm["direction"] == "UP"].set_index("bin")
null_down = perm[perm["direction"] == "DOWN"].set_index("bin")

for offset, direction, color, label in [
    (-bar_width / 2, "UP", up_color, "UP-regulated"),
    (bar_width / 2, "DOWN", down_color, "DOWN-regulated"),
]:
    dsub = agg[agg["direction"] == direction].dropna(subset=["div_bin"])
    grouped = dsub.groupby("div_bin", observed=True)["mean_oc"]
    means = grouped.mean().reindex(bin_labels)
    sems = grouped.sem().reindex(bin_labels)
    counts = grouped.count().reindex(bin_labels).fillna(0).astype(int)

    null_df = null_up if direction == "UP" else null_down

    bars = ax.bar(x + offset, means.values, bar_width,
                  yerr=sems.values, capsize=2.5,
                  color=color, alpha=0.75, edgecolor="white", linewidth=0.5,
                  label=label, error_kw=dict(lw=0.8))

    # Significance stars from permutation test
    for i, bl in enumerate(bin_labels):
        m = means.values[i]
        s = sems.values[i] if not np.isnan(sems.values[i]) else 0
        if np.isnan(m):
            continue
        if bl in null_df.index:
            p = null_df.loc[bl, "p_value"]
            sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
            if sig:
                ax.text(x[i] + offset, m + s + 0.003, sig,
                        ha="center", va="bottom", fontsize=9,
                        fontweight="bold", color=color)

# Draw null CI bands (one per direction, slightly offset)
for direction, null_df, color in [("UP", null_up, up_color), ("DOWN", null_down, down_color)]:
    ci_lo = [null_df.loc[bl, "null_ci_2.5"] if bl in null_df.index else np.nan for bl in bin_labels]
    ci_hi = [null_df.loc[bl, "null_ci_97.5"] if bl in null_df.index else np.nan for bl in bin_labels]
    null_m = [null_df.loc[bl, "null_mean_oc"] if bl in null_df.index else np.nan for bl in bin_labels]

    ax.fill_between(x, ci_lo, ci_hi, color=color, alpha=0.08, zorder=0)
    ax.plot(x, null_m, color=color, linewidth=1, linestyle=":", alpha=0.5, zorder=0)

# Add a single "Null 95% CI" legend entry
ax.fill_between([], [], [], color="#888888", alpha=0.15, label="Null 95% CI")
ax.plot([], [], color="#888888", linewidth=1, linestyle=":", label="Null mean")

ax.set_xticks(x)
ax.set_xticklabels(bin_labels, rotation=45, ha="right", fontsize=9)
ax.set_xlabel("Divergence time (Mya)", fontsize=11)
ax.set_ylabel("Mean overlap coefficient", fontsize=11)
ax.set_title("Overlap coefficient by phylogenetic distance\n(with permutation null)",
             fontsize=12, fontweight="bold", pad=10)
ax.legend(fontsize=9, loc="upper right", frameon=True)
ax.tick_params(labelsize=9)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

plt.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "fig3c_binned_with_null.pdf"), dpi=300, bbox_inches="tight")
fig.savefig(os.path.join(OUT_DIR, "fig3c_binned_with_null.png"), dpi=300, bbox_inches="tight")
print("Saved: fig3c_binned_with_null.pdf/.png")
plt.close(fig)
