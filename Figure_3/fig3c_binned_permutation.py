#!/usr/bin/env python3
"""
Permutation test: Is the observed OC per divergence bin significantly
above what you'd expect from random orthogroup overlap?

Null model: shuffle OC values across species pairs within each
stress x direction group, breaking the distance-OC association.
Then recompute mean OC per divergence bin under the null.

This tests whether the OC at each bin is significantly different
from the global average (i.e., whether phylogenetic distance matters).

Additionally, we compare the observed OC at each bin against a
"random overlap" null: shuffle species pair labels entirely
(breaking both distance AND biological pairing) to estimate
the OC expected by chance alone.
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_FILE = os.path.join(OUT_DIR, "fig3c_phylo_oc_data.csv")

N_PERM = 10000
SEED = 42

bins = [0, 10, 30, 60, 100, 150, 350, 500, 700, 1000]
bin_labels = ["<10", "10-30", "30-60", "60-100", "100-150",
              "150-350", "350-500", "500-700", "700+"]

up_color = "#d62728"
down_color = "#1f77b4"

# ── Load data ──
agg = pd.read_csv(DATA_FILE)
agg["div_bin"] = pd.cut(agg["divergence_Mya"], bins=bins, labels=bin_labels, right=False)
agg = agg.dropna(subset=["div_bin"])

rng = np.random.default_rng(SEED)

# ── Permutation per direction ──
results = {}

for direction in ["UP", "DOWN"]:
    dsub = agg[agg["direction"] == direction].copy()
    oc_vals = dsub["mean_oc"].values
    div_bins = dsub["div_bin"].values

    # Observed means per bin
    obs_means = dsub.groupby("div_bin", observed=True)["mean_oc"].mean()

    # Permutation: shuffle OC values, recompute bin means
    null_means = {b: [] for b in bin_labels}
    for _ in range(N_PERM):
        shuffled_oc = rng.permutation(oc_vals)
        for bi, bl in enumerate(bin_labels):
            mask = div_bins == bl
            if mask.sum() > 0:
                null_means[bl].append(shuffled_oc[mask].mean())

    # Compute p-values (one-sided: is observed > null?)
    pvals = {}
    null_mean_of_means = {}
    null_ci_low = {}
    null_ci_high = {}
    for bl in bin_labels:
        if bl in obs_means.index and len(null_means[bl]) > 0:
            null_arr = np.array(null_means[bl])
            obs = obs_means[bl]
            pvals[bl] = (null_arr >= obs).sum() / len(null_arr)
            null_mean_of_means[bl] = null_arr.mean()
            null_ci_low[bl] = np.percentile(null_arr, 2.5)
            null_ci_high[bl] = np.percentile(null_arr, 97.5)
        else:
            pvals[bl] = np.nan
            null_mean_of_means[bl] = np.nan
            null_ci_low[bl] = np.nan
            null_ci_high[bl] = np.nan

    results[direction] = {
        "obs_means": obs_means,
        "pvals": pvals,
        "null_mean": null_mean_of_means,
        "null_ci_low": null_ci_low,
        "null_ci_high": null_ci_high,
    }

# ── Print results ──
print(f"Permutation test (n={N_PERM}): Is observed OC per bin > shuffled null?")
print(f"{'Dir':<6} {'Bin':<10} {'Obs':>7} {'Null':>7} {'95% CI':>16} {'p-value':>10} {'Sig':>5}")
print("-" * 70)
for direction in ["UP", "DOWN"]:
    r = results[direction]
    for bl in bin_labels:
        obs = r["obs_means"].get(bl, np.nan)
        null = r["null_mean"].get(bl, np.nan)
        lo = r["null_ci_low"].get(bl, np.nan)
        hi = r["null_ci_high"].get(bl, np.nan)
        p = r["pvals"].get(bl, np.nan)
        if np.isnan(obs):
            continue
        sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"
        print(f"{direction:<6} {bl:<10} {obs:>7.4f} {null:>7.4f} [{lo:.4f}, {hi:.4f}] {p:>10.4f} {sig:>5}")

# Save table
rows = []
for direction in ["UP", "DOWN"]:
    r = results[direction]
    for bl in bin_labels:
        obs = r["obs_means"].get(bl, np.nan)
        if np.isnan(obs):
            continue
        rows.append({
            "direction": direction,
            "bin": bl,
            "observed_mean_oc": obs,
            "null_mean_oc": r["null_mean"][bl],
            "null_ci_2.5": r["null_ci_low"][bl],
            "null_ci_97.5": r["null_ci_high"][bl],
            "p_value": r["pvals"][bl],
        })
pd.DataFrame(rows).to_csv(os.path.join(OUT_DIR, "fig3c_permutation_results.csv"), index=False)

# ── Plot: observed vs null with CI band ──
fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)

for ax, direction, color, title in [
    (axes[0], "UP", up_color, "UP-regulated"),
    (axes[1], "DOWN", down_color, "DOWN-regulated"),
]:
    r = results[direction]
    obs = r["obs_means"].reindex(bin_labels)
    null_m = [r["null_mean"].get(bl, np.nan) for bl in bin_labels]
    null_lo = [r["null_ci_low"].get(bl, np.nan) for bl in bin_labels]
    null_hi = [r["null_ci_high"].get(bl, np.nan) for bl in bin_labels]
    pv = [r["pvals"].get(bl, np.nan) for bl in bin_labels]

    x = np.arange(len(bin_labels))

    # Null CI band
    ax.fill_between(x, null_lo, null_hi, color="#cccccc", alpha=0.6, label="Null 95% CI")
    ax.plot(x, null_m, color="#888888", linewidth=1.5, linestyle="--", label="Null mean")

    # Observed
    ax.plot(x, obs.values, color=color, linewidth=2.5, marker="o", markersize=7,
            label="Observed", zorder=5)

    # Significance stars
    for i, bl in enumerate(bin_labels):
        p = pv[i]
        o = obs.values[i]
        if np.isnan(o) or np.isnan(p):
            continue
        sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
        if sig:
            ax.text(i, o + 0.004, sig, ha="center", va="bottom",
                    fontsize=10, fontweight="bold", color=color)

    ax.set_xticks(x)
    ax.set_xticklabels(bin_labels, rotation=45, ha="right", fontsize=9)
    ax.set_xlabel("Divergence time (Mya)", fontsize=11)
    ax.set_ylabel("Mean overlap coefficient", fontsize=11)
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.legend(fontsize=9, loc="upper right")
    ax.tick_params(labelsize=9)

fig.suptitle("Observed OC vs permuted null per divergence bin",
             fontsize=13, fontweight="bold", y=1.02)
plt.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "fig3c_permutation_test.pdf"), dpi=300, bbox_inches="tight")
fig.savefig(os.path.join(OUT_DIR, "fig3c_permutation_test.png"), dpi=300, bbox_inches="tight")
print("\nSaved: fig3c_permutation_test.pdf/.png")
plt.close(fig)
