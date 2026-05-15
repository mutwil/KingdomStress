#!/usr/bin/env python3
"""
Figure 3B: Cross-clade overlap coefficient heatmap -- All data (public + inhouse).
Vibrant red/blue colormaps, square cells.
Recomputes from unrestricted_one_to_one_*.csv files with Monte Carlo significance.
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from joblib import Parallel, delayed

# ── Paths ──
INPUT_FOLDER = "/tmp/kingdom_stress_oc"
CLADES_FILE = "/tmp/kingdom_stress_oc/kingdom_stress_species_clades.csv"
OUT_DIR = os.path.dirname(os.path.abspath(__file__))
CACHE_FILE = os.path.join(OUT_DIR, "_cache_cross_clade_all.csv")

# ── Parameters ──
STRESS_LIST = ["Cold", "Drought", "Heat", "Heavy metal", "Pathogen", "Salt"]
CLADE_ORDER = ["Chlorophyte", "Charophyte", "Bryophyte", "Lycophyte",
               "Gymnosperm", "Monocot", "Dicot"]
N_ITER = 1000
N_JOBS = 8
RANDOM_SEED = 42

NEEDED_COLS = [
    "Bioproject_X_bp",
    "Bioproject_X_species", "Bioproject_Y_species",
    "Bioproject_X_stress", "Bioproject_Y_stress",
    "Bioproject_X_direction", "Bioproject_Y_direction",
    "Overlap_coefficient", "Ref_species",
]

# ── 1. Load clade mapping ──
ref_df = pd.read_csv(CLADES_FILE)
species_to_clade = dict(zip(ref_df["species"], ref_df["classification"]))
print("Clades loaded.", flush=True)


# ── 2. Load & filter ──
def assign_clades(chunk):
    chunk = chunk.copy()
    chunk["Bioproject_X_clade"] = chunk["Bioproject_X_species"].map(species_to_clade)
    chunk["Bioproject_Y_clade"] = chunk["Bioproject_Y_species"].map(species_to_clade)
    return chunk


def load_data():
    if os.path.exists(CACHE_FILE):
        print(f"Loading cached: {CACHE_FILE}")
        return pd.read_csv(CACHE_FILE)

    print("Loading unrestricted files...")
    kept = []
    for fname in sorted(os.listdir(INPUT_FOLDER)):
        if not (fname.startswith("unrestricted_one_to_one_") and fname.endswith(".csv")):
            continue
        fpath = os.path.join(INPUT_FOLDER, fname)
        print(f"  {fname}", flush=True)
        chunk = pd.read_csv(fpath, usecols=NEEDED_COLS)
        chunk = assign_clades(chunk)
        mask = (
            (chunk["Bioproject_X_clade"] != chunk["Bioproject_Y_clade"]) &
            (chunk["Bioproject_X_stress"] == chunk["Bioproject_Y_stress"]) &
            (chunk["Bioproject_X_direction"] == chunk["Bioproject_Y_direction"]) &
            chunk["Bioproject_X_clade"].notna() &
            chunk["Bioproject_Y_clade"].notna()
        )
        kept.append(chunk.loc[mask])
        del chunk
    df = pd.concat(kept, ignore_index=True)
    print(f"  Total rows: {len(df):,}")
    df.to_csv(CACHE_FILE, index=False)
    return df


# ── 3. Monte Carlo ──
def run_monte_carlo(clade_df, n_iter=1000, n_jobs=8, seed=42):
    if len(clade_df) == 0:
        return pd.DataFrame(columns=["Observed", "P_value"])

    observed_means = clade_df.groupby(
        ["Bioproject_X_stress", "Bioproject_X_direction"]
    )["Overlap_coefficient"].mean()
    observed_keys = list(observed_means.index)
    key_set = set(observed_keys)

    pool_data = []
    for sp in clade_df["Ref_species"].unique():
        sp_mask = clade_df["Ref_species"] == sp
        for d in ["UP", "DOWN"]:
            sub = clade_df.loc[
                sp_mask & (clade_df["Bioproject_X_direction"] == d),
                ["Bioproject_X_stress", "Overlap_coefficient"]
            ]
            if len(sub) < 2:
                continue
            sub_s = sub.sort_values("Bioproject_X_stress")
            coeffs = sub_s["Overlap_coefficient"].values.astype(np.float64)
            stress_slices = {}
            pos = 0
            for stress, grp in sub_s.groupby("Bioproject_X_stress", sort=False):
                n = len(grp)
                stress_slices[stress] = (pos, pos + n)
                pos += n
            pool_data.append((d, coeffs, stress_slices))

    def _single_iter(i):
        local_rng = np.random.default_rng(seed + i)
        sums = {key: 0.0 for key in observed_keys}
        cnts = {key: 0.0 for key in observed_keys}
        for d, coeffs, stress_slices in pool_data:
            shuffled = coeffs[local_rng.permutation(len(coeffs))]
            for stress, (start, end) in stress_slices.items():
                key = (stress, d)
                if key in key_set:
                    sums[key] += shuffled[start:end].sum()
                    cnts[key] += (end - start)
        return {key: sums[key] / cnts[key] if cnts[key] > 0 else np.nan
                for key in observed_keys}

    results = Parallel(n_jobs=n_jobs, prefer="threads", verbose=0)(
        delayed(_single_iter)(i) for i in range(n_iter)
    )
    null_dist = {key: np.array([r[key] for r in results]) for key in observed_keys}

    p_values = {}
    for key in observed_means.index:
        obs = observed_means.loc[key]
        null = null_dist[key]
        null = null[~np.isnan(null)]
        p_values[key] = (null >= obs).sum() / len(null) if len(null) > 0 else np.nan

    return pd.DataFrame({
        "Observed": observed_means,
        "P_value": pd.Series(p_values),
    })


def sig_stars(p):
    if p is None or (isinstance(p, float) and np.isnan(p)):
        return ""
    if p < 0.001: return "***"
    if p < 0.01: return "**"
    if p < 0.05: return "*"
    return ""


# ── 4. Load and process ──
real_all = load_data()

present_clades = set(real_all["Bioproject_X_clade"].unique()) | set(real_all["Bioproject_Y_clade"].unique())
plot_clades = [c for c in CLADE_ORDER if c in present_clades]
print(f"Clades with data: {plot_clades}")

# Monte Carlo per clade
mc_results = {}
for clade in plot_clades:
    clade_df = real_all[real_all["Bioproject_X_clade"] == clade].reset_index(drop=True)
    print(f"  MC for {clade} ({len(clade_df):,} rows) ...", flush=True)
    mc_results[clade] = run_monte_carlo(clade_df, n_iter=N_ITER, n_jobs=N_JOBS, seed=RANDOM_SEED)

# Build matrices
n_c = len(plot_clades)
n_s = len(STRESS_LIST)
mean_up = np.full((n_c, n_s), np.nan)
pval_up = np.full((n_c, n_s), np.nan)
mean_down = np.full((n_c, n_s), np.nan)
pval_down = np.full((n_c, n_s), np.nan)

for ci, clade in enumerate(plot_clades):
    mc = mc_results.get(clade, pd.DataFrame())
    for si, stress in enumerate(STRESS_LIST):
        for direction, mean_mat, pval_mat in [
            ("UP", mean_up, pval_up),
            ("DOWN", mean_down, pval_down),
        ]:
            try:
                mean_mat[ci, si] = mc.loc[(stress, direction), "Observed"]
                pval_mat[ci, si] = mc.loc[(stress, direction), "P_value"]
            except (KeyError, TypeError):
                pass

# Save data
rows = []
for ci, clade in enumerate(plot_clades):
    for si, stress in enumerate(STRESS_LIST):
        for direction, mean_mat, pval_mat in [("UP", mean_up, pval_up), ("DOWN", mean_down, pval_down)]:
            rows.append({
                "clade": clade, "stress": stress, "direction": direction,
                "mean_oc": mean_mat[ci, si], "p_value": pval_mat[ci, si],
            })
pd.DataFrame(rows).to_csv(os.path.join(OUT_DIR, "fig3b_cross_clade_data.csv"), index=False)

# ── 5. Plot: vibrant square-cell heatmap ──
all_vals = np.concatenate([mean_up.ravel(), mean_down.ravel()])
all_vals = all_vals[~np.isnan(all_vals)]
vmin = all_vals.min() - 0.005
vmax = all_vals.max() + 0.005

up_cmap = mcolors.LinearSegmentedColormap.from_list(
    "vibrant_red", ["#fff5f0", "#fee0d2", "#fc9272", "#de2d26", "#a50f15"]
)
down_cmap = mcolors.LinearSegmentedColormap.from_list(
    "vibrant_blue", ["#f7fbff", "#deebf7", "#9ecae1", "#3182bd", "#08519c"]
)


def plot_heatmap(ax, vals, pvals, title, cmap, vmin, vmax):
    n_rows, n_cols = vals.shape
    for i in range(n_rows):
        for j in range(n_cols):
            v = vals[i, j]
            if np.isnan(v):
                ax.add_patch(plt.Rectangle(
                    (j - 0.5, i - 0.5), 1, 1,
                    facecolor="#e0e0e0", edgecolor="white", linewidth=1.5
                ))
                ax.text(j, i, "NA", ha="center", va="center",
                        fontsize=9, color="#999999", fontstyle="italic")
            else:
                norm_v = np.clip((v - vmin) / (vmax - vmin), 0, 1)
                color = cmap(norm_v)
                ax.add_patch(plt.Rectangle(
                    (j - 0.5, i - 0.5), 1, 1,
                    facecolor=color, edgecolor="white", linewidth=1.5
                ))
                ax.text(j, i - 0.08, f"{v:.3f}", ha="center", va="center",
                        fontsize=9, color="black")
                p = pvals[i, j]
                s = sig_stars(p)
                if s:
                    ax.text(j, i + 0.25, s, ha="center", va="center",
                            fontsize=11, fontweight="bold", color="black")

    ax.set_xlim(-0.5, n_cols - 0.5)
    ax.set_ylim(n_rows - 0.5, -0.5)
    ax.set_xticks(range(n_cols))
    ax.set_xticklabels(STRESS_LIST, rotation=45, ha="right", fontsize=12, fontweight="bold")
    ax.set_yticks(range(n_rows))
    ax.set_yticklabels(plot_clades, fontsize=12, fontweight="bold")
    ax.set_aspect("equal")
    ax.set_title(title, fontsize=15, fontweight="bold", pad=12)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(length=0)


fig, (ax_up, ax_down) = plt.subplots(
    2, 1, figsize=(9, 11),
    gridspec_kw={"hspace": 0.45}
)

plot_heatmap(ax_up, mean_up, pval_up, "UP-regulated genes", up_cmap, vmin, vmax)
plot_heatmap(ax_down, mean_down, pval_down, "DOWN-regulated genes", down_cmap, vmin, vmax)

fig.suptitle("Cross-clade overlap coefficient -- All data",
             fontsize=17, fontweight="bold", y=0.98)

sm_up = plt.cm.ScalarMappable(cmap=up_cmap, norm=plt.Normalize(vmin, vmax))
sm_down = plt.cm.ScalarMappable(cmap=down_cmap, norm=plt.Normalize(vmin, vmax))

cbar_ax_up = fig.add_axes([0.92, 0.53, 0.02, 0.35])
cbar_up = fig.colorbar(sm_up, cax=cbar_ax_up)
cbar_up.set_label("Mean overlap coefficient", fontsize=11)
cbar_up.ax.tick_params(labelsize=10)

cbar_ax_down = fig.add_axes([0.92, 0.08, 0.02, 0.35])
cbar_down = fig.colorbar(sm_down, cax=cbar_ax_down)
cbar_down.set_label("Mean overlap coefficient", fontsize=11)
cbar_down.ax.tick_params(labelsize=10)

fig.text(0.92, 0.02, "*** p<0.001", fontsize=9, va="bottom", ha="left")

fig.savefig(os.path.join(OUT_DIR, "fig3b_cross_clade_heatmap.pdf"), dpi=300, bbox_inches="tight")
fig.savefig(os.path.join(OUT_DIR, "fig3b_cross_clade_heatmap.png"), dpi=300, bbox_inches="tight")
print("\nSaved: fig3b_cross_clade_heatmap.pdf/.png")
plt.close(fig)
