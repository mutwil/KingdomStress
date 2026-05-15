#!/usr/bin/env python3
"""
Figure 3C: Phylogenetic distance vs overlap coefficient decay.
Scatter + LOESS of cross-species OC against TimeTree divergence times.

Reads unrestricted_one_to_one_*.csv files, filters to cross-species
same-stress same-direction pairs, maps species pairs to divergence
times, and plots OC decay with evolutionary distance.
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
from statsmodels.nonparametric.smoothers_lowess import lowess

# ── Paths ──
INPUT_FOLDER = "/tmp/kingdom_stress_oc"
CLADES_FILE = "/tmp/kingdom_stress_oc/kingdom_stress_species_clades.csv"
OUT_DIR = os.path.dirname(os.path.abspath(__file__))
CACHE_FILE = os.path.join(OUT_DIR, "_cache_cross_species_oc.csv")

STRESS_LIST = ["Cold", "Drought", "Heat", "Heavy metal", "Pathogen", "Salt"]
INHOUSE_TAGS = ["Inhouse", "PRJEB74997", "PRJEB48734"]

NEEDED_COLS = [
    "Bioproject_X_bp",
    "Bioproject_X_species", "Bioproject_Y_species",
    "Bioproject_X_stress", "Bioproject_Y_stress",
    "Bioproject_X_direction", "Bioproject_Y_direction",
    "Overlap_coefficient", "Ref_species",
]

# ═══════════════════════════════════════════════════════════════════════════════
# TimeTree divergence times (Mya) -- pairwise MRCA approach
# ═══════════════════════════════════════════════════════════════════════════════
# Encoded as a taxonomy tree: each species is assigned to nested clades.
# Divergence time = age of the most recent common ancestor (MRCA).
# Sources: TimeTree.org (Kumar et al. 2022), supplemented with literature.

# Tree structure: (clade_name, divergence_time_Mya, children)
# Leaf nodes: (species_name,)

PHYLO_TREE = (
    "Viridiplantae", 950, [
        ("Chlorophyta", 0, [
            ("Chlamydomonas_reinhardtii",),
        ]),
        ("Streptophyta", 550, [
            ("Charophyta", 0, [
                ("Klebsormidium_nitens",),
            ]),
            ("Embryophyta", 480, [
                ("Bryophyta", 400, [
                    ("Marchantiophyta", 0, [
                        ("Marchantia_polymorpha",),
                    ]),
                    ("Bryophyta_moss", 0, [
                        ("Physcomitrium_patens",),
                    ]),
                ]),
                ("Tracheophyta", 420, [
                    ("Lycophyta", 0, [
                        ("Selaginella_moellendorffii",),
                    ]),
                    ("Euphyllophyta", 350, [
                        ("Gymnospermae", 0, [
                            ("Picea_abies",),
                        ]),
                        ("Angiospermae", 150, [
                            ("Monocotyledoneae", 0, [
                                ("Poaceae", 55, [
                                    ("BEP", 45, [
                                        ("Oryzeae", 2, [
                                            ("Oryza_sativa",),
                                            ("Oryza_sativa_Japonica_Group",),
                                        ]),
                                        ("Pooideae", 35, [
                                            ("Triticeae", 10, [
                                                ("Hordeum_vulgare",),
                                                ("Triticum_aestivum",),
                                            ]),
                                            ("Brachypodieae", 0, [
                                                ("Brachypodium_distachyon",),
                                            ]),
                                        ]),
                                    ]),
                                    ("PACMAD", 30, [
                                        ("Panicoideae", 25, [
                                            ("Andropogoneae", 15, [
                                                ("Zea_mays",),
                                                ("Sorghum_bicolor",),
                                            ]),
                                            ("Paniceae", 20, [
                                                ("Setaria_italica",),
                                                ("Cenchrus_americanus",),
                                                ("Panicum_virgatum",),
                                            ]),
                                        ]),
                                    ]),
                                ]),
                            ]),
                            ("Eudicotyledoneae", 125, [
                                ("Rosidae", 115, [
                                    ("Fabidae", 105, [
                                        ("Brassicales", 0, [
                                            ("Brassicaceae", 20, [
                                                ("Arabidopsis_thaliana",),
                                                ("Brassica_clade", 8, [
                                                    ("Brassica_napus",),
                                                    ("Brassica_oleracea",),
                                                ]),
                                            ]),
                                        ]),
                                        ("Fabales", 0, [
                                            ("Fabaceae", 55, [
                                                ("Hologalegina", 45, [
                                                    ("Medicageae", 8, [
                                                        ("Medicago_truncatula",),
                                                        ("Medicago_sativa",),
                                                    ]),
                                                    ("Cicereae", 0, [
                                                        ("Cicer_arietinum",),
                                                    ]),
                                                    ("Phaseoleae", 20, [
                                                        ("Glycine_max",),
                                                        ("Phaseolus_vulgaris",),
                                                    ]),
                                                ]),
                                                ("Dalbergioid", 0, [
                                                    ("Arachis_hypogaea",),
                                                ]),
                                            ]),
                                        ]),
                                        ("Cucurbitales", 0, [
                                            ("Cucumis_sativus",),
                                        ]),
                                        ("Rosales", 0, [
                                            ("Prunus_persica",),
                                        ]),
                                        ("Malvales", 0, [
                                            ("Gossypium_hirsutum",),
                                        ]),
                                    ]),
                                    ("Malvidae_Vitales", 0, [
                                        ("Vitales", 0, [
                                            ("Vitis_vinifera",),
                                        ]),
                                        ("Caryophyllales", 0, [
                                            ("Spinacia_oleracea",),
                                        ]),
                                    ]),
                                ]),
                                ("Asteridae", 0, [
                                    ("Lamiidae", 85, [
                                        ("Solanales", 0, [
                                            ("Solanaceae", 30, [
                                                ("Solaneae", 8, [
                                                    ("Solanum_lycopersicum",),
                                                    ("Solanum_tuberosum",),
                                                ]),
                                                ("Capsiceae", 0, [
                                                    ("Capsicum_annuum",),
                                                ]),
                                                ("Nicotianeae", 0, [
                                                    ("Nicotiana_tabacum",),
                                                ]),
                                            ]),
                                        ]),
                                        ("Asterales", 0, [
                                            ("Lactuca_sativa",),
                                        ]),
                                    ]),
                                    ("Ericales", 0, [
                                        ("Camellia_sinensis",),
                                    ]),
                                ]),
                            ]),
                        ]),
                    ]),
                ]),
            ]),
        ]),
    ]
)


def _parse_tree(node, depth=0):
    """Parse tree into {species: path_of_(clade, age)} for MRCA computation."""
    if len(node) == 1:
        # Leaf
        return {node[0]: []}
    name, age, children = node
    result = {}
    for child in children:
        child_leaves = _parse_tree(child, depth + 1)
        for sp, path in child_leaves.items():
            result[sp] = [(name, age)] + path
    return result


def build_divergence_lookup():
    """Build dict of (sp1, sp2) -> divergence time in Mya."""
    species_paths = _parse_tree(PHYLO_TREE)
    species_list = sorted(species_paths.keys())
    div_times = {}

    for i, sp1 in enumerate(species_list):
        for sp2 in species_list[i + 1:]:
            path1 = species_paths[sp1]
            path2 = species_paths[sp2]
            # Find MRCA: deepest shared clade
            mrca_age = 0
            for (c1, a1), (c2, a2) in zip(path1, path2):
                if c1 == c2:
                    mrca_age = a1
                else:
                    break
            key = tuple(sorted([sp1, sp2]))
            div_times[key] = mrca_age

    return div_times, species_list


# ═══════════════════════════════════════════════════════════════════════════════
# Load cross-species OC data
# ═══════════════════════════════════════════════════════════════════════════════

def load_cross_species_data():
    """Load unrestricted files, filter to cross-species same-stress pairs."""
    if os.path.exists(CACHE_FILE):
        print(f"Loading cached data: {CACHE_FILE}")
        return pd.read_csv(CACHE_FILE)

    print("Loading unrestricted files (this may take a while on Google Drive)...")
    kept = []
    for fname in sorted(os.listdir(INPUT_FOLDER)):
        if not (fname.startswith("unrestricted_one_to_one_") and fname.endswith(".csv")):
            continue
        fpath = os.path.join(INPUT_FOLDER, fname)
        print(f"  Loading: {fname}", flush=True)
        chunk = pd.read_csv(fpath, usecols=NEEDED_COLS)
        # Filter: cross-species, same stress, same direction
        mask = (
            (chunk["Bioproject_X_species"] != chunk["Bioproject_Y_species"]) &
            (chunk["Bioproject_X_stress"] == chunk["Bioproject_Y_stress"]) &
            (chunk["Bioproject_X_direction"] == chunk["Bioproject_Y_direction"]) &
            (chunk["Bioproject_X_stress"].isin(STRESS_LIST))
        )
        kept.append(chunk.loc[mask])
        del chunk

    df = pd.concat(kept, ignore_index=True)

    # Tag inhouse
    pattern = "|".join(INHOUSE_TAGS)
    df["is_inhouse"] = df["Bioproject_X_bp"].str.contains(pattern, na=False)

    # Create canonical species pair key (sorted)
    df["sp_pair"] = df.apply(
        lambda r: tuple(sorted([r["Bioproject_X_species"], r["Bioproject_Y_species"]])),
        axis=1,
    )
    df["sp_pair_str"] = df["sp_pair"].apply(lambda x: f"{x[0]}|{x[1]}")

    print(f"  Total cross-species rows: {len(df):,}")
    df.to_csv(CACHE_FILE, index=False)
    print(f"  Cached to: {CACHE_FILE}")
    return df


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

print("Building divergence time lookup...")
div_times, all_species = build_divergence_lookup()
print(f"  {len(all_species)} species, {len(div_times)} pairwise divergence times")

# Show a few examples
examples = [
    ("Oryza_sativa", "Oryza_sativa_Japonica_Group"),
    ("Oryza_sativa", "Zea_mays"),
    ("Arabidopsis_thaliana", "Glycine_max"),
    ("Arabidopsis_thaliana", "Oryza_sativa"),
    ("Arabidopsis_thaliana", "Picea_abies"),
    ("Arabidopsis_thaliana", "Physcomitrium_patens"),
    ("Arabidopsis_thaliana", "Chlamydomonas_reinhardtii"),
]
for sp1, sp2 in examples:
    key = tuple(sorted([sp1, sp2]))
    print(f"  {sp1} -- {sp2}: {div_times.get(key, 'N/A')} Mya")

# Load data
df = load_cross_species_data()

# Use all data (public + inhouse merged)
print(f"\nAll data: {len(df):,} rows")

# Map divergence times
df["divergence_Mya"] = df["sp_pair_str"].apply(
    lambda x: div_times.get(tuple(sorted(x.split("|"))), np.nan)
)
df = df.dropna(subset=["divergence_Mya"])
print(f"After mapping divergence times: {len(df):,} rows")

# ── Aggregate: mean OC per species_pair x stress x direction ──
agg = (
    df.groupby(["sp_pair_str", "Bioproject_X_stress", "Bioproject_X_direction", "divergence_Mya"])
    ["Overlap_coefficient"]
    .mean()
    .reset_index()
)
agg.columns = ["sp_pair", "stress", "direction", "divergence_Mya", "mean_oc"]
print(f"Aggregated: {len(agg):,} species-pair x stress x direction combinations")

# Save aggregated data
agg.to_csv(os.path.join(OUT_DIR, "fig3c_phylo_oc_data.csv"), index=False)

# ═══════════════════════════════════════════════════════════════════════════════
# Plot 1: Faceted by stress, colored by direction
# ═══════════════════════════════════════════════════════════════════════════════

print("\nGenerating phylo-OC decay plot...")

fig, axes = plt.subplots(2, 3, figsize=(14, 9), sharey=True, sharex=True)
axes = axes.ravel()

up_color = "#d62728"
down_color = "#1f77b4"

for idx, stress in enumerate(STRESS_LIST):
    ax = axes[idx]
    sub = agg[agg["stress"] == stress]

    for direction, color, label in [("UP", up_color, "UP"), ("DOWN", down_color, "DOWN")]:
        dsub = sub[sub["direction"] == direction]
        if len(dsub) == 0:
            continue

        # Scatter (small, transparent)
        ax.scatter(dsub["divergence_Mya"], dsub["mean_oc"],
                   c=color, alpha=0.15, s=8, edgecolors="none", rasterized=True)

        # LOWESS trend
        if len(dsub) >= 10:
            smoothed = lowess(dsub["mean_oc"].values, dsub["divergence_Mya"].values,
                              frac=0.4, return_sorted=True)
            ax.plot(smoothed[:, 0], smoothed[:, 1], color=color, linewidth=2.5,
                    label=label, zorder=5)

        # Spearman correlation
        if len(dsub) >= 5:
            rho, pval = spearmanr(dsub["divergence_Mya"], dsub["mean_oc"])
            sig = "***" if pval < 0.001 else "**" if pval < 0.01 else "*" if pval < 0.05 else "ns"
            # Position text for UP vs DOWN
            y_pos = 0.95 if direction == "UP" else 0.87
            ax.text(0.98, y_pos, f"{label}: rho={rho:.2f} {sig}",
                    transform=ax.transAxes, ha="right", va="top",
                    fontsize=7.5, color=color,
                    bbox=dict(boxstyle="round,pad=0.2", facecolor="white",
                              edgecolor=color, alpha=0.8, linewidth=0.5))

    ax.set_title(stress, fontsize=12, fontweight="bold")
    ax.set_xlim(-20, 1000)
    ax.set_ylim(-0.02, 0.55)

    if idx >= 3:
        ax.set_xlabel("Divergence time (Mya)", fontsize=10)
    if idx % 3 == 0:
        ax.set_ylabel("Mean overlap coefficient", fontsize=10)

    ax.tick_params(labelsize=9)

# Legend
handles = [
    plt.Line2D([0], [0], color=up_color, linewidth=2.5, label="UP-regulated"),
    plt.Line2D([0], [0], color=down_color, linewidth=2.5, label="DOWN-regulated"),
]
fig.legend(handles=handles, loc="upper right", fontsize=10,
           bbox_to_anchor=(0.98, 0.98), frameon=True)

fig.suptitle("Stress response conservation decays with phylogenetic distance",
             fontsize=14, fontweight="bold", y=1.01)
plt.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "fig3c_phylo_oc_decay.pdf"), dpi=300, bbox_inches="tight")
fig.savefig(os.path.join(OUT_DIR, "fig3c_phylo_oc_decay.png"), dpi=300, bbox_inches="tight")
print("Saved: fig3c_phylo_oc_decay.pdf/.png")
plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════════════
# Plot 2: All stresses combined, binned by divergence time
# ═══════════════════════════════════════════════════════════════════════════════

print("\nGenerating binned summary plot...")

# Define bins
bins = [0, 10, 30, 60, 100, 150, 350, 500, 700, 1000]
bin_labels = ["<10", "10-30", "30-60", "60-100", "100-150",
              "150-350", "350-500", "500-700", "700+"]

agg["div_bin"] = pd.cut(agg["divergence_Mya"], bins=bins, labels=bin_labels, right=False)

fig2, ax = plt.subplots(figsize=(10, 5))

n_bins = len(bin_labels)
bar_width = 0.38
x_pos = np.arange(n_bins)

for offset, direction, color, label in [
    (-bar_width / 2, "UP", up_color, "UP-regulated"),
    (bar_width / 2, "DOWN", down_color, "DOWN-regulated"),
]:
    dsub = agg[agg["direction"] == direction].dropna(subset=["div_bin"])
    grouped = dsub.groupby("div_bin", observed=True)["mean_oc"]
    means = grouped.mean().reindex(bin_labels)
    sems = grouped.sem().reindex(bin_labels)
    counts = grouped.count().reindex(bin_labels).fillna(0).astype(int)

    bars = ax.bar(x_pos + offset, means.values, bar_width,
                  yerr=sems.values, capsize=2.5,
                  color=color, alpha=0.75, edgecolor="white", linewidth=0.5,
                  label=label, error_kw=dict(lw=0.8))

    # Add sample sizes above bars
    for i in range(n_bins):
        m = means.values[i]
        s = sems.values[i]
        n = counts.values[i]
        if np.isnan(m):
            continue
        ax.text(x_pos[i] + offset, m + (s if not np.isnan(s) else 0) + 0.003,
                f"n={n}", ha="center", va="bottom", fontsize=5.5, color="#555555")

ax.set_xticks(x_pos)
ax.set_xticklabels(bin_labels, rotation=45, ha="right", fontsize=9)
ax.set_xlabel("Divergence time (Mya)", fontsize=11)
ax.set_ylabel("Mean overlap coefficient", fontsize=11)
ax.set_title("Overlap coefficient by phylogenetic distance (all stresses pooled)",
             fontsize=12, fontweight="bold", pad=10)
ax.legend(fontsize=10, frameon=True)
ax.tick_params(labelsize=9)

plt.tight_layout()
fig2.savefig(os.path.join(OUT_DIR, "fig3c_phylo_oc_binned.pdf"), dpi=300, bbox_inches="tight")
fig2.savefig(os.path.join(OUT_DIR, "fig3c_phylo_oc_binned.png"), dpi=300, bbox_inches="tight")
print("Saved: fig3c_phylo_oc_binned.pdf/.png")
plt.close(fig2)

print("\nDone.")
