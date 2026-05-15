#!/usr/bin/env python3
"""
Two additional Figure 2 panels:
  Panel X: Data hierarchy pyramid (Species -> Stress -> ... -> DEGs)
  Panel Y: Overlap coefficient heatmap between stress types at OG level
"""

import pandas as pd
import numpy as np
import os
from collections import defaultdict
from itertools import combinations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import matplotlib.patheffects as pe

# ── Paths ──
DEG_FILE = "/Users/vjx443/Downloads/kingdom_stress_dict v3.csv"
OUT_DIR = "/Users/vjx443/Library/CloudStorage/GoogleDrive-mutwil@plant.tools/" \
          "My Drive/Projects/2026_KingdomStress/CC-Fig 2 data"

MAIN_STRESSES = [
    "Heat", "Cold", "Drought", "Salt", "High light",
    "Pathogen", "Flooding", "Heavy metal", "Herbivory",
]
STRESS_NORMALIZE = {"High Light": "High light", "high light": "High light"}

STRESS_COLORS = {
    "Heat": "#d62728", "Cold": "#1f77b4", "Drought": "#ff7f0e",
    "Salt": "#8c564b", "High light": "#ffbb33", "Pathogen": "#2ca02c",
    "Flooding": "#17becf", "Heavy metal": "#7f7f7f",
    "Herbivory": "#bcbd22",
}


# ═══════════════════════════════════════════════════════════════════════════════
# 1. Count hierarchy levels from DEG table
# ═══════════════════════════════════════════════════════════════════════════════

print("=" * 60)
print("Computing data hierarchy counts...")
print("=" * 60)

# Use cached OG-species-stress data for overlap, but need full DEG file for hierarchy
og_cache = os.path.join(OUT_DIR, "_cache_og_species_stress.csv")
og_ss_df = pd.read_csv(og_cache)

# For hierarchy counts, read from DEG file in chunks
hierarchy_cache = os.path.join(OUT_DIR, "_cache_hierarchy_counts.csv")

if os.path.exists(hierarchy_cache):
    print("  Loading cached hierarchy counts...")
    hc = pd.read_csv(hierarchy_cache)
    counts = dict(zip(hc["level"], hc["count"]))
else:
    print("  Reading DEG file for hierarchy counts...")
    species_set = set()
    stress_set = set()
    organ_set = set()
    bioproject_set = set()
    expgroup_set = set()  # species + bioproject + experiment
    direction_set = set()
    total_degs = 0

    for chunk in pd.read_csv(DEG_FILE, chunksize=500_000, low_memory=False):
        chunk["stress"] = chunk["stress"].replace(STRESS_NORMALIZE)
        chunk = chunk[chunk["stress"].isin(MAIN_STRESSES)]

        species_set.update(chunk["species"].unique())
        stress_set.update(chunk["stress"].unique())
        organ_set.update(chunk["organ"].dropna().unique())
        bioproject_set.update(
            (chunk["species"] + "|" + chunk["bioproject"]).unique()
        )
        expgroup_set.update(
            (chunk["species"] + "|" + chunk["bioproject"] + "|" + chunk["experiment"]).unique()
        )
        direction_set.update(chunk["direction"].unique())
        total_degs += len(chunk)

    counts = {
        "Species": len(species_set),
        "Stress types": len(stress_set),
        "Organs": len(organ_set),
        "BioProjects": len(bioproject_set),
        "Experiments": len(expgroup_set),
        "Directions": len(direction_set),
        "DEGs": total_degs,
    }
    pd.DataFrame({"level": list(counts.keys()), "count": list(counts.values())}).to_csv(
        hierarchy_cache, index=False
    )

print("  Hierarchy counts:")
for level, count in counts.items():
    print(f"    {level}: {count:,}")


# ═══════════════════════════════════════════════════════════════════════════════
# PANEL X: Data hierarchy pyramid
# ═══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 60)
print("Generating hierarchy pyramid...")
print("=" * 60)

# Pyramid layers from top (fewest) to bottom (most)
layers = [
    ("Species", counts["Species"], "#ff6b6b"),
    ("Stresses", counts["Stress types"], "#ffa94d"),
    ("Organs", counts["Organs"], "#ffd43b"),
    ("BioProjects", counts["BioProjects"], "#a9e34b"),
    ("Experiments", counts["Experiments"], "#69db7c"),
    ("Directions", counts["Directions"], "#74c0fc"),
    ("DEGs", counts["DEGs"], "#b197fc"),
]

n_layers = len(layers)
fig_p, ax_p = plt.subplots(figsize=(6, 5))

# Draw trapezoids from top to bottom
max_width = 0.9
min_width = 0.15
y_spacing = 1.0
total_height = n_layers * y_spacing

for i, (label, count, color) in enumerate(layers):
    # Width increases from top to bottom
    frac = i / (n_layers - 1) if n_layers > 1 else 0
    width = min_width + frac * (max_width - min_width)
    # Next layer width (for trapezoid bottom)
    if i < n_layers - 1:
        next_frac = (i + 1) / (n_layers - 1)
        next_width = min_width + next_frac * (max_width - next_width if 'next_width' in dir() else (max_width - min_width))
    next_width_val = min_width + ((i + 1) / (n_layers - 1)) * (max_width - min_width) if i < n_layers - 1 else width

    y_top = total_height - i * y_spacing
    y_bot = y_top - y_spacing * 0.85

    # Trapezoid vertices
    top_half = width / 2
    bot_half = next_width_val / 2 if i < n_layers - 1 else width / 2

    trap_x = [-top_half, top_half, bot_half, -bot_half]
    trap_y = [y_top, y_top, y_bot, y_bot]

    ax_p.fill(trap_x, trap_y, color=color, edgecolor="white", linewidth=1.5,
              alpha=0.9, zorder=2)

    # Label with count
    y_mid = (y_top + y_bot) / 2
    if count >= 1_000_000:
        count_str = f"{count / 1_000_000:.1f}M"
    elif count >= 1_000:
        count_str = f"{count:,}"
    else:
        count_str = str(count)

    ax_p.text(0, y_mid, f"{label}\n({count_str})", ha="center", va="center",
              fontsize=9, fontweight="bold", zorder=3,
              path_effects=[pe.withStroke(linewidth=2, foreground="white")])

ax_p.set_xlim(-0.6, 0.6)
ax_p.set_ylim(-0.2, total_height + 0.5)
ax_p.set_aspect("equal")
ax_p.axis("off")
ax_p.set_title("Data hierarchy", fontsize=11, pad=10)

fig_p.savefig(os.path.join(OUT_DIR, "fig2_hierarchy.pdf"), dpi=300, bbox_inches="tight")
fig_p.savefig(os.path.join(OUT_DIR, "fig2_hierarchy.png"), dpi=300, bbox_inches="tight")
print("  Saved: fig2_hierarchy.pdf/.png")
plt.close(fig_p)


# ═══════════════════════════════════════════════════════════════════════════════
# PANEL Y: Overlap coefficient schematic (Venn diagram with example)
# ═══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 60)
print("Computing overlap coefficients between stresses...")
print("=" * 60)

# For each stress, get the set of OGs with DEGs
stress_ogs = {}
for stress in MAIN_STRESSES:
    ogs = set(og_ss_df[og_ss_df["stress"] == stress]["og"].unique())
    stress_ogs[stress] = ogs
    print(f"  {stress}: {len(ogs):,} OGs")

# Pick a concrete example pair for the schematic
ex_a, ex_b = "Heat", "Cold"
set_a = stress_ogs[ex_a]
set_b = stress_ogs[ex_b]
inter = set_a & set_b
only_a = set_a - set_b
only_b = set_b - set_a
min_size = min(len(set_a), len(set_b))
oc_val = len(inter) / min_size

print(f"\n  Example: {ex_a} vs {ex_b}")
print(f"    |{ex_a}| = {len(set_a):,}")
print(f"    |{ex_b}| = {len(set_b):,}")
print(f"    |intersection| = {len(inter):,}")
print(f"    min(|A|,|B|) = {min_size:,}")
print(f"    OC = {oc_val:.2f}")

# ── Draw schematic ──
print("\nGenerating overlap coefficient schematic...")

fig_s, ax_s = plt.subplots(figsize=(7, 4.5))
ax_s.set_xlim(-3.5, 3.5)
ax_s.set_ylim(-2.2, 2.8)
ax_s.set_aspect("equal")
ax_s.axis("off")

# Two overlapping circles
from matplotlib.patches import Circle

r = 1.5
offset = 1.0  # center-to-center distance

circle_a = Circle((-offset/2, 0), r, facecolor=STRESS_COLORS[ex_a], alpha=0.25,
                  edgecolor=STRESS_COLORS[ex_a], linewidth=2)
circle_b = Circle((offset/2, 0), r, facecolor=STRESS_COLORS[ex_b], alpha=0.25,
                  edgecolor=STRESS_COLORS[ex_b], linewidth=2)
ax_s.add_patch(circle_a)
ax_s.add_patch(circle_b)

# Labels for circles
ax_s.text(-offset/2 - 0.7, 0.05, f"{ex_a}\nOGs",
          ha="center", va="center", fontsize=11, fontweight="bold",
          color=STRESS_COLORS[ex_a])
ax_s.text(offset/2 + 0.7, 0.05, f"{ex_b}\nOGs",
          ha="center", va="center", fontsize=11, fontweight="bold",
          color=STRESS_COLORS[ex_b])

# Counts in regions
ax_s.text(-offset/2 - 0.7, -0.65, f"{len(only_a):,}",
          ha="center", va="center", fontsize=9, color="#555555")
ax_s.text(0, -0.05, f"{len(inter):,}",
          ha="center", va="center", fontsize=10, fontweight="bold", color="#333333")
ax_s.text(offset/2 + 0.7, -0.65, f"{len(only_b):,}",
          ha="center", va="center", fontsize=9, color="#555555")

# Bracket labels above circles
ax_s.annotate("", xy=(-offset/2 - r, 1.7), xytext=(-offset/2 + r, 1.7),
              arrowprops=dict(arrowstyle="|-|", color=STRESS_COLORS[ex_a], lw=1.5))
ax_s.text(-offset/2, 1.9, f"|A| = {len(set_a):,}",
          ha="center", va="bottom", fontsize=8, color=STRESS_COLORS[ex_a], fontweight="bold")

ax_s.annotate("", xy=(offset/2 - r, 2.1), xytext=(offset/2 + r, 2.1),
              arrowprops=dict(arrowstyle="|-|", color=STRESS_COLORS[ex_b], lw=1.5))
ax_s.text(offset/2, 2.3, f"|B| = {len(set_b):,}",
          ha="center", va="bottom", fontsize=8, color=STRESS_COLORS[ex_b], fontweight="bold")

# Formula below
formula_y = -1.6
ax_s.text(0, formula_y,
          r"$OC(A, B) = \frac{|A \cap B|}{\min(|A|, |B|)}$"
          f"  =  "
          r"$\frac{" + f"{len(inter):,}" + r"}{" + f"{min_size:,}" + r"}$"
          f"  =  {oc_val:.2f}",
          ha="center", va="center", fontsize=12,
          bbox=dict(boxstyle="round,pad=0.4", facecolor="#f0f0f0", edgecolor="#999999", linewidth=0.5))

# Arrow pointing to intersection
ax_s.annotate("|A $\\cap$ B|", xy=(0, 0.5), xytext=(0, 1.2),
              ha="center", fontsize=8, color="#333333",
              arrowprops=dict(arrowstyle="->", color="#333333", lw=0.8))

ax_s.set_title("Overlap coefficient: shared stress-responsive orthogroups", fontsize=10, pad=15)

fig_s.savefig(os.path.join(OUT_DIR, "fig2_overlap_schematic.pdf"), dpi=300, bbox_inches="tight")
fig_s.savefig(os.path.join(OUT_DIR, "fig2_overlap_schematic.png"), dpi=300, bbox_inches="tight")
print("  Saved: fig2_overlap_schematic.pdf/.png")
plt.close(fig_s)


# ═══════════════════════════════════════════════════════════════════════════════
# PANEL Y continued: Full overlap coefficient heatmap
# ═══════════════════════════════════════════════════════════════════════════════

# Compute pairwise overlap coefficient: OC(A,B) = |A ∩ B| / min(|A|, |B|)
n = len(MAIN_STRESSES)
oc_matrix = np.zeros((n, n))
intersection_matrix = np.zeros((n, n), dtype=int)

for i in range(n):
    for j in range(n):
        a = stress_ogs[MAIN_STRESSES[i]]
        b = stress_ogs[MAIN_STRESSES[j]]
        inter = len(a & b)
        intersection_matrix[i, j] = inter
        if i == j:
            oc_matrix[i, j] = 1.0
        else:
            min_size = min(len(a), len(b))
            oc_matrix[i, j] = inter / min_size if min_size > 0 else 0

print(f"\n  Overlap coefficient matrix:")
for i, s in enumerate(MAIN_STRESSES):
    vals = " ".join(f"{oc_matrix[i, j]:.2f}" for j in range(n))
    print(f"    {s:>12s}: {vals}")

# ── Plot heatmap ──
print("\nGenerating overlap heatmap...")

fig_oc, ax_oc = plt.subplots(figsize=(7, 6))

# Mask diagonal for cleaner look
mask = np.eye(n, dtype=bool)
oc_display = oc_matrix.copy()

im = ax_oc.imshow(oc_display, cmap="YlOrRd", vmin=0, vmax=1, interpolation="nearest")

# Annotate cells
for i in range(n):
    for j in range(n):
        val = oc_matrix[i, j]
        if i == j:
            # Diagonal: show OG count
            ax_oc.text(j, i, f"{len(stress_ogs[MAIN_STRESSES[i]]):,}",
                       ha="center", va="center", fontsize=6,
                       fontweight="bold", color="white")
        else:
            text_c = "white" if val > 0.6 else "black"
            ax_oc.text(j, i, f"{val:.2f}", ha="center", va="center",
                       fontsize=6, color=text_c)

# Cell borders
for i in range(n):
    for j in range(n):
        ax_oc.add_patch(plt.Rectangle(
            (j - 0.5, i - 0.5), 1, 1,
            facecolor="none", edgecolor="#cccccc", linewidth=0.3,
        ))

ax_oc.set_xticks(range(n))
ax_oc.set_xticklabels(MAIN_STRESSES, rotation=45, ha="right", fontsize=8)
ax_oc.set_yticks(range(n))
ax_oc.set_yticklabels(MAIN_STRESSES, fontsize=8)

# Color the tick labels
for i, label in enumerate(ax_oc.get_xticklabels()):
    label.set_color(STRESS_COLORS[MAIN_STRESSES[i]])
    label.set_fontweight("bold")
for i, label in enumerate(ax_oc.get_yticklabels()):
    label.set_color(STRESS_COLORS[MAIN_STRESSES[i]])
    label.set_fontweight("bold")

ax_oc.set_title(
    "Overlap coefficient of stress-responsive orthogroups\n"
    "OC(A,B) = |A $\\cap$ B| / min(|A|, |B|); diagonal = OG count",
    fontsize=10, pad=10,
)

cbar = plt.colorbar(im, ax=ax_oc, shrink=0.7, pad=0.02)
cbar.set_label("Overlap coefficient", fontsize=9)
cbar.ax.tick_params(labelsize=7)

plt.tight_layout()
fig_oc.savefig(os.path.join(OUT_DIR, "fig2_overlap_coefficient.pdf"), dpi=300, bbox_inches="tight")
fig_oc.savefig(os.path.join(OUT_DIR, "fig2_overlap_coefficient.png"), dpi=300, bbox_inches="tight")
print("  Saved: fig2_overlap_coefficient.pdf/.png")

# Save data
oc_df = pd.DataFrame(oc_matrix, index=MAIN_STRESSES, columns=MAIN_STRESSES)
oc_df.to_csv(os.path.join(OUT_DIR, "fig2_overlap_coefficient_data.csv"))
inter_df = pd.DataFrame(intersection_matrix, index=MAIN_STRESSES, columns=MAIN_STRESSES)
inter_df.to_csv(os.path.join(OUT_DIR, "fig2_overlap_intersection_data.csv"))
print("  Saved: fig2_overlap_coefficient_data.csv, fig2_overlap_intersection_data.csv")

print("\n" + "=" * 60)
print("Done!")
print("=" * 60)
