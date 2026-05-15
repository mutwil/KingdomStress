#!/usr/bin/env python3
"""
Figure 1A: Species phylogeny coupled with stress experiment count heatmap.
Draws a cladogram of the 36 species and a heatmap of experiment group counts
per species per stress condition, ordered by phylogenetic position.
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
from collections import OrderedDict

# ── 1. Define species phylogeny as nested tuples ─────────────────────────────
# Each leaf is a string; each internal node is a tuple of children.
# Simplified to family/subfamily-level topology (APG IV).

# ── Monocots (Poaceae) ──
# Oryzoideae
oryzoideae = ("Oryza_sativa", "Oryza_sativa_Japonica_Group")
# Pooideae: BEP clade with Oryzoideae
pooideae = ("Brachypodium_distachyon", ("Hordeum_vulgare", "Triticum_aestivum"))
bep = (oryzoideae, pooideae)
# Panicoideae: PACMAD clade
panicoideae = (
    ("Zea_mays", "Sorghum_bicolor"),
    ("Setaria_italica", ("Cenchrus_americanus", "Panicum_virgatum")),
)
monocots = (bep, panicoideae)

# ── Eudicots ──
# Fabaceae (Papilionoideae)
fabaceae = (
    ("Glycine_max", "Phaseolus_vulgaris"),
    (("Medicago_sativa", "Medicago_truncatula"),
     ("Cicer_arietinum", "Arachis_hypogaea")),
)
# Rosales + Cucurbitales (nitrogen-fixing clade with Fabaceae)
rosales_cucurbitales = ("Prunus_persica", "Cucumis_sativus")
fabids = (fabaceae, rosales_cucurbitales)

# Brassicaceae + Malvaceae (Malvids)
brassicaceae = ("Arabidopsis_thaliana", ("Brassica_napus", "Brassica_oleracea"))
malvids = (brassicaceae, "Gossypium_hirsutum")

# Vitis is basal rosid (outgroup to fabids + malvids)
rosids = ("Vitis_vinifera", (fabids, malvids))

# Solanaceae (Lamiids / core asterids)
solanaceae = (
    ("Solanum_lycopersicum", "Solanum_tuberosum"),
    ("Capsicum_annuum", "Nicotiana_tabacum"),
)
# Asteraceae + Amaranthaceae (Campanulids)
campanulids = ("Lactuca_sativa", "Spinacia_oleracea")
# Camellia = Ericales (basal asterid)
asterids = ("Camellia_sinensis", (solanaceae, campanulids))

# Core eudicots
eudicots = (rosids, asterids)

# Angiosperms = monocots + eudicots
angiosperms = (monocots, eudicots)

# ── Gymnosperms ──
seed_plants = (angiosperms, "Picea_abies")

# ── Lycophytes ──
vascular = (seed_plants, "Selaginella_moellendorffii")

# ── Bryophytes ──
bryophytes = ("Physcomitrium_patens", "Marchantia_polymorpha")
land_plants = (vascular, bryophytes)

# ── Streptophyte algae ──
streptophytes = (land_plants, "Klebsormidium_nitens")

# ── Full tree ──
TREE = (streptophytes, "Chlamydomonas_reinhardtii")

# ── 2. Flatten tree to get leaf order and compute coordinates ────────────────

def get_leaves(node):
    """Return ordered list of leaves."""
    if isinstance(node, str):
        return [node]
    leaves = []
    for child in node:
        leaves.extend(get_leaves(child))
    return leaves

def get_max_depth(node, depth=0):
    """Get the maximum depth of the tree."""
    if isinstance(node, str):
        return depth
    return max(get_max_depth(child, depth + 1) for child in node)

def compute_coords(node, x=0, y_counter=[0], max_depth=None):
    """
    Compute (x, y) for each node. Leaves get incrementing y values.
    All leaves are extended to max_depth so branches reach the species names.
    Returns: {node_id: (x, y)}, list of edges [(x1,y1,x2,y2)]
    """
    coords = {}
    edges = []

    if isinstance(node, str):
        y = y_counter[0]
        y_counter[0] += 1
        leaf_x = max_depth if max_depth is not None else x
        coords[id(node)] = (leaf_x, y)
        coords[node] = (leaf_x, y)
        return coords, edges

    child_ys = []
    for child in node:
        child_coords, child_edges = compute_coords(child, x + 1, y_counter, max_depth)
        coords.update(child_coords)
        edges.extend(child_edges)

        if isinstance(child, str):
            child_x, child_y = coords[child]
        else:
            child_x, child_y = coords[id(child)]
        child_ys.append(child_y)

        # Horizontal line from parent x to child x
        edges.append((x, child_y, child_x, child_y))

    # Parent y = midpoint of children
    parent_y = (min(child_ys) + max(child_ys)) / 2
    coords[id(node)] = (x, parent_y)

    # Vertical line connecting children
    edges.append((x, min(child_ys), x, max(child_ys)))

    return coords, edges


# ── 3. Clade coloring ───────────────────────────────────────────────────────

CLADE_COLORS = {
    "Monocot":     "#a8c5e2",
    "Dicot":       "#d4d4d4",
    "Gymnosperm":  "#b8d4a8",
    "Lycophyte":   "#c8e0c0",
    "Bryophyte":   "#f0c8b0",
    "Charophyte":  "#f5e6b8",
    "Chlorophyte": "#f5e6b8",
}

def get_clade(species):
    monocot_spp = get_leaves(monocots)
    dicot_spp = get_leaves(eudicots)
    bryophyte_spp = get_leaves(bryophytes)
    if species in monocot_spp:
        return "Monocot"
    elif species in dicot_spp:
        return "Dicot"
    elif species == "Picea_abies":
        return "Gymnosperm"
    elif species == "Selaginella_moellendorffii":
        return "Lycophyte"
    elif species in bryophyte_spp:
        return "Bryophyte"
    elif species == "Klebsormidium_nitens":
        return "Charophyte"
    elif species == "Chlamydomonas_reinhardtii":
        return "Chlorophyte"
    return "Unknown"


# ── 4. Count experiment groups from DEG table ────────────────────────────────

print("Counting experiment groups per species x stress...")

DEG_FILE = "/Users/vjx443/Downloads/kingdom_stress_dict v3.csv"
cols = ["species", "stress", "bioproject", "experiment"]
chunks = pd.read_csv(DEG_FILE, usecols=cols, chunksize=500_000, low_memory=False)

all_data = []
for chunk in chunks:
    all_data.append(chunk.drop_duplicates())

df = pd.concat(all_data).drop_duplicates()

# Normalize stress names
stress_normalize = {
    "High Light": "High light",
}
df["stress"] = df["stress"].replace(stress_normalize)

# Count unique control/treatment pairs: species + bioproject + experiment
df["bio_exp"] = df["species"] + df["bioproject"] + df["experiment"]
counts = df.groupby(["species", "stress"])["bio_exp"].nunique().reset_index()
counts.columns = ["species", "stress", "n_groups"]

# Main stresses — same set and order as the Fig 1C boxplot
MAIN_STRESSES = ["Heat", "Cold", "Drought", "Salt", "High light",
                 "Pathogen", "Flooding", "Heavy metal", "Herbivory"]

counts = counts[counts["stress"].isin(MAIN_STRESSES)]

# ── 5. Build heatmap matrix in phylogenetic order ────────────────────────────

leaf_order = get_leaves(TREE)

pivot = counts.pivot_table(index="species", columns="stress", values="n_groups", fill_value=0)
# Reindex to phylogenetic order, add missing species as zeros
pivot = pivot.reindex(index=leaf_order, columns=MAIN_STRESSES, fill_value=0)

print(f"Heatmap: {pivot.shape[0]} species x {pivot.shape[1]} stresses")

# ── 6. Draw figure ───────────────────────────────────────────────────────────

max_depth = get_max_depth(TREE)
y_counter = [0]
coords, edges = compute_coords(TREE, x=0, y_counter=y_counter, max_depth=max_depth)

n_species = len(leaf_order)
n_stresses = len(MAIN_STRESSES)

fig = plt.figure(figsize=(12, max(8, n_species * 0.28)))
gs = GridSpec(1, 3, width_ratios=[0.2, 0.35, 0.45], wspace=0.0, figure=fig)

ax_tree = fig.add_subplot(gs[0])
ax_labels = fig.add_subplot(gs[1], sharey=ax_tree)
ax_heat = fig.add_subplot(gs[2], sharey=ax_tree)

# ── Draw tree ──
max_x = max(c[0] for c in coords.values())

for (x1, y1, x2, y2) in edges:
    ax_tree.plot([x1, x2], [y1, y2], color="black", linewidth=0.8, solid_capstyle="round")

# Clade background shading
clade_ranges = {}
for i, sp in enumerate(leaf_order):
    clade = get_clade(sp)
    if clade not in clade_ranges:
        clade_ranges[clade] = [i, i]
    else:
        clade_ranges[clade][1] = i

for clade, (ymin, ymax) in clade_ranges.items():
    color = CLADE_COLORS.get(clade, "#ffffff")
    for a in [ax_tree, ax_labels, ax_heat]:
        a.axhspan(ymin - 0.5, ymax + 0.5, color=color, alpha=0.4, zorder=0)

# Species labels in dedicated middle panel
DISPLAY_OVERRIDES = {
    "Oryza_sativa": "Oryza sativa (indica)",
    "Oryza_sativa_Japonica_Group": "Oryza sativa (japonica)",
}
for i, sp in enumerate(leaf_order):
    display_name = DISPLAY_OVERRIDES.get(sp, sp.replace("_", " "))
    ax_labels.text(0.05, i, display_name, va="center", ha="left",
                   fontsize=7, fontstyle="italic")

ax_tree.set_xlim(-0.5, max_x + 0.5)
ax_tree.set_ylim(-0.8, n_species - 0.2)
ax_tree.invert_yaxis()
ax_tree.axis("off")

ax_labels.set_xlim(0, 1)
ax_labels.axis("off")

# ── Add Total column ──
pivot["Total"] = pivot[MAIN_STRESSES].sum(axis=1)
ALL_COLS = MAIN_STRESSES + ["Total"]
n_cols = len(ALL_COLS)

# ── Draw heatmap ──
heatmap_data = pivot[ALL_COLS].values.astype(float)
# Separate vmax for stress columns (exclude Total for color scaling)
stress_data = pivot[MAIN_STRESSES].values.astype(float)
vmax = np.percentile(stress_data[stress_data > 0], 95) if (stress_data > 0).any() else 1

im = ax_heat.imshow(
    heatmap_data[:, :n_stresses],  # only stress columns for color
    aspect="auto",
    cmap="Blues",
    vmin=0,
    vmax=vmax,
    interpolation="nearest",
    extent=[-0.5, n_stresses - 0.5, n_species - 0.5, -0.5],
)

# Draw Total column separately with different color
total_data = heatmap_data[:, -1:]
total_vmax = np.percentile(total_data[total_data > 0], 95) if (total_data > 0).any() else 1
for i in range(n_species):
    val = int(total_data[i, 0])
    intensity = min(val / total_vmax, 1.0) if total_vmax > 0 else 0
    color = plt.cm.Oranges(0.15 + intensity * 0.7)
    ax_heat.add_patch(plt.Rectangle((n_stresses - 0.5, i - 0.5), 1, 1,
                      facecolor=color, edgecolor="#cccccc", linewidth=0.3, zorder=1))
    text_color = "white" if intensity > 0.5 else "black"
    ax_heat.text(n_stresses, i, str(val), ha="center", va="center",
                fontsize=6, color=text_color, fontweight="bold", zorder=2)

# Cell borders for stress columns
for i in range(n_species):
    for j in range(n_stresses):
        ax_heat.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1,
                          facecolor="none", edgecolor="#cccccc", linewidth=0.3, zorder=1))

# Annotate cells with counts
for i in range(n_species):
    for j in range(n_stresses):
        val = int(heatmap_data[i, j])
        if val > 0:
            text_color = "white" if val > vmax * 0.6 else "black"
            ax_heat.text(j, i, str(val), ha="center", va="center",
                        fontsize=6, color=text_color, fontweight="bold", zorder=2)

# Horizontal guide lines from labels to heatmap
for i in range(n_species):
    ax_labels.plot([0.95, 1.0], [i, i], color="#dddddd", linewidth=0.4,
                   transform=ax_labels.get_yaxis_transform(), clip_on=False)

# Axes
ax_heat.set_xlim(-0.5, n_cols - 0.5)
ax_heat.set_xticks(range(n_cols))
ax_heat.set_xticklabels(ALL_COLS, rotation=45, ha="right", fontsize=8)
ax_heat.xaxis.set_ticks_position("bottom")
ax_heat.set_yticks([])
ax_heat.set_xlabel("Stress condition", fontsize=10, labelpad=8)

# Colorbar
cbar = plt.colorbar(im, ax=ax_heat, shrink=0.3, pad=0.02, aspect=15)
cbar.set_label("# control/treatment pairs", fontsize=8)
cbar.ax.tick_params(labelsize=7)

# Title
fig.suptitle("Species phylogeny and stress experiment coverage",
             fontsize=12, y=0.98)

# Legend for clades
legend_patches = [mpatches.Patch(color=CLADE_COLORS[c], alpha=0.5, label=c)
                  for c in ["Monocot", "Dicot", "Gymnosperm", "Lycophyte",
                            "Bryophyte", "Charophyte", "Chlorophyte"]
                  if c in [get_clade(sp) for sp in leaf_order]]
ax_tree.legend(handles=legend_patches, loc="upper left", fontsize=6,
               frameon=True, facecolor="white", edgecolor="grey",
               title="Clade", title_fontsize=7)

plt.subplots_adjust(left=0.02, right=0.95, top=0.96, bottom=0.08)

OUT_DIR = "/Users/vjx443/Library/CloudStorage/GoogleDrive-mutwil@plant.tools/My Drive/Projects/2026_KingdomStress/CC_Fig 1 data"
fig.savefig(f"{OUT_DIR}/fig1a_phylo_heatmap.pdf", dpi=300, bbox_inches="tight")
fig.savefig(f"{OUT_DIR}/fig1a_phylo_heatmap.png", dpi=300, bbox_inches="tight")
print(f"Saved: fig1a_phylo_heatmap.pdf, fig1a_phylo_heatmap.png")

# Save underlying data
pivot.to_csv(f"{OUT_DIR}/fig1a_experiment_counts.csv")
print(f"Saved: fig1a_experiment_counts.csv")

print("\nDone!")
