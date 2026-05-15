#!/usr/bin/env python3
"""
Figure 1C: Marker gene validation swarmplot
Extracts experimentally verified stress-response genes from TAIR GO annotations,
maps them via OrthoFinder orthogroups to orthologs per species,
then pulls log2FC from the kingdom-wide DEG table and plots as a swarmplot.
"""

import pandas as pd
import numpy as np
import re
import csv
from collections import defaultdict

# ── 1. GO-verified canonical marker genes ────────────────────────────────────

GO_FILE = "/Users/vjx443/Downloads/ATH_GO_tmp/ATH_GO_GOSLIM.txt"

# Curated canonical markers per stress
CURATED_MARKERS = {
    "Heat": {
        "HSP101": "AT1G74310", "HSP70": "AT3G12580", "HSP18.2": "AT5G59720",
        "HSP21": "AT4G27670", "HSFA2": "AT2G26150",
    },
    "Cold": {
        "CBF1": "AT4G25490", "CBF3": "AT4G25480", "COR15A": "AT2G42540",
        "COR47": "AT1G20440", "KIN1": "AT5G15960",
    },
    "Drought": {
        "RD29A": "AT5G52310", "RD29B": "AT5G52300", "RAB18": "AT5G66400",
        "P5CS1": "AT2G39800", "DREB2A": "AT5G05410",
    },
    "Salt": {
        "SOS1": "AT2G01980", "NHX1": "AT5G27150", "HKT1": "AT4G10310",
    },
    "High light": {
        "APX2": "AT3G09640", "ELIP1": "AT3G22840", "ELIP2": "AT4G14690",
        "ZAT10": "AT1G27730",
    },
    "Pathogen": {
        "PR1": "AT2G14610", "PR2": "AT3G57260", "PR5": "AT1G75040",
        "PDF1.2": "AT5G44420", "WRKY33": "AT2G38470",
    },
    "Flooding": {
        "ADH1": "AT1G77120", "PDC1": "AT4G33070", "SUS1": "AT5G20830",
        "HRE1": "AT1G72360",
    },
    "Heavy metal": {
        "MT2A": "AT3G09390", "PCS1": "AT5G44070", "NRAMP3": "AT2G23150",
    },
    "Herbivory": {
        "VSP2": "AT5G24770", "LOX2": "AT3G45140", "MYC2": "AT1G32640",
    },
}

# GO terms that validate each stress category (broad enough to capture canonical markers)
GO_TO_STRESS = {
    # Heat
    "response to heat": "Heat", "cellular response to heat": "Heat",
    # Cold
    "response to cold": "Cold", "cellular response to cold": "Cold",
    # Drought
    "response to water deprivation": "Drought",
    "cellular response to water deprivation": "Drought",
    # Salt
    "response to salt stress": "Salt", "cellular response to salt stress": "Salt",
    # High light
    "response to high light intensity": "High light",
    "cellular response to high light intensity": "High light",
    "response to oxidative stress": "High light",
    # Pathogen
    "defense response to bacterium": "Pathogen",
    "defense response to fungus": "Pathogen",
    "defense response to oomycetes": "Pathogen",
    "defense response to virus": "Pathogen",
    "systemic acquired resistance": "Pathogen",
    # Flooding / hypoxia
    "response to hypoxia": "Flooding", "cellular response to hypoxia": "Flooding",
    # Nitrogen
    "cellular response to nitrogen starvation": "Nitrogen",
    "response to nitrate": "Nitrogen",
    "nitrate assimilation": "Nitrogen",
    # Heavy metal
    "response to cadmium ion": "Heavy metal",
    "response to zinc ion": "Heavy metal",
    # Herbivory / wounding
    "response to wounding": "Herbivory",
    "defense response to insect": "Herbivory",
    "response to insect": "Herbivory",
    "response to jasmonic acid": "Herbivory",
}

# Only keep experimentally verified evidence codes
EXPERIMENTAL_EVIDENCE = {"IMP", "IDA", "IEP", "IGI"}

print("Extracting GO-verified canonical stress markers...")

# Parse GO file: {AGI_locus: set of stresses it is experimentally annotated to}
go_verified_stresses = defaultdict(set)

with open(GO_FILE, "r") as f:
    for line in f:
        if line.startswith("!"):
            continue
        fields = line.strip().split("\t")
        if len(fields) < 10:
            continue
        locus = fields[0]
        go_term_name = fields[4]
        evidence = fields[9]

        if evidence in EXPERIMENTAL_EVIDENCE and go_term_name in GO_TO_STRESS:
            go_verified_stresses[locus].add(GO_TO_STRESS[go_term_name])

# Intersect: keep only curated markers that have GO experimental evidence for their stress
MARKERS = {}  # {stress: {name: AGI}} -- final verified set
marker_lookup = {}  # {AGI: (stress, name)}

print("\nCurated markers with GO experimental verification:")
for stress, genes in CURATED_MARKERS.items():
    for name, agi in genes.items():
        verified_stresses = go_verified_stresses.get(agi, set())
        if stress in verified_stresses:
            MARKERS.setdefault(stress, {})[name] = agi
            marker_lookup[agi] = (stress, name)
            print(f"  {stress:15s} {name:10s} {agi}  [GO-verified]")
        else:
            print(f"  {stress:15s} {name:10s} {agi}  [not in GO for this stress]")

print(f"\nTotal GO-verified markers: {len(marker_lookup)}")

# ── 2. Parse OrthoFinder orthogroups ─────────────────────────────────────────

print("Parsing orthogroups...")

ORTHOGROUPS_FILE = "/Users/vjx443/Downloads/deepcre_local/Orthogroups.txt"

# For each marker AGI, find its orthogroup and collect all member genes
# AGIs in the file have transcript suffix (.1, .2), so match on locus ID prefix

marker_to_og = {}        # {AGI_locus: OG_id}
og_to_genes = {}         # {OG_id: [gene_id, ...]}
markers_found = set()

with open(ORTHOGROUPS_FILE, "r") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        og_id, genes_str = line.split(": ", 1)
        gene_ids = genes_str.split()

        # Check if any gene in this OG matches an Arabidopsis marker
        og_has_marker = False
        for gid in gene_ids:
            # Strip transcript suffix to get locus ID (AT1G74310.1 -> AT1G74310)
            if gid.startswith("AT"):
                locus = re.sub(r"\.\d+$", "", gid)
                if locus in marker_lookup:
                    marker_to_og[locus] = og_id
                    markers_found.add(locus)
                    og_has_marker = True

        if og_has_marker:
            og_to_genes[og_id] = gene_ids

print(f"Markers mapped to orthogroups: {len(markers_found)}/{len(marker_lookup)}")

missing = set(marker_lookup.keys()) - markers_found
if missing:
    print(f"  Missing from orthogroups: {[marker_lookup[m] for m in missing]}")

# ── 3. Build gene → (stress, marker_name) lookup for all orthologs ───────────

gene_locus_to_marker = defaultdict(list)

for agi_locus, og_id in marker_to_og.items():
    stress, name = marker_lookup[agi_locus]
    for gid in og_to_genes[og_id]:
        locus = re.sub(r"\.\d+$", "", gid)
        entry = (stress, name)
        if entry not in gene_locus_to_marker[locus]:
            gene_locus_to_marker[locus].append(entry)

print(f"Total ortholog gene IDs to search in DEG table: {len(gene_locus_to_marker)}")

# ── 4. Read DEG table and extract matching rows ─────────────────────────────

print("Reading DEG table (this may take a few minutes for 13.8M rows)...")

DEG_FILE = "/Users/vjx443/Downloads/kingdom_stress_dict v3.csv"

# Read in chunks to manage memory
matched_rows = []
chunk_size = 500_000
chunks_processed = 0

for chunk in pd.read_csv(DEG_FILE, chunksize=chunk_size, low_memory=False):
    chunks_processed += 1
    if chunks_processed % 5 == 0:
        print(f"  Processed {chunks_processed * chunk_size / 1e6:.1f}M rows...")

    # Strip transcript suffix from gene column for matching
    chunk["gene_locus"] = chunk["gene"].str.replace(r"\.\d+$", "", regex=True)

    # Filter to genes that are orthologs of our markers
    mask = chunk["gene_locus"].isin(gene_locus_to_marker)
    if mask.any():
        matched_rows.append(chunk[mask].copy())

print(f"  Done. Total chunks: {chunks_processed}")

deg_matched = pd.concat(matched_rows, ignore_index=True)
print(f"Matched DEG rows: {len(deg_matched):,}")

# ── 5. Annotate matched rows with marker info ───────────────────────────────

deg_matched["marker_info"] = deg_matched["gene_locus"].apply(
    lambda x: gene_locus_to_marker.get(x, [])
)

deg_exploded = deg_matched.explode("marker_info")
deg_exploded["marker_stress"] = deg_exploded["marker_info"].apply(lambda x: x[0] if isinstance(x, tuple) else None)
deg_exploded["marker_name"] = deg_exploded["marker_info"].apply(lambda x: x[1] if isinstance(x, tuple) else None)
deg_exploded = deg_exploded.dropna(subset=["marker_stress"])

print(f"Annotated rows after explode: {len(deg_exploded):,}")

# ── 6. Filter: keep only rows where DEG stress matches marker stress ─────────

# Normalize stress names for matching
stress_normalize = {
    "High Light": "High light",
    "high light": "High light",
}

deg_exploded["stress_norm"] = deg_exploded["stress"].replace(stress_normalize)
deg_exploded = deg_exploded[deg_exploded["stress_norm"] == deg_exploded["marker_stress"]]

print(f"Rows after stress-matching filter: {len(deg_exploded):,}")

# ── 7. Prepare data for swarmplot ────────────────────────────────────────────

# For the swarmplot, we want one point per ortholog per species per experiment.
# To avoid overplotting, summarize: one value per species × marker gene × experiment.
# (A species may have multiple orthologs in the same OG; take the max |log2FC| one.)

plot_data = (
    deg_exploded
    .groupby(["species", "marker_stress", "marker_name", "experiment"])
    .agg(log2FC=("log2FC", lambda x: x.loc[x.abs().idxmax()]))
    .reset_index()
)

print(f"\nPlot data points: {len(plot_data):,}")
print(f"Species with data: {plot_data['species'].nunique()}")
print(f"Stresses with data: {plot_data['marker_stress'].nunique()}")

# ── 8. Save outputs ─────────────────────────────────────────────────────────

OUT_DIR = "/Users/vjx443/Library/CloudStorage/GoogleDrive-mutwil@plant.tools/My Drive/Projects/2026_KingdomStress/CC_Fig 1 data"

plot_data.to_csv(f"{OUT_DIR}/fig1c_swarm_data.csv", index=False)
print(f"Saved: fig1c_swarm_data.csv")

# ── 9. Plot swarmplot ────────────────────────────────────────────────────────

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

# Stress order (only stresses that have GO-verified markers)
STRESS_DISPLAY_ORDER = [
    "Heat", "Cold", "Drought", "Salt", "High light",
    "Pathogen", "Flooding", "Heavy metal", "Herbivory",
]
stress_order = [s for s in STRESS_DISPLAY_ORDER if s in plot_data["marker_stress"].values]

# Color palette: warm for abiotic, cool for biotic
stress_colors = {
    "Heat":        "#d62728",
    "Cold":        "#1f77b4",
    "Drought":     "#ff7f0e",
    "Salt":        "#8c564b",
    "High light":  "#ffbb33",
    "Pathogen":    "#2ca02c",
    "Flooding":    "#17becf",
    "Heavy metal": "#7f7f7f",
    "Herbivory":   "#bcbd22",
}

from matplotlib.gridspec import GridSpec

fig = plt.figure(figsize=(6, 5.5))
gs = GridSpec(3, 1, height_ratios=[1, 0.06, 0.06], hspace=0.05, figure=fig)

ax = fig.add_subplot(gs[0])
ax_n = fig.add_subplot(gs[1], sharex=ax)
ax_spp = fig.add_subplot(gs[2], sharex=ax)

# ── Main swarmplot ──
sns.stripplot(
    data=plot_data,
    x="marker_stress",
    y="log2FC",
    order=stress_order,
    hue="marker_stress",
    hue_order=stress_order,
    palette=stress_colors,
    size=1.2,
    alpha=0.4,
    jitter=0.3,
    legend=False,
    ax=ax,
    zorder=2,
)

bp = ax.boxplot(
    [plot_data[plot_data["marker_stress"] == s]["log2FC"].values for s in stress_order],
    positions=range(len(stress_order)),
    widths=0.4,
    showfliers=False,
    patch_artist=True,
    zorder=3,
)
for i, (box, stress) in enumerate(zip(bp["boxes"], stress_order)):
    color = stress_colors[stress]
    box.set_facecolor(color)
    box.set_alpha(0.35)
    box.set_edgecolor(color)
    box.set_linewidth(1)
for element in ["whiskers", "caps"]:
    for i, line in enumerate(bp[element]):
        line.set_color(stress_colors[stress_order[i // 2]])
        line.set_linewidth(0.8)
for line in bp["medians"]:
    line.set_color("black")
    line.set_linewidth(1.2)

ax.axhline(0, color="grey", linewidth=0.5, linestyle="--", zorder=1)
ax.set_xlabel("")
ax.set_ylabel("log$_2$FC", fontsize=11)
ax.set_title("Canonical stress markers:\northolog response across species", fontsize=10, pad=10)
ax.tick_params(axis="x", labelbottom=False, bottom=False)
sns.despine(ax=ax, bottom=True)

# ── Compute stats per stress ──
n_vals = []
spp_vals = []
colors_list = []
for stress in stress_order:
    subset = plot_data[plot_data["marker_stress"] == stress]
    n_vals.append(len(subset))
    spp_vals.append(subset["species"].nunique())
    colors_list.append(stress_colors[stress])

# ── n heatmap row ──
n_arr = np.array(n_vals).reshape(1, -1)
ax_n.imshow(n_arr, aspect="auto", cmap="Greys", alpha=0.3)
for i, val in enumerate(n_vals):
    ax_n.text(i, 0, f"{val:,}", ha="center", va="center", fontsize=7, fontweight="bold")
ax_n.set_yticks([0])
ax_n.set_yticklabels(["# genes"], fontsize=8)
ax_n.tick_params(axis="x", labelbottom=False, bottom=False)
ax_n.tick_params(axis="y", length=0)
for spine in ax_n.spines.values():
    spine.set_visible(False)

# ── spp heatmap row ──
spp_arr = np.array(spp_vals).reshape(1, -1)
ax_spp.imshow(spp_arr, aspect="auto", cmap="Greys", alpha=0.3)
for i, val in enumerate(spp_vals):
    ax_spp.text(i, 0, str(val), ha="center", va="center", fontsize=7, fontweight="bold")
ax_spp.set_yticks([0])
ax_spp.set_yticklabels(["Species"], fontsize=8)
ax_spp.set_xticks(range(len(stress_order)))
ax_spp.set_xticklabels(stress_order, rotation=45, ha="right", fontsize=8)
ax_spp.tick_params(axis="y", length=0)
for spine in ax_spp.spines.values():
    spine.set_visible(False)

fig.savefig(f"{OUT_DIR}/fig1c_swarmplot.pdf", dpi=300, bbox_inches="tight")
fig.savefig(f"{OUT_DIR}/fig1c_swarmplot.png", dpi=300, bbox_inches="tight")
print(f"Saved: fig1c_swarmplot.pdf, fig1c_swarmplot.png")

print("\nDone!")
