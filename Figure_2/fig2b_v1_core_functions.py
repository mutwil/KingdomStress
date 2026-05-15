#!/usr/bin/env python3
"""
Figure 2B option 1: Functional breakdown of core stress responders.
Bar chart showing which biological functions are represented among
OGs that respond to >= 7 of 10 stress types.
"""

import pandas as pd
import numpy as np
import re
import os
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── Paths ──
OG_FILE = "/Users/vjx443/Library/CloudStorage/GoogleDrive-mutwil@plant.tools/My Drive/Projects/2026_KingdomStress/Orthogroups.txt"
GO_FILE = "/Users/vjx443/Downloads/ATH_GO_tmp/ATH_GO_GOSLIM.txt"
OUT_DIR = "/Users/vjx443/Library/CloudStorage/GoogleDrive-mutwil@plant.tools/" \
          "My Drive/Projects/2026_KingdomStress/CC-Fig 2 data"

MAIN_STRESSES = [
    "Heat", "Cold", "Drought", "Salt", "High light",
    "Pathogen", "Flooding", "Heavy metal", "Herbivory",
]
CORE_THRESHOLD = 7

# ── Functional category definitions ──
_categories = {
    "Transcription regulation": [
        "regulation of DNA-templated transcription",
        "regulation of transcription by RNA polymerase II",
        "positive regulation of DNA-templated transcription",
        "negative regulation of DNA-templated transcription",
        "regulation of gene expression",
    ],
    "Protein folding & chaperones": [
        "protein folding", "chaperone-mediated protein folding",
        "cellular response to heat", "response to heat",
        "response to unfolded protein",
    ],
    "Protein degradation": [
        "ubiquitin-dependent protein catabolic process",
        "protein ubiquitination", "proteolysis",
        "proteasome-mediated ubiquitin-dependent protein catabolic process",
        "SCF-dependent proteasomal ubiquitin-dependent protein catabolic process",
        "protein catabolic process",
    ],
    "Kinase & phosphorylation": [
        "phosphorylation", "protein phosphorylation",
        "protein autophosphorylation", "dephosphorylation",
        "protein dephosphorylation",
    ],
    "Hormone signaling": [
        "response to abscisic acid", "abscisic acid-activated signaling pathway",
        "negative regulation of abscisic acid-activated signaling pathway",
        "response to auxin", "auxin-activated signaling pathway",
        "response to ethylene", "ethylene-activated signaling pathway",
        "response to gibberellin",
        "response to cytokinin", "cytokinin-activated signaling pathway",
        "brassinosteroid mediated signaling pathway",
    ],
    "JA/SA defense signaling": [
        "response to jasmonic acid", "jasmonic acid mediated signaling pathway",
        "response to salicylic acid", "salicylic acid mediated signaling pathway",
        "systemic acquired resistance",
    ],
    "Calcium signaling": [
        "calcium-mediated signaling", "intracellular calcium ion homeostasis",
        "calcium ion transport",
    ],
    "ROS & redox": [
        "response to oxidative stress", "response to hydrogen peroxide",
        "cellular oxidant detoxification", "hydrogen peroxide catabolic process",
        "superoxide metabolic process",
    ],
    "Defense & immunity": [
        "defense response to bacterium", "defense response to fungus",
        "defense response to oomycetes", "defense response",
        "defense response to other organism", "defense response to virus",
        "plant-type hypersensitive response", "immune response",
        "response to bacterium", "response to fungus",
        "regulation of defense response", "defense response to insect",
    ],
    "Osmotic & salt stress": [
        "response to salt stress", "response to osmotic stress",
        "hyperosmotic salinity response", "response to water deprivation",
        "cellular response to water deprivation",
    ],
    "Cold stress": [
        "response to cold", "cold acclimation",
        "cellular response to cold",
    ],
    "Hypoxia & flooding": [
        "cellular response to hypoxia", "response to hypoxia",
        "anaerobic respiration", "fermentation",
    ],
    "Light & UV response": [
        "response to light stimulus", "response to UV-B",
        "response to red light", "response to blue light",
        "response to far red light",
        "photomorphogenesis", "response to high light intensity",
    ],
    "Wounding & herbivory": [
        "response to wounding", "response to insect",
        "response to herbivore",
    ],
    "Membrane transport": [
        "transmembrane transport", "ion transmembrane transport",
        "ion transport", "potassium ion transport",
        "proton transmembrane transport",
    ],
    "Cell wall modification": [
        "plant-type cell wall modification",
        "plant-type secondary cell wall biogenesis",
        "plant-type cell wall organization",
        "cell wall organization",
        "cellulose biosynthetic process",
    ],
    "Secondary metabolism": [
        "secondary metabolic process", "flavonoid biosynthetic process",
        "phenylpropanoid metabolic process",
        "anthocyanin-containing compound biosynthetic process",
        "glucosinolate biosynthetic process", "terpenoid biosynthetic process",
        "carotenoid biosynthetic process",
    ],
    "Stomatal regulation": [
        "regulation of stomatal movement", "stomatal closure",
        "regulation of stomatal closure",
    ],
    "Translation & ribosome": [
        "translation", "translational elongation",
        "cytoplasmic translation", "ribosome biogenesis",
    ],
    "Lipid metabolism": [
        "fatty acid metabolic process", "lipid metabolic process",
        "fatty acid biosynthetic process", "lipid biosynthetic process",
        "wax biosynthetic process", "cutin biosynthetic process",
    ],
    "DNA repair & chromatin": [
        "DNA repair", "chromatin remodeling",
        "epigenetic regulation of gene expression",
        "DNA damage response",
        "double-strand break repair via homologous recombination",
    ],
    "Circadian rhythm": [
        "circadian rhythm", "regulation of circadian rhythm",
        "photoperiodism, flowering",
    ],
}

GO_TERM_TO_CATEGORY = {}
for category, terms in _categories.items():
    for term in terms:
        GO_TERM_TO_CATEGORY[term] = category
FUNC_CATEGORIES = list(_categories.keys())

# ── 1. Parse orthogroups ──
print("Parsing orthogroups...")
og_ath_genes = defaultdict(set)
with open(OG_FILE) as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        og_id, genes_str = line.split(": ", 1)
        for gid in genes_str.split():
            locus = re.sub(r"\.\d+$", "", gid)
            if re.match(r"^AT[1-5CM]G\d{5}$", locus):
                og_ath_genes[og_id].add(locus)

# ── 2. Parse GO -> functional categories ──
print("Parsing GO annotations...")
ath_func_cats = defaultdict(set)
with open(GO_FILE) as f:
    for line in f:
        if line.startswith("!"):
            continue
        fields = line.strip().split("\t")
        if len(fields) < 10:
            continue
        locus = fields[0]
        go_term = fields[4]
        cat = fields[7] if len(fields) > 7 else ""
        if cat == "P" and go_term in GO_TERM_TO_CATEGORY:
            ath_func_cats[locus].add(GO_TERM_TO_CATEGORY[go_term])

# Map OG -> categories
og_func_cats = {}
for og_id, ath_genes in og_ath_genes.items():
    cats = set()
    for gene in ath_genes:
        cats.update(ath_func_cats.get(gene, set()))
    if cats:
        og_func_cats[og_id] = cats

print(f"  OGs with functional annotations: {len(og_func_cats):,}")

# ── 3. Identify core OGs ──
print("Identifying core responders...")
og_cache = os.path.join(OUT_DIR, "_cache_og_species_stress.csv")
og_ss_df = pd.read_csv(og_cache)

og_n_stresses = og_ss_df.groupby("og")["stress"].nunique().reset_index(name="n_stresses")
core_ogs = set(og_n_stresses[og_n_stresses["n_stresses"] >= CORE_THRESHOLD]["og"])
core_annotated = core_ogs & set(og_func_cats.keys())

print(f"  Core OGs (>= {CORE_THRESHOLD} stresses): {len(core_ogs):,}")
print(f"  Core OGs with annotations: {len(core_annotated):,}")

# ── 4. Count categories and stress profiles among core OGs ──
cat_counts = defaultdict(int)
for og_id in core_annotated:
    for cat in og_func_cats[og_id]:
        cat_counts[cat] += 1

cat_fracs = {cat: count / len(core_annotated) for cat, count in cat_counts.items()}

# For each category, compute which stresses its core OGs are perturbed in
# Weight = number of species with DEGs per OG x stress (from og_ss_df)
core_og_stress = og_ss_df[og_ss_df["og"].isin(core_annotated)]
# Aggregate: per OG x stress, count species
og_stress_weight = (
    core_og_stress.groupby(["og", "stress"])["species"]
    .nunique()
    .reset_index(name="n_species")
)

# For each category x stress: sum of species-level DEG signals across OGs
cat_stress_weights = {}
for cat in FUNC_CATEGORIES:
    ogs_with_cat = {og for og, cats in og_func_cats.items() if cat in cats}
    ogs_core_cat = ogs_with_cat & core_annotated
    if not ogs_core_cat:
        continue
    sub = og_stress_weight[og_stress_weight["og"].isin(ogs_core_cat)]
    stress_sums = sub.groupby("stress")["n_species"].sum()
    total = stress_sums.sum()
    if total > 0:
        cat_stress_weights[cat] = {s: stress_sums.get(s, 0) / total for s in MAIN_STRESSES}

# Sort by fraction
sorted_cats = sorted(cat_fracs.items(), key=lambda x: -x[1])
cat_names = [c[0] for c in sorted_cats]
cat_vals = [c[1] for c in sorted_cats]

STRESS_COLORS = {
    "Heat": "#d62728", "Cold": "#1f77b4", "Drought": "#ff7f0e",
    "Salt": "#8c564b", "High light": "#ffbb33", "Pathogen": "#2ca02c",
    "Flooding": "#17becf", "Heavy metal": "#7f7f7f",
    "Herbivory": "#bcbd22",
}

# ── 5. Plot ──
print("Generating figure...")

fig, ax = plt.subplots(figsize=(8, 7))

y_pos = np.arange(len(cat_names))

# Stacked horizontal bars: each bar is the category's total fraction,
# subdivided by stress type
for i, cat in enumerate(cat_names):
    total_frac = cat_fracs[cat]
    stress_profile = cat_stress_weights.get(cat, {})

    left = 0.0
    for stress in MAIN_STRESSES:
        seg_frac = stress_profile.get(stress, 0) * total_frac
        if seg_frac > 0:
            ax.barh(i, seg_frac, left=left, color=STRESS_COLORS[stress],
                    edgecolor="white", linewidth=0.2, height=0.7)
            left += seg_frac

# Annotate with percentage and count
for i, (frac, cat) in enumerate(zip(cat_vals, cat_names)):
    count = cat_counts[cat]
    ax.text(frac + 0.004, i, f"{frac:.0%} ({count:,})", va="center", fontsize=7.5)

ax.set_yticks(y_pos)
ax.set_yticklabels(cat_names, fontsize=8)
ax.set_xlabel("Fraction of annotated core OGs", fontsize=10)
ax.set_title(
    f"Functional profile of core stress responders\n"
    f"({len(core_annotated):,} OGs responding to >= {CORE_THRESHOLD} stress types)",
    fontsize=10,
)
ax.invert_yaxis()
ax.set_xlim(0, max(cat_vals) * 1.25)

# Legend for stress colors
from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor=STRESS_COLORS[s], label=s) for s in MAIN_STRESSES
]
ax.legend(handles=legend_elements, fontsize=6.5, loc="lower right", ncol=2,
          title="Stress type", title_fontsize=7)

plt.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "fig2b_v1_core_functions.pdf"), dpi=300, bbox_inches="tight")
fig.savefig(os.path.join(OUT_DIR, "fig2b_v1_core_functions.png"), dpi=300, bbox_inches="tight")
print("  Saved: fig2b_v1_core_functions.pdf/.png")
plt.close(fig)

# Save data
pd.DataFrame({"category": cat_names, "fraction": cat_vals,
              "count": [cat_counts[c] for c in cat_names]}).to_csv(
    os.path.join(OUT_DIR, "fig2b_v1_core_functions_data.csv"), index=False)


# ═══════════════════════════════════════════════════════════════════════════════
# Figure 2B': Fold-enrichment heatmap -- stress bias per functional category
# ═══════════════════════════════════════════════════════════════════════════════

print("\nGenerating fold-enrichment figure...")

# Baseline: stress profile across ALL core annotated OGs (= sampling expectation)
all_core_stress = og_stress_weight.groupby("stress")["n_species"].sum()
baseline_total = all_core_stress.sum()
baseline = {s: all_core_stress.get(s, 0) / baseline_total for s in MAIN_STRESSES}

print("  Baseline stress distribution (all core OGs):")
for s in MAIN_STRESSES:
    print(f"    {s}: {baseline[s]:.1%}")

# Compute fold-enrichment: observed / expected for each category x stress
fe_rows = []
for cat in cat_names:
    profile = cat_stress_weights.get(cat, {})
    row = {"category": cat}
    for s in MAIN_STRESSES:
        obs = profile.get(s, 0)
        exp = baseline.get(s, 0)
        if exp > 0:
            row[s] = obs / exp
        else:
            row[s] = np.nan
    fe_rows.append(row)

fe_df = pd.DataFrame(fe_rows).set_index("category")
fe_matrix = fe_df.loc[cat_names, MAIN_STRESSES].values.astype(float)

# Plot heatmap
n_cats = len(cat_names)
n_stresses = len(MAIN_STRESSES)

fig2, ax2 = plt.subplots(figsize=(8, n_cats * 0.35 + 1.5))

# Diverging colormap centered at 1.0 (no enrichment)
# log2 transform for symmetric color scale
fe_log2 = np.log2(np.where(fe_matrix > 0, fe_matrix, np.nan))
vmax = np.nanpercentile(np.abs(fe_log2), 95)
vmax = max(vmax, 0.5)

im = ax2.imshow(
    fe_log2, aspect="auto", cmap="RdBu_r", vmin=-vmax, vmax=vmax,
    interpolation="nearest",
)

# Cell annotations: show fold-enrichment value
for i in range(n_cats):
    for j in range(n_stresses):
        val = fe_matrix[i, j]
        if np.isnan(val):
            continue
        log_val = fe_log2[i, j]
        text_c = "white" if abs(log_val) > vmax * 0.6 else "black"
        # Show as fold value (e.g., 1.3x or 0.7x)
        ax2.text(j, i, f"{val:.1f}", ha="center", va="center",
                 fontsize=5.5, color=text_c)

# Cell borders
for i in range(n_cats):
    for j in range(n_stresses):
        ax2.add_patch(plt.Rectangle(
            (j - 0.5, i - 0.5), 1, 1,
            facecolor="none", edgecolor="#dddddd", linewidth=0.3,
        ))

ax2.set_xticks(range(n_stresses))
ax2.set_xticklabels(MAIN_STRESSES, rotation=45, ha="right", fontsize=8)
for i, label in enumerate(ax2.get_xticklabels()):
    label.set_color(STRESS_COLORS[MAIN_STRESSES[i]])
    label.set_fontweight("bold")

ax2.set_yticks(range(n_cats))
ax2.set_yticklabels(cat_names, fontsize=7.5)
ax2.set_title(
    "Stress enrichment in core OG functional categories\n"
    "(fold over sampling baseline; red = enriched, blue = depleted)",
    fontsize=10, pad=10,
)

cbar = plt.colorbar(im, ax=ax2, shrink=0.4, pad=0.02, aspect=15)
cbar.set_label("log$_2$(fold enrichment)", fontsize=8)
cbar.ax.tick_params(labelsize=7)

plt.tight_layout()
fig2.savefig(os.path.join(OUT_DIR, "fig2b_fold_enrichment.pdf"), dpi=300, bbox_inches="tight")
fig2.savefig(os.path.join(OUT_DIR, "fig2b_fold_enrichment.png"), dpi=300, bbox_inches="tight")
fe_df.to_csv(os.path.join(OUT_DIR, "fig2b_fold_enrichment_data.csv"))
print("  Saved: fig2b_fold_enrichment.pdf/.png/.csv")

print("Done!")
