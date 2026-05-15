#!/usr/bin/env python3
"""
Figure 2: Gene family analysis across the plant kingdom.
Panel A: Organ x stress experiment distribution (stacked bar)
Panel B: Top conserved orthogroups heatmap (species breadth x stress)
Panel C: Conservation ratio distribution across orthogroups
Panel D: GO enrichment of conserved vs lineage-specific stress OGs
"""

import pandas as pd
import numpy as np
import re
import os
from collections import defaultdict
from scipy.stats import fisher_exact

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D

# ── Paths ──────────────────────────────────────────────────────────────────────

DEG_FILE = "/Users/vjx443/Downloads/kingdom_stress_dict v3.csv"
OG_FILE = "/Users/vjx443/Library/CloudStorage/GoogleDrive-mutwil@plant.tools/My Drive/Projects/2026_KingdomStress/Orthogroups.txt"
GO_FILE = "/Users/vjx443/Downloads/ATH_GO_tmp/ATH_GO_GOSLIM.txt"
OUT_DIR = "/Users/vjx443/Library/CloudStorage/GoogleDrive-mutwil@plant.tools/" \
          "My Drive/Projects/2026_KingdomStress/CC-Fig 2 data"

os.makedirs(OUT_DIR, exist_ok=True)

MAIN_STRESSES = [
    "Heat", "Cold", "Drought", "Salt", "High light",
    "Pathogen", "Flooding", "Heavy metal", "Herbivory",
]

STRESS_NORMALIZE = {"High Light": "High light", "high light": "High light"}

N_SPECIES_TOTAL = 36  # species in the atlas

# Stress colors (consistent with Fig 1C)
STRESS_COLORS = {
    "Heat": "#d62728", "Cold": "#1f77b4", "Drought": "#ff7f0e",
    "Salt": "#8c564b", "High light": "#ffbb33", "Pathogen": "#2ca02c",
    "Flooding": "#17becf", "Nitrogen": "#9467bd", "Heavy metal": "#7f7f7f",
    "Herbivory": "#bcbd22",
}


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 1: Parse Orthogroups  -->  gene_locus -> OG mapping
# ═══════════════════════════════════════════════════════════════════════════════

print("=" * 60)
print("SECTION 1: Parsing orthogroups...")
print("=" * 60)

gene_to_og = {}
og_ath_genes = defaultdict(set)  # OG_id -> set of Arabidopsis locus IDs

with open(OG_FILE) as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        og_id, genes_str = line.split(": ", 1)
        for gid in genes_str.split():
            locus = re.sub(r"\.\d+$", "", gid)
            gene_to_og[locus] = og_id
            # Arabidopsis nuclear + organellar genes
            if re.match(r"^AT[1-5CMG]\w+$", locus):
                og_ath_genes[og_id].add(locus)

print(f"  Gene loci mapped: {len(gene_to_og):,}")
print(f"  Unique orthogroups: {len(set(gene_to_og.values())):,}")
print(f"  OGs with Arabidopsis genes: {len(og_ath_genes):,}")


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 2: Parse GO slim annotations
# ═══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 60)
print("SECTION 2: Parsing GO slim annotations...")
print("=" * 60)

ath_go_slim = defaultdict(set)  # ATH locus -> set of GO slim term names

with open(GO_FILE) as f:
    for line in f:
        if line.startswith("!"):
            continue
        fields = line.strip().split("\t")
        if len(fields) < 10:
            continue
        locus = fields[0]
        evidence_category = fields[7] if len(fields) > 7 else ""
        go_slim_name = fields[8] if len(fields) > 8 else ""

        # Biological process (P) GO slim terms only
        if evidence_category == "P" and go_slim_name:
            ath_go_slim[locus].add(go_slim_name)

print(f"  ATH loci with GO slim (P): {len(ath_go_slim):,}")

# Map each OG to GO slim terms via its Arabidopsis members
og_go_slim = {}
for og_id, ath_genes in og_ath_genes.items():
    terms = set()
    for gene in ath_genes:
        terms.update(ath_go_slim.get(gene, set()))
    if terms:
        og_go_slim[og_id] = terms

print(f"  OGs with GO slim annotations: {len(og_go_slim):,}")


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 3: Load DEG table in chunks and aggregate
# ═══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 60)
print("SECTION 3: Loading DEG table (takes a few minutes for 13.8M rows)...")
print("=" * 60)

# Cache paths for intermediate data
organ_cache = os.path.join(OUT_DIR, "_cache_organ_stress.csv")
og_cache = os.path.join(OUT_DIR, "_cache_og_species_stress.csv")
sps_cache = os.path.join(OUT_DIR, "_cache_species_per_stress.csv")

if all(os.path.exists(f) for f in [organ_cache, og_cache, sps_cache]):
    print("  Found cached intermediates, loading...")
    organ_df = pd.read_csv(organ_cache)
    og_ss_df = pd.read_csv(og_cache)
    sps_df = pd.read_csv(sps_cache)
else:
    organ_exp_rows = []    # for Panel A
    og_ss_rows = []        # for Panels B/C/D
    species_stress_rows = []

    chunk_i = 0
    for chunk in pd.read_csv(DEG_FILE, chunksize=500_000, low_memory=False):
        chunk_i += 1
        if chunk_i % 5 == 0:
            print(f"  {chunk_i * 500_000 / 1e6:.1f}M rows...")

        chunk["stress"] = chunk["stress"].replace(STRESS_NORMALIZE)
        chunk = chunk[chunk["stress"].isin(MAIN_STRESSES)]

        # ---- Panel A: unique experiments per (stress, organ) ----
        panel_a = chunk[["stress", "organ", "species", "bioproject", "experiment"]].drop_duplicates()
        organ_exp_rows.append(panel_a)

        # ---- Species tested per stress ----
        ss = chunk[["stress", "species"]].drop_duplicates()
        species_stress_rows.append(ss)

        # ---- OG x species x stress ----
        chunk["gene_locus"] = chunk["gene"].str.replace(r"\.\d+$", "", regex=True)
        chunk["og"] = chunk["gene_locus"].map(gene_to_og)
        matched = chunk.dropna(subset=["og"])
        if len(matched) > 0:
            unique_oss = matched[["og", "species", "stress"]].drop_duplicates()
            og_ss_rows.append(unique_oss)

    print(f"  Done. {chunk_i} chunks processed.")
    print("  Aggregating...")

    # Panel A: count unique experiments per stress x organ
    organ_all = pd.concat(organ_exp_rows).drop_duplicates()
    organ_df = organ_all.groupby(["stress", "organ"]).size().reset_index(name="n_experiments")
    organ_df.to_csv(organ_cache, index=False)

    # OG x species x stress (deduplicate across chunks)
    og_ss_df = pd.concat(og_ss_rows).drop_duplicates()
    og_ss_df.to_csv(og_cache, index=False)

    # Species per stress
    sps_all = pd.concat(species_stress_rows).drop_duplicates()
    sps_df = sps_all.groupby("stress")["species"].nunique().reset_index(name="n_species")
    sps_df.to_csv(sps_cache, index=False)

print(f"  Organ-stress entries: {len(organ_df):,}")
print(f"  OG-species-stress tuples: {len(og_ss_df):,}")
n_species_per_stress = dict(zip(sps_df["stress"], sps_df["n_species"]))
print(f"  Species per stress: {n_species_per_stress}")


# ═══════════════════════════════════════════════════════════════════════════════
# PANEL A: Organ x Stress stacked bar chart
# ═══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 60)
print("PANEL A: Organ x Stress stacked bar chart")
print("=" * 60)

pivot_a = organ_df.pivot_table(
    index="stress", columns="organ", values="n_experiments", fill_value=0
)

# Top 10 organs by total experiments across stresses
organ_totals = pivot_a.sum().sort_values(ascending=False)
top_organs = organ_totals.head(10).index.tolist()
other_cols = [c for c in pivot_a.columns if c not in top_organs]
if other_cols:
    pivot_a["Others"] = pivot_a[other_cols].sum(axis=1)
plot_organs = top_organs + (["Others"] if other_cols else [])
pivot_a = pivot_a[plot_organs]
pivot_a = pivot_a.reindex(MAIN_STRESSES, fill_value=0)

# Stacked bar
organ_cmap = plt.cm.tab20(np.linspace(0, 1, len(plot_organs)))

fig_a, ax_a = plt.subplots(figsize=(8, 5))
bottom = np.zeros(len(MAIN_STRESSES))
for i, organ in enumerate(plot_organs):
    vals = pivot_a[organ].values
    ax_a.bar(
        range(len(MAIN_STRESSES)), vals, bottom=bottom,
        label=organ, color=organ_cmap[i], edgecolor="white", linewidth=0.3,
    )
    bottom += vals

ax_a.set_xticks(range(len(MAIN_STRESSES)))
ax_a.set_xticklabels(MAIN_STRESSES, rotation=45, ha="right", fontsize=9)
ax_a.set_ylabel("Number of experiments", fontsize=10)
ax_a.set_xlabel("Stress type", fontsize=10)
ax_a.set_title("Experiment coverage by organ and stress type", fontsize=11)
ax_a.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8, frameon=True)
plt.tight_layout()

fig_a.savefig(os.path.join(OUT_DIR, "fig2a_organ_stress.pdf"), dpi=300, bbox_inches="tight")
fig_a.savefig(os.path.join(OUT_DIR, "fig2a_organ_stress.png"), dpi=300, bbox_inches="tight")
pivot_a.to_csv(os.path.join(OUT_DIR, "fig2a_organ_stress_data.csv"))
print("  Saved: fig2a_organ_stress.pdf/.png/.csv")
plt.close(fig_a)


# ═══════════════════════════════════════════════════════════════════════════════
# PANEL B: Top conserved orthogroups heatmap
# ═══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 60)
print("PANEL B: Top conserved orthogroups heatmap")
print("=" * 60)

# Species count per OG x stress
og_stress_counts = (
    og_ss_df.groupby(["og", "stress"])["species"]
    .nunique()
    .reset_index(name="n_species")
)

# Total species breadth per OG (across all stresses, unique)
og_breadth = (
    og_ss_df.groupby("og")["species"]
    .nunique()
    .reset_index(name="total_species")
    .sort_values("total_species", ascending=False)
)

print(f"  Total stress-responsive OGs: {len(og_breadth):,}")
print(f"  OGs in >50% species: {(og_breadth['total_species'] > N_SPECIES_TOTAL / 2).sum():,}")

# Select top 40 most broadly responsive OGs
TOP_N = 40
top_ogs = og_breadth.head(TOP_N)["og"].tolist()

# Build heatmap matrix
hm_data = og_stress_counts[og_stress_counts["og"].isin(top_ogs)]
hm_pivot = hm_data.pivot_table(
    index="og", columns="stress", values="n_species", fill_value=0
)
hm_pivot = hm_pivot.reindex(columns=MAIN_STRESSES, fill_value=0)
hm_pivot = hm_pivot.reindex(top_ogs)

# Add Total column
hm_pivot["Total"] = og_breadth.set_index("og").reindex(top_ogs)["total_species"].values

# Label each OG with its top GO slim term
BORING_TERMS = {
    "other cellular processes", "other metabolic processes",
    "other biological processes", "biological_process",
}

og_labels = []
for og_id in top_ogs:
    terms = og_go_slim.get(og_id, set())
    good_terms = sorted(t for t in terms if t not in BORING_TERMS)
    label_term = good_terms[0] if good_terms else (sorted(terms)[0] if terms else "unannotated")
    # Truncate long labels
    if len(label_term) > 35:
        label_term = label_term[:32] + "..."
    n_spp = hm_pivot.loc[og_id, "Total"]
    og_labels.append(f"{og_id}  {label_term} ({n_spp} spp)")

# Plot
fig_b, ax_b = plt.subplots(figsize=(9, TOP_N * 0.3 + 1.5))

# Main heatmap (stress columns only)
stress_vals = hm_pivot[MAIN_STRESSES].values.astype(float)
vmax = np.percentile(stress_vals[stress_vals > 0], 95) if (stress_vals > 0).any() else 1

im = ax_b.imshow(
    stress_vals, aspect="auto", cmap="YlOrRd", vmin=0, vmax=vmax,
    interpolation="nearest",
    extent=[-0.5, len(MAIN_STRESSES) - 0.5, TOP_N - 0.5, -0.5],
)

# Cell annotations
for i in range(TOP_N):
    for j in range(len(MAIN_STRESSES)):
        val = int(stress_vals[i, j])
        if val > 0:
            text_c = "white" if val > vmax * 0.6 else "black"
            ax_b.text(j, i, str(val), ha="center", va="center",
                      fontsize=5.5, color=text_c, fontweight="bold")

# Cell borders
for i in range(TOP_N):
    for j in range(len(MAIN_STRESSES)):
        ax_b.add_patch(plt.Rectangle(
            (j - 0.5, i - 0.5), 1, 1,
            facecolor="none", edgecolor="#cccccc", linewidth=0.3,
        ))

ax_b.set_xticks(range(len(MAIN_STRESSES)))
ax_b.set_xticklabels(MAIN_STRESSES, rotation=45, ha="right", fontsize=8)
ax_b.set_yticks(range(TOP_N))
ax_b.set_yticklabels(og_labels, fontsize=5.5)
ax_b.set_title(
    f"Top {TOP_N} orthogroups by cross-species stress responsiveness",
    fontsize=10, pad=10,
)

cbar = plt.colorbar(im, ax=ax_b, shrink=0.4, pad=0.02, aspect=15)
cbar.set_label("Species with DEGs", fontsize=8)
cbar.ax.tick_params(labelsize=7)

plt.tight_layout()
fig_b.savefig(os.path.join(OUT_DIR, "fig2b_og_heatmap.pdf"), dpi=300, bbox_inches="tight")
fig_b.savefig(os.path.join(OUT_DIR, "fig2b_og_heatmap.png"), dpi=300, bbox_inches="tight")
hm_pivot.to_csv(os.path.join(OUT_DIR, "fig2b_og_heatmap_data.csv"))
print("  Saved: fig2b_og_heatmap.pdf/.png/.csv")
plt.close(fig_b)


# ═══════════════════════════════════════════════════════════════════════════════
# PANEL C: Conservation ratio distribution
# ═══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 60)
print("PANEL C: Conservation ratio distribution")
print("=" * 60)

og_breadth["conservation"] = og_breadth["total_species"] / N_SPECIES_TOTAL


def categorize(score):
    if score >= 0.5:
        return "Universal (>50%)"
    elif score >= 0.1:
        return "Moderate (10-50%)"
    else:
        return "Lineage-specific (<10%)"


og_breadth["category"] = og_breadth["conservation"].apply(categorize)

cat_order = ["Universal (>50%)", "Moderate (10-50%)", "Lineage-specific (<10%)"]
cat_colors = {
    "Universal (>50%)": "#d62728",
    "Moderate (10-50%)": "#ff7f0e",
    "Lineage-specific (<10%)": "#4c72b0",
}

print(f"  Category counts:")
for cat in cat_order:
    n = (og_breadth["category"] == cat).sum()
    print(f"    {cat}: {n:,}")

# ---- Left: histogram of conservation scores ----
fig_c, (ax_c1, ax_c2) = plt.subplots(
    1, 2, figsize=(11, 4.5), gridspec_kw={"width_ratios": [1.3, 1]}
)

ax_c1.hist(
    og_breadth["conservation"], bins=36,
    color="#4c72b0", edgecolor="white", linewidth=0.3, alpha=0.85,
)
ax_c1.set_yscale("log")
ax_c1.set_xlabel("Conservation score (fraction of 36 species)", fontsize=9)
ax_c1.set_ylabel("Number of orthogroups (log$_{10}$)", fontsize=9)
ax_c1.set_title("Distribution of stress response conservation", fontsize=10)
ax_c1.axvline(0.5, color="#d62728", linestyle="--", linewidth=0.8, label=">50% threshold")
ax_c1.axvline(0.1, color="#ff7f0e", linestyle="--", linewidth=0.8, label="10% threshold")
ax_c1.legend(fontsize=7)

# ---- Right: per-stress stacked bar of conservation categories ----
stress_cats = []
for stress in MAIN_STRESSES:
    sub = og_ss_df[og_ss_df["stress"] == stress]
    stress_og_spp = sub.groupby("og")["species"].nunique().reset_index(name="n_spp")
    stress_og_spp["category"] = (stress_og_spp["n_spp"] / N_SPECIES_TOTAL).apply(categorize)
    for cat in cat_order:
        stress_cats.append({
            "stress": stress,
            "category": cat,
            "n_ogs": (stress_og_spp["category"] == cat).sum(),
        })

stress_cat_df = pd.DataFrame(stress_cats)
stress_cat_pivot = stress_cat_df.pivot_table(
    index="stress", columns="category", values="n_ogs", fill_value=0
)
stress_cat_pivot = stress_cat_pivot.reindex(
    index=MAIN_STRESSES, columns=cat_order, fill_value=0
)

# Normalize to fractions
totals = stress_cat_pivot.sum(axis=1)
stress_cat_frac = stress_cat_pivot.div(totals, axis=0)

left = np.zeros(len(MAIN_STRESSES))
for cat in cat_order:
    vals = stress_cat_frac[cat].values
    ax_c2.barh(
        range(len(MAIN_STRESSES)), vals, left=left,
        label=cat, color=cat_colors[cat], edgecolor="white", linewidth=0.3,
    )
    left += vals

ax_c2.set_yticks(range(len(MAIN_STRESSES)))
ax_c2.set_yticklabels(MAIN_STRESSES, fontsize=8)
ax_c2.set_xlabel("Fraction of responsive OGs", fontsize=9)
ax_c2.set_title("Conservation profile by stress", fontsize=10)
ax_c2.legend(fontsize=6, loc="lower right")
ax_c2.invert_yaxis()

plt.tight_layout()
fig_c.savefig(os.path.join(OUT_DIR, "fig2c_conservation.pdf"), dpi=300, bbox_inches="tight")
fig_c.savefig(os.path.join(OUT_DIR, "fig2c_conservation.png"), dpi=300, bbox_inches="tight")
og_breadth.to_csv(os.path.join(OUT_DIR, "fig2c_conservation_data.csv"), index=False)
stress_cat_pivot.to_csv(os.path.join(OUT_DIR, "fig2c_stress_categories.csv"))
print("  Saved: fig2c_conservation.pdf/.png/.csv")
plt.close(fig_c)


# ═══════════════════════════════════════════════════════════════════════════════
# PANEL D: GO enrichment dot plot -- conserved vs lineage-specific OGs
# ═══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 60)
print("PANEL D: GO enrichment -- conserved vs lineage-specific OGs")
print("=" * 60)

conserved_ogs = set(og_breadth[og_breadth["conservation"] >= 0.5]["og"])
lineage_ogs = set(og_breadth[og_breadth["conservation"] < 0.1]["og"])
all_responsive_ogs = set(og_breadth["og"])

print(f"  Conserved OGs (>=50% spp): {len(conserved_ogs):,}")
print(f"  Lineage-specific OGs (<10% spp): {len(lineage_ogs):,}")
print(f"  All responsive OGs: {len(all_responsive_ogs):,}")


def count_go_terms(og_set):
    counts = defaultdict(int)
    for og_id in og_set:
        for term in og_go_slim.get(og_id, set()):
            counts[term] += 1
    return counts


conserved_terms = count_go_terms(conserved_ogs)
lineage_terms = count_go_terms(lineage_ogs)

# Fisher's exact test for each GO slim term
enrichment_rows = []
all_go_terms = set(list(conserved_terms.keys()) + list(lineage_terms.keys()))
skip_terms = {
    "biological_process", "other biological processes",
    "other cellular processes", "other metabolic processes",
}

for term in all_go_terms:
    if term in skip_terms:
        continue

    a = conserved_terms.get(term, 0)     # conserved with term
    b = len(conserved_ogs) - a           # conserved without
    c = lineage_terms.get(term, 0)       # lineage with term
    d = len(lineage_ogs) - c             # lineage without

    if a + c < 5:
        continue

    odds, pval = fisher_exact([[a, b], [c, d]], alternative="two-sided")

    enrichment_rows.append({
        "go_term": term,
        "conserved_count": a,
        "conserved_frac": a / max(len(conserved_ogs), 1),
        "lineage_count": c,
        "lineage_frac": c / max(len(lineage_ogs), 1),
        "odds_ratio": odds,
        "pvalue": pval,
    })

enrich_df = pd.DataFrame(enrichment_rows)

if len(enrich_df) > 0:
    # Benjamini-Hochberg FDR correction
    enrich_df = enrich_df.sort_values("pvalue").reset_index(drop=True)
    n_tests = len(enrich_df)
    enrich_df["rank"] = range(1, n_tests + 1)
    enrich_df["padj"] = (enrich_df["pvalue"] * n_tests / enrich_df["rank"]).clip(upper=1.0)
    # Enforce monotonicity (cumulative min from bottom up)
    enrich_df["padj"] = enrich_df["padj"].iloc[::-1].cummin().iloc[::-1]
    enrich_df["-log10_padj"] = -np.log10(enrich_df["padj"].clip(lower=1e-50))
    enrich_df = enrich_df.sort_values("-log10_padj", ascending=False)

    print(f"  GO terms tested: {len(enrich_df)}")
    print(f"  Significant (padj < 0.05): {(enrich_df['padj'] < 0.05).sum()}")

    # ---- Dot plot: top 20 enriched terms ----
    top_terms = enrich_df.head(20).copy()
    top_terms = top_terms.sort_values("-log10_padj", ascending=True)

    fig_d, ax_d = plt.subplots(figsize=(7, 6))

    colors = [
        "#d62728" if row["odds_ratio"] > 1 else "#4c72b0"
        for _, row in top_terms.iterrows()
    ]
    raw_sizes = (top_terms["conserved_count"] + top_terms["lineage_count"]).values.astype(float)
    sizes = 30 + (raw_sizes / max(raw_sizes.max(), 1)) * 250

    ax_d.scatter(
        top_terms["-log10_padj"], range(len(top_terms)),
        c=colors, s=sizes, alpha=0.7, edgecolors="black", linewidth=0.3,
    )

    ax_d.set_yticks(range(len(top_terms)))
    ax_d.set_yticklabels(top_terms["go_term"].values, fontsize=7)
    ax_d.set_xlabel("$-\\log_{10}$(adjusted p-value)", fontsize=9)
    ax_d.set_title("GO enrichment: conserved vs lineage-specific stress OGs", fontsize=10)
    ax_d.axvline(
        -np.log10(0.05), color="grey", linestyle="--", linewidth=0.5, label="padj = 0.05"
    )

    legend_elements = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#d62728",
               markersize=8, label="Enriched in conserved OGs"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#4c72b0",
               markersize=8, label="Enriched in lineage-specific OGs"),
    ]
    ax_d.legend(handles=legend_elements, fontsize=7, loc="lower right")

    plt.tight_layout()
    fig_d.savefig(os.path.join(OUT_DIR, "fig2d_go_enrichment.pdf"), dpi=300, bbox_inches="tight")
    fig_d.savefig(os.path.join(OUT_DIR, "fig2d_go_enrichment.png"), dpi=300, bbox_inches="tight")
    enrich_df.to_csv(os.path.join(OUT_DIR, "fig2d_go_enrichment_data.csv"), index=False)
    print("  Saved: fig2d_go_enrichment.pdf/.png/.csv")
    plt.close(fig_d)
else:
    print("  WARNING: No enrichment results. Check data.")


# ═══════════════════════════════════════════════════════════════════════════════
# Summary
# ═══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 60)
print("DONE -- All Figure 2 panels generated:")
print("=" * 60)
print(f"  {OUT_DIR}/")
print("    fig2a_organ_stress.pdf/.png/.csv")
print("    fig2b_og_heatmap.pdf/.png/.csv")
print("    fig2c_conservation.pdf/.png/.csv")
print("    fig2d_go_enrichment.pdf/.png/.csv")
print("=" * 60)
