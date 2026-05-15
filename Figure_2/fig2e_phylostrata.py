#!/usr/bin/env python3
"""
Figure 2E: Phylostratigraphic analysis of stress response.
Assigns each orthogroup an evolutionary age (phylostratum) based on species
composition, then tests whether older gene families respond more strongly.

Workflow:
  1. Build gene_locus -> species mapping from DEG file
  2. Parse orthogroups, identify species per OG, assign phylostrata
  3. Collect |log2FC| per OG from DEG file
  4. Plot: response strength and breadth by phylostratum
"""

import pandas as pd
import numpy as np
import re
import os
from collections import defaultdict
from scipy.stats import spearmanr, kruskal

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

# ── Paths ──────────────────────────────────────────────────────────────────────

DEG_FILE = "/Users/vjx443/Downloads/kingdom_stress_dict v3.csv"
OG_FILE = "/Users/vjx443/Library/CloudStorage/GoogleDrive-mutwil@plant.tools/My Drive/Projects/2026_KingdomStress/Orthogroups.txt"
OUT_DIR = "/Users/vjx443/Library/CloudStorage/GoogleDrive-mutwil@plant.tools/" \
          "My Drive/Projects/2026_KingdomStress/CC-Fig 2 data"

os.makedirs(OUT_DIR, exist_ok=True)

MAIN_STRESSES = [
    "Heat", "Cold", "Drought", "Salt", "High light",
    "Pathogen", "Flooding", "Heavy metal", "Herbivory",
]
STRESS_NORMALIZE = {"High Light": "High light", "high light": "High light"}

# ── Species -> clade mapping (from Fig 1A phylogeny) ──────────────────────────

MONOCOTS = [
    "Oryza_sativa", "Oryza_sativa_Japonica_Group",
    "Brachypodium_distachyon", "Hordeum_vulgare", "Triticum_aestivum",
    "Zea_mays", "Sorghum_bicolor", "Setaria_italica",
    "Cenchrus_americanus", "Panicum_virgatum",
]

EUDICOTS = [
    "Glycine_max", "Phaseolus_vulgaris", "Medicago_sativa", "Medicago_truncatula",
    "Cicer_arietinum", "Arachis_hypogaea", "Prunus_persica", "Cucumis_sativus",
    "Arabidopsis_thaliana", "Brassica_napus", "Brassica_oleracea",
    "Gossypium_hirsutum", "Vitis_vinifera",
    "Solanum_lycopersicum", "Solanum_tuberosum", "Capsicum_annuum",
    "Nicotiana_tabacum", "Lactuca_sativa", "Spinacia_oleracea", "Camellia_sinensis",
]

SPECIES_CLADE = {}
for sp in MONOCOTS:
    SPECIES_CLADE[sp] = "Monocot"
for sp in EUDICOTS:
    SPECIES_CLADE[sp] = "Eudicot"
SPECIES_CLADE["Picea_abies"] = "Gymnosperm"
SPECIES_CLADE["Selaginella_moellendorffii"] = "Lycophyte"
SPECIES_CLADE["Physcomitrium_patens"] = "Bryophyte"
SPECIES_CLADE["Marchantia_polymorpha"] = "Bryophyte"
SPECIES_CLADE["Klebsormidium_nitens"] = "Charophyte"
SPECIES_CLADE["Chlamydomonas_reinhardtii"] = "Chlorophyte"

# Phylostratum labels and numeric rank (1=oldest)
PS_ORDER = [
    "PS1: Viridiplantae",
    "PS2: Streptophyta",
    "PS3: Embryophyta",
    "PS4: Tracheophyta",
    "PS5: Spermatophyta",
    "PS6: Angiospermae",
    "PS7: Class-specific",
    "PS8: Narrow",
]
PS_RANK = {ps: i + 1 for i, ps in enumerate(PS_ORDER)}


def assign_phylostratum(clades):
    """Assign OG age based on most basal clade represented."""
    if "Chlorophyte" in clades:
        return "PS1: Viridiplantae"
    if "Charophyte" in clades:
        return "PS2: Streptophyta"
    if "Bryophyte" in clades:
        return "PS3: Embryophyta"
    if "Lycophyte" in clades:
        return "PS4: Tracheophyta"
    if "Gymnosperm" in clades:
        return "PS5: Spermatophyta"
    if "Monocot" in clades and "Eudicot" in clades:
        return "PS6: Angiospermae"
    if len(clades) > 0:
        return "PS7: Class-specific"
    return "PS8: Narrow"


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 1 + 3 combined: Read DEG file once
#   -> gene_locus -> species mapping
#   -> (og, abs_log2FC) per gene for response strength
# ═══════════════════════════════════════════════════════════════════════════════

gene_sp_cache = os.path.join(OUT_DIR, "_cache_gene_species.csv")
og_fc_cache = os.path.join(OUT_DIR, "_cache_og_log2fc.csv")

# First, parse orthogroups for gene_to_og (needed during DEG read)
print("=" * 60)
print("Parsing orthogroups for gene -> OG mapping...")
print("=" * 60)

gene_to_og = {}
og_all_genes = defaultdict(list)  # OG -> list of gene_loci

with open(OG_FILE) as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        og_id, genes_str = line.split(": ", 1)
        for gid in genes_str.split():
            locus = re.sub(r"\.\d+$", "", gid)
            gene_to_og[locus] = og_id
            og_all_genes[og_id].append(locus)

print(f"  Gene loci: {len(gene_to_og):,}")
print(f"  Orthogroups: {len(og_all_genes):,}")

# Read DEG file
if os.path.exists(gene_sp_cache) and os.path.exists(og_fc_cache):
    print("\nLoading cached gene-species and OG-log2FC data...")
    gene_species_df = pd.read_csv(gene_sp_cache)
    og_fc_df = pd.read_csv(og_fc_cache)
else:
    print("\n" + "=" * 60)
    print("Reading DEG file for gene-species mapping and |log2FC| per OG...")
    print("=" * 60)

    gene_sp_rows = []  # unique (gene_locus, species)
    og_fc_rows = []    # (og, abs_log2FC, stress)

    chunk_i = 0
    for chunk in pd.read_csv(DEG_FILE, chunksize=500_000, low_memory=False):
        chunk_i += 1
        if chunk_i % 5 == 0:
            print(f"  {chunk_i * 500_000 / 1e6:.1f}M rows...")

        chunk["stress"] = chunk["stress"].replace(STRESS_NORMALIZE)
        chunk = chunk[chunk["stress"].isin(MAIN_STRESSES)]

        chunk["gene_locus"] = chunk["gene"].str.replace(r"\.\d+$", "", regex=True)

        # Gene -> species (unique pairs per chunk)
        gs = chunk[["gene_locus", "species"]].drop_duplicates()
        gene_sp_rows.append(gs)

        # OG -> |log2FC|
        chunk["og"] = chunk["gene_locus"].map(gene_to_og)
        matched = chunk.dropna(subset=["og"]).copy()
        if len(matched) > 0:
            matched["abs_log2FC"] = matched["log2FC"].abs()
            # Keep one value per gene x experiment x stress (already unique in DEG table)
            fc_sub = matched[["og", "abs_log2FC", "stress", "species"]].copy()
            og_fc_rows.append(fc_sub)

    print(f"  Done. {chunk_i} chunks.")
    print("  Aggregating...")

    # Gene -> species mapping
    gene_species_df = pd.concat(gene_sp_rows).drop_duplicates()
    gene_species_df.to_csv(gene_sp_cache, index=False)
    print(f"  Unique (gene, species) pairs: {len(gene_species_df):,}")

    # OG log2FC data -- aggregate to median |log2FC| per OG
    og_fc_all = pd.concat(og_fc_rows, ignore_index=True)
    # Compute per-OG summary: median |log2FC|, max |log2FC|, n_DEGs
    og_fc_df = (
        og_fc_all.groupby("og")
        .agg(
            median_abs_log2FC=("abs_log2FC", "median"),
            mean_abs_log2FC=("abs_log2FC", "mean"),
            max_abs_log2FC=("abs_log2FC", "max"),
            n_degs=("abs_log2FC", "size"),
            n_species=("species", "nunique"),
            n_stresses=("stress", "nunique"),
        )
        .reset_index()
    )
    og_fc_df.to_csv(og_fc_cache, index=False)
    print(f"  OGs with log2FC data: {len(og_fc_df):,}")

print(f"  Gene-species pairs: {len(gene_species_df):,}")
print(f"  OGs with fold-change data: {len(og_fc_df):,}")


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 2: Assign species to each OG, then compute phylostrata
# ═══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 60)
print("Building prefix-based species classifier for ALL OG genes...")
print("=" * 60)

# Step A: Direct lookup from DEG-derived mapping
gene_to_species = dict(zip(gene_species_df["gene_locus"], gene_species_df["species"]))
print(f"  Direct gene-to-species lookup: {len(gene_to_species):,} entries")

# Step B: Learn prefix -> species patterns from DEG data
# For each species, collect all gene loci and find discriminating prefixes
from collections import Counter

species_genes = defaultdict(set)
for gene, sp in gene_to_species.items():
    species_genes[sp].add(gene)

# For each prefix length (5 down to 3), find prefixes unique to one species
prefix_to_species = {}
for prefix_len in range(8, 2, -1):
    prefix_species = defaultdict(set)
    for sp, genes in species_genes.items():
        for gene in genes:
            if len(gene) >= prefix_len:
                prefix_species[gene[:prefix_len]].add(sp)
    # Keep only unambiguous prefixes (map to exactly one species)
    for prefix, spp in prefix_species.items():
        if len(spp) == 1 and prefix not in prefix_to_species:
            prefix_to_species[prefix] = list(spp)[0]

# Sort by length (longest first) for greedy matching
sorted_prefixes = sorted(prefix_to_species.keys(), key=len, reverse=True)
print(f"  Learned {len(sorted_prefixes):,} species-discriminating prefixes")

# Build a fast prefix lookup: group by first 3 chars for speed
prefix_index = defaultdict(list)
for p in sorted_prefixes:
    prefix_index[p[:3]].append(p)


def identify_species(gene_locus):
    """Identify species: try direct lookup, then prefix matching."""
    sp = gene_to_species.get(gene_locus)
    if sp:
        return sp
    # Prefix matching (longest first within same 3-char bucket)
    key3 = gene_locus[:3]
    for prefix in prefix_index.get(key3, []):
        if gene_locus.startswith(prefix):
            return prefix_to_species[prefix]
    return None


print("\n" + "=" * 60)
print("Assigning phylostrata to orthogroups...")
print("=" * 60)

# For each OG, determine which species (and thus clades) are present
og_phylo_rows = []
n_mapped = 0
n_total_genes = 0

for og_id, genes in og_all_genes.items():
    species_in_og = set()
    for gene in genes:
        n_total_genes += 1
        sp = identify_species(gene)
        if sp:
            species_in_og.add(sp)
            n_mapped += 1

    clades = set()
    for sp in species_in_og:
        clade = SPECIES_CLADE.get(sp)
        if clade:
            clades.add(clade)

    ps = assign_phylostratum(clades)
    og_phylo_rows.append({
        "og": og_id,
        "phylostratum": ps,
        "n_species_in_og": len(species_in_og),
        "n_clades": len(clades),
        "clades": ",".join(sorted(clades)) if clades else "none",
    })

og_phylo = pd.DataFrame(og_phylo_rows)

pct_mapped = 100 * n_mapped / max(n_total_genes, 1)
print(f"  Genes mapped to species: {n_mapped:,}/{n_total_genes:,} ({pct_mapped:.1f}%)")
print(f"\n  Phylostratum distribution (all {len(og_phylo):,} OGs):")
for ps in PS_ORDER:
    n = (og_phylo["phylostratum"] == ps).sum()
    print(f"    {ps}: {n:,}")


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 4: Merge phylostrata with response data
# ═══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 60)
print("Merging phylostrata with response strength data...")
print("=" * 60)

# Merge phylostratum with OG fold-change data
merged = og_fc_df.merge(og_phylo[["og", "phylostratum", "n_species_in_og"]], on="og", how="left")
merged = merged.dropna(subset=["phylostratum"])
merged["ps_rank"] = merged["phylostratum"].map(PS_RANK)

# Normalized breadth: fraction of species in OG that show DEGs
# This controls for the trivial confound that ancient OGs span more species
merged["frac_responding"] = merged["n_species"] / merged["n_species_in_og"].clip(lower=1)
merged["frac_responding"] = merged["frac_responding"].clip(upper=1.0)

# Filter to phylostrata with enough data
ps_counts = merged["phylostratum"].value_counts()
valid_ps = [ps for ps in PS_ORDER if ps_counts.get(ps, 0) >= 10]
merged_valid = merged[merged["phylostratum"].isin(valid_ps)]

print(f"  Merged OGs: {len(merged):,}")
print(f"  Valid phylostrata (n >= 10): {len(valid_ps)}")
print(f"\n  Response strength and normalized breadth by phylostratum:")
for ps in valid_ps:
    sub = merged_valid[merged_valid["phylostratum"] == ps]
    print(f"    {ps}: n={len(sub):,}, "
          f"median |log2FC|={sub['median_abs_log2FC'].median():.2f}, "
          f"median frac_responding={sub['frac_responding'].median():.2f}, "
          f"mean n_spp_in_og={sub['n_species_in_og'].mean():.1f}")


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 5: Statistical tests
# ═══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 60)
print("Statistical tests...")
print("=" * 60)

# Spearman correlation: phylostratum rank vs median |log2FC|
rho_fc, p_fc = spearmanr(merged_valid["ps_rank"], merged_valid["median_abs_log2FC"])
print(f"  Spearman (PS rank vs median |log2FC|): rho={rho_fc:.4f}, p={p_fc:.2e}")

# Spearman: phylostratum rank vs stress breadth (n_stresses)
rho_sb, p_sb = spearmanr(merged_valid["ps_rank"], merged_valid["n_stresses"])
print(f"  Spearman (PS rank vs n_stresses): rho={rho_sb:.4f}, p={p_sb:.2e}")

# Kruskal-Wallis: |log2FC| across phylostrata
groups_fc = [
    merged_valid[merged_valid["phylostratum"] == ps]["median_abs_log2FC"].values
    for ps in valid_ps
]
h_fc, p_kw_fc = kruskal(*groups_fc)
print(f"  Kruskal-Wallis (|log2FC| across PS): H={h_fc:.1f}, p={p_kw_fc:.2e}")

# Kruskal-Wallis: n_stresses across phylostrata
groups_sb = [
    merged_valid[merged_valid["phylostratum"] == ps]["n_stresses"].values
    for ps in valid_ps
]
h_sb, p_kw_sb = kruskal(*groups_sb)
print(f"  Kruskal-Wallis (n_stresses across PS): H={h_sb:.1f}, p={p_kw_sb:.2e}")


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 6: Plot
# ═══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 60)
print("Generating figure...")
print("=" * 60)

# Short labels for x-axis
PS_SHORT = {
    "PS1: Viridiplantae": "Viridiplantae",
    "PS2: Streptophyta": "Streptophyta",
    "PS3: Embryophyta": "Embryophyta",
    "PS4: Tracheophyta": "Tracheophyta",
    "PS5: Spermatophyta": "Spermatophyta",
    "PS6: Angiospermae": "Angiospermae",
    "PS7: Class-specific": "Class-specific",
    "PS8: Narrow": "Narrow",
}

# Color gradient: old = dark, young = light
ps_palette = {
    "Viridiplantae": "#1b7837",
    "Streptophyta": "#2ca25f",
    "Embryophyta": "#5ab4ac",
    "Tracheophyta": "#c7e9b4",
    "Spermatophyta": "#fdd49e",
    "Angiospermae": "#fc8d59",
    "Class-specific": "#d7301f",
    "Narrow": "#7f0000",
}

merged_valid = merged_valid.copy()
merged_valid["ps_short"] = merged_valid["phylostratum"].map(PS_SHORT)

short_order = [PS_SHORT[ps] for ps in valid_ps]

fig, axes = plt.subplots(1, 3, figsize=(15, 5.5), gridspec_kw={"width_ratios": [0.6, 1.2, 1.2]})

# ---- Panel 1: OG count per phylostratum (context) ----
ax0 = axes[0]
counts = [len(merged_valid[merged_valid["ps_short"] == s]) for s in short_order]
colors = [ps_palette[s] for s in short_order]
ax0.barh(range(len(short_order)), counts, color=colors, edgecolor="white", linewidth=0.3)
ax0.set_xscale("log")
ax0.set_yticks(range(len(short_order)))
ax0.set_yticklabels(short_order, fontsize=8)
ax0.set_xlabel("Stress-responsive OGs (log scale)", fontsize=9)
ax0.set_title("OG count", fontsize=10)
ax0.invert_yaxis()

# Annotate counts: inside bar if wide enough, else outside
for i, c in enumerate(counts):
    if c >= 2500:
        ax0.text(c * 0.85, i, f"{c:,}", va="center", ha="right",
                 fontsize=7, color="white", fontweight="bold")
    else:
        ax0.text(c * 1.2, i, f"{c:,}", va="center", ha="left",
                 fontsize=7, color="black")

# ---- Panel 2: |log2FC| by phylostratum (strength), clipped ----
ax1 = axes[1]
FC_CLIP = 6  # clip at |log2FC| = 6 to avoid outlier-dominated violins
plot_df = merged_valid.copy()
plot_df["median_abs_log2FC_clip"] = plot_df["median_abs_log2FC"].clip(upper=FC_CLIP)

sns.violinplot(
    data=plot_df, y="ps_short", x="median_abs_log2FC_clip",
    order=short_order, hue="ps_short", hue_order=short_order,
    palette=ps_palette, legend=False,
    inner="box", linewidth=0.5, cut=0, density_norm="width",
    ax=ax1,
)
ax1.set_ylabel("")
ax1.set_xlabel("Median |log$_2$FC| per orthogroup", fontsize=9)
ax1.set_title(
    f"Response strength by gene age\n"
    f"(Spearman $\\rho$={rho_fc:.3f}, p={p_fc:.1e})",
    fontsize=10,
)
ax1.set_yticklabels([])
ax1.axvline(
    merged_valid["median_abs_log2FC"].median(),
    color="grey", linestyle="--", linewidth=0.5, alpha=0.5,
)
# Mark the clip boundary
ax1.set_xlim(0.5, FC_CLIP + 0.3)

# ---- Panel 3: Stress breadth -- ridgeline of n_stresses ----
ax2 = axes[2]

n_ps = len(short_order)
overlap = 0.65

for i, ps_name in enumerate(short_order):
    sub = merged_valid[merged_valid["ps_short"] == ps_name]["n_stresses"].values
    color = ps_palette[ps_name]

    # Histogram: bins at 0.5, 1.5, ..., 10.5 (integer stress counts 1-10)
    bins = np.arange(0.5, 11.5, 1)
    hist_vals, bin_edges = np.histogram(sub, bins=bins, density=True)
    bin_centers = np.arange(1, 11)

    # Each ridge scaled to its own max
    scale = overlap / max(hist_vals.max(), 1e-9)
    y_base = n_ps - 1 - i
    y_vals = y_base + hist_vals * scale

    ax2.fill_between(bin_centers, y_base, y_vals, color=color, alpha=0.7,
                     edgecolor="white", linewidth=0.3, step="mid")
    ax2.step(bin_centers, y_vals, color="black", linewidth=0.4, where="mid")

    # Median marker
    med = np.median(sub)
    ax2.plot(med, y_base + 0.02, marker="v", color="black", markersize=4, zorder=5)

ax2.set_yticks([n_ps - 1 - i for i in range(n_ps)])
ax2.set_yticklabels([])
ax2.set_xticks(range(1, 11))
ax2.set_xlabel("Number of stress types with DEGs", fontsize=9)
ax2.set_title(
    f"Stress breadth by gene age\n"
    f"(Spearman $\\rho$={rho_sb:.3f}, p={p_sb:.1e})",
    fontsize=10,
)
ax2.set_xlim(0.5, 10.5)

plt.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "fig2e_phylostrata.pdf"), dpi=300, bbox_inches="tight")
fig.savefig(os.path.join(OUT_DIR, "fig2e_phylostrata.png"), dpi=300, bbox_inches="tight")
print("  Saved: fig2e_phylostrata.pdf/.png")

# ---- Save data ----
merged_valid.to_csv(os.path.join(OUT_DIR, "fig2e_phylostrata_data.csv"), index=False)
og_phylo.to_csv(os.path.join(OUT_DIR, "fig2e_og_phylostrata.csv"), index=False)

# Stats summary
stats_txt = (
    f"Phylostratigraphic analysis of stress response\n"
    f"{'=' * 50}\n\n"
    f"Total stress-responsive OGs: {len(merged):,}\n"
    f"OGs with valid phylostratum: {len(merged_valid):,}\n\n"
    f"Response STRENGTH (median |log2FC| per OG):\n"
    f"  Spearman rho = {rho_fc:.4f}, p = {p_fc:.2e}\n"
    f"  Kruskal-Wallis H = {h_fc:.1f}, p = {p_kw_fc:.2e}\n"
    f"  {'Older OGs respond MORE strongly' if rho_fc < 0 else 'Younger OGs respond MORE strongly'}\n\n"
    f"Stress BREADTH (n_stresses per OG):\n"
    f"  Spearman rho = {rho_sb:.4f}, p = {p_sb:.2e}\n"
    f"  Kruskal-Wallis H = {h_sb:.1f}, p = {p_kw_sb:.2e}\n"
    f"  {'Older OGs respond to MORE stresses' if rho_sb < 0 else 'Younger OGs respond to MORE stresses'}\n\n"
    f"Per-phylostratum summary:\n"
    f"{'Phylostratum':<25s} {'n_OGs':>8s} {'med|log2FC|':>12s} {'med_stresses':>13s}\n"
    f"{'-' * 62}\n"
)
for ps in valid_ps:
    sub = merged_valid[merged_valid["phylostratum"] == ps]
    stats_txt += (
        f"{PS_SHORT[ps]:<25s} {len(sub):>8,d} "
        f"{sub['median_abs_log2FC'].median():>12.2f} "
        f"{sub['n_stresses'].median():>13.1f}\n"
    )

with open(os.path.join(OUT_DIR, "fig2e_stats.txt"), "w") as f:
    f.write(stats_txt)

print("  Saved: fig2e_stats.txt")
print(f"\n{stats_txt}")

print("\n" + "=" * 60)
print("Done!")
print("=" * 60)
