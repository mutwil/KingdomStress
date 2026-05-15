"""
Map top hub MapMan bins to Arabidopsis genes and summarize their functions.
Uses Mercator4 annotations.
"""

import os
import re
import pandas as pd
from collections import defaultdict, Counter

OUT = "/tmp/mercator_outputs"
os.makedirs(OUT, exist_ok=True)

# ── Load Mercator annotations ─────────────────────────────────────────────
print("Loading Mercator annotations...")
merc = pd.read_csv("/tmp/Arabidopsis_thaliana_Mercator.txt", sep='\t',
                    names=['BINCODE', 'NAME', 'IDENTIFIER', 'DESCRIPTION', 'TYPE'],
                    skiprows=1)

# Strip quotes
for col in merc.columns:
    merc[col] = merc[col].astype(str).str.strip("'")

# Keep only rows with gene assignments (TYPE == 'T')
genes = merc[merc['TYPE'] == 'T'].copy()
genes['gene'] = genes['IDENTIFIER'].str.upper().str.replace(r'\.\d+$', '', regex=True)
genes['NAME'] = genes['NAME'].str.strip("'")

# Extract Level0 and Level1 bin names
genes['L0'] = genes['NAME'].str.split('.').str[0]
genes['L1'] = genes['NAME'].str.split('.').str[:2].str.join('.')

print(f"  Total gene-bin assignments: {len(genes)}")
print(f"  Unique genes: {genes['gene'].nunique()}")
print(f"  Unique Level0 bins: {genes['L0'].nunique()}")
print(f"  Unique Level1 bins: {genes['L1'].nunique()}")

# ── Load GO annotations (optional) ────────────────────────────────────────
go_file = "/Users/vjx443/Downloads/ATH_GO_GOSLIM.txt"
gene_go = defaultdict(set)
if os.path.exists(go_file):
    print("\nLoading GO slim annotations...")
    go_rows = []
    with open(go_file, 'r') as f:
        for line in f:
            if line.startswith('!'):
                continue
            parts = line.strip().split('\t')
            if len(parts) >= 10:
                locus = parts[0].upper()
                go_term = parts[4]
                aspect = parts[8]
                if aspect == 'P':
                    gene_go[locus].add(go_term)
    print(f"  GO BP annotations loaded for {len(gene_go)} genes")
else:
    print("\n  GO file not found, using Mercator descriptions only")


# ── Hub bins from analysis ────────────────────────────────────────────────
hub_L0 = pd.read_csv(os.path.join(OUT, "analysis6_hubs_level0.csv"),
                      index_col=0, header=0)
hub_L0.columns = ['weighted_degree']

hub_L1 = pd.read_csv(os.path.join(OUT, "analysis6_hubs_level1.csv"),
                      index_col=0, header=0)
hub_L1.columns = ['weighted_degree']


# ── Summarize each Level0 hub ────────────────────────────────────────────
print("\n" + "=" * 70)
print("LEVEL 0 HUB BINS: Arabidopsis gene content and GO enrichment")
print("=" * 70)

l0_summary_rows = []

for bin_name in hub_L0.index:
    wdeg = hub_L0.loc[bin_name, 'weighted_degree']
    bin_genes = genes[genes['L0'] == bin_name]
    unique_genes = bin_genes['gene'].unique()
    n_genes = len(unique_genes)

    # Get sub-bins (Level1)
    sub_bins = bin_genes['L1'].value_counts()

    # GO term enrichment for genes in this bin
    go_terms = Counter()
    for g in unique_genes:
        for term in gene_go.get(g, []):
            go_terms[term] += 1

    top_go = go_terms.most_common(5)

    # Example genes (with descriptions)
    example_genes = bin_genes.drop_duplicates('gene').head(10)

    print(f"\n{'─' * 70}")
    print(f"{bin_name}")
    print(f"  Weighted degree: {wdeg:.2f} | Arabidopsis genes: {n_genes}")
    print(f"  Top Level1 sub-bins:")
    for sb, count in sub_bins.head(5).items():
        print(f"    {sb}: {count} genes")
    print(f"  Top GO biological processes:")
    for term, count in top_go:
        pct = count / n_genes * 100
        print(f"    {term}: {count}/{n_genes} genes ({pct:.0f}%)")
    print(f"  Example genes:")
    for _, row in example_genes.iterrows():
        desc = row['DESCRIPTION']
        # Clean up description
        desc = desc.replace('mercator4v7.0: ', '').split(' & original description:')[0]
        print(f"    {row['gene']}: {desc[:70]}")

    l0_summary_rows.append({
        'bin': bin_name,
        'weighted_degree': wdeg,
        'n_genes': n_genes,
        'top_subbins': '; '.join(f"{k} ({v})" for k, v in sub_bins.head(5).items()),
        'top_GO': '; '.join(f"{t} ({c})" for t, c in top_go),
    })

l0_df = pd.DataFrame(l0_summary_rows)
l0_df.to_csv(os.path.join(OUT, "hub_L0_gene_summary.csv"), index=False)


# ── Summarize top 20 Level1 hubs ────────────────────────────────────────
print("\n\n" + "=" * 70)
print("LEVEL 1 HUB BINS: Arabidopsis gene content and GO enrichment")
print("=" * 70)

l1_summary_rows = []

for bin_name in hub_L1.index[:20]:
    wdeg = hub_L1.loc[bin_name, 'weighted_degree']
    bin_genes = genes[genes['L1'] == bin_name]
    unique_genes = bin_genes['gene'].unique()
    n_genes = len(unique_genes)

    # GO terms
    go_terms = Counter()
    for g in unique_genes:
        for term in gene_go.get(g, []):
            go_terms[term] += 1

    top_go = go_terms.most_common(5)

    # Example genes
    example_genes = bin_genes.drop_duplicates('gene').head(8)

    print(f"\n{'─' * 70}")
    print(f"{bin_name}")
    print(f"  Weighted degree: {wdeg:.2f} | Arabidopsis genes: {n_genes}")
    print(f"  Top GO biological processes:")
    for term, count in top_go:
        pct = count / n_genes * 100
        print(f"    {term}: {count}/{n_genes} genes ({pct:.0f}%)")
    print(f"  Example genes:")
    for _, row in example_genes.iterrows():
        desc = row['DESCRIPTION']
        desc = desc.replace('mercator4v7.0: ', '').split(' & original description:')[0]
        print(f"    {row['gene']}: {desc[:70]}")

    l1_summary_rows.append({
        'bin': bin_name,
        'weighted_degree': wdeg,
        'n_genes': n_genes,
        'top_GO': '; '.join(f"{t} ({c})" for t, c in top_go),
    })

l1_df = pd.DataFrame(l1_summary_rows)
l1_df.to_csv(os.path.join(OUT, "hub_L1_gene_summary.csv"), index=False)


# ── Copy to Google Drive ─────────────────────────────────────────────────
import shutil
GDRIVE_OUT = "/Users/vjx443/Library/CloudStorage/GoogleDrive-mutwil@plant.tools/My Drive/Projects/2026_KingdomStress/CC-Fig 4 data/outputs"
for f in ['hub_L0_gene_summary.csv', 'hub_L1_gene_summary.csv']:
    shutil.copy2(os.path.join(OUT, f), os.path.join(GDRIVE_OUT, f))

print("\n\nSaved: hub_L0_gene_summary.csv, hub_L1_gene_summary.csv")
