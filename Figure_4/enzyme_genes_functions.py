"""
What do the Enzyme classification genes actually do?
Cross-reference EC genes with their other MapMan bin assignments
to get meaningful functional descriptions.
"""

import os
import pandas as pd
from collections import Counter, defaultdict

OUT = "/tmp/mercator_outputs"

# ── Load Mercator ─────────────────────────────────────────────────────────
merc = pd.read_csv("/tmp/Arabidopsis_thaliana_Mercator.txt", sep='\t',
                    names=['BINCODE', 'NAME', 'IDENTIFIER', 'DESCRIPTION', 'TYPE'],
                    skiprows=1)
for col in merc.columns:
    merc[col] = merc[col].astype(str).str.strip("'")

genes_all = merc[merc['TYPE'] == 'T'].copy()
genes_all['gene'] = genes_all['IDENTIFIER'].str.upper().str.replace(r'\.\d+$', '', regex=True)
genes_all['L0'] = genes_all['NAME'].str.split('.').str[0]
genes_all['L1'] = genes_all['NAME'].str.split('.').str[:2].str.join('.')
genes_all['L2'] = genes_all['NAME'].str.split('.').str[:3].str.join('.')

genes_all['func'] = genes_all['DESCRIPTION'].str.replace('mercator4v7.0: ', '', regex=False)
genes_all['func'] = genes_all['func'].str.split(' & original description:').str[0]

# EC genes
ec_genes = set(genes_all[genes_all['L0'] == 'Enzyme classification']['gene'].unique())
print(f"Enzyme classification genes: {len(ec_genes)}")

# ── Cross-reference: what other bins are EC genes assigned to? ────────────
# Many genes have assignments in BOTH Enzyme classification AND a pathway-specific bin
# The pathway-specific bin tells us what the gene actually does

ec_gene_other_bins = genes_all[
    (genes_all['gene'].isin(ec_genes)) & (genes_all['L0'] != 'Enzyme classification')
].copy()

print(f"EC genes with additional pathway assignments: {ec_gene_other_bins['gene'].nunique()}")
print(f"EC genes with ONLY EC assignment: {len(ec_genes) - ec_gene_other_bins['gene'].nunique()}")

# ── Which pathways do EC genes participate in? ────────────────────────────
print("\n" + "=" * 70)
print("PATHWAY BINS CONTAINING ENZYME CLASSIFICATION GENES")
print("=" * 70)

pathway_counts = ec_gene_other_bins.groupby('L0')['gene'].nunique().sort_values(ascending=False)
print(f"\n{'Pathway':<50} {'EC genes in pathway':>20}")
print("-" * 72)
for pathway, count in pathway_counts.items():
    pct = count / len(ec_genes) * 100
    print(f"  {pathway:<48} {count:>6} ({pct:>4.1f}%)")

# ── For each EC subclass, what pathways are the genes involved in? ────────
print("\n" + "=" * 70)
print("EC SUBCLASS -> PATHWAY MAPPING")
print("=" * 70)

ec_data = genes_all[genes_all['L0'] == 'Enzyme classification'].copy()
ec_subclasses = ec_data.groupby('L2')['gene'].nunique().sort_values(ascending=False)

cross_rows = []
for ec_sub in ec_subclasses.head(15).index:
    sub_genes = set(ec_data[ec_data['L2'] == ec_sub]['gene'].unique())
    sub_other = genes_all[
        (genes_all['gene'].isin(sub_genes)) & (genes_all['L0'] != 'Enzyme classification')
    ]

    sub_short = ec_sub.replace('Enzyme classification.', '')
    n_total = len(sub_genes)
    n_mapped = sub_other['gene'].nunique()

    print(f"\n  {sub_short} ({n_total} genes, {n_mapped} with pathway assignments):")

    # Top pathway L1 assignments
    pathway_l1 = sub_other.groupby('L1')['gene'].nunique().sort_values(ascending=False)
    for pathway, count in pathway_l1.head(8).items():
        pct = count / n_total * 100
        print(f"    {pathway:<55} {count:>4} ({pct:>4.1f}%)")

        cross_rows.append({
            'EC_subclass': ec_sub, 'EC_short': sub_short,
            'pathway_L1': pathway, 'n_genes': count, 'pct': pct
        })

    # Show specific functions (from their non-EC bin descriptions)
    named_funcs = sub_other[sub_other['func'].str.contains(r'\*\(', regex=True)]
    if not named_funcs.empty:
        func_counts = named_funcs.groupby('func')['gene'].nunique().sort_values(ascending=False)
        print(f"    Key named proteins:")
        for func, cnt in func_counts.head(5).items():
            print(f"      {func[:70]} ({cnt})")

cross_df = pd.DataFrame(cross_rows)
cross_df.to_csv(os.path.join(OUT, "enzyme_pathway_crossref.csv"), index=False)


# ── Detailed: the most common specific functions of EC genes ──────────────
print("\n" + "=" * 70)
print("MOST COMMON SPECIFIC FUNCTIONS OF EC GENES (from pathway bins)")
print("=" * 70)

# Get descriptive functions from non-EC bins
named_all = ec_gene_other_bins[ec_gene_other_bins['func'].str.contains(r'\*\(', regex=True)]
func_counter = named_all.groupby('func')['gene'].nunique().sort_values(ascending=False)

print(f"\nTop 40 named functions:")
for func, count in func_counter.head(40).items():
    # Also get which EC class these genes are in
    func_genes = named_all[named_all['func'] == func]['gene'].unique()
    ec_classes = ec_data[ec_data['gene'].isin(func_genes)]['L1'].value_counts()
    top_ec = ec_classes.index[0].replace('Enzyme classification.', '') if len(ec_classes) > 0 else '?'
    print(f"  {count:>4} genes: {func[:65]} [{top_ec}]")


# ── EC-only genes: what are they? ────────────────────────────────────────
print("\n" + "=" * 70)
print("EC-ONLY GENES (no other pathway assignment)")
print("=" * 70)

ec_only_genes = ec_genes - set(ec_gene_other_bins['gene'].unique())
ec_only_data = ec_data[ec_data['gene'].isin(ec_only_genes)]
ec_only_by_class = ec_only_data.groupby('L2')['gene'].nunique().sort_values(ascending=False)

print(f"\n{len(ec_only_genes)} genes assigned ONLY to Enzyme classification:")
print(f"These are likely general-purpose enzymes not assigned to specific metabolic pathways.\n")
for sub, count in ec_only_by_class.head(15).items():
    sub_short = sub.replace('Enzyme classification.', '')
    print(f"  {sub_short:<55} {count:>5} genes")


# ── Summary interpretation ────────────────────────────────────────────────
print("\n" + "=" * 70)
print("SUMMARY: WHY ENZYME CLASSIFICATION IS THE TOP HUB")
print("=" * 70)

dual_assigned = ec_gene_other_bins['gene'].nunique()
total = len(ec_genes)
print(f"""
Of {total} Arabidopsis genes in Enzyme classification:
- {dual_assigned} ({dual_assigned/total*100:.0f}%) also belong to specific pathway bins
- {total - dual_assigned} ({(total-dual_assigned)/total*100:.0f}%) are classified only by EC number

The dual-assigned genes span nearly every biological process:""")

for pathway, count in pathway_counts.head(10).items():
    pct = count / total * 100
    print(f"  {pathway:<45} {count:>5} genes ({pct:.1f}%)")

print(f"""
This explains why Enzyme classification is the #1 hub: it contains genes
from virtually every metabolic and signaling pathway. When any pathway is
perturbed under stress, its enzymatic genes simultaneously register as
Enzyme classification DEGs, creating co-occurrence links with all other
active pathways.

The largest subgroup is EC_2-7 phosphotransferases (1,210 genes = 30%),
which includes protein kinases -- the backbone of signal transduction.
""")


# Copy to Google Drive
import shutil
GDRIVE_OUT = "/Users/vjx443/Library/CloudStorage/GoogleDrive-mutwil@plant.tools/My Drive/Projects/2026_KingdomStress/CC-Fig 4 data/outputs"
shutil.copy2(os.path.join(OUT, "enzyme_pathway_crossref.csv"),
             os.path.join(GDRIVE_OUT, "enzyme_pathway_crossref.csv"))
print("Saved: enzyme_pathway_crossref.csv")
