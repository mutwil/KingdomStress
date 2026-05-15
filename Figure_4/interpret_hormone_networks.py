"""
Use PlantConnectome KG to interpret the hormone-stress co-occurrence networks.
For each hormone: which KG-annotated genes fall in which MapMan bins?
This reveals WHY certain bins co-occur with specific hormones.
"""

import os
import re
import pandas as pd
import numpy as np
from collections import defaultdict, Counter

OUT = "/tmp/mercator_outputs"

# ── Load Mercator annotations ─────────────────────────────────────────────
print("Loading Mercator annotations...")
merc = pd.read_csv("/tmp/Arabidopsis_thaliana_Mercator.txt", sep='\t',
                    names=['BINCODE', 'NAME', 'IDENTIFIER', 'DESCRIPTION', 'TYPE'],
                    skiprows=1)
for col in merc.columns:
    merc[col] = merc[col].astype(str).str.strip("'")

merc_genes = merc[merc['TYPE'] == 'T'].copy()
merc_genes['gene'] = merc_genes['IDENTIFIER'].str.upper().str.replace(r'\.\d+$', '', regex=True)
merc_genes['L0'] = merc_genes['NAME'].str.split('.').str[0]
merc_genes['L1'] = merc_genes['NAME'].str.split('.').str[:2].str.join('.')

# Build gene -> L1 bins mapping
gene_to_l1 = defaultdict(set)
for _, r in merc_genes.iterrows():
    gene_to_l1[r['gene']].add(r['L1'])

# ── Load KG hormone-gene links ───────────────────────────────────────────
print("Loading KG hormone-gene entries...")
hdf = pd.read_csv('/tmp/kg/hormone_kg_entries.csv', low_memory=False)

HORMONE_SEARCH = {
    'Abscisic acid': ['abscisic acid', 'ABA'],
    'Salicylic acid': ['salicyl', 'SA '],
    'Brassinosteroid': ['brassinosteroid', 'brassinolide', 'BRI1'],
    'Gibberellin': ['gibberellin', 'GA3', 'GA20'],
    'Jasmonic acid': ['jasmonic acid', 'jasmonate', 'JA '],
    'Ethylene': ['ethylene', 'ACC '],
    'Auxin': ['auxin', 'IAA '],
    'Cytokinin': ['cytokinin'],
}

def extract_agi_ids(df):
    """Extract AT-style gene IDs from gene alias columns."""
    agis = set()
    for col in ['source gene alias', 'target gene alias']:
        for val in df[col].dropna():
            for part in str(val).split(','):
                part = part.strip().strip('()')
                if re.match(r'AT[1-5CM]G\d{5}', part.upper()):
                    agis.add(part.upper())
    return agis

# Extract gene lists per hormone
hormone_genes = {}
for hormone, search_terms in HORMONE_SEARCH.items():
    mask = pd.Series(False, index=hdf.index)
    for term in search_terms:
        mask |= hdf['source resolved'].str.contains(term, case=False, na=False)
        mask |= hdf['target resolved'].str.contains(term, case=False, na=False)

    sub = hdf[mask]

    # Filter to gene-related entries
    gene_sub = sub[
        (sub['source type resolved'].isin(['gene', 'gene identifier'])) |
        (sub['target type resolved'].isin(['gene', 'gene identifier']))
    ]

    agis = extract_agi_ids(gene_sub)
    hormone_genes[hormone] = agis

    # Also get the relationships for context
    relationships = gene_sub['relationship resolved'].value_counts()

    print(f"  {hormone}: {len(agis)} AGI IDs from {len(gene_sub)} KG edges")

# ── Cross-reference: which MapMan bins contain KG-linked hormone genes? ───
print("\n" + "=" * 70)
print("MAPMAN BINS CONTAINING KG-LINKED HORMONE GENES")
print("=" * 70)

interpretation_rows = []

for hormone in HORMONE_SEARCH:
    agis = hormone_genes.get(hormone, set())
    if not agis:
        print(f"\n  {hormone}: no AGI IDs found in KG")
        continue

    # Map genes to L1 bins
    bin_genes = defaultdict(set)
    for gene in agis:
        for l1 in gene_to_l1.get(gene, set()):
            bin_genes[l1].add(gene)

    # Sort by gene count
    bin_counts = {b: len(g) for b, g in bin_genes.items()}
    sorted_bins = sorted(bin_counts.items(), key=lambda x: -x[1])

    def short(name):
        s = name.split('.')[-1].strip()
        return s[0].upper() + s[1:]

    print(f"\n  {hormone.upper()} ({len(agis)} KG genes):")
    print(f"  {'MapMan L1 bin':<55} {'KG genes':>8} {'Example genes'}")
    print(f"  {'-'*90}")

    for l1_bin, count in sorted_bins[:20]:
        examples = sorted(bin_genes[l1_bin])[:3]
        ex_str = ', '.join(examples)
        print(f"  {short(l1_bin):<55} {count:>8}   {ex_str}")

        interpretation_rows.append({
            'hormone': hormone, 'L1_bin': l1_bin,
            'L1_short': short(l1_bin),
            'n_KG_genes': count,
            'example_genes': ex_str,
            'L0_bin': l1_bin.split('.')[0],
        })

interp_df = pd.DataFrame(interpretation_rows)
interp_df.to_csv(os.path.join(OUT, "hormone_kg_bin_interpretation.csv"), index=False)

# ── Per-hormone: what phenotypes does the KG link to each hormone? ────────
print("\n" + "=" * 70)
print("HORMONE -> PHENOTYPE LINKS FROM KG")
print("=" * 70)

for hormone, search_terms in HORMONE_SEARCH.items():
    mask = pd.Series(False, index=hdf.index)
    for term in search_terms:
        mask |= hdf['source resolved'].str.contains(term, case=False, na=False)

    sub = hdf[mask & (hdf['target type resolved'].isin(['phenotype', 'process']))]

    if sub.empty:
        # Try reverse
        mask2 = pd.Series(False, index=hdf.index)
        for term in search_terms:
            mask2 |= hdf['target resolved'].str.contains(term, case=False, na=False)
        sub = hdf[mask2 & (hdf['source type resolved'].isin(['phenotype', 'process']))]

    if sub.empty:
        print(f"\n  {hormone}: no phenotype links")
        continue

    targets = sub['target resolved'].value_counts().head(12)
    print(f"\n  {hormone.upper()} -> phenotypes ({len(sub)} edges):")
    for t, c in targets.items():
        print(f"    {t}: {c}")

# ── Interpretation summary: link network structure to KG knowledge ────────
print("\n" + "=" * 70)
print("INTERPRETATION: WHY THESE BINS CO-OCCUR WITH THESE HORMONES")
print("=" * 70)

# For each hormone, show the overlap between KG-linked bins and network neighbors
# Load the per-stress network edge data
stress_bias_df = pd.read_csv(os.path.join(OUT, "bin_per_stress_L1.csv"))

for hormone in ['Abscisic acid', 'Salicylic acid', 'Brassinosteroid', 'Gibberellin',
                 'Jasmonic acid', 'Ethylene', 'Auxin']:
    hfull = f"Phytohormone action.{hormone.lower()}"
    agis = hormone_genes.get(hormone, set())
    if not agis:
        continue

    # KG-linked bins
    bin_genes_map = defaultdict(set)
    for gene in agis:
        for l1 in gene_to_l1.get(gene, set()):
            bin_genes_map[l1].add(gene)

    kg_bins = set(bin_genes_map.keys())

    print(f"\n  {hormone.upper()}:")
    print(f"    KG genes: {len(agis)}, mapped to {len(kg_bins)} L1 bins")

    # Top KG bins that are NOT the hormone bin itself
    kg_other = {b: len(g) for b, g in bin_genes_map.items()
                if not b.startswith('Phytohormone action')}
    top_kg = sorted(kg_other.items(), key=lambda x: -x[1])[:10]

    print(f"    Top non-hormone bins containing {hormone} KG genes:")
    for b, c in top_kg:
        short_b = b.split('.')[-1]
        short_b = short_b[0].upper() + short_b[1:]
        # Get the direction bias for this bin (all-stress average)
        bias_vals = stress_bias_df[stress_bias_df['bin_name'] == b]['bias']
        avg_bias = bias_vals.mean() if not bias_vals.empty else 0
        direction = 'UP' if avg_bias > 0.02 else ('DOWN' if avg_bias < -0.02 else 'neutral')
        print(f"      {short_b:<45} {c:>3} genes  ({direction})")


# Copy
import shutil
GDRIVE_OUT = "/Users/vjx443/Library/CloudStorage/GoogleDrive-mutwil@plant.tools/My Drive/Projects/2026_KingdomStress/CC-Fig 4 data/outputs"
shutil.copy2(os.path.join(OUT, "hormone_kg_bin_interpretation.csv"),
             os.path.join(GDRIVE_OUT, "hormone_kg_bin_interpretation.csv"))
print(f"\nSaved: hormone_kg_bin_interpretation.csv")
