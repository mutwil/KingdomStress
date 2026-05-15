"""
Link individual Arabidopsis TFs to their target MapMan bins.
Uses dual Mercator annotations: TFs assigned to both TF bin and pathway bins.
"""

import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

OUT = "/tmp/mercator_outputs"

plt.rcParams.update({
    'font.family': 'Arial', 'font.size': 10, 'axes.titlesize': 12,
    'axes.labelsize': 10, 'figure.dpi': 300, 'savefig.dpi': 300,
    'savefig.bbox': 'tight', 'figure.facecolor': 'white',
})

# ── Load Mercator ─────────────────────────────────────────────────────────
merc = pd.read_csv("/tmp/Arabidopsis_thaliana_Mercator.txt", sep='\t',
                    names=['BINCODE', 'NAME', 'IDENTIFIER', 'DESCRIPTION', 'TYPE'],
                    skiprows=1)
for col in merc.columns:
    merc[col] = merc[col].astype(str).str.strip("'")

genes = merc[merc['TYPE'] == 'T'].copy()
genes['gene'] = genes['IDENTIFIER'].str.upper().str.replace(r'\.\d+$', '', regex=True)

parts = genes['NAME'].str.split('.')
genes['L0'] = parts.str[0]
genes['L1'] = parts.str[:2].str.join('.')
genes['L2'] = parts.str[:3].str.join('.')

genes['func'] = genes['DESCRIPTION'].str.replace('mercator4v7.0: ', '', regex=False)
genes['func'] = genes['func'].str.split(' & original description:').str[0]

# ── Identify TF genes and their families ──────────────────────────────────
TF_PREFIX = 'RNA biosynthesis.DNA-binding transcriptional regulation'
tf_entries = genes[genes['NAME'].str.startswith(TF_PREFIX)].copy()
tf_parts = tf_entries['NAME'].str.split('.')
tf_entries['tf_family'] = tf_parts.str[3]
tf_entries['tf_subclass'] = tf_parts.str[4]

tf_genes = tf_entries[['gene', 'tf_family', 'tf_subclass', 'func']].drop_duplicates('gene')
tf_gene_set = set(tf_genes['gene'])
tf_family_map = dict(zip(tf_genes['gene'], tf_genes['tf_family']))
tf_func_map = dict(zip(tf_genes['gene'], tf_genes['func']))

print(f"TF genes: {len(tf_gene_set)}")

# ── Find pathway assignments for TF genes ─────────────────────────────────
tf_pathways = genes[
    (genes['gene'].isin(tf_gene_set)) &
    (~genes['NAME'].str.startswith(TF_PREFIX)) &
    (genes['L0'] != 'RNA biosynthesis')  # exclude other RNA biosynthesis sub-bins
].copy()

tf_with_pathway = tf_pathways['gene'].nunique()
tf_without = len(tf_gene_set) - tf_with_pathway
print(f"TFs with pathway assignments: {tf_with_pathway} ({tf_with_pathway/len(tf_gene_set)*100:.0f}%)")
print(f"TFs without pathway assignments: {tf_without}")

# ── Build TF -> target bins table ─────────────────────────────────────────
rows = []
for _, r in tf_pathways.iterrows():
    gene = r['gene']
    fam = tf_family_map.get(gene, 'unknown')
    tf_desc = tf_func_map.get(gene, '')
    target_l0 = r['L0']
    target_l1 = r['L1']
    target_full = r['NAME']
    pathway_func = r['func']

    rows.append({
        'TF_gene': gene,
        'TF_family': fam,
        'TF_description': tf_desc,
        'target_L0': target_l0,
        'target_L1': target_l1,
        'target_full_bin': target_full,
        'target_function': pathway_func,
    })

tf_target_df = pd.DataFrame(rows)
tf_target_df.to_csv(os.path.join(OUT, "tf_target_bins.csv"), index=False)

# ── Summary: TF family -> target L0 bins ──────────────────────────────────
print("\n" + "=" * 70)
print("TF FAMILY -> TARGET PATHWAY BINS (Level 0)")
print("=" * 70)

# Build matrix: TF family x target L0
cross = tf_target_df.groupby(['TF_family', 'target_L0'])['TF_gene'].nunique().reset_index()
cross.columns = ['TF_family', 'target_L0', 'n_genes']

# Get top TF families by total pathway-assigned genes
fam_totals = cross.groupby('TF_family')['n_genes'].sum().sort_values(ascending=False)
top_families = fam_totals.head(20).index.tolist()

# Pivot
cross_pivot = cross[cross['TF_family'].isin(top_families)].pivot_table(
    index='TF_family', columns='target_L0', values='n_genes', fill_value=0
)
# Sort rows by total
cross_pivot['_total'] = cross_pivot.sum(axis=1)
cross_pivot = cross_pivot.sort_values('_total', ascending=False)
cross_pivot = cross_pivot.drop('_total', axis=1)

# Sort columns by total
col_order = cross_pivot.sum().sort_values(ascending=False).index
cross_pivot = cross_pivot[col_order]
# Remove columns with all zeros
cross_pivot = cross_pivot.loc[:, (cross_pivot > 0).any()]

for fam in top_families:
    targets = cross[(cross['TF_family'] == fam)].sort_values('n_genes', ascending=False)
    total = targets['n_genes'].sum()
    print(f"\n  {fam} ({total} pathway-assigned TFs):")
    for _, r in targets.head(5).iterrows():
        print(f"    {r['target_L0']:<45} {r['n_genes']:>3} TFs")

# ── Summary: TF family -> target L1 bins (finer resolution) ──────────────
print("\n" + "=" * 70)
print("TF FAMILY -> TARGET PATHWAY BINS (Level 1, top connections)")
print("=" * 70)

cross_l1 = tf_target_df.groupby(['TF_family', 'target_L1'])['TF_gene'].nunique().reset_index()
cross_l1.columns = ['TF_family', 'target_L1', 'n_genes']

for fam in top_families[:12]:
    targets = cross_l1[cross_l1['TF_family'] == fam].sort_values('n_genes', ascending=False)
    total = targets['n_genes'].sum()
    print(f"\n  {fam} ({total} links):")
    for _, r in targets.head(8).iterrows():
        short = r['target_L1'].split('.')[-1] if '.' in r['target_L1'] else r['target_L1']
        short = short[0].upper() + short[1:]
        print(f"    {short:<50} {r['n_genes']:>3} TFs")

# ── Named TF examples with their targets ─────────────────────────────────
print("\n" + "=" * 70)
print("NOTABLE TF-TARGET CONNECTIONS")
print("=" * 70)

# Find TFs with named descriptions and pathway assignments
named_tfs = tf_target_df[tf_target_df['TF_description'].str.contains(r'\*\(', regex=True)]
named_tfs = named_tfs.sort_values('TF_family')

# Group by TF gene, show gene + family + all targets
for gene in named_tfs['TF_gene'].unique()[:40]:
    gene_rows = named_tfs[named_tfs['TF_gene'] == gene]
    fam = gene_rows.iloc[0]['TF_family']
    desc = gene_rows.iloc[0]['TF_description'][:60]
    targets = gene_rows['target_L1'].unique()
    target_str = '; '.join(t.split('.')[-1] for t in targets[:4])
    if len(targets) > 4:
        target_str += f' (+{len(targets)-4} more)'
    print(f"  {gene} [{fam}] {desc}")
    print(f"    -> {target_str}")

# ═══════════════════════════════════════════════════════════════════════════
# Visualizations
# ═══════════════════════════════════════════════════════════════════════════

def clean_fam(name):
    s = str(name)
    for suffix in [' transcription factor activity', ' transcription factor', ' domain', ' family']:
        s = s.replace(suffix, '')
    return s[0].upper() + s[1:] if s else name

# ── Heatmap: TF family x target L0 ──────────────────────────────────────
fig, ax = plt.subplots(figsize=(16, 10))
plot_data = cross_pivot.copy()
plot_data.index = [clean_fam(f) for f in plot_data.index]

sns.heatmap(plot_data, cmap='YlOrRd', ax=ax, linewidths=0.5, linecolor='white',
            annot=True, fmt='g', annot_kws={'size': 7},
            cbar_kws={'label': 'Number of TF genes', 'shrink': 0.5})
ax.set_title("TF families linked to pathway bins via dual Mercator annotation\n(Arabidopsis)",
             fontsize=13, fontweight='bold')
ax.set_xlabel("Target pathway bin (Level 0)", fontsize=11)
ax.set_ylabel("TF family", fontsize=11)
ax.tick_params(axis='x', labelsize=8, rotation=45)
ax.tick_params(axis='y', labelsize=9)
plt.tight_layout()
fig.savefig(os.path.join(OUT, "tf_target_heatmap_L0.png"))
fig.savefig(os.path.join(OUT, "tf_target_heatmap_L0.pdf"))
plt.close(fig)

# ── Heatmap: TF family x target L1 (top targets only) ───────────────────
cross_l1_pivot = cross_l1[cross_l1['TF_family'].isin(top_families[:15])].pivot_table(
    index='TF_family', columns='target_L1', values='n_genes', fill_value=0
)
# Keep top 25 target L1 bins by total
col_totals = cross_l1_pivot.sum().nlargest(25).index
cross_l1_pivot = cross_l1_pivot[col_totals]
cross_l1_pivot['_total'] = cross_l1_pivot.sum(axis=1)
cross_l1_pivot = cross_l1_pivot.sort_values('_total', ascending=False).drop('_total', axis=1)

fig, ax = plt.subplots(figsize=(20, 10))
plot_l1 = cross_l1_pivot.copy()
plot_l1.index = [clean_fam(f) for f in plot_l1.index]
plot_l1.columns = [c.split('.')[-1][0].upper() + c.split('.')[-1][1:] for c in plot_l1.columns]

sns.heatmap(plot_l1, cmap='YlOrRd', ax=ax, linewidths=0.5, linecolor='white',
            annot=True, fmt='g', annot_kws={'size': 6},
            cbar_kws={'label': 'Number of TF genes', 'shrink': 0.4})
ax.set_title("TF families linked to pathway sub-bins (Level 1)\n(Arabidopsis, top 15 TF families x top 25 targets)",
             fontsize=13, fontweight='bold')
ax.set_xlabel("Target pathway (Level 1)", fontsize=11)
ax.set_ylabel("TF family", fontsize=11)
ax.tick_params(axis='x', labelsize=7, rotation=60)
ax.tick_params(axis='y', labelsize=9)
plt.tight_layout()
fig.savefig(os.path.join(OUT, "tf_target_heatmap_L1.png"))
fig.savefig(os.path.join(OUT, "tf_target_heatmap_L1.pdf"))
plt.close(fig)

# ── Copy to Google Drive ─────────────────────────────────────────────────
import shutil
GDRIVE_OUT = "/Users/vjx443/Library/CloudStorage/GoogleDrive-mutwil@plant.tools/My Drive/Projects/2026_KingdomStress/CC-Fig 4 data/outputs"
for f in os.listdir(OUT):
    if f.startswith("tf_target"):
        shutil.copy2(os.path.join(OUT, f), os.path.join(GDRIVE_OUT, f))

print("\n" + "=" * 60)
print("TF-TARGET BIN ANALYSIS COMPLETE")
print("=" * 60)
for f in sorted(os.listdir(OUT)):
    if f.startswith("tf_target"):
        size = os.path.getsize(os.path.join(OUT, f))
        print(f"  {f} ({size/1024:.1f} KB)")
