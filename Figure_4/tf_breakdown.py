"""
Detailed breakdown of DNA-binding transcriptional regulation TF classes.
"""

import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
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

# Filter to TF bin
tf = genes[genes['NAME'].str.startswith('RNA biosynthesis.DNA-binding transcriptional regulation')].copy()
print(f"Total TF gene assignments: {len(tf)}, unique genes: {tf['gene'].nunique()}")

# Parse hierarchy
parts = tf['NAME'].str.split('.')
tf['domain_type'] = parts.str[2]  # Level 3: DNA-binding domain class
tf['tf_family'] = parts.str[3]    # Level 4: TF family
tf['tf_subclass'] = parts.str[4]  # Level 5: specific class/activity

# Clean description
tf['func'] = tf['DESCRIPTION'].str.replace('mercator4v7.0: ', '', regex=False)
tf['func'] = tf['func'].str.split(' & original description:').str[0]

# ── Summary by domain type ────────────────────────────────────────────────
print("\n" + "=" * 70)
print("TF CLASSES BY DNA-BINDING DOMAIN TYPE")
print("=" * 70)

domain_counts = tf.groupby('domain_type')['gene'].nunique().sort_values(ascending=False)
for domain, count in domain_counts.items():
    print(f"\n  {domain} ({count} genes):")
    families = tf[tf['domain_type'] == domain].groupby('tf_family')['gene'].nunique().sort_values(ascending=False)
    for fam, fc in families.items():
        if pd.isna(fam) or fam == 'nan':
            continue
        # Get subclasses
        sub = tf[(tf['domain_type'] == domain) & (tf['tf_family'] == fam)]
        subclasses = sub.groupby('tf_subclass')['gene'].nunique().sort_values(ascending=False)
        sub_str = ''
        valid_subs = [(s, c) for s, c in subclasses.items() if not pd.isna(s) and s != 'nan']
        if valid_subs:
            top_subs = valid_subs[:3]
            sub_str = ' [' + ', '.join(f"{s} ({c})" for s, c in top_subs) + ']'
        print(f"    {fam:<55} {fc:>4} genes{sub_str}")

# ── Flat ranking of all TF families ──────────────────────────────────────
print("\n" + "=" * 70)
print("ALL TF FAMILIES RANKED BY GENE COUNT")
print("=" * 70)

family_summary = tf.groupby('tf_family').agg(
    n_genes=('gene', 'nunique'),
    domain=('domain_type', 'first'),
).sort_values('n_genes', ascending=False)
family_summary = family_summary[family_summary.index != 'nan']

print(f"\n{'TF Family':<55} {'Domain':<35} {'Genes':>5}")
print("-" * 98)
for fam, row in family_summary.iterrows():
    if pd.isna(fam):
        continue
    print(f"  {fam:<53} {row['domain']:<35} {row['n_genes']:>5}")

# ── Example genes per top TF family ──────────────────────────────────────
print("\n" + "=" * 70)
print("EXAMPLE GENES PER TOP TF FAMILY")
print("=" * 70)

for fam in family_summary.head(15).index:
    fam_genes = tf[tf['tf_family'] == fam].drop_duplicates('gene')
    # Prefer named entries
    named = fam_genes[fam_genes['func'].str.contains(r'\*\(', regex=True)]
    if len(named) >= 5:
        sample = named.head(8)
    else:
        sample = fam_genes.head(8)

    print(f"\n  {fam} ({family_summary.loc[fam, 'n_genes']} genes):")
    for _, row in sample.iterrows():
        print(f"    {row['gene']}: {row['func'][:70]}")

# ── Visualization ────────────────────────────────────────────────────────

# Bar chart of TF families
top_families = family_summary.head(25)

# Color by domain type
domain_colors = {
    'basic DNA-binding domain': '#E63946',
    'helix-turn-helix DNA-binding domain': '#457B9D',
    'beta-hairpin exposed by alpha/beta-scaffold structure': '#2A9D8F',
    'zinc-coordinating DNA-binding domain': '#E9C46A',
    'alpha-helix exposed by beta-structure': '#F4A261',
    'beta-barrel DNA-binding domain': '#264653',
    'other all-alpha-helix DNA-binding domain': '#8338EC',
    'undefined DNA-binding domain': '#AAAAAA',
    'beta-sheet DNA-binding domain': '#6D6875',
}

fig, ax = plt.subplots(figsize=(12, 10))
colors = [domain_colors.get(top_families.loc[f, 'domain'], '#CCCCCC') for f in top_families.index]
ax.barh(range(len(top_families)), top_families['n_genes'].values, color=colors, alpha=0.85)
ax.set_yticks(range(len(top_families)))

# Clean family names
def clean_fam(name):
    s = str(name)
    # Remove redundant suffixes
    for suffix in [' transcription factor activity', ' transcription factor', ' domain', ' family']:
        s = s.replace(suffix, '')
    return s[0].upper() + s[1:] if s else name

ax.set_yticklabels([clean_fam(f) for f in top_families.index], fontsize=9)
ax.set_xlabel("Number of Arabidopsis genes", fontsize=11)
ax.set_title("DNA-binding transcription factor families\n(MapMan Level4, Arabidopsis)", fontsize=13, fontweight='bold')
ax.invert_yaxis()

# Legend for domain types
from matplotlib.patches import Patch
legend_elements = []
used_domains = set()
for f in top_families.index:
    d = top_families.loc[f, 'domain']
    if d not in used_domains:
        used_domains.add(d)
        # Shorten domain name
        d_short = d.replace(' DNA-binding domain', '').replace(' structure', '')
        legend_elements.append(Patch(facecolor=domain_colors.get(d, '#CCC'), alpha=0.85, label=d_short))

ax.legend(handles=legend_elements, loc='lower right', fontsize=7, title='DNA-binding domain', title_fontsize=8)

plt.tight_layout()
fig.savefig(os.path.join(OUT, "tf_families_breakdown.png"))
fig.savefig(os.path.join(OUT, "tf_families_breakdown.pdf"))
plt.close(fig)

# Save table
family_summary.to_csv(os.path.join(OUT, "tf_families_summary.csv"))

# Copy to Google Drive
import shutil
GDRIVE_OUT = "/Users/vjx443/Library/CloudStorage/GoogleDrive-mutwil@plant.tools/My Drive/Projects/2026_KingdomStress/CC-Fig 4 data/outputs"
for f in ['tf_families_breakdown.png', 'tf_families_breakdown.pdf', 'tf_families_summary.csv']:
    shutil.copy2(os.path.join(OUT, f), os.path.join(GDRIVE_OUT, f))
print("\nSaved: tf_families_breakdown.png/pdf, tf_families_summary.csv")
