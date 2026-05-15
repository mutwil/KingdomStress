"""
Deep breakdown of Enzyme classification MapMan bin:
What do the Arabidopsis genes in this top hub bin actually do?
Uses Mercator4 annotations at all hierarchy levels.
"""

import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter, defaultdict
import textwrap
import warnings
warnings.filterwarnings('ignore')

OUT = "/tmp/mercator_outputs"
os.makedirs(OUT, exist_ok=True)

plt.rcParams.update({
    'font.family': 'Arial', 'font.size': 10, 'axes.titlesize': 12,
    'axes.labelsize': 10, 'figure.dpi': 300, 'savefig.dpi': 300,
    'savefig.bbox': 'tight', 'figure.facecolor': 'white',
})

# ── Load Mercator ─────────────────────────────────────────────────────────
print("Loading Mercator annotations...")
merc = pd.read_csv("/tmp/Arabidopsis_thaliana_Mercator.txt", sep='\t',
                    names=['BINCODE', 'NAME', 'IDENTIFIER', 'DESCRIPTION', 'TYPE'],
                    skiprows=1)
for col in merc.columns:
    merc[col] = merc[col].astype(str).str.strip("'")

# Gene assignments only
genes = merc[merc['TYPE'] == 'T'].copy()
genes['gene'] = genes['IDENTIFIER'].str.upper().str.replace(r'\.\d+$', '', regex=True)

# Parse hierarchy levels
genes['full_name'] = genes['NAME']
name_parts = genes['NAME'].str.split('.')
genes['L0'] = name_parts.str[0]
genes['L1'] = name_parts.str[:2].str.join('.')
genes['L2'] = name_parts.str[:3].str.join('.')
genes['L3'] = name_parts.str[:4].str.join('.')

# Clean description
genes['func'] = genes['DESCRIPTION'].str.replace('mercator4v7.0: ', '', regex=False)
genes['func'] = genes['func'].str.split(' & original description:').str[0]

# ── Filter to Enzyme classification ───────────────────────────────────────
ec = genes[genes['L0'] == 'Enzyme classification'].copy()
print(f"Enzyme classification: {ec['gene'].nunique()} unique Arabidopsis genes, {len(ec)} assignments")

# ═══════════════════════════════════════════════════════════════════════════
# Level 1 breakdown (EC classes)
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("EC CLASS BREAKDOWN (Level 1)")
print("=" * 70)

l1_counts = ec.groupby('L1')['gene'].nunique().sort_values(ascending=False)
print(f"\n{'EC Class':<55} {'Genes':>6}")
print("-" * 65)
for name, count in l1_counts.items():
    print(f"  {name:<53} {count:>6}")

# ═══════════════════════════════════════════════════════════════════════════
# Level 2 breakdown (EC subclasses)
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("EC SUBCLASS BREAKDOWN (Level 2) -- top entries per EC class")
print("=" * 70)

l2_counts = ec.groupby(['L1', 'L2'])['gene'].nunique().reset_index()
l2_counts.columns = ['L1', 'L2', 'n_genes']

ec_classes = l1_counts.index.tolist()
l2_summary_rows = []

for ec_class in ec_classes:
    subset = l2_counts[l2_counts['L1'] == ec_class].sort_values('n_genes', ascending=False)
    print(f"\n  {ec_class} ({l1_counts[ec_class]} genes):")
    for _, row in subset.head(10).iterrows():
        subclass = row['L2'].replace(ec_class + '.', '')
        print(f"    {subclass:<50} {row['n_genes']:>5} genes")
        l2_summary_rows.append({
            'EC_class': ec_class, 'subclass': row['L2'],
            'subclass_short': subclass, 'n_genes': row['n_genes']
        })

l2_df = pd.DataFrame(l2_summary_rows)

# ═══════════════════════════════════════════════════════════════════════════
# Level 3+ detailed functions (what do the genes actually catalyze?)
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("DETAILED FUNCTIONS (Level 3+) -- top entries per EC subclass")
print("=" * 70)

# For each major EC subclass, show the most common Level3+ functions
l3_summary = []
for ec_class in ec_classes:
    class_genes = ec[ec['L1'] == ec_class]
    l2_top = class_genes.groupby('L2')['gene'].nunique().nlargest(5)

    for l2_name in l2_top.index:
        l2_genes = class_genes[class_genes['L2'] == l2_name]
        # Get unique functions at deepest level
        func_counts = l2_genes.groupby('func')['gene'].nunique().sort_values(ascending=False)

        l2_short = l2_name.replace(ec_class + '.', '')
        print(f"\n  {ec_class} > {l2_short} ({l2_top[l2_name]} genes):")
        for func, count in func_counts.head(8).items():
            func_clean = func[:75]
            print(f"    {func_clean:<73} {count:>4}")
            l3_summary.append({
                'EC_class': ec_class, 'L2': l2_name,
                'function': func, 'n_genes': count
            })

l3_df = pd.DataFrame(l3_summary)

# ═══════════════════════════════════════════════════════════════════════════
# Example well-known Arabidopsis genes per EC class
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("NOTABLE ARABIDOPSIS GENES per EC class")
print("=" * 70)

# Select genes with informative descriptions (containing known gene names)
for ec_class in ec_classes:
    class_genes = ec[ec['L1'] == ec_class].drop_duplicates('gene')

    # Prefer genes with named protein descriptions (contain asterisks)
    named = class_genes[class_genes['func'].str.contains(r'\*\(', regex=True)]
    if len(named) > 0:
        sample = named.sample(min(12, len(named)), random_state=42)
    else:
        sample = class_genes.head(12)

    print(f"\n  {ec_class}:")
    for _, row in sample.iterrows():
        print(f"    {row['gene']}: {row['func'][:70]}")


# ═══════════════════════════════════════════════════════════════════════════
# Visualizations
# ═══════════════════════════════════════════════════════════════════════════

# ── Plot 1: Treemap-style stacked bar of EC hierarchy ─────────────────────
fig, ax = plt.subplots(figsize=(14, 8))

# Stacked horizontal bars: EC class -> top subclasses
colors = plt.cm.Set3(np.linspace(0, 1, 12))
y_pos = 0
y_labels = []
y_positions = []

for i, ec_class in enumerate(ec_classes):
    subset = l2_counts[l2_counts['L1'] == ec_class].sort_values('n_genes', ascending=False)
    x_offset = 0
    ec_short = ec_class.replace('Enzyme classification.', '')

    for j, (_, row) in enumerate(subset.iterrows()):
        subclass = row['L2'].replace(ec_class + '.', '')
        width = row['n_genes']
        color = colors[j % len(colors)]
        bar = ax.barh(y_pos, width, left=x_offset, height=0.7,
                      color=color, alpha=0.8, edgecolor='white', linewidth=0.5)
        if width > 30:
            ax.text(x_offset + width/2, y_pos, f"{subclass}\n({width})",
                    ha='center', va='center', fontsize=5, fontweight='bold')
        x_offset += width

    y_labels.append(f"{ec_short} ({l1_counts[ec_class]})")
    y_positions.append(y_pos)
    y_pos += 1

ax.set_yticks(y_positions)
ax.set_yticklabels(y_labels, fontsize=9)
ax.set_xlabel("Number of Arabidopsis genes")
ax.set_title("Enzyme classification: gene count breakdown by EC subclass")
ax.invert_yaxis()
plt.tight_layout()
fig.savefig(os.path.join(OUT, "enzyme_L1_L2_breakdown.png"))
fig.savefig(os.path.join(OUT, "enzyme_L1_L2_breakdown.pdf"))
plt.close(fig)


# ── Plot 2: Top 30 specific functions by gene count ──────────────────────
top_funcs = l3_df.groupby('function')['n_genes'].sum().nlargest(30)

fig, ax = plt.subplots(figsize=(12, 10))
colors_by_ec = []
func_ec_map = {}
for _, row in l3_df.iterrows():
    func_ec_map[row['function']] = row['EC_class']

ec_color_map = {ec: plt.cm.tab10(i/len(ec_classes)) for i, ec in enumerate(ec_classes)}

for func in top_funcs.index:
    colors_by_ec.append(ec_color_map.get(func_ec_map.get(func, ''), 'grey'))

ax.barh(range(len(top_funcs)), top_funcs.values, color=colors_by_ec, alpha=0.85)
ax.set_yticks(range(len(top_funcs)))

def shorten(s, n=55):
    return s if len(s) <= n else s[:n-2] + '..'

ax.set_yticklabels([shorten(f) for f in top_funcs.index], fontsize=7)
ax.set_xlabel("Number of Arabidopsis genes")
ax.set_title("Top 30 enzyme functions in the Enzyme classification hub bin")
ax.invert_yaxis()

# Legend
from matplotlib.patches import Patch
legend_elements = [Patch(facecolor=ec_color_map[ec], alpha=0.85,
                         label=ec.replace('Enzyme classification.', ''))
                   for ec in ec_classes if ec in ec_color_map]
ax.legend(handles=legend_elements, loc='lower right', fontsize=7, title='EC class')

plt.tight_layout()
fig.savefig(os.path.join(OUT, "enzyme_top_functions.png"))
fig.savefig(os.path.join(OUT, "enzyme_top_functions.pdf"))
plt.close(fig)


# ── Plot 3: Per-EC-class pie of Level2 composition ──────────────────────
fig, axes = plt.subplots(2, 3, figsize=(18, 12))
axes_flat = axes.flatten()

for i, ec_class in enumerate(ec_classes[:6]):
    ax = axes_flat[i]
    subset = l2_counts[l2_counts['L1'] == ec_class].sort_values('n_genes', ascending=False)
    labels = [r['L2'].replace(ec_class + '.', '') for _, r in subset.iterrows()]
    sizes = subset['n_genes'].values

    # Group small slices
    threshold = sizes.sum() * 0.03
    main_mask = sizes >= threshold
    main_labels = [l for l, m in zip(labels, main_mask) if m]
    main_sizes = sizes[main_mask]
    if (~main_mask).any():
        main_labels.append('other')
        main_sizes = np.append(main_sizes, sizes[~main_mask].sum())

    wedges, texts, autotexts = ax.pie(main_sizes, labels=None, autopct='%1.0f%%',
                                       pctdistance=0.8, startangle=90,
                                       colors=plt.cm.Set3(np.linspace(0, 1, len(main_labels))))
    for t in autotexts:
        t.set_fontsize(6)

    ec_short = ec_class.replace('Enzyme classification.', '')
    ax.set_title(f"{ec_short}\n({l1_counts[ec_class]} genes)", fontsize=10, fontweight='bold')

    # Legend instead of labels (cleaner)
    ax.legend(main_labels, loc='center left', bbox_to_anchor=(0.9, 0.5), fontsize=6)

fig.suptitle("EC subclass composition (Level 2)", fontsize=14, y=1.01)
plt.tight_layout()
fig.savefig(os.path.join(OUT, "enzyme_EC_pies.png"))
fig.savefig(os.path.join(OUT, "enzyme_EC_pies.pdf"))
plt.close(fig)


# ── Save summary tables ──────────────────────────────────────────────────
l2_df.to_csv(os.path.join(OUT, "enzyme_L2_breakdown.csv"), index=False)
l3_df.to_csv(os.path.join(OUT, "enzyme_detailed_functions.csv"), index=False)

# Full gene list
ec_gene_list = ec[['gene', 'L0', 'L1', 'L2', 'L3', 'full_name', 'func']].drop_duplicates()
ec_gene_list.to_csv(os.path.join(OUT, "enzyme_gene_list.csv"), index=False)

# ── Copy to Google Drive ─────────────────────────────────────────────────
import shutil
GDRIVE_OUT = "/Users/vjx443/Library/CloudStorage/GoogleDrive-mutwil@plant.tools/My Drive/Projects/2026_KingdomStress/CC-Fig 4 data/outputs"
for f in os.listdir(OUT):
    if f.startswith("enzyme_"):
        shutil.copy2(os.path.join(OUT, f), os.path.join(GDRIVE_OUT, f))

print("\n" + "=" * 60)
print("ENZYME CLASSIFICATION BREAKDOWN COMPLETE")
print("=" * 60)
for f in sorted(os.listdir(OUT)):
    if f.startswith("enzyme_"):
        size = os.path.getsize(os.path.join(OUT, f))
        print(f"  {f} ({size/1024:.1f} KB)")
