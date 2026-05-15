"""
Figure: KG validation of hormone-L0 co-occurrence network.
Panel A: Heatmap of KG support per hormone-L0 pair
Panel B: Direction comparison (KG normal vs stress network)
Panel C: Summary bar chart
"""

import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import Normalize, LogNorm
import matplotlib.patches as mpatches
import warnings
warnings.filterwarnings('ignore')

OUT = "/tmp/mercator_outputs"

plt.rcParams.update({
    'font.family': 'Arial', 'font.size': 12,
    'axes.titlesize': 16, 'axes.labelsize': 14,
    'figure.dpi': 300, 'savefig.dpi': 300,
    'savefig.bbox': 'tight', 'figure.facecolor': 'white',
})

# ── Load data ─────────────────────────────────────────────────────────────
val = pd.read_csv(os.path.join(OUT, "hormone_L0_KG_validation.csv"))

# Aggregate across stresses: mean co-occurrence, mean bias, total KG entries
pair = val.groupby(['hormone', 'L0_bin']).agg(
    mean_weight=('mean_weight', 'mean'),
    mean_bias=('edge_bias', 'mean'),
    KG_total=('KG_entries', 'sum'),
).reset_index()

HORMONES = ['Abscisic acid', 'Salicylic acid', 'Brassinosteroid', 'Gibberellin',
            'Jasmonic acid', 'Ethylene', 'Auxin']

# Get top L0 bins by total KG entries
l0_totals = pair.groupby('L0_bin')['KG_total'].sum().sort_values(ascending=False)
TOP_L0 = l0_totals.head(15).index.tolist()

# ── Panel A: KG support heatmap (log scale) ──────────────────────────────
kg_pivot = pair[pair['L0_bin'].isin(TOP_L0)].pivot_table(
    index='L0_bin', columns='hormone', values='KG_total', fill_value=0)
kg_pivot = kg_pivot.reindex(columns=HORMONES)
kg_pivot = kg_pivot.loc[TOP_L0]

# ── Panel B: Direction comparison ────────────────────────────────────────
bias_pivot = pair[pair['L0_bin'].isin(TOP_L0)].pivot_table(
    index='L0_bin', columns='hormone', values='mean_bias', fill_value=0)
bias_pivot = bias_pivot.reindex(columns=HORMONES)
bias_pivot = bias_pivot.loc[TOP_L0]

# ── Panel C: Co-occurrence weight heatmap ────────────────────────────────
weight_pivot = pair[pair['L0_bin'].isin(TOP_L0)].pivot_table(
    index='L0_bin', columns='hormone', values='mean_weight', fill_value=0)
weight_pivot = weight_pivot.reindex(columns=HORMONES)
weight_pivot = weight_pivot.loc[TOP_L0]

# ── Build figure ─────────────────────────────────────────────────────────
fig = plt.figure(figsize=(24, 16))
gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 1], wspace=0.35)

# Short hormone labels
h_short = ['ABA', 'SA', 'BR', 'GA', 'JA', 'Ethylene', 'Auxin']

# Panel A: Co-occurrence frequency
ax1 = fig.add_subplot(gs[0])
sns.heatmap(weight_pivot, cmap='YlOrRd', ax=ax1, linewidths=0.5, linecolor='white',
            annot=True, fmt='.2f', annot_kws={'size': 9},
            xticklabels=h_short,
            cbar_kws={'label': 'Co-occurrence frequency', 'shrink': 0.4})
ax1.set_title("A) Co-occurrence\nfrequency", fontweight='bold')
ax1.set_xlabel("")
ax1.set_ylabel("")
ax1.tick_params(axis='y', labelsize=11)
ax1.tick_params(axis='x', labelsize=11, rotation=45)

# Panel B: Direction bias
ax2 = fig.add_subplot(gs[1])
max_b = max(abs(bias_pivot.min().min()), abs(bias_pivot.max().max()))
sns.heatmap(bias_pivot, cmap='RdBu_r', center=0, vmin=-max_b, vmax=max_b,
            ax=ax2, linewidths=0.5, linecolor='white',
            annot=True, fmt='+.2f', annot_kws={'size': 9},
            xticklabels=h_short,
            yticklabels=False,
            cbar_kws={'label': 'UP - DOWN bias (stress)', 'shrink': 0.4})
ax2.set_title("B) Direction bias\n(stress network)", fontweight='bold')
ax2.set_xlabel("")
ax2.tick_params(axis='x', labelsize=11, rotation=45)

# Panel C: KG literature support (log scale)
ax3 = fig.add_subplot(gs[2])
kg_log = np.log10(kg_pivot.replace(0, np.nan))
sns.heatmap(kg_log, cmap='Greens', ax=ax3, linewidths=0.5, linecolor='white',
            annot=kg_pivot.values.astype(int), fmt='d', annot_kws={'size': 8},
            xticklabels=h_short,
            yticklabels=False,
            cbar_kws={'label': 'KG entries (log10)', 'shrink': 0.4},
            mask=kg_pivot == 0)
# Grey out zeros
ax3.set_title("C) Knowledge graph\nsupport", fontweight='bold')
ax3.set_xlabel("")
ax3.tick_params(axis='x', labelsize=11, rotation=45)

fig.suptitle("Hormone-pathway associations: co-occurrence network vs knowledge graph validation",
             fontsize=18, fontweight='bold', y=1.02)

fig.savefig(os.path.join(OUT, "kg_validation_heatmaps.png"))
fig.savefig(os.path.join(OUT, "kg_validation_heatmaps.pdf"))
plt.close(fig)

# ── Second figure: Direction mismatch analysis ───────────────────────────
# For key hormone-L0 pairs, show KG direction vs network direction

test_cases = [
    ('Abscisic acid', 'Protein homeostasis', +1, 'Chaperones/HSPs'),
    ('Abscisic acid', 'Redox homeostasis', +1, 'ROS scavenging'),
    ('Abscisic acid', 'Photosynthesis', -1, 'Photosynthesis suppressed'),
    ('Salicylic acid', 'External stimuli response', +1, 'SAR/defense'),
    ('Salicylic acid', 'Redox homeostasis', +1, 'ROS burst'),
    ('Brassinosteroid', 'Cell wall organisation', -1, 'Growth suppressed'),
    ('Brassinosteroid', 'Cell division', -1, 'Division suppressed'),
    ('Brassinosteroid', 'Photosynthesis', -1, 'Photosynthesis suppressed'),
    ('Auxin', 'Cell wall organisation', -1, 'Growth suppressed'),
    ('Auxin', 'Plant organogenesis', -1, 'Organ formation halted'),
    ('Auxin', 'Solute transport', -1, 'PIN transport reduced'),
    ('Gibberellin', 'Solute transport', +1, 'Transporter activation'),
    ('Gibberellin', 'External stimuli response', +1, 'Defense via DELLA'),
    ('Jasmonic acid', 'Photosynthesis', -1, 'Growth-defense tradeoff'),
    ('Ethylene', 'Photosynthesis', -1, 'Senescence'),
    ('Ethylene', 'Cell wall organisation', -1, 'Abscission'),
]

# KG direction: growth hormones (BR, Auxin, GA) PROMOTE their targets normally
# So KG direction is +1 for most
kg_direction = {
    ('Abscisic acid', 'Protein homeostasis'): +1,
    ('Abscisic acid', 'Redox homeostasis'): +1,
    ('Abscisic acid', 'Photosynthesis'): -1,
    ('Salicylic acid', 'External stimuli response'): +1,
    ('Salicylic acid', 'Redox homeostasis'): +1,
    ('Brassinosteroid', 'Cell wall organisation'): +1,  # BR promotes wall expansion normally
    ('Brassinosteroid', 'Cell division'): +1,
    ('Brassinosteroid', 'Photosynthesis'): +1,
    ('Auxin', 'Cell wall organisation'): +1,  # auxin promotes wall loosening
    ('Auxin', 'Plant organogenesis'): +1,
    ('Auxin', 'Solute transport'): +1,  # PIN transport
    ('Gibberellin', 'Solute transport'): +1,
    ('Gibberellin', 'External stimuli response'): +1,
    ('Jasmonic acid', 'Photosynthesis'): -1,
    ('Ethylene', 'Photosynthesis'): -1,
    ('Ethylene', 'Cell wall organisation'): -1,  # ethylene promotes abscission
}

fig, ax = plt.subplots(figsize=(14, 10))

y_labels = []
kg_vals = []
net_vals = []
colors = []

for i, (hormone, l0, net_dir, desc) in enumerate(test_cases):
    row = pair[(pair['hormone'] == hormone) & (pair['L0_bin'] == l0)]
    if row.empty:
        continue
    net_bias = row.iloc[0]['mean_bias']
    kg_dir = kg_direction.get((hormone, l0), 0)

    y_labels.append(f"{hormone[:3]}. - {l0}")
    kg_vals.append(kg_dir * 0.3)  # scale for visibility
    net_vals.append(net_bias)

    # Color: green = KG matches network, red = mismatch
    if (kg_dir > 0 and net_bias > 0) or (kg_dir < 0 and net_bias < 0):
        colors.append('#2A9D8F')  # match
    else:
        colors.append('#E63946')  # mismatch (growth-defense flip)

y_pos = range(len(y_labels))

# KG direction (background bars)
ax.barh(y_pos, kg_vals, height=0.4, color='#AAAAAA', alpha=0.4, label='KG direction (normal)')
# Network direction
ax.barh(y_pos, net_vals, height=0.4, color=colors, alpha=0.85, label='Network bias (stress)')

ax.set_yticks(y_pos)
ax.set_yticklabels(y_labels, fontsize=10)
ax.axvline(0, color='black', lw=0.8)
ax.set_xlabel("Direction (UP <-- | --> DOWN)", fontsize=13)
ax.set_title("Hormone-pathway direction: normal physiology (KG) vs stress response (network)\n"
             "Green = consistent, Red = reversed under stress (growth-defense tradeoff)",
             fontsize=14, fontweight='bold')

# Custom legend
legend_elements = [
    mpatches.Patch(facecolor='#AAAAAA', alpha=0.4, label='KG: normal physiology direction'),
    mpatches.Patch(facecolor='#2A9D8F', alpha=0.85, label='Network: matches KG (defense hormones)'),
    mpatches.Patch(facecolor='#E63946', alpha=0.85, label='Network: reversed (growth hormones suppressed)'),
]
ax.legend(handles=legend_elements, loc='lower right', fontsize=10)

plt.tight_layout()
fig.savefig(os.path.join(OUT, "kg_direction_comparison.png"))
fig.savefig(os.path.join(OUT, "kg_direction_comparison.pdf"))
plt.close(fig)

# Copy to Google Drive
import shutil
GDRIVE_OUT = "/Users/vjx443/Library/CloudStorage/GoogleDrive-mutwil@plant.tools/My Drive/Projects/2026_KingdomStress/CC-Fig 4 data/outputs"
for f in ['kg_validation_heatmaps.png', 'kg_validation_heatmaps.pdf',
          'kg_direction_comparison.png', 'kg_direction_comparison.pdf']:
    shutil.copy2(os.path.join(OUT, f), os.path.join(GDRIVE_OUT, f))

print("Saved: kg_validation_heatmaps.png/pdf, kg_direction_comparison.png/pdf")
