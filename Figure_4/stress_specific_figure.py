"""
Per-stress hormone-L0 heatmaps: 6 panels (one per stress),
each showing hormone (columns) x L0 bin (rows) with edge bias color.
"""

import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

OUT = "/tmp/mercator_outputs"

plt.rcParams.update({
    'font.family': 'Arial', 'font.size': 11,
    'figure.dpi': 300, 'savefig.dpi': 300,
    'savefig.bbox': 'tight', 'figure.facecolor': 'white',
})

agg = pd.read_csv(os.path.join(OUT, "hormone_L0_edges.csv"))

EXCLUDE = {'Phytohormone action', 'Enzyme classification', 'not assigned',
           'Protein modification', 'Protein biosynthesis'}
agg = agg[~agg['target_L0'].isin(EXCLUDE)]

STRESSES = ['Heat', 'Cold', 'Drought', 'Salt', 'Pathogen', 'Heavy metal']
H_SHORT = {'Abscisic acid': 'ABA', 'Salicylic acid': 'SA', 'Brassinosteroid': 'BR',
           'Gibberellin': 'GA', 'Jasmonic acid': 'JA', 'Ethylene': 'Eth.',
           'Auxin': 'Auxin'}
H_ORDER = ['ABA', 'SA', 'BR', 'GA', 'JA', 'Eth.', 'Auxin']

agg['h_short'] = agg['hormone_short'].map(H_SHORT)

# Get consistent L0 bin order (by mean absolute bias across all)
l0_mean_bias = agg.groupby('target_L0')['edge_bias'].apply(lambda x: x.mean()).sort_values()
L0_ORDER = l0_mean_bias.index.tolist()

# Global color range
max_b = max(abs(agg['edge_bias'].min()), abs(agg['edge_bias'].max()))

# ── Figure: 2x3 grid of heatmaps ─────────────────────────────────────────
fig, axes = plt.subplots(2, 3, figsize=(24, 20))

for i, stress in enumerate(STRESSES):
    ax = axes[i // 3, i % 3]
    sdf = agg[agg['stress'] == stress]

    pivot = sdf.pivot_table(index='target_L0', columns='h_short',
                             values='edge_bias', aggfunc='first')
    pivot = pivot.reindex(index=L0_ORDER, columns=H_ORDER)

    sns.heatmap(pivot, cmap='RdBu_r', center=0, vmin=-max_b, vmax=max_b,
                ax=ax, linewidths=0.4, linecolor='white',
                annot=True, fmt='+.2f', annot_kws={'size': 8},
                cbar=i == 0,
                cbar_kws={'label': 'UP - DOWN bias', 'shrink': 0.6} if i == 0 else {})
    ax.set_title(stress, fontsize=18, fontweight='bold')
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.tick_params(axis='x', labelsize=12, rotation=45)
    ax.tick_params(axis='y', labelsize=10)

fig.suptitle("Hormone-pathway co-occurrence bias per stress\n"
             "(Red = co-upregulated, Blue = co-downregulated)",
             fontsize=20, fontweight='bold', y=1.01)
plt.tight_layout()
fig.savefig(os.path.join(OUT, "stress_specific_hormone_L0.png"))
fig.savefig(os.path.join(OUT, "stress_specific_hormone_L0.pdf"))
plt.close(fig)

# ── Second figure: co-occurrence weight (not bias) ───────────────────────
fig, axes = plt.subplots(2, 3, figsize=(24, 20))

for i, stress in enumerate(STRESSES):
    ax = axes[i // 3, i % 3]
    sdf = agg[agg['stress'] == stress]

    pivot = sdf.pivot_table(index='target_L0', columns='h_short',
                             values='mean_weight', aggfunc='first')
    pivot = pivot.reindex(index=L0_ORDER, columns=H_ORDER)

    sns.heatmap(pivot, cmap='YlOrRd', vmin=0,
                ax=ax, linewidths=0.4, linecolor='white',
                annot=True, fmt='.2f', annot_kws={'size': 8},
                cbar=i == 0,
                cbar_kws={'label': 'Co-occurrence frequency', 'shrink': 0.6} if i == 0 else {})
    ax.set_title(stress, fontsize=18, fontweight='bold')
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.tick_params(axis='x', labelsize=12, rotation=45)
    ax.tick_params(axis='y', labelsize=10)

fig.suptitle("Hormone-pathway co-occurrence frequency per stress\n"
             "(How often hormone and pathway DEGs co-occur)",
             fontsize=20, fontweight='bold', y=1.01)
plt.tight_layout()
fig.savefig(os.path.join(OUT, "stress_specific_hormone_L0_weight.png"))
fig.savefig(os.path.join(OUT, "stress_specific_hormone_L0_weight.pdf"))
plt.close(fig)

# ── Third figure: stress-specificity (deviation from cross-stress mean) ──
cross_mean = agg.groupby(['h_short', 'target_L0'])['edge_bias'].mean()

fig, axes = plt.subplots(2, 3, figsize=(24, 20))

for i, stress in enumerate(STRESSES):
    ax = axes[i // 3, i % 3]
    sdf = agg[agg['stress'] == stress].copy()
    sdf['cross_mean'] = sdf.apply(
        lambda r: cross_mean.get((r['h_short'], r['target_L0']), 0), axis=1)
    sdf['deviation'] = sdf['edge_bias'] - sdf['cross_mean']

    pivot = sdf.pivot_table(index='target_L0', columns='h_short',
                             values='deviation', aggfunc='first')
    pivot = pivot.reindex(index=L0_ORDER, columns=H_ORDER)

    max_dev = 0.15
    sns.heatmap(pivot, cmap='PiYG', center=0, vmin=-max_dev, vmax=max_dev,
                ax=ax, linewidths=0.4, linecolor='white',
                annot=True, fmt='+.2f', annot_kws={'size': 8},
                cbar=i == 0,
                cbar_kws={'label': 'Deviation from mean', 'shrink': 0.6} if i == 0 else {})
    ax.set_title(stress, fontsize=18, fontweight='bold')
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.tick_params(axis='x', labelsize=12, rotation=45)
    ax.tick_params(axis='y', labelsize=10)

fig.suptitle("Stress-specific hormone-pathway associations\n"
             "(Deviation of bias from cross-stress mean: green = more UP here, pink = more DOWN here)",
             fontsize=20, fontweight='bold', y=1.01)
plt.tight_layout()
fig.savefig(os.path.join(OUT, "stress_specific_deviation.png"))
fig.savefig(os.path.join(OUT, "stress_specific_deviation.pdf"))
plt.close(fig)

# Copy
import shutil
GDRIVE_OUT = "/Users/vjx443/Library/CloudStorage/GoogleDrive-mutwil@plant.tools/My Drive/Projects/2026_KingdomStress/CC-Fig 4 data/outputs"
for f in os.listdir(OUT):
    if f.startswith("stress_specific_"):
        shutil.copy2(os.path.join(OUT, f), os.path.join(GDRIVE_OUT, f))

print("Saved: stress_specific_hormone_L0.png/pdf")
print("Saved: stress_specific_hormone_L0_weight.png/pdf")
print("Saved: stress_specific_deviation.png/pdf")
