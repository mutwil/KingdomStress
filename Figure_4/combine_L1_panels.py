"""
Combine Level1 direction bias bar chart and bias heatmap into a single figure.
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
    'font.family': 'Arial', 'font.size': 10, 'axes.titlesize': 12,
    'axes.labelsize': 10, 'figure.dpi': 300, 'savefig.dpi': 300,
    'savefig.bbox': 'tight', 'figure.facecolor': 'white',
})

STRESSES = ['Heat', 'Cold', 'Drought', 'Salt', 'Pathogen', 'Heavy metal']

def shorten(s, n=40):
    return s if len(s) <= n else s[:n-2] + '..'

# ── Load data ─────────────────────────────────────────────────────────────
summary = pd.read_csv(os.path.join(OUT, "bin_summary_L1.csv"))
stress_df = pd.read_csv(os.path.join(OUT, "bin_per_stress_L1.csv"))

# Get top/bottom 25 by direction bias
summary = summary.dropna(subset=['bin_name']).sort_values('direction_bias')
top25 = summary.tail(25)
bot25 = summary.head(25)
selected = pd.concat([bot25, top25])
bin_order = selected['bin_name'].tolist()

# Build bias heatmap pivot for the same bins
bias_pivot = stress_df.pivot_table(index='bin_name', columns='stress', values='bias', aggfunc='first')
bias_pivot = bias_pivot.reindex(bin_order)
stress_order = [s for s in STRESSES if s in bias_pivot.columns]
bias_pivot = bias_pivot[stress_order]

# ── Combined figure ───────────────────────────────────────────────────────
# 3-column layout: bar chart | labels | heatmap
fig = plt.figure(figsize=(24, 22))
gs = fig.add_gridspec(1, 3, width_ratios=[0.35, 1.6, 0.5], wspace=0.02)
ax1 = fig.add_subplot(gs[0])
ax_labels = fig.add_subplot(gs[1], sharey=ax1)
ax2 = fig.add_subplot(gs[2], sharey=ax1)

# Panel A: Diverging bar chart (mirrored so bars grow leftward)
colors = ['#E63946' if b > 0 else '#457B9D' for b in selected['direction_bias']]
ax1.barh(range(len(selected)), -selected['direction_bias'].values, color=colors, alpha=0.85,
         height=0.75)
ax1.set_yticks(range(len(selected)))
ax1.set_yticklabels([])
ax1.axvline(0, color='black', lw=0.5)
# Flip x so positive bias (UP) extends right visually but axis reads correctly
xticks = ax1.get_xticks()
ax1.set_xticklabels([f'{abs(x):.1f}' for x in xticks])
ax1.set_xlabel("Direction bias\n(mean UP - mean DOWN)", fontsize=20)
ax1.set_title("A) Overall bias", fontweight='bold', fontsize=24)
ax1.tick_params(axis='x', labelsize=16)

# Center column: bin name labels (short form, capitalized, left-aligned)
ax_labels.set_xlim(0, 1)
ax_labels.set_yticks(range(len(selected)))
for i, name in enumerate(selected['bin_name']):
    # Take part after last dot, capitalize first letter
    short_name = name.split('.')[-1].strip()
    short_name = short_name[0].upper() + short_name[1:] if short_name else name
    ax_labels.text(0.05, i, short_name, ha='left', va='center', fontsize=17)
ax_labels.set_yticklabels([])
ax_labels.tick_params(left=False, bottom=False, labelbottom=False)
ax_labels.spines['top'].set_visible(False)
ax_labels.spines['bottom'].set_visible(False)
ax_labels.spines['left'].set_visible(False)
ax_labels.spines['right'].set_visible(False)

# Panel B: Heatmap
max_abs = max(abs(bias_pivot.min().min()), abs(bias_pivot.max().max()))
sns.heatmap(bias_pivot, cmap='RdBu_r', center=0, vmin=-max_abs, vmax=max_abs,
            ax=ax2, linewidths=0.4, linecolor='white',
            yticklabels=False,
            cbar_kws={'label': 'UP - DOWN bias', 'shrink': 0.3, 'aspect': 20})
ax2.set_title("B) Bias per stress", fontweight='bold', fontsize=24)
ax2.set_xlabel("Stress", fontsize=20)
ax2.tick_params(axis='x', labelsize=16, rotation=45)
ax2.set_ylabel("")

# Set y limits with padding so bars aren't clipped
ax1.set_ylim(-0.8, len(selected) - 0.2)

fig.suptitle("MapMan Level1 bin direction bias across stresses", fontsize=28, y=0.98)
fig.subplots_adjust(top=0.95, bottom=0.06, left=0.03, right=0.97)
fig.savefig(os.path.join(OUT, "combined_L1_direction.png"))
fig.savefig(os.path.join(OUT, "combined_L1_direction.pdf"))
plt.close(fig)

# Copy to Google Drive
import shutil
GDRIVE_OUT = "/Users/vjx443/Library/CloudStorage/GoogleDrive-mutwil@plant.tools/My Drive/Projects/2026_KingdomStress/CC-Fig 4 data/outputs"
for f in ['combined_L1_direction.png', 'combined_L1_direction.pdf']:
    shutil.copy2(os.path.join(OUT, f), os.path.join(GDRIVE_OUT, f))
print("Saved: combined_L1_direction.png/pdf")
