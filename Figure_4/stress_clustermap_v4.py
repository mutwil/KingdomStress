"""
Stress clustermap v4:
- Split-diagonal heatmap: upper triangle = UP overlap, lower triangle = DOWN overlap
- Dendrogram on the left
- Bar chart with UP and DOWN unique edges
"""

import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.cluster.hierarchy import linkage, dendrogram, leaves_list
from scipy.spatial.distance import squareform
import matplotlib.patches as mpatches
import warnings
warnings.filterwarnings('ignore')

OUT = "/tmp/mercator_outputs"
BASE = "/tmp/mercator_data"

plt.rcParams.update({
    'font.family': 'Arial', 'font.size': 12,
    'figure.dpi': 300, 'savefig.dpi': 300,
    'savefig.bbox': 'tight', 'figure.facecolor': 'white',
})

STRESSES = ['Heat', 'Cold', 'Drought', 'Salt', 'Pathogen', 'Heavy metal']
EXCLUDE_L0 = {'Enzyme classification', 'not assigned', 'Protein modification',
              'Protein biosynthesis'}
STRESS_COLORS = {
    'Heat': '#E63946', 'Cold': '#457B9D', 'Drought': '#E9C46A',
    'Salt': '#2A9D8F', 'Pathogen': '#8338EC', 'Heavy metal': '#6D6875',
}
BIAS_THRESH = 0.02

# ── Load per-stress L1 edges ─────────────────────────────────────────────
print("Loading per-stress L1 edges...")
stress_edge_bias = {}
for stress in STRESSES:
    edges = {}
    for fname, d in [('Mercator_network_UP_Level1 (All_organ).csv', 'up'),
                      ('Mercator_network_DOWN_Level1 (All_organ).csv', 'down')]:
        path = os.path.join(BASE, 'Stresses', stress, fname)
        if not os.path.exists(path):
            continue
        df = pd.read_csv(path, index_col=0)
        for _, r in df.iterrows():
            if r['source'].split('.')[0] in EXCLUDE_L0 or r['target'].split('.')[0] in EXCLUDE_L0:
                continue
            key = tuple(sorted([r['source'], r['target']]))
            edges.setdefault(key, {'up': 0, 'down': 0})
            edges[key][d] = r['weight']
    stress_edge_bias[stress] = {k: v['up'] - v['down'] for k, v in edges.items()}

# ── Build directional edge sets, remove universal ─────────────────────────
stress_up = {}
stress_down = {}
for stress in STRESSES:
    stress_up[stress] = {k for k, b in stress_edge_bias[stress].items() if b > BIAS_THRESH}
    stress_down[stress] = {k for k, b in stress_edge_bias[stress].items() if b < -BIAS_THRESH}

universal_up = set.intersection(*stress_up.values())
universal_down = set.intersection(*stress_down.values())

for stress in STRESSES:
    stress_up[stress] -= universal_up
    stress_down[stress] -= universal_down
    print(f"  {stress}: {len(stress_up[stress])} UP, {len(stress_down[stress])} DOWN (non-universal)")

# ── Overlap coefficient matrices: separate for UP and DOWN ────────────────
def overlap_coef(set_dict, stresses):
    mat = pd.DataFrame(1.0, index=stresses, columns=stresses)
    for s1 in stresses:
        for s2 in stresses:
            if s1 == s2:
                continue
            a, b = set_dict[s1], set_dict[s2]
            min_size = min(len(a), len(b))
            mat.loc[s1, s2] = len(a & b) / min_size if min_size > 0 else 0
    return mat

oc_up = overlap_coef(stress_up, STRESSES)
oc_down = overlap_coef(stress_down, STRESSES)

# Combined OC for clustering (average of UP and DOWN)
oc_combined = (oc_up + oc_down) / 2

print("\nUP overlap coefficients:")
print(oc_up.to_string(float_format='{:.2f}'.format))
print("\nDOWN overlap coefficients:")
print(oc_down.to_string(float_format='{:.2f}'.format))

# ── Linkage from combined OC ─────────────────────────────────────────────
oc_dist = 1 - oc_combined.values
np.fill_diagonal(oc_dist, 0)
dist_condensed = squareform(oc_dist)
Z = linkage(dist_condensed, method='average')
leaf_order = leaves_list(Z)
reordered = [STRESSES[i] for i in leaf_order]

# ── Unique edge counts (UP and DOWN separate) ────────────────────────────
unique_up = {}
unique_down = {}
for stress in STRESSES:
    others_up = set.union(*[stress_up[s] for s in STRESSES if s != stress])
    others_down = set.union(*[stress_down[s] for s in STRESSES if s != stress])
    unique_up[stress] = len(stress_up[stress] - others_up)
    unique_down[stress] = len(stress_down[stress] - others_down)

# ── Figure ────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(22, 10))
gs = fig.add_gridspec(1, 3, width_ratios=[0.5, 2, 1.3], wspace=0.08)

# ── Panel A: Dendrogram ──────────────────────────────────────────────────
ax_dendro = fig.add_subplot(gs[0])
dendro_data = dendrogram(Z, labels=STRESSES, orientation='left', ax=ax_dendro,
                          leaf_font_size=14, color_threshold=0,
                          above_threshold_color='#555555')

for lbl in ax_dendro.get_ymajorticklabels():
    lbl.set_color(STRESS_COLORS.get(lbl.get_text(), 'black'))
    lbl.set_fontweight('bold')
    lbl.set_fontsize(14)

ax_dendro.set_xlabel("1 - Overlap coefficient", fontsize=11)
ax_dendro.set_title("A) Dendrogram", fontweight='bold', fontsize=14)
ax_dendro.spines['top'].set_visible(False)
ax_dendro.spines['right'].set_visible(False)

# Get dendrogram order for heatmap
dendro_leaves = [STRESSES[i] for i in dendro_data['leaves']]

# ── Panel B: Split-diagonal heatmap ──────────────────────────────────────
ax_heat = fig.add_subplot(gs[1])

n = len(dendro_leaves)
# Reorder both matrices
up_ordered = oc_up.loc[dendro_leaves, dendro_leaves].values
down_ordered = oc_down.loc[dendro_leaves, dendro_leaves].values

# Build split matrix: upper triangle = UP, lower triangle = DOWN
split = np.zeros((n, n))
for i in range(n):
    for j in range(n):
        if i == j:
            split[i, j] = np.nan  # diagonal
        elif i < j:
            split[i, j] = up_ordered[i, j]  # upper = UP
        else:
            split[i, j] = down_ordered[i, j]  # lower = DOWN

# Custom colormap plotting with imshow
im = ax_heat.imshow(split, cmap='YlOrRd', vmin=0, vmax=0.8, aspect='equal')

# Annotate cells
for i in range(n):
    for j in range(n):
        if i == j:
            # Diagonal: stress name
            ax_heat.text(j, i, dendro_leaves[i], ha='center', va='center',
                        fontsize=12, fontweight='bold',
                        color=STRESS_COLORS.get(dendro_leaves[i], 'black'))
        else:
            val = split[i, j]
            color = 'white' if val > 0.5 else 'black'
            ax_heat.text(j, i, f'{val:.2f}', ha='center', va='center',
                        fontsize=13, fontweight='bold', color=color)

# Draw diagonal line
ax_heat.plot([-0.5, n-0.5], [-0.5, n-0.5], 'k-', lw=2)

# Grid
for i in range(n + 1):
    ax_heat.axhline(i - 0.5, color='white', lw=2)
    ax_heat.axvline(i - 0.5, color='white', lw=2)

ax_heat.set_xticks(range(n))
ax_heat.set_xticklabels(dendro_leaves, fontsize=13, rotation=45, ha='right')
ax_heat.set_yticks(range(n))
ax_heat.set_yticklabels(dendro_leaves, fontsize=13)

for lbl in ax_heat.get_xticklabels():
    lbl.set_color(STRESS_COLORS.get(lbl.get_text(), 'black'))
    lbl.set_fontweight('bold')
for lbl in ax_heat.get_yticklabels():
    lbl.set_color(STRESS_COLORS.get(lbl.get_text(), 'black'))
    lbl.set_fontweight('bold')

# Triangle labels
ax_heat.text(n - 1.2, 0.3, 'UP edges', fontsize=14, fontweight='bold',
             ha='center', color='#8B0000', style='italic')
ax_heat.text(0.3, n - 1.2, 'DOWN edges', fontsize=14, fontweight='bold',
             ha='center', color='#8B0000', style='italic')

# Colorbar
cbar = plt.colorbar(im, ax=ax_heat, shrink=0.6, pad=0.02)
cbar.set_label('Overlap coefficient', fontsize=12)

ax_heat.set_title("B) Split overlap: UP (upper) vs DOWN (lower)\n"
                   "(universal edges removed)",
                   fontweight='bold', fontsize=14)

# ── Panel C: UP and DOWN unique edges ────────────────────────────────────
ax_bars = fig.add_subplot(gs[2])

y_pos = np.arange(len(dendro_leaves))
bar_h = 0.35

up_vals = [unique_up[s] for s in dendro_leaves]
down_vals = [unique_down[s] for s in dendro_leaves]
total_up = [len(stress_up[s]) for s in dendro_leaves]
total_down = [len(stress_down[s]) for s in dendro_leaves]
colors = [STRESS_COLORS[s] for s in dendro_leaves]

# Total (light)
ax_bars.barh(y_pos - bar_h/2, total_up, height=bar_h, color=colors, alpha=0.2)
ax_bars.barh(y_pos + bar_h/2, [-t for t in total_down], height=bar_h, color=colors, alpha=0.2)

# Unique (solid)
bars_up = ax_bars.barh(y_pos - bar_h/2, up_vals, height=bar_h, color=colors, alpha=0.9,
                         label='Unique UP')
bars_dn = ax_bars.barh(y_pos + bar_h/2, [-d for d in down_vals], height=bar_h, color=colors,
                         alpha=0.6, label='Unique DOWN', hatch='///')

# Labels
for i in range(len(dendro_leaves)):
    if up_vals[i] > 0:
        ax_bars.text(up_vals[i] + 30, y_pos[i] - bar_h/2,
                     str(up_vals[i]), va='center', fontsize=9, fontweight='bold')
    if down_vals[i] > 0:
        ax_bars.text(-down_vals[i] - 30, y_pos[i] + bar_h/2,
                     str(down_vals[i]), va='center', fontsize=9, fontweight='bold', ha='right')

ax_bars.axvline(0, color='black', lw=0.8)
ax_bars.set_yticks(y_pos)
ax_bars.set_yticklabels(dendro_leaves, fontsize=13, fontweight='bold')
for i, s in enumerate(dendro_leaves):
    ax_bars.get_yticklabels()[i].set_color(STRESS_COLORS[s])

ax_bars.set_xlabel("UP edges -->          <-- DOWN edges", fontsize=11)
ax_bars.set_title("C) Unique directional edges\n(solid = unique, light = total)",
                   fontweight='bold', fontsize=14)
ax_bars.invert_yaxis()

# Legend
legend_elements = [
    mpatches.Patch(facecolor='grey', alpha=0.9, label='Unique UP'),
    mpatches.Patch(facecolor='grey', alpha=0.6, hatch='///', label='Unique DOWN'),
    mpatches.Patch(facecolor='grey', alpha=0.2, label='Total (non-universal)'),
]
ax_bars.legend(handles=legend_elements, fontsize=9, loc='lower right')

fig.suptitle("Stress response similarity based on directional L1 co-occurrence edges",
             fontsize=18, fontweight='bold', y=1.02)

fig.savefig(os.path.join(OUT, "stress_clustermap_v4.png"))
fig.savefig(os.path.join(OUT, "stress_clustermap_v4.pdf"))
plt.close(fig)

# Copy
import shutil
GDRIVE_OUT = "/Users/vjx443/Library/CloudStorage/GoogleDrive-mutwil@plant.tools/My Drive/Projects/2026_KingdomStress/CC-Fig 4 data/outputs"
shutil.copy2(os.path.join(OUT, "stress_clustermap_v4.png"), os.path.join(GDRIVE_OUT, "stress_clustermap_v4.png"))
shutil.copy2(os.path.join(OUT, "stress_clustermap_v4.pdf"), os.path.join(GDRIVE_OUT, "stress_clustermap_v4.pdf"))
print("\nSaved: stress_clustermap_v4.png/pdf")
