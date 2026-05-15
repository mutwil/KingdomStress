"""
Stress clustermap v3:
- Use sns.clustermap for proper dendrogram-heatmap alignment
- Compute overlap on DIRECTIONAL edges (UP and DOWN separately)
- Exclude universal edges (present in all 6 stresses with same direction)
"""

import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.spatial.distance import squareform
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

# ── Build directional edge sets per stress ────────────────────────────────
# An edge is "UP" if bias > threshold, "DOWN" if bias < -threshold
stress_up = {}
stress_down = {}
for stress in STRESSES:
    stress_up[stress] = {k for k, b in stress_edge_bias[stress].items() if b > BIAS_THRESH}
    stress_down[stress] = {k for k, b in stress_edge_bias[stress].items() if b < -BIAS_THRESH}
    print(f"  {stress}: {len(stress_up[stress])} UP, {len(stress_down[stress])} DOWN")

# ── Remove universal edges (same direction in ALL stresses) ──────────────
universal_up = set.intersection(*stress_up.values())
universal_down = set.intersection(*stress_down.values())
print(f"\nUniversal UP edges (all 6 stresses): {len(universal_up)}")
print(f"Universal DOWN edges (all 6 stresses): {len(universal_down)}")

for stress in STRESSES:
    stress_up[stress] -= universal_up
    stress_down[stress] -= universal_down

print("\nAfter removing universal edges:")
for stress in STRESSES:
    print(f"  {stress}: {len(stress_up[stress])} UP, {len(stress_down[stress])} DOWN")

# ── Compute overlap coefficient on directional edges ─────────────────────
# Combine UP and DOWN as separate "tagged" edges
def tagged_edges(stress):
    up = {('UP', e) for e in stress_up[stress]}
    down = {('DOWN', e) for e in stress_down[stress]}
    return up | down

stress_tagged = {s: tagged_edges(s) for s in STRESSES}

oc_mat = pd.DataFrame(1.0, index=STRESSES, columns=STRESSES)
for i, s1 in enumerate(STRESSES):
    for j, s2 in enumerate(STRESSES):
        if i == j:
            continue
        a = stress_tagged[s1]
        b = stress_tagged[s2]
        min_size = min(len(a), len(b))
        if min_size == 0:
            oc_mat.loc[s1, s2] = 0
        else:
            oc_mat.loc[s1, s2] = len(a & b) / min_size

print("\nOverlap coefficient matrix (directional, non-universal):")
print(oc_mat.to_string(float_format='{:.3f}'.format))

# ── Stress-unique counts (directional) ───────────────────────────────────
stress_unique = {}
for stress in STRESSES:
    others = set()
    for s in STRESSES:
        if s != stress:
            others |= stress_tagged[s]
    unique = stress_tagged[stress] - others
    stress_unique[stress] = len(unique)
    print(f"  {stress} unique directional edges: {len(unique)}")

stress_totals = {s: len(stress_tagged[s]) for s in STRESSES}

# ── Figure 1: sns.clustermap of overlap coefficient ──────────────────────
# Convert to distance for linkage
oc_dist = 1 - oc_mat.values
np.fill_diagonal(oc_dist, 0)

from scipy.cluster.hierarchy import linkage

row_colors = [STRESS_COLORS[s] for s in STRESSES]
dist_condensed = squareform(oc_dist)
Z = linkage(dist_condensed, method='average')

g = sns.clustermap(oc_mat, cmap='YlOrRd', vmin=0, vmax=1,
                    linewidths=1.5, linecolor='white',
                    annot=True, fmt='.2f', annot_kws={'size': 16, 'fontweight': 'bold'},
                    figsize=(10, 10),
                    row_colors=row_colors, col_colors=row_colors,
                    dendrogram_ratio=(0.15, 0.15),
                    cbar_kws={'label': 'Overlap coefficient'},
                    row_linkage=Z, col_linkage=Z)

g.ax_heatmap.set_xticklabels(g.ax_heatmap.get_xmajorticklabels(),
                               fontsize=14, fontweight='bold', rotation=45)
g.ax_heatmap.set_yticklabels(g.ax_heatmap.get_ymajorticklabels(),
                               fontsize=14, fontweight='bold', rotation=0)

# Color tick labels
reordered = [STRESSES[i] for i in g.dendrogram_row.reordered_ind]
for lbl in g.ax_heatmap.get_yticklabels():
    lbl.set_color(STRESS_COLORS.get(lbl.get_text(), 'black'))
for lbl in g.ax_heatmap.get_xticklabels():
    lbl.set_color(STRESS_COLORS.get(lbl.get_text(), 'black'))

g.fig.suptitle("Stress similarity: overlap of directional L1 edges\n"
               "(universal edges removed, UP and DOWN counted separately)",
               fontsize=15, fontweight='bold', y=1.03)

g.savefig(os.path.join(OUT, "stress_clustermap_v3.png"))
g.savefig(os.path.join(OUT, "stress_clustermap_v3.pdf"))
plt.close()

# ── Figure 2: Clustermap + unique edges side by side ─────────────────────
fig = plt.figure(figsize=(18, 8))
gs = fig.add_gridspec(1, 2, width_ratios=[2, 1], wspace=0.3)

# Left: re-draw heatmap with dendrogram using the same order
ax_heat = fig.add_subplot(gs[0])

oc_ordered = oc_mat.loc[reordered, reordered]
sns.heatmap(oc_ordered, cmap='YlOrRd', vmin=0, vmax=1,
            ax=ax_heat, linewidths=1.5, linecolor='white',
            annot=True, fmt='.2f', annot_kws={'size': 16, 'fontweight': 'bold'},
            square=True,
            cbar_kws={'label': 'Overlap coefficient', 'shrink': 0.6})

ax_heat.set_title("A) Stress similarity\n(directional overlap, universal removed)",
                   fontweight='bold', fontsize=14)
ax_heat.tick_params(axis='x', labelsize=13, rotation=45)
ax_heat.tick_params(axis='y', labelsize=13, rotation=0)

for lbl in ax_heat.get_yticklabels():
    lbl.set_color(STRESS_COLORS.get(lbl.get_text(), 'black'))
    lbl.set_fontweight('bold')
for lbl in ax_heat.get_xticklabels():
    lbl.set_color(STRESS_COLORS.get(lbl.get_text(), 'black'))
    lbl.set_fontweight('bold')

# Right: unique edges bar chart (same order)
ax_bars = fig.add_subplot(gs[1])

y_pos = range(len(reordered))
total_vals = [stress_totals[s] for s in reordered]
unique_vals = [stress_unique[s] for s in reordered]
colors = [STRESS_COLORS[s] for s in reordered]

ax_bars.barh(y_pos, total_vals, color=colors, alpha=0.3, height=0.7, label='Total non-universal')
ax_bars.barh(y_pos, unique_vals, color=colors, alpha=0.9, height=0.7, label='Stress-unique')

ax_bars.set_yticks(y_pos)
ax_bars.set_yticklabels(reordered, fontsize=13, fontweight='bold')
for i, s in enumerate(reordered):
    ax_bars.get_yticklabels()[i].set_color(STRESS_COLORS[s])

for i, (t, u) in enumerate(zip(total_vals, unique_vals)):
    ax_bars.text(t + 50, i, f'{t:,}', va='center', fontsize=10, alpha=0.5)
    ax_bars.text(u + 50, i - 0.15, f'{u}', va='center', fontsize=10, fontweight='bold')

ax_bars.set_xlabel("Number of directional L1 edges", fontsize=12)
ax_bars.set_title("B) Total & unique edges\n(after removing universal)", fontweight='bold', fontsize=14)
ax_bars.legend(fontsize=10, loc='lower right')
ax_bars.invert_yaxis()

fig.suptitle("Stress response similarity based on directional L1 co-occurrence edges",
             fontsize=17, fontweight='bold', y=1.02)

fig.savefig(os.path.join(OUT, "stress_clustermap_combined_v3.png"))
fig.savefig(os.path.join(OUT, "stress_clustermap_combined_v3.pdf"))
plt.close(fig)

# Copy
import shutil
GDRIVE_OUT = "/Users/vjx443/Library/CloudStorage/GoogleDrive-mutwil@plant.tools/My Drive/Projects/2026_KingdomStress/CC-Fig 4 data/outputs"
for f in os.listdir(OUT):
    if 'clustermap' in f and 'v3' in f:
        shutil.copy2(os.path.join(OUT, f), os.path.join(GDRIVE_OUT, f))

print("\nSaved: stress_clustermap_v3.png/pdf, stress_clustermap_combined_v3.png/pdf")
