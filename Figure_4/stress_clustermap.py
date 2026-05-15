"""
Stress response clustermap based on L1 edge bias profiles.
Left: clustermap showing how similar stresses are based on their edge patterns.
Right: stress-unique edge counts.
"""

import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.cluster.hierarchy import linkage, dendrogram
from scipy.spatial.distance import pdist, squareform
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

# ── Load per-stress L1 edge bias ──────────────────────────────────────────
print("Loading per-stress L1 edges...")
stress_edge_data = {}
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

    rows = {}
    for (src, tgt), vals in edges.items():
        edge = f'{src} -- {tgt}'
        rows[edge] = vals['up'] - vals['down']
    stress_edge_data[stress] = rows

# Build edge x stress bias matrix
all_edges = set()
for v in stress_edge_data.values():
    all_edges.update(v.keys())

bias_mat = pd.DataFrame(0.0, index=sorted(all_edges), columns=STRESSES)
for stress in STRESSES:
    for edge, bias in stress_edge_data[stress].items():
        bias_mat.loc[edge, stress] = bias

print(f"  Edge x stress matrix: {bias_mat.shape}")

# ── Compute overlap coefficient between stresses ──────────────────────────
# OC(A,B) = |A intersect B| / min(|A|, |B|)
# Use edge sets (edges with |bias| > threshold = active)
BIAS_THRESH = 0.02

stress_active = {}
for stress in STRESSES:
    active_up = set(bias_mat.index[bias_mat[stress] > BIAS_THRESH])
    active_down = set(bias_mat.index[bias_mat[stress] < -BIAS_THRESH])
    stress_active[stress] = {'up': active_up, 'down': active_down,
                              'all': active_up | active_down}

# Overlap coefficient matrix (on all active edges)
oc_mat = pd.DataFrame(1.0, index=STRESSES, columns=STRESSES)
for i, s1 in enumerate(STRESSES):
    for j, s2 in enumerate(STRESSES):
        if i == j:
            continue
        a = stress_active[s1]['all']
        b = stress_active[s2]['all']
        if min(len(a), len(b)) == 0:
            oc_mat.loc[s1, s2] = 0
        else:
            oc_mat.loc[s1, s2] = len(a & b) / min(len(a), len(b))

# Linkage from overlap (convert similarity to distance)
oc_dist = 1 - squareform(oc_mat.values, checks=False)
# Fix diagonal issues
oc_dist_full = 1 - oc_mat.values
np.fill_diagonal(oc_dist_full, 0)
oc_dist = squareform(oc_dist_full)
Z = linkage(oc_dist, method='average')

# Stress-unique counts
stress_unique = {'Heat': 313, 'Cold': 89, 'Drought': 522, 'Salt': 314,
                 'Pathogen': 33, 'Heavy metal': 131}

# Total edges per stress
stress_totals = {s: len(stress_active[s]['all']) for s in STRESSES}

# ── Figure: Combined clustermap + unique edges ───────────────────────────
fig = plt.figure(figsize=(18, 10))
gs = fig.add_gridspec(1, 3, width_ratios=[0.8, 2, 1.2], wspace=0.15)

# Panel 1: Dendrogram
ax_dendro = fig.add_subplot(gs[0])
dendro = dendrogram(Z, labels=STRESSES, orientation='left', ax=ax_dendro,
                     leaf_font_size=14, color_threshold=0.3,
                     above_threshold_color='grey')

ylbls = ax_dendro.get_ymajorticklabels()
for lbl in ylbls:
    lbl.set_color(STRESS_COLORS.get(lbl.get_text(), 'black'))
    lbl.set_fontweight('bold')

ax_dendro.set_xlabel("1 - Overlap coefficient", fontsize=12)
ax_dendro.set_title("A) Stress similarity", fontweight='bold', fontsize=14)
ax_dendro.spines['top'].set_visible(False)
ax_dendro.spines['right'].set_visible(False)

# Get dendrogram leaf order
dendro_order = [STRESSES[i] for i in dendro['leaves']]

# Panel 2: Overlap coefficient heatmap (match dendrogram order)
ax_corr = fig.add_subplot(gs[1])
oc_ordered = oc_mat.loc[dendro_order, dendro_order]

sns.heatmap(oc_ordered, cmap='YlOrRd', vmin=0.5, vmax=1,
            ax=ax_corr, linewidths=1.5, linecolor='white',
            annot=True, fmt='.2f', annot_kws={'size': 16, 'fontweight': 'bold'},
            square=True,
            yticklabels=dendro_order, xticklabels=dendro_order,
            cbar_kws={'label': 'Overlap coefficient', 'shrink': 0.6})
ax_corr.set_title("B) Pairwise overlap coefficient\n(shared active L1 edges)", fontweight='bold', fontsize=14)
ax_corr.tick_params(axis='x', labelsize=13, rotation=45)
ax_corr.tick_params(axis='y', labelsize=13)

# Color axis labels
for lbl in ax_corr.get_xmajorticklabels():
    lbl.set_color(STRESS_COLORS.get(lbl.get_text(), 'black'))
    lbl.set_fontweight('bold')
for lbl in ax_corr.get_ymajorticklabels():
    lbl.set_color(STRESS_COLORS.get(lbl.get_text(), 'black'))
    lbl.set_fontweight('bold')

# Panel 3: Unique + total edges per stress (match dendrogram order)
ax_bars = fig.add_subplot(gs[2])

y_pos = range(len(dendro_order))
total_vals = [stress_totals[s] for s in dendro_order]
unique_vals = [stress_unique[s] for s in dendro_order]
colors = [STRESS_COLORS[s] for s in dendro_order]

ax_bars.barh(y_pos, total_vals, color=colors, alpha=0.3, height=0.7, label='Total edges')
ax_bars.barh(y_pos, unique_vals, color=colors, alpha=0.9, height=0.7, label='Unique edges')

ax_bars.set_yticks(y_pos)
ax_bars.set_yticklabels(dendro_order, fontsize=13, fontweight='bold')
for i, s in enumerate(dendro_order):
    ax_bars.get_yticklabels()[i].set_color(STRESS_COLORS[s])

for i, (t, u) in enumerate(zip(total_vals, unique_vals)):
    ax_bars.text(t + 100, i, f'{t:,}', va='center', fontsize=10, alpha=0.5)
    ax_bars.text(u + 100, i - 0.15, f'{u}', va='center', fontsize=10, fontweight='bold')

ax_bars.set_xlabel("Number of L1 edges", fontsize=12)
ax_bars.set_title("C) Total & unique edges", fontweight='bold', fontsize=14)
ax_bars.legend(fontsize=10, loc='lower right')
ax_bars.invert_yaxis()

fig.suptitle("Stress response similarity based on L1 co-occurrence edge profiles",
             fontsize=17, fontweight='bold', y=1.02)

fig.savefig(os.path.join(OUT, "stress_clustermap.png"))
fig.savefig(os.path.join(OUT, "stress_clustermap.pdf"))
plt.close(fig)

# Print overlap matrix
print("\nStress-stress overlap coefficient (on L1 active edges):")
print(oc_ordered.to_string(float_format='{:.3f}'.format))

# Copy
import shutil
GDRIVE_OUT = "/Users/vjx443/Library/CloudStorage/GoogleDrive-mutwil@plant.tools/My Drive/Projects/2026_KingdomStress/CC-Fig 4 data/outputs"
shutil.copy2(os.path.join(OUT, "stress_clustermap.png"), os.path.join(GDRIVE_OUT, "stress_clustermap.png"))
shutil.copy2(os.path.join(OUT, "stress_clustermap.pdf"), os.path.join(GDRIVE_OUT, "stress_clustermap.pdf"))
print("\nSaved: stress_clustermap.png/pdf")
