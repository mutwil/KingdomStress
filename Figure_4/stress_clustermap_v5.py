"""
Stress clustermap v5:
- Use sns.clustermap for proper alignment
- Draw split triangle manually on top of the clustermap
- Separate bar chart figure for unique edges
"""

import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.cluster.hierarchy import linkage, leaves_list
from scipy.spatial.distance import squareform
import matplotlib.patches as mpatches
from matplotlib.patches import Polygon
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
# Threshold: at least 2 experiments difference, scaled by N per stress
# bias = (n_up - n_down) / N, so bias > 2/N means at least 2 experiments
STRESS_N = {'Heat': 96, 'Cold': 70, 'Drought': 92, 'Salt': 68,
            'Pathogen': 61, 'Heavy metal': 46}
MIN_EXPERIMENTS = 2

# ── Load and process (same as v4) ─────────────────────────────────────────
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
    stress_edge_bias[stress] = edges  # keep full up/down weights

# Load species co-occurrence data
import pickle
with open(os.path.join(OUT, "species_cooccurrence_L1.pkl"), 'rb') as f:
    species_cooc = pickle.load(f)

MIN_SPECIES = 5
stress_up, stress_down = {}, {}
for stress in STRESSES:
    n = STRESS_N[stress]
    bias_thresh = MIN_EXPERIMENTS / n
    sp_data = species_cooc.get(stress, {})

    up_set = set()
    down_set = set()
    for k, v in stress_edge_bias[stress].items():
        bias = v['up'] - v['down']
        sp = sp_data.get(k, {'n_species_up': 0, 'n_species_down': 0})

        # UP edge: bias exceeds threshold AND co-occurs in >= 2 species as UP
        if bias > bias_thresh and sp['n_species_up'] >= MIN_SPECIES:
            up_set.add(k)
        # DOWN edge: bias exceeds threshold AND co-occurs in >= 2 species as DOWN
        if bias < -bias_thresh and sp['n_species_down'] >= MIN_SPECIES:
            down_set.add(k)

    stress_up[stress] = up_set
    stress_down[stress] = down_set
    print(f"  {stress}: bias_thresh={bias_thresh:.4f}, min_species={MIN_SPECIES} -> {len(up_set)} UP, {len(down_set)} DOWN")

universal_up = set.intersection(*stress_up.values())
universal_down = set.intersection(*stress_down.values())
for stress in STRESSES:
    stress_up[stress] -= universal_up
    stress_down[stress] -= universal_down
    print(f"  {stress}: {len(stress_up[stress])} UP, {len(stress_down[stress])} DOWN")

def jaccard(set_dict):
    mat = pd.DataFrame(1.0, index=STRESSES, columns=STRESSES)
    for s1 in STRESSES:
        for s2 in STRESSES:
            if s1 == s2:
                continue
            a, b = set_dict[s1], set_dict[s2]
            union = len(a | b)
            mat.loc[s1, s2] = len(a & b) / union if union > 0 else 0
    return mat

jac_up = jaccard(stress_up)
jac_down = jaccard(stress_down)
jac_combined = (jac_up + jac_down) / 2

# Linkage
oc_dist = 1 - jac_combined.values
np.fill_diagonal(oc_dist, 0)
Z = linkage(squareform(oc_dist), method='average')
leaf_order = leaves_list(Z)
reordered = [STRESSES[i] for i in leaf_order]

# Unique counts
unique_up, unique_down = {}, {}
for stress in STRESSES:
    others_up = set.union(*[stress_up[s] for s in STRESSES if s != stress])
    others_down = set.union(*[stress_down[s] for s in STRESSES if s != stress])
    unique_up[stress] = len(stress_up[stress] - others_up)
    unique_down[stress] = len(stress_down[stress] - others_down)

# ── Figure 1: sns.clustermap with UP overlap ──────────────────────────────
# We'll make two clustermaps (UP and DOWN) and a combined bar chart

# UP clustermap
row_colors = pd.Series({s: STRESS_COLORS[s] for s in STRESSES})

g_up = sns.clustermap(jac_up, cmap='Reds', vmin=0, vmax=0.5,
                       linewidths=1.5, linecolor='white',
                       annot=True, fmt='.2f', annot_kws={'size': 16, 'fontweight': 'bold'},
                       figsize=(9, 8),
                       row_linkage=Z, col_linkage=Z,
                       dendrogram_ratio=(0.18, 0.18),
                       cbar_kws={'label': 'Jaccard index (UP edges)'},
                       row_colors=row_colors, col_colors=row_colors)

g_up.ax_heatmap.set_xticklabels(g_up.ax_heatmap.get_xmajorticklabels(),
                                  fontsize=13, fontweight='bold', rotation=45)
g_up.ax_heatmap.set_yticklabels(g_up.ax_heatmap.get_ymajorticklabels(),
                                  fontsize=13, fontweight='bold', rotation=0)
for lbl in g_up.ax_heatmap.get_yticklabels():
    lbl.set_color(STRESS_COLORS.get(lbl.get_text(), 'black'))
for lbl in g_up.ax_heatmap.get_xticklabels():
    lbl.set_color(STRESS_COLORS.get(lbl.get_text(), 'black'))

g_up.fig.suptitle("UP edges: Jaccard index\n(universal removed)", fontsize=15, fontweight='bold', y=1.02)
g_up.savefig(os.path.join(OUT, "stress_clustermap_UP.png"))
g_up.savefig(os.path.join(OUT, "stress_clustermap_UP.pdf"))
plt.close()

# DOWN clustermap
g_dn = sns.clustermap(jac_down, cmap='Blues', vmin=0, vmax=0.5,
                       linewidths=1.5, linecolor='white',
                       annot=True, fmt='.2f', annot_kws={'size': 16, 'fontweight': 'bold'},
                       figsize=(9, 8),
                       row_linkage=Z, col_linkage=Z,
                       dendrogram_ratio=(0.18, 0.18),
                       cbar_kws={'label': 'Jaccard index (DOWN edges)'},
                       row_colors=row_colors, col_colors=row_colors)

g_dn.ax_heatmap.set_xticklabels(g_dn.ax_heatmap.get_xmajorticklabels(),
                                  fontsize=13, fontweight='bold', rotation=45)
g_dn.ax_heatmap.set_yticklabels(g_dn.ax_heatmap.get_ymajorticklabels(),
                                  fontsize=13, fontweight='bold', rotation=0)
for lbl in g_dn.ax_heatmap.get_yticklabels():
    lbl.set_color(STRESS_COLORS.get(lbl.get_text(), 'black'))
for lbl in g_dn.ax_heatmap.get_xticklabels():
    lbl.set_color(STRESS_COLORS.get(lbl.get_text(), 'black'))

g_dn.fig.suptitle("DOWN edges: Jaccard index\n(universal removed)", fontsize=15, fontweight='bold', y=1.02)
g_dn.savefig(os.path.join(OUT, "stress_clustermap_DOWN.png"))
g_dn.savefig(os.path.join(OUT, "stress_clustermap_DOWN.pdf"))
plt.close()

# ── Figure 2: Unique edges bar chart ──────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 6))

y_pos = np.arange(len(reordered))
bar_h = 0.35
colors = [STRESS_COLORS[s] for s in reordered]

up_vals = [unique_up[s] for s in reordered]
down_vals = [unique_down[s] for s in reordered]

bars_up = ax.barh(y_pos - bar_h/2, up_vals, height=bar_h, color=colors, alpha=0.9,
                   label='Unique UP edges')
bars_dn = ax.barh(y_pos + bar_h/2, [-d for d in down_vals], height=bar_h, color=colors,
                   alpha=0.6, hatch='///', label='Unique DOWN edges')

for i in range(len(reordered)):
    ax.text(up_vals[i] + 20, y_pos[i] - bar_h/2, str(up_vals[i]),
            va='center', fontsize=11, fontweight='bold')
    ax.text(-down_vals[i] - 20, y_pos[i] + bar_h/2, str(down_vals[i]),
            va='center', fontsize=11, fontweight='bold', ha='right')

ax.axvline(0, color='black', lw=0.8)
ax.set_yticks(y_pos)
ax.set_yticklabels(reordered, fontsize=14, fontweight='bold')
for i, s in enumerate(reordered):
    ax.get_yticklabels()[i].set_color(STRESS_COLORS[s])

ax.set_xlabel("UP edges -->                              <-- DOWN edges", fontsize=12)
ax.set_title("Stress-unique directional edges (non-universal)", fontweight='bold', fontsize=15)
ax.invert_yaxis()
ax.legend(fontsize=11, loc='lower right')

plt.tight_layout()
fig.savefig(os.path.join(OUT, "stress_unique_edges_updown.png"))
fig.savefig(os.path.join(OUT, "stress_unique_edges_updown.pdf"))
plt.close(fig)

# Copy
import shutil
GDRIVE_OUT = "/Users/vjx443/Library/CloudStorage/GoogleDrive-mutwil@plant.tools/My Drive/Projects/2026_KingdomStress/CC-Fig 4 data/outputs"
for f in os.listdir(OUT):
    if 'clustermap_UP' in f or 'clustermap_DOWN' in f or 'unique_edges_updown' in f:
        shutil.copy2(os.path.join(OUT, f), os.path.join(GDRIVE_OUT, f))

print("\nSaved: stress_clustermap_UP, stress_clustermap_DOWN, stress_unique_edges_updown")
