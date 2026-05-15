"""
Organ-specific rewiring v2: focus on direction-flipping edges only.
These are edges that are UP-biased in one organ but DOWN-biased in the other.
Exclude trivially organ-specific bins (photosynthesis, root formation, etc.)
"""

import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.patches as mpatches
import networkx as nx
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
import warnings
warnings.filterwarnings('ignore')

OUT = "/tmp/mercator_outputs"
BASE = "/tmp/mercator_data"

plt.rcParams.update({
    'font.family': 'Arial', 'font.size': 11,
    'figure.dpi': 300, 'savefig.dpi': 300,
    'savefig.bbox': 'tight', 'figure.facecolor': 'white',
})

EXCLUDE_L0 = {'Enzyme classification', 'not assigned', 'Protein modification',
              'Protein biosynthesis'}

# Also exclude trivially organ-specific bins
TRIVIAL_BINS = {
    'Photosynthesis.photophosphorylation', 'Photosynthesis.calvin cycle',
    'Photosynthesis.photorespiration', 'Photosynthesis.CAM/C4 photosynthesis',
    'Plant organogenesis.root formation', 'Plant organogenesis.leaf formation',
    'Plant organogenesis.flower formation', 'Plant organogenesis.stem formation',
    'Plant organogenesis.vascular system formation',
    'Plant reproduction.gametogenesis', 'Plant reproduction.seed formation',
    'Coenzyme metabolism.chlorophyll metabolism',
}


def short_label(name):
    s = str(name).split('.')[-1].strip()
    return s[0].upper() + s[1:] if s else str(name)


def load_organ_edges(organ):
    edges = {}
    for fname, d in [(f'Mercator_network_UP_Level1 (Normalised)({organ}).csv', 'up'),
                      (f'Mercator_network_DOWN_Level1 (Normalised)({organ}).csv', 'down')]:
        path = os.path.join(BASE, 'All_stress', fname)
        if not os.path.exists(path):
            continue
        df = pd.read_csv(path, index_col=0)
        for _, r in df.iterrows():
            src_l0 = r['source'].split('.')[0]
            tgt_l0 = r['target'].split('.')[0]
            if src_l0 in EXCLUDE_L0 or tgt_l0 in EXCLUDE_L0:
                continue
            # Skip trivial bins
            if r['source'] in TRIVIAL_BINS or r['target'] in TRIVIAL_BINS:
                continue
            key = tuple(sorted([r['source'], r['target']]))
            edges.setdefault(key, {'up': 0, 'down': 0})
            edges[key][d] = r['weight']

    rows = []
    for (src, tgt), vals in edges.items():
        rows.append({
            'source': src, 'target': tgt,
            'edge': f'{src} -- {tgt}',
            'UP': vals['up'], 'DOWN': vals['down'],
            'mean_weight': (vals['up'] + vals['down']) / 2,
            'bias': vals['up'] - vals['down'],
        })
    return pd.DataFrame(rows)


# ── Load ──────────────────────────────────────────────────────────────────
print("Loading organ-specific L1 edges (excluding trivial bins)...")
leaf = load_organ_edges('Leaf')
root = load_organ_edges('Root')
print(f"  Leaf: {len(leaf)} edges, Root: {len(root)} edges")

leaf_d = leaf.set_index('edge')[['bias', 'mean_weight']].rename(
    columns={'bias': 'leaf_bias', 'mean_weight': 'leaf_weight'})
root_d = root.set_index('edge')[['bias', 'mean_weight']].rename(
    columns={'bias': 'root_bias', 'mean_weight': 'root_weight'})

merged = leaf_d.join(root_d, how='outer').fillna(0)
merged['bias_diff'] = merged['leaf_bias'] - merged['root_bias']

# Direction flips: UP in one, DOWN in the other (threshold 0.02)
THRESH = 0.02
merged['flips'] = ((merged['leaf_bias'] > THRESH) & (merged['root_bias'] < -THRESH)) | \
                   ((merged['leaf_bias'] < -THRESH) & (merged['root_bias'] > THRESH))

# Also require both organs to have non-trivial weight
merged['both_active'] = (merged['leaf_weight'] > 0.01) & (merged['root_weight'] > 0.01)
flips = merged[merged['flips'] & merged['both_active']].copy()
flips['abs_diff'] = flips['bias_diff'].abs()
flips['flip_type'] = np.where(flips['leaf_bias'] > 0, 'UP in Leaf, DOWN in Root',
                                                         'DOWN in Leaf, UP in Root')

print(f"\n  Direction-flipping edges (both active): {len(flips)}")
print(f"    UP in Leaf / DOWN in Root: {(flips['flip_type'] == 'UP in Leaf, DOWN in Root').sum()}")
print(f"    DOWN in Leaf / UP in Root: {(flips['flip_type'] == 'DOWN in Leaf, UP in Root').sum()}")

flips.to_csv(os.path.join(OUT, "organ_flips_L1.csv"))

# ── Figure 1: Top flipping edges bar chart ────────────────────────────────
top_flips = flips.nlargest(40, 'abs_diff')

def short_edge(e):
    parts = e.split(' -- ')
    return f"{short_label(parts[0])} -- {short_label(parts[1])}"

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 14), sharey=True,
                                 gridspec_kw={'wspace': 0.05})

# Left: leaf bias
colors_leaf = ['#E63946' if b > 0 else '#457B9D' for b in top_flips['leaf_bias']]
ax1.barh(range(len(top_flips)), top_flips['leaf_bias'].values, color=colors_leaf, alpha=0.85)
ax1.set_yticks(range(len(top_flips)))
ax1.set_yticklabels([short_edge(e) for e in top_flips.index], fontsize=8)
ax1.axvline(0, color='black', lw=0.5)
ax1.set_xlabel("Leaf bias (UP - DOWN)", fontsize=12)
ax1.set_title("Leaf", fontweight='bold', fontsize=16, color='#2A9D8F')
ax1.invert_xaxis()

# Right: root bias
colors_root = ['#E63946' if b > 0 else '#457B9D' for b in top_flips['root_bias']]
ax2.barh(range(len(top_flips)), top_flips['root_bias'].values, color=colors_root, alpha=0.85)
ax2.axvline(0, color='black', lw=0.5)
ax2.set_xlabel("Root bias (UP - DOWN)", fontsize=12)
ax2.set_title("Root", fontweight='bold', fontsize=16, color='#E9C46A')

fig.suptitle("Direction-flipping edges: opposite regulation in Leaf vs Root\n"
             "(Top 40 by magnitude, trivially organ-specific bins removed)",
             fontsize=16, fontweight='bold', y=1.01)
plt.tight_layout()
fig.savefig(os.path.join(OUT, "organ_flips_bars_L1.png"))
fig.savefig(os.path.join(OUT, "organ_flips_bars_L1.pdf"))
plt.close(fig)

# ── Figure 2: Which bins are most involved in flips? ──────────────────────
flip_bin_counts = {}
for edge in flips.index:
    parts = edge.split(' -- ')
    for p in parts:
        flip_bin_counts[p] = flip_bin_counts.get(p, 0) + 1

flip_bins = pd.Series(flip_bin_counts).sort_values(ascending=False).head(25)

fig, ax = plt.subplots(figsize=(12, 10))
ax.barh(range(len(flip_bins)), flip_bins.values, color='#8338EC', alpha=0.8)
ax.set_yticks(range(len(flip_bins)))
ax.set_yticklabels([short_label(b) for b in flip_bins.index], fontsize=10)
ax.set_xlabel("Number of flipping edges involving this bin", fontsize=12)
ax.set_title("Bins most involved in organ-specific direction flips\n"
             "(co-upregulated with different partners in leaf vs root)",
             fontsize=14, fontweight='bold')
ax.invert_yaxis()
plt.tight_layout()
fig.savefig(os.path.join(OUT, "organ_flip_bins_L1.png"))
fig.savefig(os.path.join(OUT, "organ_flip_bins_L1.pdf"))
plt.close(fig)

# ── Figure 3: Network of flipping edges ──────────────────────────────────
# Take top 50 flips
top_net = flips.nlargest(50, 'abs_diff')

G = nx.Graph()
for edge in top_net.index:
    parts = edge.split(' -- ')
    if len(parts) != 2:
        continue
    r = top_net.loc[edge]
    G.add_edge(parts[0], parts[1],
               leaf_bias=r['leaf_bias'], root_bias=r['root_bias'],
               flip_type=r['flip_type'], abs_diff=r['abs_diff'])

pos = nx.spring_layout(G, k=2.2, seed=42, iterations=150)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(28, 14))

# Shared layout, two views: leaf coloring and root coloring
for ax, organ, bias_col, title_color in [
    (ax1, 'Leaf', 'leaf_bias', '#2A9D8F'),
    (ax2, 'Root', 'root_bias', '#E9C46A'),
]:
    # Edge colors by organ-specific bias
    for (u, v, data) in G.edges(data=True):
        b = data[bias_col]
        color = '#E63946' if b > 0 else '#457B9D'
        width = abs(b) * 15
        ax.plot([pos[u][0], pos[v][0]], [pos[u][1], pos[v][1]],
                color=color, linewidth=width, alpha=0.4, solid_capstyle='round')

    # Node colors by mean bias in this organ
    node_biases = {}
    edf = leaf if organ == 'Leaf' else root
    for _, r in edf.iterrows():
        for n in [r['source'], r['target']]:
            node_biases.setdefault(n, []).append(r['bias'])
    node_biases = {n: np.mean(v) for n, v in node_biases.items()}

    node_list = list(G.nodes)
    cmap = plt.cm.RdBu_r
    max_nb = max(abs(node_biases.get(n, 0)) for n in node_list) if node_list else 0.3
    n_norm = Normalize(vmin=-max_nb, vmax=max_nb)
    nc = [cmap(n_norm(node_biases.get(n, 0))) for n in node_list]

    wdeg = dict(G.degree())
    max_deg = max(wdeg.values()) if wdeg else 1
    ns = [wdeg[n] / max_deg * 1200 + 150 for n in node_list]

    nx.draw_networkx_nodes(G, pos, ax=ax, nodelist=node_list,
                           node_color=nc, node_size=ns,
                           alpha=0.9, edgecolors='black', linewidths=0.7)

    labels = {n: short_label(n) for n in node_list if wdeg[n] >= 2}
    nx.draw_networkx_labels(G, pos, labels=labels, ax=ax, font_size=7, font_weight='bold')

    ax.set_title(f"{organ} view", fontsize=18, fontweight='bold', color=title_color)
    ax.axis('off')

    sm = ScalarMappable(cmap=cmap, norm=n_norm)
    sm.set_array([])
    plt.colorbar(sm, ax=ax, shrink=0.3, label=f'{organ} bias')

fig.suptitle("Direction-flipping edges: same network, different regulation\n"
             "Red edges/nodes = UP-biased | Blue = DOWN-biased | Same layout, two organ views",
             fontsize=16, fontweight='bold', y=1.02)
plt.tight_layout()
fig.savefig(os.path.join(OUT, "organ_flip_network_L1.png"))
fig.savefig(os.path.join(OUT, "organ_flip_network_L1.pdf"))
plt.close(fig)

# ── Print top flips ──────────────────────────────────────────────────────
print("\nTop 20 direction-flipping edges:")
print(f"{'Edge':<60} {'Leaf':>8} {'Root':>8} {'Flip type'}")
print("-" * 100)
for edge in top_flips.head(20).index:
    r = top_flips.loc[edge]
    print(f"  {short_edge(edge):<58} {r['leaf_bias']:>+.3f} {r['root_bias']:>+.3f}   {r['flip_type']}")

# Copy
import shutil
GDRIVE_OUT = "/Users/vjx443/Library/CloudStorage/GoogleDrive-mutwil@plant.tools/My Drive/Projects/2026_KingdomStress/CC-Fig 4 data/outputs"
for f in os.listdir(OUT):
    if 'organ_flip' in f:
        shutil.copy2(os.path.join(OUT, f), os.path.join(GDRIVE_OUT, f))
print("\nSaved: organ_flips_bars_L1, organ_flip_bins_L1, organ_flip_network_L1")
