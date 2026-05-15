"""
Panel B: Level0 MapMan co-occurrence network showing UP and DOWN edges.
Edges colored red (UP-biased) or blue (DOWN-biased), thickness by weight.
"""

import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from matplotlib.patches import FancyArrowPatch
import matplotlib.patches as mpatches
import networkx as nx
import warnings
warnings.filterwarnings('ignore')

OUT = "/tmp/mercator_outputs"
BASE = "/tmp/mercator_data"

plt.rcParams.update({
    'font.family': 'Arial', 'font.size': 10,
    'figure.dpi': 300, 'savefig.dpi': 300,
    'savefig.bbox': 'tight', 'figure.facecolor': 'white',
})

# ── Load UP and DOWN Level0 networks ─────────────────────────────────────
up_df = pd.read_csv(os.path.join(BASE, "All_stress",
                     "Mercator_network_UP_Level0 (Normalised).csv"), index_col=0)
dn_df = pd.read_csv(os.path.join(BASE, "All_stress",
                     "Mercator_network_DOWN_Level0 (Normalised).csv"), index_col=0)

# Also load the bin direction summary to size/color nodes
bin_summary = pd.read_csv(os.path.join(OUT, "bin_summary_L0.csv"))
bin_bias = dict(zip(bin_summary['bin_name'], bin_summary['direction_bias']))
bin_responsive = dict(zip(bin_summary['bin_name'], bin_summary['total_responsive']))

# ── Build edge data: UP weight, DOWN weight, bias ────────────────────────
edges = {}
for _, r in up_df.iterrows():
    key = tuple(sorted([r['source'], r['target']]))
    edges.setdefault(key, {'up': 0, 'down': 0})
    edges[key]['up'] = r['weight']

for _, r in dn_df.iterrows():
    key = tuple(sorted([r['source'], r['target']]))
    edges.setdefault(key, {'up': 0, 'down': 0})
    edges[key]['down'] = r['weight']

# Compute edge bias
edge_data = []
for (src, tgt), vals in edges.items():
    bias = vals['up'] - vals['down']
    mean_w = (vals['up'] + vals['down']) / 2
    edge_data.append({
        'source': src, 'target': tgt,
        'up': vals['up'], 'down': vals['down'],
        'bias': bias, 'mean_weight': mean_w,
    })

edf = pd.DataFrame(edge_data)

# Filter: keep top 3 edges per node (by mean_weight)
TOP_K = 3
keep_edges = set()
all_nodes = set(edf['source'].tolist() + edf['target'].tolist())
for node in all_nodes:
    node_edges = edf[(edf['source'] == node) | (edf['target'] == node)]
    top_edges = node_edges.nlargest(TOP_K, 'mean_weight')
    for _, r in top_edges.iterrows():
        keep_edges.add(tuple(sorted([r['source'], r['target']])))

edf['edge_key'] = edf.apply(lambda r: tuple(sorted([r['source'], r['target']])), axis=1)
edf = edf[edf['edge_key'].isin(keep_edges)].drop(columns='edge_key')
print(f"  Edges after top-{TOP_K} per node filter: {len(edf)}")

# ── Build network ────────────────────────────────────────────────────────
G = nx.Graph()
for _, r in edf.iterrows():
    G.add_edge(r['source'], r['target'],
               up=r['up'], down=r['down'],
               bias=r['bias'], weight=r['mean_weight'])

# ── Node properties ──────────────────────────────────────────────────────
# Size by total responsiveness, color by direction bias
node_bias = {n: bin_bias.get(n, 0) for n in G.nodes}
node_resp = {n: bin_responsive.get(n, 0) for n in G.nodes}

# ── Layout ───────────────────────────────────────────────────────────────
pos = nx.spring_layout(G, k=2.8, seed=42, iterations=150, weight='weight')

# ── Plot ─────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(16, 14))

# Edge colors: red-blue diverging by bias
edge_biases = [G[u][v]['bias'] for u, v in G.edges()]
edge_weights = [G[u][v]['weight'] for u, v in G.edges()]
max_bias = max(abs(min(edge_biases)), abs(max(edge_biases)))

# Normalize bias for colormap
cmap = plt.cm.RdBu_r
norm = Normalize(vmin=-max_bias, vmax=max_bias)
edge_colors = [cmap(norm(b)) for b in edge_biases]

# Draw edges with varying width and color
for (u, v), color, w, bias in zip(G.edges(), edge_colors, edge_weights, edge_biases):
    x = [pos[u][0], pos[v][0]]
    y = [pos[u][1], pos[v][1]]
    linewidth = w * 6
    ax.plot(x, y, color=color, linewidth=linewidth, alpha=0.5, solid_capstyle='round')

# Node colors by direction bias
node_list = list(G.nodes)
node_bias_vals = [node_bias[n] for n in node_list]
max_node_bias = max(abs(min(node_bias_vals)), abs(max(node_bias_vals)))
node_norm = Normalize(vmin=-max_node_bias, vmax=max_node_bias)
node_colors = [cmap(node_norm(node_bias[n])) for n in node_list]

# Node sizes by total responsiveness
max_resp = max(node_resp.values()) if node_resp else 1
node_sizes = [node_resp.get(n, 0) / max_resp * 2500 + 400 for n in node_list]

nx.draw_networkx_nodes(G, pos, ax=ax, nodelist=node_list,
                       node_color=node_colors, node_size=node_sizes,
                       alpha=0.9, edgecolors='black', linewidths=1.2)

# Labels
short_labels = {}
for n in node_list:
    label = n
    if len(label) > 20:
        # Try to wrap
        words = label.split(' ')
        lines = []
        current = ''
        for w in words:
            if len(current + ' ' + w) > 18:
                lines.append(current)
                current = w
            else:
                current = (current + ' ' + w).strip()
        lines.append(current)
        label = '\n'.join(lines)
    short_labels[n] = label

nx.draw_networkx_labels(G, pos, labels=short_labels, ax=ax,
                        font_size=7.5, font_weight='bold')

# Colorbars
# Edge colorbar
sm_edge = ScalarMappable(cmap=cmap, norm=norm)
sm_edge.set_array([])
cbar_edge = plt.colorbar(sm_edge, ax=ax, shrink=0.4, aspect=20, pad=0.02,
                          location='right')
cbar_edge.set_label('Edge bias (UP - DOWN co-occurrence)', fontsize=10)

# Node colorbar
sm_node = ScalarMappable(cmap=cmap, norm=node_norm)
sm_node.set_array([])
cbar_node = plt.colorbar(sm_node, ax=ax, shrink=0.4, aspect=20, pad=0.06,
                          location='right')
cbar_node.set_label('Node bias (UP - DOWN enrichment)', fontsize=10)

# Legend for edge thickness
for w, label in [(0.1, '0.1'), (0.3, '0.3'), (0.5, '0.5')]:
    ax.plot([], [], color='grey', linewidth=w*6, alpha=0.5, label=f'Weight = {label}')
ax.legend(loc='lower left', fontsize=9, title='Edge thickness', title_fontsize=9,
          framealpha=0.9)

ax.set_title("MapMan Level0 co-occurrence network\n"
             "Red = UP-biased, Blue = DOWN-biased (all stresses, normalised)",
             fontsize=13, fontweight='bold')
ax.axis('off')

plt.tight_layout()
fig.savefig(os.path.join(OUT, "panel_B_network_L0.png"))
fig.savefig(os.path.join(OUT, "panel_B_network_L0.pdf"))
plt.close(fig)

# Copy to Google Drive
import shutil
GDRIVE_OUT = "/Users/vjx443/Library/CloudStorage/GoogleDrive-mutwil@plant.tools/My Drive/Projects/2026_KingdomStress/CC-Fig 4 data/outputs"
for f in ['panel_B_network_L0.png', 'panel_B_network_L0.pdf']:
    shutil.copy2(os.path.join(OUT, f), os.path.join(GDRIVE_OUT, f))
print("Saved: panel_B_network_L0.png/pdf")
