"""
Panel B (Level1): MapMan co-occurrence network showing UP and DOWN edges.
Top 3 edges per node to keep it readable.
"""

import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
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

# ── Load UP and DOWN Level1 networks ─────────────────────────────────────
print("Loading Level1 UP/DOWN networks...")
up_df = pd.read_csv(os.path.join(BASE, "All_stress",
                     "Mercator_network_UP_Level1 (Normalised).csv"), index_col=0)
dn_df = pd.read_csv(os.path.join(BASE, "All_stress",
                     "Mercator_network_DOWN_Level1 (Normalised).csv"), index_col=0)

# Load bin direction summary for L1
bin_summary = pd.read_csv(os.path.join(OUT, "bin_summary_L1.csv"))
bin_bias = dict(zip(bin_summary['bin_name'], bin_summary['direction_bias']))
bin_responsive = dict(zip(bin_summary['bin_name'], bin_summary['total_responsive']))

# ── Build edge data ──────────────────────────────────────────────────────
print("Building edge data...")
edges = {}
for _, r in up_df.iterrows():
    key = tuple(sorted([r['source'], r['target']]))
    edges.setdefault(key, {'up': 0, 'down': 0})
    edges[key]['up'] = r['weight']

for _, r in dn_df.iterrows():
    key = tuple(sorted([r['source'], r['target']]))
    edges.setdefault(key, {'up': 0, 'down': 0})
    edges[key]['down'] = r['weight']

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
print(f"  Total edges: {len(edf)}")

# ── Filter: only keep nodes with strong direction bias, exclude EC_ bins ──
BIAS_THRESHOLD = 0.04  # keep nodes with |bias| > threshold
strong_nodes = set()
for node in set(edf['source'].tolist() + edf['target'].tolist()):
    b = bin_bias.get(node, 0)
    # Skip Enzyme classification bins (EC genes span multiple bins)
    if 'Enzyme classification' in node:
        continue
    if abs(b) >= BIAS_THRESHOLD:
        strong_nodes.add(node)
print(f"  Nodes with |bias| >= {BIAS_THRESHOLD}: {len(strong_nodes)}")

# Keep only edges where BOTH nodes pass the bias threshold
edf = edf[edf['source'].isin(strong_nodes) & edf['target'].isin(strong_nodes)]
print(f"  Edges after node bias filter: {len(edf)}")

# Then keep top 3 edges per node
TOP_K = 3
keep_edges = set()
for node in strong_nodes:
    node_edges = edf[(edf['source'] == node) | (edf['target'] == node)]
    top_edges = node_edges.nlargest(TOP_K, 'mean_weight')
    for _, r in top_edges.iterrows():
        keep_edges.add(tuple(sorted([r['source'], r['target']])))

edf['edge_key'] = edf.apply(lambda r: tuple(sorted([r['source'], r['target']])), axis=1)
edf = edf[edf['edge_key'].isin(keep_edges)].drop(columns='edge_key')
print(f"  Edges after top-{TOP_K} per node filter: {len(edf)}")

# Final active nodes
active_nodes = set(edf['source'].tolist() + edf['target'].tolist())
print(f"  Active nodes: {len(active_nodes)}")

# ── Build network ────────────────────────────────────────────────────────
G = nx.Graph()
for _, r in edf.iterrows():
    G.add_edge(r['source'], r['target'],
               up=r['up'], down=r['down'],
               bias=r['bias'], weight=r['mean_weight'])

# ── Node properties ──────────────────────────────────────────────────────
node_bias = {n: bin_bias.get(n, 0) for n in G.nodes}
node_resp = {n: bin_responsive.get(n, 0) for n in G.nodes}

# ── Short labels: part after last dot, capitalized ───────────────────────
def short_label(name, max_len=22):
    s = name.split('.')[-1].strip()
    s = s[0].upper() + s[1:] if s else name
    if len(s) > max_len:
        # Wrap at space nearest to midpoint
        mid = max_len
        space = s.rfind(' ', 0, mid)
        if space > 5:
            s = s[:space] + '\n' + s[space+1:]
        else:
            s = s[:max_len-1] + '..'
    return s

# ── Concentric layout: UP-biased outside, DOWN-biased inside (or vice versa)
print("Computing concentric layout...")

# Sort nodes by bias: most DOWN in center, most UP on outside
sorted_nodes = sorted(G.nodes, key=lambda n: node_bias.get(n, 0))

# Assign to concentric rings based on bias
import math

n_rings = 4
ring_assignment = {}
nodes_per_ring = len(sorted_nodes) / n_rings
for i, node in enumerate(sorted_nodes):
    ring = min(int(i / nodes_per_ring), n_rings - 1)
    ring_assignment[node] = ring

# Build positions: each ring at increasing radius, nodes evenly spaced
pos = {}
ring_radii = [0.8, 1.6, 2.4, 3.2]
ring_nodes = {r: [] for r in range(n_rings)}
for node, ring in ring_assignment.items():
    ring_nodes[ring].append(node)

for ring in range(n_rings):
    nodes = ring_nodes[ring]
    # Sort within ring by bias for smooth color transition
    nodes.sort(key=lambda n: node_bias.get(n, 0))
    radius = ring_radii[ring]
    for j, node in enumerate(nodes):
        angle = 2 * math.pi * j / max(len(nodes), 1) - math.pi / 2
        pos[node] = (radius * math.cos(angle), radius * math.sin(angle))

# ── Plot ─────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(18, 16))

# Edge colors
edge_biases = [G[u][v]['bias'] for u, v in G.edges()]
edge_weights = [G[u][v]['weight'] for u, v in G.edges()]
max_bias = max(abs(min(edge_biases)), abs(max(edge_biases)))

cmap = plt.cm.RdBu_r
norm = Normalize(vmin=-max_bias, vmax=max_bias)

# Draw edges
for (u, v), color_val, w in zip(G.edges(), edge_biases, edge_weights):
    color = cmap(norm(color_val))
    x = [pos[u][0], pos[v][0]]
    y = [pos[u][1], pos[v][1]]
    linewidth = w * 8
    ax.plot(x, y, color=color, linewidth=linewidth, alpha=0.45, solid_capstyle='round')

# Node colors and sizes
node_list = list(G.nodes)
node_bias_vals = [node_bias[n] for n in node_list]
max_node_bias = max(abs(min(node_bias_vals)), abs(max(node_bias_vals))) if node_bias_vals else 0.1
node_norm = Normalize(vmin=-max_node_bias, vmax=max_node_bias)
node_colors = [cmap(node_norm(node_bias[n])) for n in node_list]

max_resp = max(node_resp.values()) if node_resp else 1
node_sizes = [node_resp.get(n, 0) / max_resp * 1800 + 200 for n in node_list]

nx.draw_networkx_nodes(G, pos, ax=ax, nodelist=node_list,
                       node_color=node_colors, node_size=node_sizes,
                       alpha=0.9, edgecolors='black', linewidths=0.8)

# Labels (only for nodes with degree >= 2 or high responsiveness to reduce clutter)
degree = dict(G.degree())
labels = {}
for n in node_list:
    if degree[n] >= 2 or node_resp.get(n, 0) > 0.15:
        labels[n] = short_label(n)

nx.draw_networkx_labels(G, pos, labels=labels, ax=ax,
                        font_size=6.5, font_weight='bold')

# Colorbars
sm_edge = ScalarMappable(cmap=cmap, norm=norm)
sm_edge.set_array([])
cbar_edge = plt.colorbar(sm_edge, ax=ax, shrink=0.35, aspect=20, pad=0.02,
                          location='right')
cbar_edge.set_label('Edge bias (UP - DOWN)', fontsize=10)

sm_node = ScalarMappable(cmap=cmap, norm=node_norm)
sm_node.set_array([])
cbar_node = plt.colorbar(sm_node, ax=ax, shrink=0.35, aspect=20, pad=0.06,
                          location='right')
cbar_node.set_label('Node bias (UP - DOWN)', fontsize=10)

# Edge thickness legend
for w, label in [(0.1, '0.1'), (0.3, '0.3'), (0.5, '0.5')]:
    ax.plot([], [], color='grey', linewidth=w*8, alpha=0.45, label=f'Weight = {label}')
ax.legend(loc='lower left', fontsize=8, title='Edge thickness', title_fontsize=8,
          framealpha=0.9)

ax.set_title("MapMan Level1 co-occurrence network (top 3 edges per node)\n"
             "Red = UP-biased, Blue = DOWN-biased (all stresses, normalised)",
             fontsize=13, fontweight='bold')
ax.axis('off')

plt.tight_layout()
fig.savefig(os.path.join(OUT, "panel_B_network_L1.png"))
fig.savefig(os.path.join(OUT, "panel_B_network_L1.pdf"))
plt.close(fig)

# Copy to Google Drive
import shutil
GDRIVE_OUT = "/Users/vjx443/Library/CloudStorage/GoogleDrive-mutwil@plant.tools/My Drive/Projects/2026_KingdomStress/CC-Fig 4 data/outputs"
for f in ['panel_B_network_L1.png', 'panel_B_network_L1.pdf']:
    shutil.copy2(os.path.join(OUT, f), os.path.join(GDRIVE_OUT, f))
print("Saved: panel_B_network_L1.png/pdf")
