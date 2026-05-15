"""
Stress-specific L0 bin responses (no hormones).
Panel A: Direction bias heatmap (L0 bins x stresses)
Panel B: Stress-specificity (deviation from cross-stress mean)
Panel C: Per-stress co-occurrence networks at L0
"""

import os
import math
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
import networkx as nx
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

# ── Load per-stress L0 bias data ──────────────────────────────────────────
stress_L0 = pd.read_csv(os.path.join(OUT, "bin_per_stress_L0.csv"))

# Pivot: bins x stresses
bias_pivot = stress_L0.pivot_table(index='bin_name', columns='stress',
                                    values='bias', aggfunc='first')
bias_pivot = bias_pivot.reindex(columns=STRESSES)

# Remove 'not assigned'
bias_pivot = bias_pivot.drop('not assigned', errors='ignore')

# Sort by mean bias
bias_pivot['_mean'] = bias_pivot.mean(axis=1)
bias_pivot = bias_pivot.sort_values('_mean')
bias_pivot = bias_pivot.drop('_mean', axis=1)

# Compute deviation from cross-stress mean
cross_mean = bias_pivot.mean(axis=1)
dev_pivot = bias_pivot.sub(cross_mean, axis=0)

# ── Figure 1: Combined bias + deviation ──────────────────────────────────
fig = plt.figure(figsize=(20, 14))
gs = fig.add_gridspec(1, 3, width_ratios=[0.3, 1, 1], wspace=0.03)

ax_bar = fig.add_subplot(gs[0])
ax1 = fig.add_subplot(gs[1], sharey=ax_bar)
ax2 = fig.add_subplot(gs[2], sharey=ax_bar)

# Bar chart of mean bias
mean_bias = bias_pivot.mean(axis=1)
colors = ['#E63946' if b > 0 else '#457B9D' for b in mean_bias]
ax_bar.barh(range(len(mean_bias)), mean_bias.values, color=colors, alpha=0.85, height=0.75)
ax_bar.set_yticks(range(len(mean_bias)))
ax_bar.set_yticklabels(mean_bias.index, fontsize=11)
ax_bar.axvline(0, color='black', lw=0.5)
ax_bar.set_xlabel("Mean bias", fontsize=12)
ax_bar.set_title("Mean", fontweight='bold', fontsize=14)

# Heatmap: direction bias
max_b = max(abs(bias_pivot.min().min()), abs(bias_pivot.max().max()))
sns.heatmap(bias_pivot, cmap='RdBu_r', center=0, vmin=-max_b, vmax=max_b,
            ax=ax1, linewidths=0.5, linecolor='white',
            annot=True, fmt='+.2f', annot_kws={'size': 10},
            yticklabels=False,
            cbar_kws={'label': 'UP - DOWN bias', 'shrink': 0.4})
ax1.set_title("Direction bias per stress", fontweight='bold', fontsize=14)
ax1.set_xlabel("")
ax1.tick_params(axis='x', labelsize=12, rotation=45)

# Heatmap: stress-specificity (deviation)
max_dev = max(abs(dev_pivot.min().min()), abs(dev_pivot.max().max()))
sns.heatmap(dev_pivot, cmap='PiYG', center=0, vmin=-max_dev, vmax=max_dev,
            ax=ax2, linewidths=0.5, linecolor='white',
            annot=True, fmt='+.2f', annot_kws={'size': 10},
            yticklabels=False,
            cbar_kws={'label': 'Deviation from mean', 'shrink': 0.4})
ax2.set_title("Stress-specific deviation", fontweight='bold', fontsize=14)
ax2.set_xlabel("")
ax2.tick_params(axis='x', labelsize=12, rotation=45)

ax_bar.set_ylim(-0.8, len(mean_bias) - 0.2)

fig.suptitle("MapMan Level0 bin responses across stresses\n"
             "(Left: mean direction bias | Center: per-stress bias | Right: stress-specific deviations)",
             fontsize=16, fontweight='bold', y=1.02)
fig.subplots_adjust(top=0.92, bottom=0.08)

fig.savefig(os.path.join(OUT, "stress_L0_combined.png"))
fig.savefig(os.path.join(OUT, "stress_L0_combined.pdf"))
plt.close(fig)

# ── Figure 2: Per-stress L0 co-occurrence networks ──────────────────────
print("Building per-stress L0 networks...")

cmap = plt.cm.RdBu_r

fig, axes = plt.subplots(2, 3, figsize=(30, 20))

for idx, stress in enumerate(STRESSES):
    ax = axes[idx // 3, idx % 3]

    # Load UP and DOWN edges
    edges = {}
    for fname, direction in [
        (f"Mercator_network_UP_Level0 (All_organ).csv", 'up'),
        (f"Mercator_network_DOWN_Level0 (All_organ).csv", 'down'),
    ]:
        path = os.path.join(BASE, "Stresses", stress, fname)
        if not os.path.exists(path):
            continue
        df = pd.read_csv(path, index_col=0)
        for _, r in df.iterrows():
            key = tuple(sorted([r['source'], r['target']]))
            edges.setdefault(key, {'up': 0, 'down': 0})
            edges[key][direction] = r['weight']

    # Build edge list
    erows = []
    for (src, tgt), vals in edges.items():
        if src == 'not assigned' or tgt == 'not assigned':
            continue
        mean_w = (vals['up'] + vals['down']) / 2
        bias = vals['up'] - vals['down']
        erows.append({'source': src, 'target': tgt, 'weight': mean_w, 'bias': bias})

    edf = pd.DataFrame(erows)

    # Top 3 edges per node
    keep = set()
    for node in set(edf['source'].tolist() + edf['target'].tolist()):
        ne = edf[(edf['source'] == node) | (edf['target'] == node)]
        for _, r in ne.nlargest(3, 'weight').iterrows():
            keep.add(tuple(sorted([r['source'], r['target']])))
    edf['_key'] = edf.apply(lambda r: tuple(sorted([r['source'], r['target']])), axis=1)
    edf = edf[edf['_key'].isin(keep)].drop(columns='_key')

    # Build graph
    G = nx.Graph()
    for _, r in edf.iterrows():
        G.add_edge(r['source'], r['target'], weight=r['weight'], bias=r['bias'])

    # Node bias from PEA data
    stress_bins = stress_L0[stress_L0['stress'] == stress]
    node_bias = dict(zip(stress_bins['bin_name'], stress_bins['bias']))

    # Layout
    pos = nx.spring_layout(G, k=2.5, seed=42, iterations=100, weight='weight')

    # Draw edges
    edge_biases = [G[u][v]['bias'] for u, v in G.edges()]
    edge_weights = [G[u][v]['weight'] for u, v in G.edges()]
    max_eb = max(abs(min(edge_biases)), abs(max(edge_biases))) if edge_biases else 0.3
    e_norm = Normalize(vmin=-max_eb, vmax=max_eb)

    for (u, v), eb, ew in zip(G.edges(), edge_biases, edge_weights):
        color = cmap(e_norm(eb))
        ax.plot([pos[u][0], pos[v][0]], [pos[u][1], pos[v][1]],
                color=color, linewidth=ew * 6, alpha=0.5, solid_capstyle='round')

    # Node colors and sizes
    node_list = list(G.nodes)
    max_nb = max(abs(node_bias.get(n, 0)) for n in node_list) if node_list else 0.3
    if max_nb == 0:
        max_nb = 0.3
    n_norm = Normalize(vmin=-max_nb, vmax=max_nb)
    node_colors = [cmap(n_norm(node_bias.get(n, 0))) for n in node_list]

    # Size by weighted degree
    wdeg = dict(G.degree(weight='weight'))
    max_wd = max(wdeg.values()) if wdeg else 1
    node_sizes = [wdeg[n] / max_wd * 1500 + 200 for n in node_list]

    nx.draw_networkx_nodes(G, pos, ax=ax, nodelist=node_list,
                           node_color=node_colors, node_size=node_sizes,
                           alpha=0.9, edgecolors='black', linewidths=0.8)

    # Labels
    def wrap_label(s, n=18):
        if len(s) <= n:
            return s
        words = s.split()
        lines, cur = [], ''
        for w in words:
            if len(cur + ' ' + w) > n:
                lines.append(cur)
                cur = w
            else:
                cur = (cur + ' ' + w).strip()
        lines.append(cur)
        return '\n'.join(lines)

    labels = {n: wrap_label(n) for n in node_list}
    nx.draw_networkx_labels(G, pos, labels=labels, ax=ax, font_size=7, font_weight='bold')

    ax.set_title(stress, fontsize=18, fontweight='bold')
    ax.axis('off')

fig.suptitle("MapMan Level0 co-occurrence networks per stress\n"
             "Node color: UP (red) / DOWN (blue) bias | Edge color: co-occurrence direction | Size: weighted degree",
             fontsize=18, fontweight='bold', y=1.01)
plt.tight_layout()
fig.savefig(os.path.join(OUT, "stress_L0_networks.png"))
fig.savefig(os.path.join(OUT, "stress_L0_networks.pdf"))
plt.close(fig)

# Copy
import shutil
GDRIVE_OUT = "/Users/vjx443/Library/CloudStorage/GoogleDrive-mutwil@plant.tools/My Drive/Projects/2026_KingdomStress/CC-Fig 4 data/outputs"
for f in os.listdir(OUT):
    if f.startswith("stress_L0_"):
        shutil.copy2(os.path.join(OUT, f), os.path.join(GDRIVE_OUT, f))

print("Saved: stress_L0_combined.png/pdf, stress_L0_networks.png/pdf")
