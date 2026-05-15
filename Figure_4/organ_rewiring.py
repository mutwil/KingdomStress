"""
Organ-specific rewiring: Leaf vs Root stress co-occurrence networks at L1.
Which edges are leaf-specific, root-specific, or shared?
Which edges flip direction between organs?
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
from xml.sax.saxutils import escape
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


def short_label(name):
    s = str(name).split('.')[-1].strip()
    return s[0].upper() + s[1:] if s else str(name)


def load_organ_edges(organ, level=1):
    """Load UP and DOWN edges for an organ (all stresses, normalised)."""
    edges = {}
    for fname, d in [(f'Mercator_network_UP_Level{level} (Normalised)({organ}).csv', 'up'),
                      (f'Mercator_network_DOWN_Level{level} (Normalised)({organ}).csv', 'down')]:
        path = os.path.join(BASE, 'All_stress', fname)
        if not os.path.exists(path):
            print(f"  Missing: {path}")
            continue
        try:
            df = pd.read_csv(path, index_col=0)
        except Exception:
            continue
        for _, r in df.iterrows():
            src_l0 = r['source'].split('.')[0]
            tgt_l0 = r['target'].split('.')[0]
            if src_l0 in EXCLUDE_L0 or tgt_l0 in EXCLUDE_L0:
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


# ── Load data ─────────────────────────────────────────────────────────────
print("Loading organ-specific L1 edges...")
leaf = load_organ_edges('Leaf')
root = load_organ_edges('Root')
print(f"  Leaf: {len(leaf)} edges")
print(f"  Root: {len(root)} edges")

# ── Merge into single comparison table ────────────────────────────────────
leaf_dict = leaf.set_index('edge')[['bias', 'mean_weight']].rename(
    columns={'bias': 'leaf_bias', 'mean_weight': 'leaf_weight'})
root_dict = root.set_index('edge')[['bias', 'mean_weight']].rename(
    columns={'bias': 'root_bias', 'mean_weight': 'root_weight'})

merged = leaf_dict.join(root_dict, how='outer').fillna(0)
merged['bias_diff'] = merged['leaf_bias'] - merged['root_bias']  # positive = more UP in leaf
merged['weight_diff'] = merged['leaf_weight'] - merged['root_weight']
merged['leaf_only'] = (merged['leaf_weight'] > 0) & (merged['root_weight'] == 0)
merged['root_only'] = (merged['root_weight'] > 0) & (merged['leaf_weight'] == 0)
merged['shared'] = (merged['leaf_weight'] > 0) & (merged['root_weight'] > 0)
# Direction flip: UP in one organ, DOWN in the other
merged['flips'] = ((merged['leaf_bias'] > 0.02) & (merged['root_bias'] < -0.02)) | \
                   ((merged['leaf_bias'] < -0.02) & (merged['root_bias'] > 0.02))

n_shared = merged['shared'].sum()
n_leaf_only = merged['leaf_only'].sum()
n_root_only = merged['root_only'].sum()
n_flips = merged['flips'].sum()
r_corr = merged['leaf_bias'].corr(merged['root_bias'])

print(f"\n  Shared edges: {n_shared}")
print(f"  Leaf-only: {n_leaf_only}")
print(f"  Root-only: {n_root_only}")
print(f"  Direction flips: {n_flips}")
print(f"  Bias correlation: r = {r_corr:.3f}")

merged.to_csv(os.path.join(OUT, "organ_edge_comparison_L1.csv"))

# ── Figure 1: Scatter leaf vs root bias ──────────────────────────────────
fig, ax = plt.subplots(figsize=(12, 12))

shared = merged[merged['shared'] & ~merged['flips']]
flips = merged[merged['flips']]

ax.scatter(shared['root_bias'], shared['leaf_bias'], alpha=0.15, s=5,
           c='#457B9D', edgecolors='none', label=f'Shared, same direction ({len(shared)})')
ax.scatter(flips['root_bias'], flips['leaf_bias'], alpha=0.6, s=20,
           c='#E63946', edgecolors='none', marker='D',
           label=f'Direction flips ({len(flips)})')

max_b = max(abs(merged[['leaf_bias', 'root_bias']].min().min()),
            abs(merged[['leaf_bias', 'root_bias']].max().max()))
ax.plot([-max_b, max_b], [-max_b, max_b], 'k--', alpha=0.3, lw=1)
ax.axhline(0, color='grey', lw=0.5, alpha=0.3)
ax.axvline(0, color='grey', lw=0.5, alpha=0.3)

# Label top flips
top_flips = flips.nlargest(10, 'bias_diff').index.tolist() + \
            flips.nsmallest(10, 'bias_diff').index.tolist()
for edge in top_flips[:12]:
    r = merged.loc[edge]
    short = f"{short_label(edge.split(' -- ')[0])} --\n{short_label(edge.split(' -- ')[1])}"
    ax.annotate(short, (r['root_bias'], r['leaf_bias']), fontsize=5, alpha=0.7)

ax.set_xlabel("Root direction bias (UP - DOWN)", fontsize=13)
ax.set_ylabel("Leaf direction bias (UP - DOWN)", fontsize=13)
ax.set_title(f"Leaf vs Root edge direction bias (L1, all stresses)\n"
             f"r = {r_corr:.3f} | {n_flips} edges flip direction between organs",
             fontsize=14, fontweight='bold')
ax.legend(fontsize=10, loc='upper left')

plt.tight_layout()
fig.savefig(os.path.join(OUT, "organ_scatter_L1.png"))
fig.savefig(os.path.join(OUT, "organ_scatter_L1.pdf"))
plt.close(fig)


# ── Figure 2: Top organ-divergent edges (bar chart) ──────────────────────
fig, ax = plt.subplots(figsize=(14, 12))

# Top edges by absolute bias difference
top_div = merged[merged['shared']].nlargest(15, 'bias_diff')
bot_div = merged[merged['shared']].nsmallest(15, 'bias_diff')
div_edges = pd.concat([bot_div, top_div])

colors = ['#E63946' if d > 0 else '#2A9D8F' for d in div_edges['bias_diff']]
ax.barh(range(len(div_edges)), div_edges['bias_diff'], color=colors, alpha=0.85)
ax.set_yticks(range(len(div_edges)))

def short_edge(e):
    parts = e.split(' -- ')
    return f"{short_label(parts[0])} -- {short_label(parts[1])}"

ax.set_yticklabels([short_edge(e) for e in div_edges.index], fontsize=8)
ax.axvline(0, color='black', lw=0.5)
ax.set_xlabel("Leaf - Root bias difference\n(Red = more UP in leaf | Teal = more UP in root)", fontsize=12)
ax.set_title("Most organ-divergent L1 edges\n(edges with largest direction difference between leaf and root)",
             fontsize=14, fontweight='bold')

plt.tight_layout()
fig.savefig(os.path.join(OUT, "organ_divergent_edges_L1.png"))
fig.savefig(os.path.join(OUT, "organ_divergent_edges_L1.pdf"))
plt.close(fig)


# ── Figure 3: Heatmap of key bins -- leaf vs root bias side by side ──────
# Get bins that appear in the most divergent edges
div_bins = set()
for e in div_edges.index:
    parts = e.split(' -- ')
    div_bins.update(parts)

# Also add universally important bins
important_bins = [
    'Photosynthesis.photophosphorylation',
    'Cell wall organisation.pectin',
    'Cell wall organisation.cell wall proteins',
    'Protein homeostasis.protein quality control',
    'Redox homeostasis.glutathione-based redox regulation',
    'Multi-process regulation.circadian clock system',
    'Phytohormone action.abscisic acid',
    'Phytohormone action.auxin',
    'Nutrient uptake.nitrogen assimilation',
    'Nutrient uptake.iron uptake',
    'Nutrient uptake.phosphorus assimilation',
    'Solute transport.carrier-mediated transport',
    'Cell wall organisation.cellulose',
    'Cell wall organisation.lignin',
    'Cell wall organisation.cutin and suberin',
    'Secondary metabolism.phenolics biosynthesis',
    'External stimuli response.pathogen',
]

# Compute per-bin bias for each organ
def bin_bias_from_edges(edf):
    biases = {}
    for _, r in edf.iterrows():
        for node in [r['source'], r['target']]:
            biases.setdefault(node, []).append(r['bias'])
    return {b: np.mean(v) for b, v in biases.items()}

leaf_bin_bias = bin_bias_from_edges(leaf)
root_bin_bias = bin_bias_from_edges(root)

all_key_bins = sorted(set(important_bins) & set(leaf_bin_bias.keys()) & set(root_bin_bias.keys()))

bin_data = pd.DataFrame({
    'Leaf': [leaf_bin_bias.get(b, 0) for b in all_key_bins],
    'Root': [root_bin_bias.get(b, 0) for b in all_key_bins],
}, index=all_key_bins)
bin_data['diff'] = bin_data['Leaf'] - bin_data['Root']
bin_data = bin_data.sort_values('diff')

fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 10),
                                      gridspec_kw={'width_ratios': [1, 1, 0.6], 'wspace': 0.05},
                                      sharey=True)

max_b = max(abs(bin_data[['Leaf', 'Root']].min().min()),
            abs(bin_data[['Leaf', 'Root']].max().max()))

# Leaf
sns.heatmap(bin_data[['Leaf']], cmap='RdBu_r', center=0, vmin=-max_b, vmax=max_b,
            ax=ax1, linewidths=0.5, linecolor='white',
            annot=True, fmt='+.3f', annot_kws={'size': 9},
            yticklabels=[short_label(b) for b in bin_data.index],
            cbar=False)
ax1.set_title("Leaf", fontsize=16, fontweight='bold', color='#2A9D8F')

# Root
sns.heatmap(bin_data[['Root']], cmap='RdBu_r', center=0, vmin=-max_b, vmax=max_b,
            ax=ax2, linewidths=0.5, linecolor='white',
            annot=True, fmt='+.3f', annot_kws={'size': 9},
            yticklabels=False,
            cbar_kws={'label': 'Direction bias', 'shrink': 0.4})
ax2.set_title("Root", fontsize=16, fontweight='bold', color='#E63946')

# Difference
max_d = abs(bin_data['diff']).max()
sns.heatmap(bin_data[['diff']], cmap='PiYG', center=0, vmin=-max_d, vmax=max_d,
            ax=ax3, linewidths=0.5, linecolor='white',
            annot=True, fmt='+.3f', annot_kws={'size': 9},
            yticklabels=False,
            cbar_kws={'label': 'Leaf - Root', 'shrink': 0.4})
ax3.set_title("Diff", fontsize=14, fontweight='bold')

fig.suptitle("Key L1 bin direction bias: Leaf vs Root\n"
             "(all stresses combined, normalised)",
             fontsize=16, fontweight='bold', y=1.02)
plt.tight_layout()
fig.savefig(os.path.join(OUT, "organ_bin_comparison_L1.png"))
fig.savefig(os.path.join(OUT, "organ_bin_comparison_L1.pdf"))
plt.close(fig)


# ── Figure 4: Network showing organ-specific edges ──────────────────────
# Edges colored by organ: leaf-enriched, root-enriched, or shared
fig, ax = plt.subplots(figsize=(18, 16))

# Select top 30 most organ-divergent + top 15 most organ-conserved edges
top_divergent = merged[merged['shared']].nlargest(20, 'bias_diff')
bot_divergent = merged[merged['shared']].nsmallest(20, 'bias_diff')
# Shared = similar bias in both organs
merged['abs_bias_diff'] = merged['bias_diff'].abs()
top_conserved = merged[merged['shared']].nsmallest(15, 'abs_bias_diff')
top_conserved = top_conserved[top_conserved['leaf_weight'] > 0.05]  # non-trivial

all_sel = pd.concat([top_divergent, bot_divergent, top_conserved]).drop_duplicates()

G = nx.Graph()
for edge in all_sel.index:
    parts = edge.split(' -- ')
    if len(parts) != 2:
        continue
    r = all_sel.loc[edge]
    bd = r['bias_diff']
    if abs(bd) > 0.15:
        if bd > 0:
            color = '#E63946'  # leaf-enriched
            label = 'leaf'
        else:
            color = '#2A9D8F'  # root-enriched
            label = 'root'
    else:
        color = '#888888'  # shared
        label = 'shared'
    mean_w = max(r['leaf_weight'], r['root_weight'])
    G.add_edge(parts[0], parts[1], color=color, label=label,
               width=mean_w * 6, bias_diff=bd)

pos = nx.spring_layout(G, k=2.0, seed=42, iterations=150)

# Draw edges
for (u, v, data) in G.edges(data=True):
    ax.plot([pos[u][0], pos[v][0]], [pos[u][1], pos[v][1]],
            color=data['color'], linewidth=data['width'], alpha=0.5,
            solid_capstyle='round')

# Node colors by overall bias
all_bias = {**leaf_bin_bias, **root_bin_bias}
node_list = list(G.nodes)
cmap = plt.cm.RdBu_r
max_nb = max(abs(all_bias.get(n, 0)) for n in node_list) if node_list else 0.3
n_norm = Normalize(vmin=-max_nb, vmax=max_nb)
node_colors = [cmap(n_norm(all_bias.get(n, 0))) for n in node_list]

wdeg = dict(G.degree())
max_deg = max(wdeg.values()) if wdeg else 1
node_sizes = [wdeg[n] / max_deg * 1200 + 150 for n in node_list]

nx.draw_networkx_nodes(G, pos, ax=ax, nodelist=node_list,
                       node_color=node_colors, node_size=node_sizes,
                       alpha=0.9, edgecolors='black', linewidths=0.7)

labels = {n: short_label(n) for n in node_list if wdeg[n] >= 2}
nx.draw_networkx_labels(G, pos, labels=labels, ax=ax, font_size=6.5, font_weight='bold')

# Legend
legend_elements = [
    mpatches.Patch(facecolor='#E63946', alpha=0.5, label='Leaf-enriched edges'),
    mpatches.Patch(facecolor='#2A9D8F', alpha=0.5, label='Root-enriched edges'),
    mpatches.Patch(facecolor='#888888', alpha=0.5, label='Organ-conserved edges'),
]
ax.legend(handles=legend_elements, loc='lower left', fontsize=11)

sm = ScalarMappable(cmap=cmap, norm=n_norm)
sm.set_array([])
plt.colorbar(sm, ax=ax, shrink=0.3, label='Node direction bias')

ax.set_title("Organ-specific rewiring of L1 stress network\n"
             "Red edges = leaf-enriched | Teal = root-enriched | Grey = organ-conserved",
             fontsize=14, fontweight='bold')
ax.axis('off')
plt.tight_layout()
fig.savefig(os.path.join(OUT, "organ_network_L1.png"))
fig.savefig(os.path.join(OUT, "organ_network_L1.pdf"))
plt.close(fig)


# Copy
import shutil
GDRIVE_OUT = "/Users/vjx443/Library/CloudStorage/GoogleDrive-mutwil@plant.tools/My Drive/Projects/2026_KingdomStress/CC-Fig 4 data/outputs"
for f in os.listdir(OUT):
    if f.startswith("organ_") and ('L1' in f or 'comparison' in f or 'scatter' in f):
        shutil.copy2(os.path.join(OUT, f), os.path.join(GDRIVE_OUT, f))

print("\nSaved: organ_scatter_L1, organ_divergent_edges_L1, organ_bin_comparison_L1, organ_network_L1")
