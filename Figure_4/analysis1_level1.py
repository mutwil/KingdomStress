"""
Analysis 1 extended: Consensus stress network at Level 1 (MERGE).
Same method as Level0 but with MapMan sub-bin resolution.
"""

import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
import seaborn as sns
import networkx as nx
import warnings
warnings.filterwarnings('ignore')

BASE = "/tmp/mercator_data"
OUT = "/tmp/mercator_outputs"
os.makedirs(OUT, exist_ok=True)

plt.rcParams.update({
    'font.family': 'Arial',
    'font.size': 10,
    'axes.titlesize': 12,
    'axes.labelsize': 10,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'figure.facecolor': 'white',
})

STRESSES = ['Heat', 'Cold', 'Drought', 'Salt', 'Pathogen', 'Heavy metal']

def shorten_label(label, max_len=25):
    if len(label) <= max_len:
        return label
    return label[:max_len-2] + '..'

def edge_key(row):
    return tuple(sorted([row['source'], row['target']]))

# ── Load Level1 MERGE per stress ──────────────────────────────────────────
print("Loading Level1 MERGE per stress...")
stress_edge_weights = {}
all_edges = set()

for stress in STRESSES:
    path = os.path.join(BASE, "Stresses", stress,
                        f"Mercator_network_MERGE_Level1 (All_organ).csv")
    if not os.path.exists(path):
        print(f"  Missing: {stress}")
        continue
    try:
        df = pd.read_csv(path, index_col=0)
    except Exception:
        continue

    ew = {}
    for _, r in df.iterrows():
        k = tuple(sorted([r['source'], r['target']]))
        ew[k] = r['weight']
    stress_edge_weights[stress] = ew
    all_edges.update(ew.keys())
    print(f"  {stress}: {len(ew)} edges")

print(f"  Total unique Level1 edges: {len(all_edges)}")

# ── Build edge x stress weight matrix ─────────────────────────────────────
edge_list = sorted(all_edges)
edge_labels = [f"{e[0]} -- {e[1]}" for e in edge_list]

weight_matrix = pd.DataFrame(0.0, index=edge_labels, columns=STRESSES)
for i, e in enumerate(edge_list):
    for stress in STRESSES:
        weight_matrix.iloc[i, weight_matrix.columns.get_loc(stress)] = \
            stress_edge_weights.get(stress, {}).get(e, 0.0)

# ── Core edges (>= 80% of stresses) ──────────────────────────────────────
presence_count = (weight_matrix > 0).sum(axis=1)
mean_weight = weight_matrix.mean(axis=1)
threshold = int(np.ceil(len(STRESSES) * 0.8))

core_mask = presence_count >= threshold
core_edges = weight_matrix[core_mask].copy()
core_edges['mean_weight'] = mean_weight[core_mask]
core_edges['presence'] = presence_count[core_mask]
core_edges = core_edges.sort_values('mean_weight', ascending=False)

print(f"\nCore edges (present in >= {threshold}/{len(STRESSES)} stresses): {core_mask.sum()} / {len(edge_list)}")
print(f"\nTop 20 core edges by mean weight:")
for idx in core_edges.head(20).index:
    print(f"  {shorten_label(idx, 65)}: mean={core_edges.loc[idx, 'mean_weight']:.3f}, "
          f"in {int(core_edges.loc[idx, 'presence'])}/{len(STRESSES)} stresses")

core_edges.to_csv(os.path.join(OUT, "analysis1_L1_core_edges.csv"))

# ── Stress-unique edges at Level1 ────────────────────────────────────────
print(f"\nStress-unique edges (present in only 1 stress):")
for stress in STRESSES:
    unique = 0
    ew = stress_edge_weights.get(stress, {})
    for e in ew:
        present_in = sum(1 for s in STRESSES if e in stress_edge_weights.get(s, {}))
        if present_in == 1:
            unique += 1
    print(f"  {stress}: {unique} unique edges")

# ── Network visualization: top core edges ─────────────────────────────────
# Full network would be too dense; show top 200 core edges
top_n_edges = 200
top_core = core_edges.head(top_n_edges)

fig, ax = plt.subplots(figsize=(18, 14))
G = nx.Graph()
for idx in top_core.index:
    parts = idx.split(' -- ')
    if len(parts) == 2:
        w = top_core.loc[idx, 'mean_weight']
        G.add_edge(parts[0], parts[1], weight=w)

if len(G.nodes) > 0:
    pos = nx.spring_layout(G, k=1.8, seed=42, iterations=120)
    edges = G.edges(data=True)
    weights = [d['weight'] for _, _, d in edges]

    wdeg = dict(G.degree(weight='weight'))
    node_sizes = [wdeg[n] * 80 + 200 for n in G.nodes]

    nx.draw_networkx_edges(G, pos, ax=ax, width=[w*4 for w in weights],
                           alpha=0.4, edge_color=weights, edge_cmap=plt.cm.YlOrRd,
                           edge_vmin=0, edge_vmax=max(weights) if weights else 1)
    nx.draw_networkx_nodes(G, pos, ax=ax, node_size=node_sizes,
                           node_color='#4A90D9', alpha=0.85, edgecolors='black', linewidths=0.5)
    labels = {n: shorten_label(n, 22) for n in G.nodes}
    nx.draw_networkx_labels(G, pos, labels=labels, ax=ax, font_size=5, font_weight='bold')

    sm = ScalarMappable(cmap=plt.cm.YlOrRd, norm=Normalize(vmin=0, vmax=max(weights)))
    sm.set_array([])
    plt.colorbar(sm, ax=ax, label='Mean co-occurrence frequency', shrink=0.5)

ax.set_title(f"Core MapMan Level1 co-occurrence network\n"
             f"(top {min(top_n_edges, len(top_core))} edges present in >={threshold}/{len(STRESSES)} stresses)")
ax.axis('off')
fig.savefig(os.path.join(OUT, "analysis1_L1_consensus_network.png"))
fig.savefig(os.path.join(OUT, "analysis1_L1_consensus_network.pdf"))
plt.close(fig)

# ── Heatmap: top 50 most variable Level1 edges across stresses ───────────
present_mask = (weight_matrix > 0).any(axis=1)
wm_present = weight_matrix[present_mask]
variance = wm_present.var(axis=1)
top_var = variance.nlargest(50).index
wm_top = wm_present.loc[top_var]

fig, ax = plt.subplots(figsize=(10, 18))
sns.heatmap(wm_top, cmap='YlOrRd', ax=ax, linewidths=0.2, linecolor='white',
            yticklabels=[shorten_label(l, 55) for l in wm_top.index],
            cbar_kws={'label': 'Co-occurrence frequency', 'shrink': 0.4})
ax.set_title("Top 50 most variable Level1 bin pairs across stresses\n(MERGE)")
ax.set_xlabel("Stress")
ax.set_ylabel("MapMan Level1 bin pair")
plt.tight_layout()
fig.savefig(os.path.join(OUT, "analysis1_L1_variable_heatmap.png"))
fig.savefig(os.path.join(OUT, "analysis1_L1_variable_heatmap.pdf"))
plt.close(fig)

# ── Per-stress adjacency matrices (top bins only) ────────────────────────
# Get top 25 bins by weighted degree across all stresses
all_bins = set()
for stress in STRESSES:
    for e in stress_edge_weights.get(stress, {}).keys():
        all_bins.update(e)

# Compute global weighted degree
global_wdeg = {}
for b in all_bins:
    s = 0
    for stress in STRESSES:
        for e, w in stress_edge_weights.get(stress, {}).items():
            if b in e:
                s += w
    global_wdeg[b] = s

top_bins = sorted(global_wdeg, key=global_wdeg.get, reverse=True)[:25]

fig, axes = plt.subplots(2, 3, figsize=(26, 18))
for i, stress in enumerate(STRESSES):
    ax = axes[i // 3, i % 3]
    ew = stress_edge_weights.get(stress, {})
    mat = pd.DataFrame(0.0, index=top_bins, columns=top_bins)
    for e, w in ew.items():
        if e[0] in top_bins and e[1] in top_bins:
            mat.loc[e[0], e[1]] = w
            mat.loc[e[1], e[0]] = w

    short_labels = [shorten_label(l, 22) for l in mat.index]
    sns.heatmap(mat, ax=ax, cmap='YlOrRd', vmin=0, vmax=0.6,
                xticklabels=short_labels, yticklabels=short_labels,
                linewidths=0.3, linecolor='white',
                cbar_kws={'shrink': 0.4})
    ax.set_title(stress, fontsize=12, fontweight='bold')
    ax.tick_params(labelsize=6)
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=90, ha='right')
    plt.setp(ax.yaxis.get_majorticklabels(), rotation=0)

fig.suptitle("Level1 co-occurrence matrices per stress (top 25 bins, MERGE)", fontsize=14, y=1.01)
plt.tight_layout()
fig.savefig(os.path.join(OUT, "analysis1_L1_per_stress_matrices.png"))
fig.savefig(os.path.join(OUT, "analysis1_L1_per_stress_matrices.pdf"))
plt.close(fig)

# ── Save full weight matrix ──────────────────────────────────────────────
weight_matrix.to_csv(os.path.join(OUT, "analysis1_L1_weight_matrix.csv"))

# ── Copy to Google Drive ─────────────────────────────────────────────────
import shutil
GDRIVE_OUT = "/Users/vjx443/Library/CloudStorage/GoogleDrive-mutwil@plant.tools/My Drive/Projects/2026_KingdomStress/CC-Fig 4 data/outputs"
os.makedirs(GDRIVE_OUT, exist_ok=True)
for f in os.listdir(OUT):
    if f.startswith("analysis1_L1"):
        shutil.copy2(os.path.join(OUT, f), os.path.join(GDRIVE_OUT, f))

print("\n" + "="*60)
print("Level1 consensus analysis complete")
print("="*60)
for f in sorted(os.listdir(OUT)):
    if f.startswith("analysis1_L1"):
        size = os.path.getsize(os.path.join(OUT, f))
        print(f"  {f} ({size/1024:.1f} KB)")
