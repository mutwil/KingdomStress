"""
Kingdom Stress Atlas - Figure 4: MapMan bin co-occurrence network analyses
All 9 analyses of Mercator co-occurrence frequency networks.
"""

import os
import re
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
import seaborn as sns
from scipy.cluster.hierarchy import linkage, dendrogram
from scipy.spatial.distance import squareform, pdist
from itertools import combinations
import networkx as nx
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

# ── Paths ──────────────────────────────────────────────────────────────────
BASE = "/tmp/mercator_data"
OUT = "/tmp/mercator_outputs"
os.makedirs(OUT, exist_ok=True)

# ── Style ──────────────────────────────────────────────────────────────────
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
CLADES = ['Monocot', 'Dicot', 'Gymnosperm', 'Lycophyte', 'Bryophyte', 'Charophyte', 'Chlorophyte']
DIRECTIONS = ['UP', 'DOWN', 'MERGE']

# ── Helper functions ───────────────────────────────────────────────────────

def load_csv(path):
    """Load a Mercator network CSV, return DataFrame."""
    if not os.path.exists(path):
        return None
    try:
        df = pd.read_csv(path, index_col=0)
        return df
    except (pd.errors.EmptyDataError, pd.errors.ParserError):
        return None

def load_network(path):
    """Load CSV as edge list with (source, target, weight, direction)."""
    df = load_csv(path)
    if df is None or df.empty:
        return pd.DataFrame(columns=['source', 'target', 'weight', 'direction'])
    return df[['source', 'target', 'weight', 'direction']].copy()

def edge_key(row):
    """Create sorted edge key for consistent matching."""
    return tuple(sorted([row['source'], row['target']]))

def df_to_weight_dict(df, direction=None):
    """Convert edge df to {(src, tgt): weight} dict, optionally filter direction."""
    if direction:
        df = df[df['direction'] == direction]
    return {edge_key(r): r['weight'] for _, r in df.iterrows()}

def build_adjacency_matrix(df, direction=None):
    """Build symmetric adjacency matrix from edge list."""
    if direction:
        df = df[df['direction'] == direction]
    nodes = sorted(set(df['source'].tolist() + df['target'].tolist()))
    mat = pd.DataFrame(0.0, index=nodes, columns=nodes)
    for _, r in df.iterrows():
        mat.loc[r['source'], r['target']] = r['weight']
        mat.loc[r['target'], r['source']] = r['weight']
    return mat

def shorten_label(label, max_len=25):
    """Shorten long labels for plotting."""
    if len(label) <= max_len:
        return label
    return label[:max_len-2] + '..'

# ── Data loading ───────────────────────────────────────────────────────────
print("Loading data...")

# Only load Level 0 and 1 eagerly (Level2 files are ~240MB each, load on demand)
EAGER_LEVELS = [0, 1]

# Per-stress networks (from Stresses/ directory, All_organ files)
stress_networks = {}
for stress in STRESSES:
    for direction in DIRECTIONS:
        for level in EAGER_LEVELS:
            path = os.path.join(BASE, "Stresses", stress,
                                f"Mercator_network_{direction}_Level{level} (All_organ).csv")
            key = (stress, direction, level)
            stress_networks[key] = load_network(path)

# All-stress combined networks (Normalised)
allstress_networks = {}
for direction in DIRECTIONS:
    for level in EAGER_LEVELS:
        for organ in ['', '(Leaf)', '(Root)']:
            suffix = f" (Normalised){organ}.csv"
            path = os.path.join(BASE, "All_stress",
                                f"Mercator_network_{direction}_Level{level}{suffix}")
            key = (direction, level, organ.strip('()') or 'All')
            allstress_networks[key] = load_network(path)

# Clade-level networks
clade_networks = {}
for clade in CLADES:
    clade_dir = os.path.join(BASE, "Clades", clade)
    if not os.path.isdir(clade_dir):
        continue
    # Overall clade (all stresses)
    for direction in DIRECTIONS:
        for level in EAGER_LEVELS:
            path = os.path.join(clade_dir,
                                f"Mercator_network_{direction}_Level{level} (Normalised).csv")
            # Handle case-insensitive level naming
            if not os.path.exists(path):
                path = os.path.join(clade_dir,
                                    f"Mercator_network_{direction}_level{level} (Normalised).csv")
            key = (clade, 'All', direction, level)
            clade_networks[key] = load_network(path)
    # Per-stress within clade
    for stress in STRESSES:
        stress_dir = os.path.join(clade_dir, stress)
        if not os.path.isdir(stress_dir):
            continue
        for direction in DIRECTIONS:
            for level in EAGER_LEVELS:
                path = os.path.join(stress_dir,
                                    f"Mercator_network_{direction}_Level{level} (Normalised).csv")
                if not os.path.exists(path):
                    path = os.path.join(stress_dir,
                                        f"Mercator_network_{direction}_level{level} (Normalised).csv")
                key = (clade, stress, direction, level)
                clade_networks[key] = load_network(path)

print(f"  Stress networks loaded: {sum(1 for v in stress_networks.values() if v is not None and not v.empty)}")
print(f"  All-stress networks loaded: {sum(1 for v in allstress_networks.values() if v is not None and not v.empty)}")
print(f"  Clade networks loaded: {sum(1 for v in clade_networks.values() if v is not None and not v.empty)}")


# ═══════════════════════════════════════════════════════════════════════════
# ANALYSIS 1: Consensus stress network
# ═══════════════════════════════════════════════════════════════════════════
print("\n=== Analysis 1: Consensus stress network ===")

# Collect MERGE Level0 edges per stress
stress_edge_weights = {}
all_edges = set()
for stress in STRESSES:
    df = stress_networks.get((stress, 'MERGE', 0))
    if df is None or df.empty:
        continue
    ew = df_to_weight_dict(df)
    stress_edge_weights[stress] = ew
    all_edges.update(ew.keys())

# Build matrix: edges x stresses
edge_list = sorted(all_edges)
weight_matrix = pd.DataFrame(0.0, index=range(len(edge_list)),
                              columns=STRESSES)
edge_labels = []
for i, e in enumerate(edge_list):
    edge_labels.append(f"{e[0]} -- {e[1]}")
    for stress in STRESSES:
        weight_matrix.loc[i, stress] = stress_edge_weights.get(stress, {}).get(e, 0.0)

weight_matrix.index = edge_labels

# Compute presence count and mean weight
presence_count = (weight_matrix > 0).sum(axis=1)
mean_weight = weight_matrix.mean(axis=1)

# Core edges: present in >= 80% of stresses (5 out of 6)
threshold = int(np.ceil(len(STRESSES) * 0.8))
core_mask = presence_count >= threshold
core_edges = weight_matrix[core_mask].copy()
core_edges['mean_weight'] = mean_weight[core_mask]
core_edges['presence'] = presence_count[core_mask]
core_edges = core_edges.sort_values('mean_weight', ascending=False)

print(f"  Total unique edges (Level0 MERGE): {len(edge_list)}")
print(f"  Core edges (present in >= {threshold}/{len(STRESSES)} stresses): {core_mask.sum()}")
print(f"  Top 10 core edges by mean weight:")
for idx in core_edges.head(10).index:
    print(f"    {idx}: mean={core_edges.loc[idx, 'mean_weight']:.3f}, present in {int(core_edges.loc[idx, 'presence'])}/{len(STRESSES)} stresses")

# Save core edges table
core_edges.to_csv(os.path.join(OUT, "analysis1_core_edges.csv"))

# Plot: network diagram of core edges
fig, ax = plt.subplots(figsize=(14, 10))
G = nx.Graph()
for idx in core_edges.index:
    parts = idx.split(' -- ')
    if len(parts) == 2:
        w = core_edges.loc[idx, 'mean_weight']
        G.add_edge(parts[0], parts[1], weight=w)

if len(G.nodes) > 0:
    pos = nx.spring_layout(G, k=2.5, seed=42, iterations=100)
    edges = G.edges(data=True)
    weights = [d['weight'] for _, _, d in edges]

    # Node sizes by weighted degree
    wdeg = dict(G.degree(weight='weight'))
    node_sizes = [wdeg[n] * 200 + 300 for n in G.nodes]

    nx.draw_networkx_edges(G, pos, ax=ax, width=[w*5 for w in weights],
                           alpha=0.6, edge_color=weights, edge_cmap=plt.cm.YlOrRd,
                           edge_vmin=0, edge_vmax=max(weights) if weights else 1)
    nx.draw_networkx_nodes(G, pos, ax=ax, node_size=node_sizes,
                           node_color='#4A90D9', alpha=0.85, edgecolors='black', linewidths=0.5)
    labels = {n: shorten_label(n, 20) for n in G.nodes}
    nx.draw_networkx_labels(G, pos, labels=labels, ax=ax, font_size=7, font_weight='bold')

    sm = ScalarMappable(cmap=plt.cm.YlOrRd, norm=Normalize(vmin=0, vmax=max(weights) if weights else 1))
    sm.set_array([])
    plt.colorbar(sm, ax=ax, label='Mean co-occurrence frequency', shrink=0.6)

ax.set_title(f"Core MapMan bin co-occurrence network\n(edges present in >={threshold}/{len(STRESSES)} stresses, Level0 MERGE)")
ax.axis('off')
fig.savefig(os.path.join(OUT, "analysis1_consensus_network.png"))
fig.savefig(os.path.join(OUT, "analysis1_consensus_network.pdf"))
plt.close(fig)
print("  Saved: analysis1_consensus_network.png/pdf, analysis1_core_edges.csv")


# ═══════════════════════════════════════════════════════════════════════════
# ANALYSIS 2: Stress-specific vs shared edges heatmap
# ═══════════════════════════════════════════════════════════════════════════
print("\n=== Analysis 2: Stress-specific vs shared edges ===")

# Use the weight matrix from Analysis 1
# Filter to edges present in at least 1 stress, sort by variance (most variable = most stress-specific)
present_mask = (weight_matrix > 0).any(axis=1)
wm_present = weight_matrix[present_mask].copy()

# For a cleaner heatmap, build a node x node matrix per stress and show as symmetric heatmap
# Instead, show edge x stress heatmap clustered
# Filter to top edges by variance for readability
variance = wm_present.var(axis=1)
top_var = variance.nlargest(50).index
wm_top = wm_present.loc[top_var]

fig, ax = plt.subplots(figsize=(10, 16))
sns.heatmap(wm_top, cmap='YlOrRd', ax=ax, linewidths=0.3, linecolor='white',
            yticklabels=[shorten_label(l, 45) for l in wm_top.index],
            cbar_kws={'label': 'Co-occurrence frequency', 'shrink': 0.5})
ax.set_title("Top 50 most variable MapMan bin pairs across stresses\n(Level0 MERGE)")
ax.set_xlabel("Stress")
ax.set_ylabel("MapMan bin pair")
plt.tight_layout()
fig.savefig(os.path.join(OUT, "analysis2_stress_specific_heatmap.png"))
fig.savefig(os.path.join(OUT, "analysis2_stress_specific_heatmap.pdf"))
plt.close(fig)

# Also: symmetric node x node heatmap per stress
fig, axes = plt.subplots(2, 3, figsize=(24, 16))
for i, stress in enumerate(STRESSES):
    ax = axes[i // 3, i % 3]
    df = stress_networks.get((stress, 'MERGE', 0))
    if df is not None and not df.empty:
        mat = build_adjacency_matrix(df)
        short_labels = [shorten_label(l, 18) for l in mat.index]
        sns.heatmap(mat, ax=ax, cmap='YlOrRd', vmin=0, vmax=0.6,
                    xticklabels=short_labels, yticklabels=short_labels,
                    linewidths=0.3, linecolor='white',
                    cbar_kws={'shrink': 0.5})
    ax.set_title(stress, fontsize=12, fontweight='bold')
    ax.tick_params(labelsize=5, rotation=45)
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=90, ha='right')
    plt.setp(ax.yaxis.get_majorticklabels(), rotation=0)

fig.suptitle("MapMan bin co-occurrence matrices per stress (Level0 MERGE)", fontsize=14, y=1.01)
plt.tight_layout()
fig.savefig(os.path.join(OUT, "analysis2_per_stress_matrices.png"))
fig.savefig(os.path.join(OUT, "analysis2_per_stress_matrices.pdf"))
plt.close(fig)

# Unique edges per stress
print("  Stress-unique edges (present in only 1 stress):")
for stress in STRESSES:
    unique = 0
    ew = stress_edge_weights.get(stress, {})
    for e, w in ew.items():
        present_in = sum(1 for s in STRESSES if e in stress_edge_weights.get(s, {}))
        if present_in == 1:
            unique += 1
    print(f"    {stress}: {unique} unique edges")

# Save full matrix
weight_matrix.to_csv(os.path.join(OUT, "analysis2_edge_weight_matrix.csv"))
print("  Saved: analysis2_stress_specific_heatmap.png/pdf, analysis2_per_stress_matrices.png/pdf")


# ═══════════════════════════════════════════════════════════════════════════
# ANALYSIS 3: UP vs DOWN network asymmetry
# ═══════════════════════════════════════════════════════════════════════════
print("\n=== Analysis 3: UP vs DOWN asymmetry ===")

asymmetry_data = []
for stress in STRESSES:
    up_df = stress_networks.get((stress, 'UP', 0))
    dn_df = stress_networks.get((stress, 'DOWN', 0))
    if up_df is None or dn_df is None:
        continue
    up_edges = df_to_weight_dict(up_df)
    dn_edges = df_to_weight_dict(dn_df)
    all_e = set(up_edges.keys()) | set(dn_edges.keys())
    for e in all_e:
        up_w = up_edges.get(e, 0.0)
        dn_w = dn_edges.get(e, 0.0)
        asymmetry_data.append({
            'stress': stress,
            'edge': f"{e[0]} -- {e[1]}",
            'UP_weight': up_w,
            'DOWN_weight': dn_w,
            'asymmetry': up_w - dn_w,  # positive = stronger in UP
            'ratio': up_w / dn_w if dn_w > 0 else (np.inf if up_w > 0 else 0),
        })

asym_df = pd.DataFrame(asymmetry_data)
asym_df.to_csv(os.path.join(OUT, "analysis3_asymmetry_data.csv"), index=False)

# Pivot: edges x stresses, values = asymmetry
if not asym_df.empty:
    asym_pivot = asym_df.pivot_table(index='edge', columns='stress', values='asymmetry', aggfunc='first')
    asym_pivot = asym_pivot.fillna(0)

    # Top asymmetric edges by absolute mean asymmetry
    abs_mean_asym = asym_pivot.abs().mean(axis=1).nlargest(40)
    asym_top = asym_pivot.loc[abs_mean_asym.index]

    fig, ax = plt.subplots(figsize=(10, 14))
    sns.heatmap(asym_top, cmap='RdBu_r', center=0, ax=ax,
                linewidths=0.3, linecolor='white',
                yticklabels=[shorten_label(l, 45) for l in asym_top.index],
                cbar_kws={'label': 'Asymmetry (UP - DOWN)', 'shrink': 0.5})
    ax.set_title("UP vs DOWN asymmetry of MapMan bin co-occurrence\n(Top 40 most asymmetric edges, Level0)")
    ax.set_xlabel("Stress")
    plt.tight_layout()
    fig.savefig(os.path.join(OUT, "analysis3_asymmetry_heatmap.png"))
    fig.savefig(os.path.join(OUT, "analysis3_asymmetry_heatmap.pdf"))
    plt.close(fig)

    # Summary: which edges are consistently UP-biased or DOWN-biased across stresses
    consistent_up = asym_pivot[(asym_pivot > 0).all(axis=1)]
    consistent_dn = asym_pivot[(asym_pivot < 0).all(axis=1)]
    print(f"  Edges consistently UP-biased across all stresses: {len(consistent_up)}")
    print(f"  Edges consistently DOWN-biased across all stresses: {len(consistent_dn)}")
    if len(consistent_up) > 0:
        print(f"  Top 5 UP-biased:")
        for idx in consistent_up.mean(axis=1).nlargest(5).index:
            print(f"    {idx}: mean asymmetry = {consistent_up.loc[idx].mean():.3f}")
    if len(consistent_dn) > 0:
        print(f"  Top 5 DOWN-biased:")
        for idx in consistent_dn.mean(axis=1).nsmallest(5).index:
            print(f"    {idx}: mean asymmetry = {consistent_dn.loc[idx].mean():.3f}")

print("  Saved: analysis3_asymmetry_heatmap.png/pdf, analysis3_asymmetry_data.csv")


# ═══════════════════════════════════════════════════════════════════════════
# ANALYSIS 4: Phylogenetic conservation of network topology
# ═══════════════════════════════════════════════════════════════════════════
print("\n=== Analysis 4: Clade network topology comparison ===")

# Collect clade-level MERGE Level0 edge weight vectors
clade_vectors = {}
clade_all_edges = set()
available_clades = []
for clade in CLADES:
    df = clade_networks.get((clade, 'All', 'MERGE', 0))
    if df is not None and not df.empty:
        ew = df_to_weight_dict(df)
        clade_vectors[clade] = ew
        clade_all_edges.update(ew.keys())
        available_clades.append(clade)

if len(available_clades) >= 2:
    # Build edge x clade matrix
    sorted_edges = sorted(clade_all_edges)
    clade_mat = pd.DataFrame(0.0, index=[f"{e[0]} -- {e[1]}" for e in sorted_edges],
                              columns=available_clades)
    for i, e in enumerate(sorted_edges):
        for clade in available_clades:
            clade_mat.iloc[i, clade_mat.columns.get_loc(clade)] = clade_vectors.get(clade, {}).get(e, 0.0)

    # Pairwise correlation between clades
    clade_corr = clade_mat.corr(method='pearson')

    # Jaccard similarity on edge presence
    jaccard_mat = pd.DataFrame(0.0, index=available_clades, columns=available_clades)
    for c1, c2 in combinations(available_clades, 2):
        s1 = set(k for k, v in clade_vectors[c1].items() if v > 0)
        s2 = set(k for k, v in clade_vectors[c2].items() if v > 0)
        if len(s1 | s2) > 0:
            j = len(s1 & s2) / len(s1 | s2)
        else:
            j = 0
        jaccard_mat.loc[c1, c2] = j
        jaccard_mat.loc[c2, c1] = j
    for i in range(len(available_clades)):
        jaccard_mat.iloc[i, i] = 1.0

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    sns.heatmap(clade_corr, annot=True, fmt='.2f', cmap='RdYlBu_r', vmin=-0.2, vmax=1,
                ax=ax1, linewidths=0.5, linecolor='white', square=True)
    ax1.set_title("Pearson correlation of edge weights\nbetween clades (Level0 MERGE)")

    sns.heatmap(jaccard_mat, annot=True, fmt='.2f', cmap='YlGn', vmin=0, vmax=1,
                ax=ax2, linewidths=0.5, linecolor='white', square=True)
    ax2.set_title("Jaccard similarity of edge presence\nbetween clades (Level0 MERGE)")

    plt.tight_layout()
    fig.savefig(os.path.join(OUT, "analysis4_clade_similarity.png"))
    fig.savefig(os.path.join(OUT, "analysis4_clade_similarity.pdf"))
    plt.close(fig)

    # Also per-stress comparison
    stress_clade_corrs = {}
    for stress in STRESSES:
        cvecs = {}
        c_edges = set()
        for clade in available_clades:
            df = clade_networks.get((clade, stress, 'MERGE', 0))
            if df is not None and not df.empty:
                ew = df_to_weight_dict(df)
                cvecs[clade] = ew
                c_edges.update(ew.keys())
        if len(cvecs) >= 2:
            se = sorted(c_edges)
            smat = pd.DataFrame(0.0, index=range(len(se)), columns=list(cvecs.keys()))
            for i, e in enumerate(se):
                for c in cvecs:
                    smat.iloc[i, smat.columns.get_loc(c)] = cvecs[c].get(e, 0.0)
            stress_clade_corrs[stress] = smat.corr(method='pearson')

    if stress_clade_corrs:
        n_stresses = len(stress_clade_corrs)
        ncols = 3
        nrows = (n_stresses + ncols - 1) // ncols
        fig, axes = plt.subplots(nrows, ncols, figsize=(6*ncols, 5*nrows))
        axes_flat = axes.flatten() if hasattr(axes, 'flatten') else [axes]
        for i, (stress, corr) in enumerate(stress_clade_corrs.items()):
            ax = axes_flat[i]
            sns.heatmap(corr, annot=True, fmt='.2f', cmap='RdYlBu_r', vmin=-0.2, vmax=1,
                        ax=ax, linewidths=0.5, square=True)
            ax.set_title(stress, fontweight='bold')
        for j in range(i+1, len(axes_flat)):
            axes_flat[j].axis('off')
        fig.suptitle("Clade network correlation per stress (Level0 MERGE)", fontsize=14, y=1.01)
        plt.tight_layout()
        fig.savefig(os.path.join(OUT, "analysis4_clade_per_stress.png"))
        fig.savefig(os.path.join(OUT, "analysis4_clade_per_stress.pdf"))
        plt.close(fig)

    clade_corr.to_csv(os.path.join(OUT, "analysis4_clade_correlation.csv"))
    jaccard_mat.to_csv(os.path.join(OUT, "analysis4_clade_jaccard.csv"))
    print(f"  Available clades: {available_clades}")
    print(f"  Saved: analysis4_clade_similarity.png/pdf, analysis4_clade_per_stress.png/pdf")
else:
    print("  Not enough clades with data for comparison.")


# ═══════════════════════════════════════════════════════════════════════════
# ANALYSIS 5: Organ-specific rewiring (Leaf vs Root)
# ═══════════════════════════════════════════════════════════════════════════
print("\n=== Analysis 5: Organ-specific rewiring ===")

organ_data = []
for direction in ['MERGE', 'UP', 'DOWN']:
    for level in [0]:
        leaf_df = allstress_networks.get((direction, level, 'Leaf'))
        root_df = allstress_networks.get((direction, level, 'Root'))
        all_df = allstress_networks.get((direction, level, 'All'))

        if leaf_df is None or root_df is None:
            continue

        leaf_edges = df_to_weight_dict(leaf_df)
        root_edges = df_to_weight_dict(root_df)
        all_e = set(leaf_edges.keys()) | set(root_edges.keys())

        for e in all_e:
            lw = leaf_edges.get(e, 0.0)
            rw = root_edges.get(e, 0.0)
            organ_data.append({
                'direction': direction,
                'edge': f"{e[0]} -- {e[1]}",
                'Leaf_weight': lw,
                'Root_weight': rw,
                'diff': lw - rw,
            })

organ_df = pd.DataFrame(organ_data)
organ_df.to_csv(os.path.join(OUT, "analysis5_organ_data.csv"), index=False)

# Plot for MERGE
merge_organ = organ_df[organ_df['direction'] == 'MERGE'].copy()
if not merge_organ.empty:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

    # Scatter: Leaf vs Root weights
    ax1.scatter(merge_organ['Root_weight'], merge_organ['Leaf_weight'],
                alpha=0.5, s=20, c='steelblue', edgecolors='none')
    max_w = max(merge_organ['Root_weight'].max(), merge_organ['Leaf_weight'].max())
    ax1.plot([0, max_w], [0, max_w], 'k--', alpha=0.4, lw=1)
    ax1.set_xlabel("Root co-occurrence frequency")
    ax1.set_ylabel("Leaf co-occurrence frequency")
    ax1.set_title("Leaf vs Root co-occurrence weights\n(Level0 MERGE, all stresses)")

    # Annotate top divergent edges
    merge_organ['abs_diff'] = merge_organ['diff'].abs()
    top_divergent = merge_organ.nlargest(8, 'abs_diff')
    for _, r in top_divergent.iterrows():
        ax1.annotate(shorten_label(r['edge'], 30),
                     (r['Root_weight'], r['Leaf_weight']),
                     fontsize=5, alpha=0.8,
                     arrowprops=dict(arrowstyle='-', alpha=0.3))

    # Bar chart of most organ-specific edges
    top_leaf = merge_organ.nlargest(10, 'diff')
    top_root = merge_organ.nsmallest(10, 'diff')
    organ_top = pd.concat([top_leaf, top_root]).sort_values('diff')

    colors = ['#2E86AB' if d > 0 else '#A23B72' for d in organ_top['diff']]
    ax2.barh(range(len(organ_top)), organ_top['diff'], color=colors, alpha=0.8)
    ax2.set_yticks(range(len(organ_top)))
    ax2.set_yticklabels([shorten_label(e, 40) for e in organ_top['edge']], fontsize=6)
    ax2.set_xlabel("Leaf - Root weight difference")
    ax2.set_title("Most organ-specific edges\n(blue = leaf-enriched, purple = root-enriched)")
    ax2.axvline(0, color='black', lw=0.5)

    plt.tight_layout()
    fig.savefig(os.path.join(OUT, "analysis5_organ_rewiring.png"))
    fig.savefig(os.path.join(OUT, "analysis5_organ_rewiring.pdf"))
    plt.close(fig)

    r_corr = merge_organ['Leaf_weight'].corr(merge_organ['Root_weight'])
    leaf_only = ((merge_organ['Leaf_weight'] > 0) & (merge_organ['Root_weight'] == 0)).sum()
    root_only = ((merge_organ['Root_weight'] > 0) & (merge_organ['Leaf_weight'] == 0)).sum()
    print(f"  Leaf-Root Pearson r = {r_corr:.3f}")
    print(f"  Leaf-only edges: {leaf_only}, Root-only edges: {root_only}")

print("  Saved: analysis5_organ_rewiring.png/pdf, analysis5_organ_data.csv")


# ═══════════════════════════════════════════════════════════════════════════
# ANALYSIS 6: Hub analysis across levels
# ═══════════════════════════════════════════════════════════════════════════
print("\n=== Analysis 6: Hub analysis across levels ===")

hub_results = {}
for level in [0, 1]:
    df = allstress_networks.get(('MERGE', level, 'All'))
    if df is None or df.empty:
        continue
    G = nx.Graph()
    for _, r in df.iterrows():
        G.add_edge(r['source'], r['target'], weight=r['weight'])

    wdeg = pd.Series(dict(G.degree(weight='weight'))).sort_values(ascending=False)
    hub_results[level] = wdeg

    print(f"\n  Level {level} - Top 10 hubs by weighted degree:")
    for node, deg in wdeg.head(10).items():
        print(f"    {shorten_label(node, 50)}: {deg:.3f}")

# Plot hub analysis
fig, axes = plt.subplots(1, 2, figsize=(16, 8))
for level in [0, 1]:
    ax = axes[level]
    if level in hub_results:
        wdeg = hub_results[level]
        top_n = min(20, len(wdeg))
        top = wdeg.head(top_n)
        bars = ax.barh(range(top_n), top.values, color=plt.cm.viridis(np.linspace(0.3, 0.9, top_n)), alpha=0.85)
        ax.set_yticks(range(top_n))
        ax.set_yticklabels([shorten_label(n, 30) for n in top.index], fontsize=7)
        ax.set_xlabel("Weighted degree")
        ax.set_title(f"Level {level} hubs", fontweight='bold')
        ax.invert_yaxis()

fig.suptitle("MapMan bin hubs by weighted degree (MERGE, all stresses)", fontsize=14, y=1.01)
plt.tight_layout()
fig.savefig(os.path.join(OUT, "analysis6_hub_analysis.png"))
fig.savefig(os.path.join(OUT, "analysis6_hub_analysis.pdf"))
plt.close(fig)

# Save hub data
for level, wdeg in hub_results.items():
    wdeg.to_csv(os.path.join(OUT, f"analysis6_hubs_level{level}.csv"), header=['weighted_degree'])

print("  Saved: analysis6_hub_analysis.png/pdf, analysis6_hubs_level*.csv")


# ═══════════════════════════════════════════════════════════════════════════
# ANALYSIS 7: Network modularity (community detection)
# ═══════════════════════════════════════════════════════════════════════════
print("\n=== Analysis 7: Network modularity ===")

try:
    from networkx.algorithms.community import greedy_modularity_communities

    community_results = {}
    for stress in STRESSES + ['All']:
        if stress == 'All':
            df = allstress_networks.get(('MERGE', 0, 'All'))
        else:
            df = stress_networks.get((stress, 'MERGE', 0))

        if df is None or df.empty:
            continue

        G = nx.Graph()
        for _, r in df.iterrows():
            if r['weight'] > 0:
                G.add_edge(r['source'], r['target'], weight=r['weight'])

        if len(G.nodes) < 3:
            continue

        communities = list(greedy_modularity_communities(G, weight='weight'))
        modularity = nx.community.modularity(G, communities, weight='weight')
        community_results[stress] = {
            'communities': communities,
            'modularity': modularity,
            'n_communities': len(communities),
        }
        print(f"  {stress}: {len(communities)} communities, modularity = {modularity:.3f}")
        for ci, comm in enumerate(communities):
            members = sorted(comm)
            print(f"    Module {ci+1}: {', '.join(shorten_label(m, 25) for m in members[:5])}" +
                  (f" (+{len(members)-5} more)" if len(members) > 5 else ""))

    # Plot: modularity comparison + community structure for All
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

    # Bar chart of modularity per stress
    mod_vals = {s: v['modularity'] for s, v in community_results.items()}
    bars = ax1.bar(mod_vals.keys(), mod_vals.values(), color='steelblue', alpha=0.8)
    ax1.set_ylabel("Modularity (Q)")
    ax1.set_title("Network modularity per stress\n(Level0 MERGE)")
    ax1.tick_params(axis='x', rotation=45)
    for bar, v in zip(bars, mod_vals.values()):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                 f'{v:.3f}', ha='center', va='bottom', fontsize=8)

    # Network colored by community for "All"
    if 'All' in community_results:
        df = allstress_networks.get(('MERGE', 0, 'All'))
        G = nx.Graph()
        for _, r in df.iterrows():
            if r['weight'] > 0:
                G.add_edge(r['source'], r['target'], weight=r['weight'])

        comms = community_results['All']['communities']
        node_colors = {}
        cmap = plt.cm.Set2
        for ci, comm in enumerate(comms):
            for node in comm:
                node_colors[node] = cmap(ci / max(len(comms)-1, 1))

        pos = nx.spring_layout(G, k=2, seed=42, iterations=80)
        colors = [node_colors.get(n, 'grey') for n in G.nodes]
        weights = [G[u][v]['weight'] for u, v in G.edges()]

        nx.draw_networkx_edges(G, pos, ax=ax2, width=[w*3 for w in weights], alpha=0.3)
        nx.draw_networkx_nodes(G, pos, ax=ax2, node_color=colors, node_size=400,
                               alpha=0.85, edgecolors='black', linewidths=0.5)
        labels = {n: shorten_label(n, 15) for n in G.nodes}
        nx.draw_networkx_labels(G, pos, labels=labels, ax=ax2, font_size=6)
        ax2.set_title(f"Community structure (all stresses)\n{len(comms)} modules, Q={community_results['All']['modularity']:.3f}")
        ax2.axis('off')

    plt.tight_layout()
    fig.savefig(os.path.join(OUT, "analysis7_modularity.png"))
    fig.savefig(os.path.join(OUT, "analysis7_modularity.pdf"))
    plt.close(fig)

    # Save community assignments
    comm_rows = []
    for stress, res in community_results.items():
        for ci, comm in enumerate(res['communities']):
            for node in comm:
                comm_rows.append({'stress': stress, 'module': ci+1, 'bin': node})
    pd.DataFrame(comm_rows).to_csv(os.path.join(OUT, "analysis7_communities.csv"), index=False)
    print("  Saved: analysis7_modularity.png/pdf, analysis7_communities.csv")

except ImportError:
    print("  Skipped: community detection requires networkx >= 2.7")


# ═══════════════════════════════════════════════════════════════════════════
# ANALYSIS 8: Edge weight distribution shifts
# ═══════════════════════════════════════════════════════════════════════════
print("\n=== Analysis 8: Edge weight distribution shifts ===")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

# Per stress
for stress in STRESSES:
    df = stress_networks.get((stress, 'MERGE', 0))
    if df is not None and not df.empty:
        weights = df['weight'][df['weight'] > 0]
        ax1.hist(weights, bins=30, alpha=0.4, label=stress, density=True)
ax1.set_xlabel("Co-occurrence frequency")
ax1.set_ylabel("Density")
ax1.set_title("Edge weight distributions per stress\n(Level0 MERGE)")
ax1.legend(fontsize=8)

# Per clade
for clade in available_clades:
    df = clade_networks.get((clade, 'All', 'MERGE', 0))
    if df is not None and not df.empty:
        weights = df['weight'][df['weight'] > 0]
        ax2.hist(weights, bins=30, alpha=0.4, label=clade, density=True)
ax2.set_xlabel("Co-occurrence frequency")
ax2.set_ylabel("Density")
ax2.set_title("Edge weight distributions per clade\n(Level0 MERGE)")
ax2.legend(fontsize=8)

plt.tight_layout()
fig.savefig(os.path.join(OUT, "analysis8_weight_distributions.png"))
fig.savefig(os.path.join(OUT, "analysis8_weight_distributions.pdf"))
plt.close(fig)

# Violin plot version
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

# Stress violins
stress_weights_list = []
stress_labels_list = []
for stress in STRESSES:
    df = stress_networks.get((stress, 'MERGE', 0))
    if df is not None and not df.empty:
        w = df['weight'][df['weight'] > 0].values
        stress_weights_list.append(w)
        stress_labels_list.append(stress)

if stress_weights_list:
    parts = ax1.violinplot(stress_weights_list, showmeans=True, showmedians=True)
    ax1.set_xticks(range(1, len(stress_labels_list)+1))
    ax1.set_xticklabels(stress_labels_list, rotation=45, ha='right')
    ax1.set_ylabel("Co-occurrence frequency")
    ax1.set_title("Edge weight distribution per stress\n(Level0 MERGE)")

# Clade violins
clade_weights_list = []
clade_labels_list = []
for clade in available_clades:
    df = clade_networks.get((clade, 'All', 'MERGE', 0))
    if df is not None and not df.empty:
        w = df['weight'][df['weight'] > 0].values
        clade_weights_list.append(w)
        clade_labels_list.append(clade)

if clade_weights_list:
    parts = ax2.violinplot(clade_weights_list, showmeans=True, showmedians=True)
    ax2.set_xticks(range(1, len(clade_labels_list)+1))
    ax2.set_xticklabels(clade_labels_list, rotation=45, ha='right')
    ax2.set_ylabel("Co-occurrence frequency")
    ax2.set_title("Edge weight distribution per clade\n(Level0 MERGE)")

plt.tight_layout()
fig.savefig(os.path.join(OUT, "analysis8_weight_violins.png"))
fig.savefig(os.path.join(OUT, "analysis8_weight_violins.pdf"))
plt.close(fig)

# Summary stats
print("  Summary statistics (mean +/- std of edge weights):")
print("  Stresses:")
for stress in STRESSES:
    df = stress_networks.get((stress, 'MERGE', 0))
    if df is not None and not df.empty:
        w = df['weight'][df['weight'] > 0]
        print(f"    {stress}: {w.mean():.3f} +/- {w.std():.3f} (n={len(w)} edges)")
print("  Clades:")
for clade in available_clades:
    df = clade_networks.get((clade, 'All', 'MERGE', 0))
    if df is not None and not df.empty:
        w = df['weight'][df['weight'] > 0]
        print(f"    {clade}: {w.mean():.3f} +/- {w.std():.3f} (n={len(w)} edges)")

print("  Saved: analysis8_weight_distributions.png/pdf, analysis8_weight_violins.png/pdf")


# ═══════════════════════════════════════════════════════════════════════════
# ANALYSIS 9: Level0 -> Level2 drill-down for top edges
# ═══════════════════════════════════════════════════════════════════════════
print("\n=== Analysis 9: Level0 -> Level2 drill-down ===")

# Identify top 5 Level0 edges from consensus network
top_l0_edges = core_edges.head(5).index.tolist() if not core_edges.empty else []
print(f"  Drilling down top {len(top_l0_edges)} Level0 edges:")

# For each top L0 edge, find the Level1 and Level2 sub-edges
drill_results = {}

# Build set of parent bin names to filter on when reading Level2
parent_bins = set()
for l0_edge_label in top_l0_edges:
    parts = l0_edge_label.split(' -- ')
    if len(parts) == 2:
        parent_bins.add(parts[0].strip())
        parent_bins.add(parts[1].strip())

# Load Level2 in chunks, keeping only rows matching our parent bins
level2_path = os.path.join(BASE, "All_stress", "Mercator_network_MERGE_Level2 (Normalised).csv")
level2_filtered = []
if os.path.exists(level2_path):
    print("  Loading Level2 (chunked, filtering for target bins)...")
    for chunk in pd.read_csv(level2_path, index_col=0, chunksize=100000):
        for pbin in parent_bins:
            mask = chunk['source'].str.startswith(pbin + '.') | chunk['target'].str.startswith(pbin + '.')
            matched = chunk[mask]
            if not matched.empty:
                level2_filtered.append(matched)
    if level2_filtered:
        level2_df = pd.concat(level2_filtered).drop_duplicates()
        print(f"  Level2 filtered: {len(level2_df)} rows from parent bins")
    else:
        level2_df = pd.DataFrame()
else:
    level2_df = pd.DataFrame()
    print("  Level2 file not found, skipping Level2 drill-down")

for l0_edge_label in top_l0_edges:
    parts = l0_edge_label.split(' -- ')
    if len(parts) != 2:
        continue
    bin_a, bin_b = parts[0].strip(), parts[1].strip()
    print(f"\n  {bin_a} -- {bin_b}:")

    # Level1: already loaded
    for level in [1]:
        df = allstress_networks.get(('MERGE', level, 'All'))
        if df is None or df.empty:
            continue
        mask = (
            ((df['source'].str.startswith(bin_a + '.')) & (df['target'].str.startswith(bin_b + '.'))) |
            ((df['source'].str.startswith(bin_b + '.')) & (df['target'].str.startswith(bin_a + '.')))
        )
        sub = df[mask].copy()
        if not sub.empty:
            sub = sub.sort_values('weight', ascending=False)
            print(f"    Level{level}: {len(sub)} sub-edges, top 5:")
            for _, r in sub.head(5).iterrows():
                src_short = r['source'].replace(bin_a + '.', '').replace(bin_b + '.', '')
                tgt_short = r['target'].replace(bin_a + '.', '').replace(bin_b + '.', '')
                print(f"      {shorten_label(src_short, 35)} -- {shorten_label(tgt_short, 35)}: {r['weight']:.3f}")
            drill_results[(l0_edge_label, level)] = sub

    # Level2: use pre-filtered data
    if not level2_df.empty:
        mask = (
            ((level2_df['source'].str.startswith(bin_a + '.')) & (level2_df['target'].str.startswith(bin_b + '.'))) |
            ((level2_df['source'].str.startswith(bin_b + '.')) & (level2_df['target'].str.startswith(bin_a + '.')))
        )
        sub = level2_df[mask].copy()
        if not sub.empty:
            sub = sub.sort_values('weight', ascending=False)
            print(f"    Level2: {len(sub)} sub-edges, top 5:")
            for _, r in sub.head(5).iterrows():
                src_short = r['source'].replace(bin_a + '.', '').replace(bin_b + '.', '')
                tgt_short = r['target'].replace(bin_a + '.', '').replace(bin_b + '.', '')
                print(f"      {shorten_label(src_short, 35)} -- {shorten_label(tgt_short, 35)}: {r['weight']:.3f}")
            drill_results[(l0_edge_label, 2)] = sub

# Plot drill-down for top 3 edges
top_3 = top_l0_edges[:3]
if top_3:
    fig, axes = plt.subplots(len(top_3), 2, figsize=(18, 5*len(top_3)))
    if len(top_3) == 1:
        axes = axes.reshape(1, -1)

    for row, l0_edge_label in enumerate(top_3):
        parts = l0_edge_label.split(' -- ')
        bin_a, bin_b = parts[0].strip(), parts[1].strip()

        for col, level in enumerate([1, 2]):
            ax = axes[row, col]
            sub = drill_results.get((l0_edge_label, level))
            if sub is not None and not sub.empty:
                top_sub = sub.head(15)
                labels = []
                for _, r in top_sub.iterrows():
                    s = r['source'].split('.')[-1] if '.' in r['source'] else r['source']
                    t = r['target'].split('.')[-1] if '.' in r['target'] else r['target']
                    labels.append(f"{shorten_label(s,20)} -- {shorten_label(t,20)}")

                ax.barh(range(len(labels)), top_sub['weight'].values,
                        color=plt.cm.plasma(top_sub['weight'].values / max(top_sub['weight'].values.max(), 0.01)),
                        alpha=0.85)
                ax.set_yticks(range(len(labels)))
                ax.set_yticklabels(labels, fontsize=6)
                ax.set_xlabel("Co-occurrence frequency")
                ax.invert_yaxis()
            ax.set_title(f"{shorten_label(bin_a, 20)} -- {shorten_label(bin_b, 20)}\nLevel{level} sub-edges",
                         fontsize=9, fontweight='bold')

    fig.suptitle("Level0 top edges drilled down to Level1 and Level2", fontsize=13, y=1.01)
    plt.tight_layout()
    fig.savefig(os.path.join(OUT, "analysis9_drilldown.png"))
    fig.savefig(os.path.join(OUT, "analysis9_drilldown.pdf"))
    plt.close(fig)

# Save drill-down data
drill_rows = []
for (edge, level), sub in drill_results.items():
    for _, r in sub.iterrows():
        drill_rows.append({
            'L0_edge': edge, 'level': level,
            'source': r['source'], 'target': r['target'], 'weight': r['weight']
        })
if drill_rows:
    pd.DataFrame(drill_rows).to_csv(os.path.join(OUT, "analysis9_drilldown.csv"), index=False)

print("\n  Saved: analysis9_drilldown.png/pdf, analysis9_drilldown.csv")


# ═══════════════════════════════════════════════════════════════════════════
# Summary
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("ALL ANALYSES COMPLETE")
print(f"Output directory: {OUT}")
print("="*70)
output_files = sorted(os.listdir(OUT))
for f in output_files:
    size = os.path.getsize(os.path.join(OUT, f))
    print(f"  {f} ({size/1024:.1f} KB)")

# Copy results to Google Drive
import shutil
GDRIVE_OUT = "/Users/vjx443/Library/CloudStorage/GoogleDrive-mutwil@plant.tools/My Drive/Projects/2026_KingdomStress/CC-Fig 4 data/outputs"
os.makedirs(GDRIVE_OUT, exist_ok=True)
for f in output_files:
    shutil.copy2(os.path.join(OUT, f), os.path.join(GDRIVE_OUT, f))
print(f"\nCopied {len(output_files)} files to Google Drive: {GDRIVE_OUT}")
