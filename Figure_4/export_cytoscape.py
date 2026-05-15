"""
Export Level0 and Level1 networks for Cytoscape with node/edge attributes.
Produces node table + edge table CSVs per level.
"""

import os
import pandas as pd
import numpy as np

OUT = "/tmp/mercator_outputs"
BASE = "/tmp/mercator_data"

# Load bin summaries
summary_L0 = pd.read_csv(os.path.join(OUT, "bin_summary_L0.csv"))
summary_L1 = pd.read_csv(os.path.join(OUT, "bin_summary_L1.csv"))

bin_bias_L0 = dict(zip(summary_L0['bin_name'], summary_L0['direction_bias']))
bin_resp_L0 = dict(zip(summary_L0['bin_name'], summary_L0['total_responsive']))
bin_up_L0 = dict(zip(summary_L0['bin_name'], summary_L0['mean_UP']))
bin_down_L0 = dict(zip(summary_L0['bin_name'], summary_L0['mean_DOWN']))

bin_bias_L1 = dict(zip(summary_L1['bin_name'], summary_L1['direction_bias']))
bin_resp_L1 = dict(zip(summary_L1['bin_name'], summary_L1['total_responsive']))
bin_up_L1 = dict(zip(summary_L1['bin_name'], summary_L1['mean_UP']))
bin_down_L1 = dict(zip(summary_L1['bin_name'], summary_L1['mean_DOWN']))


def build_edge_data(up_path, dn_path):
    """Load UP and DOWN networks, compute edge attributes."""
    up_df = pd.read_csv(up_path, index_col=0)
    dn_df = pd.read_csv(dn_path, index_col=0)

    edges = {}
    for _, r in up_df.iterrows():
        key = tuple(sorted([r['source'], r['target']]))
        edges.setdefault(key, {'up': 0, 'down': 0})
        edges[key]['up'] = r['weight']
    for _, r in dn_df.iterrows():
        key = tuple(sorted([r['source'], r['target']]))
        edges.setdefault(key, {'up': 0, 'down': 0})
        edges[key]['down'] = r['weight']

    rows = []
    for (src, tgt), vals in edges.items():
        rows.append({
            'source': src, 'target': tgt,
            'UP_weight': vals['up'], 'DOWN_weight': vals['down'],
            'mean_weight': (vals['up'] + vals['down']) / 2,
            'edge_bias': vals['up'] - vals['down'],
        })
    return pd.DataFrame(rows)


def short_name(name):
    """Part after last dot, capitalized."""
    s = name.split('.')[-1].strip()
    return s[0].upper() + s[1:] if s else name


def export_level(level, edge_df, bin_bias, bin_resp, bin_up, bin_down,
                 bias_threshold=None, exclude_ec=False, top_k=None):
    """Export node and edge tables for Cytoscape."""

    # Optional filtering
    if exclude_ec or bias_threshold:
        valid_nodes = set()
        all_nodes = set(edge_df['source'].tolist() + edge_df['target'].tolist())
        for node in all_nodes:
            if exclude_ec and 'Enzyme classification' in node:
                continue
            if bias_threshold and abs(bin_bias.get(node, 0)) < bias_threshold:
                continue
            valid_nodes.add(node)
        edge_df = edge_df[edge_df['source'].isin(valid_nodes) & edge_df['target'].isin(valid_nodes)]

    # Optional top-K edges per node
    if top_k:
        keep_edges = set()
        for node in set(edge_df['source'].tolist() + edge_df['target'].tolist()):
            node_edges = edge_df[(edge_df['source'] == node) | (edge_df['target'] == node)]
            top_edges = node_edges.nlargest(top_k, 'mean_weight')
            for _, r in top_edges.iterrows():
                keep_edges.add(tuple(sorted([r['source'], r['target']])))
        edge_df['_key'] = edge_df.apply(lambda r: tuple(sorted([r['source'], r['target']])), axis=1)
        edge_df = edge_df[edge_df['_key'].isin(keep_edges)].drop(columns='_key')

    # Active nodes
    active_nodes = sorted(set(edge_df['source'].tolist() + edge_df['target'].tolist()))

    # ── Node table ────────────────────────────────────────────────────────
    node_rows = []
    for node in active_nodes:
        bias = bin_bias.get(node, 0)
        resp = bin_resp.get(node, 0)
        up = bin_up.get(node, 0)
        down = bin_down.get(node, 0)

        # RGB color from RdBu_r colormap
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from matplotlib.colors import Normalize
        cmap = plt.cm.RdBu_r
        max_bias = max(abs(b) for b in bin_bias.values()) if bin_bias else 0.3
        norm = Normalize(vmin=-max_bias, vmax=max_bias)
        rgba = cmap(norm(bias))
        hex_color = '#{:02x}{:02x}{:02x}'.format(int(rgba[0]*255), int(rgba[1]*255), int(rgba[2]*255))

        # Parent bin (Level0 name)
        parent = node.split('.')[0] if '.' in node else node

        node_rows.append({
            'name': node,
            'short_name': short_name(node),
            'parent_bin': parent,
            'direction_bias': round(bias, 4),
            'mean_UP': round(up, 4),
            'mean_DOWN': round(down, 4),
            'total_responsive': round(resp, 4),
            'node_color_hex': hex_color,
            'node_size': round(resp * 100, 1),  # scaled for Cytoscape
        })

    node_df = pd.DataFrame(node_rows)

    # ── Edge table ────────────────────────────────────────────────────────
    # Add hex colors for edges
    edge_rows = []
    max_edge_bias = max(abs(edge_df['edge_bias'].min()), abs(edge_df['edge_bias'].max())) if len(edge_df) > 0 else 0.3
    edge_norm = Normalize(vmin=-max_edge_bias, vmax=max_edge_bias)

    for _, r in edge_df.iterrows():
        rgba = cmap(edge_norm(r['edge_bias']))
        hex_color = '#{:02x}{:02x}{:02x}'.format(int(rgba[0]*255), int(rgba[1]*255), int(rgba[2]*255))

        edge_rows.append({
            'source': r['source'],
            'target': r['target'],
            'source_short': short_name(r['source']),
            'target_short': short_name(r['target']),
            'UP_weight': round(r['UP_weight'], 4),
            'DOWN_weight': round(r['DOWN_weight'], 4),
            'mean_weight': round(r['mean_weight'], 4),
            'edge_bias': round(r['edge_bias'], 4),
            'edge_color_hex': hex_color,
            'edge_width': round(r['mean_weight'] * 5, 2),  # scaled for Cytoscape
            'interaction': 'co-occurrence',
        })

    edge_out = pd.DataFrame(edge_rows)

    # Save
    node_file = os.path.join(OUT, f"cytoscape_nodes_{level}.csv")
    edge_file = os.path.join(OUT, f"cytoscape_edges_{level}.csv")
    node_df.to_csv(node_file, index=False)
    edge_out.to_csv(edge_file, index=False)

    print(f"\n  {level}: {len(node_df)} nodes, {len(edge_out)} edges")
    print(f"    {node_file}")
    print(f"    {edge_file}")

    return node_df, edge_out


# ═══════════════════════════════════════════════════════════════════════════
# Level 0: all nodes, top 3 edges per node
# ═══════════════════════════════════════════════════════════════════════════
print("Exporting Level 0...")
edf_L0 = build_edge_data(
    os.path.join(BASE, "All_stress", "Mercator_network_UP_Level0 (Normalised).csv"),
    os.path.join(BASE, "All_stress", "Mercator_network_DOWN_Level0 (Normalised).csv"),
)
export_level("Level0", edf_L0, bin_bias_L0, bin_resp_L0, bin_up_L0, bin_down_L0,
             top_k=3)

# Also export full (no filtering) for flexibility in Cytoscape
export_level("Level0_full", edf_L0, bin_bias_L0, bin_resp_L0, bin_up_L0, bin_down_L0)

# ═══════════════════════════════════════════════════════════════════════════
# Level 1: bias threshold + no EC + top 3
# ═══════════════════════════════════════════════════════════════════════════
print("\nExporting Level 1...")
edf_L1 = build_edge_data(
    os.path.join(BASE, "All_stress", "Mercator_network_UP_Level1 (Normalised).csv"),
    os.path.join(BASE, "All_stress", "Mercator_network_DOWN_Level1 (Normalised).csv"),
)
export_level("Level1_filtered", edf_L1, bin_bias_L1, bin_resp_L1, bin_up_L1, bin_down_L1,
             bias_threshold=0.04, exclude_ec=True, top_k=3)

# Also export full (no filtering)
export_level("Level1_full", edf_L1, bin_bias_L1, bin_resp_L1, bin_up_L1, bin_down_L1)

# ═══════════════════════════════════════════════════════════════════════════
# Copy to Google Drive
# ═══════════════════════════════════════════════════════════════════════════
import shutil
GDRIVE_OUT = "/Users/vjx443/Library/CloudStorage/GoogleDrive-mutwil@plant.tools/My Drive/Projects/2026_KingdomStress/CC-Fig 4 data/outputs"
for f in os.listdir(OUT):
    if f.startswith("cytoscape_"):
        shutil.copy2(os.path.join(OUT, f), os.path.join(GDRIVE_OUT, f))

print("\n" + "=" * 60)
print("CYTOSCAPE EXPORT COMPLETE")
print("=" * 60)
for f in sorted(os.listdir(OUT)):
    if f.startswith("cytoscape_"):
        size = os.path.getsize(os.path.join(OUT, f))
        print(f"  {f} ({size/1024:.1f} KB)")
print("""
Import instructions for Cytoscape:
1. File > Import > Network from File > select edges CSV
   - Source: 'source', Target: 'target', Interaction: 'interaction'
2. File > Import > Table from File > select nodes CSV
   - Key column: 'name'
3. Style > Node:
   - Fill Color: map 'node_color_hex' as passthrough
   - Size: map 'node_size' as continuous
   - Label: map 'short_name'
4. Style > Edge:
   - Stroke Color: map 'edge_color_hex' as passthrough
   - Width: map 'edge_width' as continuous
""")
