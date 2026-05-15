"""
Per-hormone, per-stress Level1 co-occurrence networks.
For each hormone x stress: extract all L1 edges connected to that hormone bin,
export for Cytoscape with node/edge attributes.
"""

import os
import math
import pandas as pd
import numpy as np
from xml.sax.saxutils import escape

OUT = "/tmp/mercator_outputs/hormone_networks"
BASE = "/tmp/mercator_data"
os.makedirs(OUT, exist_ok=True)

HORMONES = {
    'Phytohormone action.abscisic acid': 'Abscisic acid',
    'Phytohormone action.salicylic acid': 'Salicylic acid',
    'Phytohormone action.brassinosteroid': 'Brassinosteroid',
    'Phytohormone action.gibberellin': 'Gibberellin',
    'Phytohormone action.jasmonic acid': 'Jasmonic acid',
    'Phytohormone action.ethylene': 'Ethylene',
    'Phytohormone action.auxin': 'Auxin',
    'Phytohormone action.cytokinin': 'Cytokinin',
}

STRESSES = ['Heat', 'Cold', 'Drought', 'Salt', 'Pathogen', 'Heavy metal']

# Load bin direction summaries for node coloring
bin_summary = pd.read_csv("/tmp/mercator_outputs/bin_summary_L1.csv")
bin_bias = dict(zip(bin_summary['bin_name'], bin_summary['direction_bias']))
bin_resp = dict(zip(bin_summary['bin_name'], bin_summary['total_responsive']))

# Also load per-stress bias
stress_df = pd.read_csv("/tmp/mercator_outputs/bin_per_stress_L1.csv")


def short_label(name):
    s = str(name).split('.')[-1].strip()
    return s[0].upper() + s[1:] if s else str(name)


def load_stress_edges(stress):
    """Load UP and DOWN L1 edges for a stress, compute bias."""
    up_path = os.path.join(BASE, "Stresses", stress,
                           "Mercator_network_UP_Level1 (All_organ).csv")
    dn_path = os.path.join(BASE, "Stresses", stress,
                           "Mercator_network_DOWN_Level1 (All_organ).csv")

    edges = {}
    for path, direction in [(up_path, 'up'), (dn_path, 'down')]:
        if not os.path.exists(path):
            continue
        df = pd.read_csv(path, index_col=0)
        for _, r in df.iterrows():
            key = tuple(sorted([r['source'], r['target']]))
            edges.setdefault(key, {'up': 0, 'down': 0})
            edges[key][direction] = r['weight']

    rows = []
    for (src, tgt), vals in edges.items():
        rows.append({
            'source': src, 'target': tgt,
            'UP_weight': vals['up'], 'DOWN_weight': vals['down'],
            'mean_weight': (vals['up'] + vals['down']) / 2,
            'edge_bias': vals['up'] - vals['down'],
        })
    return pd.DataFrame(rows)


def get_stress_node_bias(stress, bin_name):
    """Get per-stress direction bias for a node."""
    row = stress_df[(stress_df['stress'] == stress) & (stress_df['bin_name'] == bin_name)]
    if not row.empty:
        return row.iloc[0]['bias']
    return 0


def write_hormone_stress_xgmml(hormone_full, hormone_short, stress, edf, top_k=3):
    """Write XGMML for one hormone x one stress."""

    # Filter edges connected to this hormone
    mask = (edf['source'] == hormone_full) | (edf['target'] == hormone_full)
    hormone_edges = edf[mask].copy()

    if hormone_edges.empty:
        return None

    # Keep top K edges by mean weight
    hormone_edges = hormone_edges.nlargest(top_k, 'mean_weight')

    # Collect all nodes
    nodes = set()
    for _, r in hormone_edges.iterrows():
        nodes.add(r['source'])
        nodes.add(r['target'])

    # Also add top edges between the non-hormone nodes (to show their interconnections)
    non_hormone = [n for n in nodes if n != hormone_full]
    inter_edges = edf[
        (edf['source'].isin(non_hormone)) & (edf['target'].isin(non_hormone))
    ]
    if not inter_edges.empty:
        inter_top = inter_edges.nlargest(min(top_k * 2, len(inter_edges)), 'mean_weight')
        hormone_edges = pd.concat([hormone_edges, inter_top]).drop_duplicates(
            subset=['source', 'target'])

    # Update node set
    nodes = set()
    for _, r in hormone_edges.iterrows():
        nodes.add(r['source'])
        nodes.add(r['target'])

    # Concentric layout: hormone in center, others in circle
    positions = {}
    positions[hormone_full] = (0, 0)
    others = sorted(nodes - {hormone_full})
    radius = 250
    for j, n in enumerate(others):
        angle = 2 * math.pi * j / max(len(others), 1) - math.pi / 2
        positions[n] = (radius * math.cos(angle), radius * math.sin(angle))

    # Color mapping
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize
    cmap = plt.cm.RdBu_r

    # Use per-stress bias for node colors
    node_biases = {n: get_stress_node_bias(stress, n) for n in nodes}
    max_nb = max(abs(v) for v in node_biases.values()) if node_biases else 0.3
    if max_nb == 0:
        max_nb = 0.3
    node_norm = Normalize(vmin=-max_nb, vmax=max_nb)

    max_eb = max(abs(hormone_edges['edge_bias'].max()), abs(hormone_edges['edge_bias'].min())) if len(hormone_edges) > 0 else 0.3
    if max_eb == 0:
        max_eb = 0.3
    edge_norm = Normalize(vmin=-max_eb, vmax=max_eb)

    def to_hex(rgba):
        return '#{:02x}{:02x}{:02x}'.format(int(rgba[0]*255), int(rgba[1]*255), int(rgba[2]*255))

    # Write XGMML
    fname = f"{hormone_short}_{stress}.xgmml".replace(' ', '_')
    fpath = os.path.join(OUT, fname)

    with open(fpath, 'w') as f:
        f.write('<?xml version="1.0" encoding="UTF-8"?>\n')
        title = f"{hormone_short} - {stress}"
        f.write(f'<graph label="{escape(title)}" xmlns="http://www.cs.rpi.edu/XGMML" directed="0">\n')

        for n in nodes:
            short = escape(short_label(n))
            x, y = positions.get(n, (0, 0))
            bias = node_biases.get(n, 0)
            resp = bin_resp.get(n, 0)
            color = to_hex(cmap(node_norm(bias)))
            size = resp * 80 + 20

            f.write(f'  <node id="{short}" label="{short}">\n')
            f.write(f'    <graphics x="{x:.1f}" y="{y:.1f}" w="{size:.0f}" h="{size:.0f}" fill="{color}"/>\n')
            f.write(f'    <att name="name" type="string" value="{short}"/>\n')
            f.write(f'    <att name="shared name" type="string" value="{short}"/>\n')
            f.write(f'    <att name="full_name" type="string" value="{escape(n)}"/>\n')
            f.write(f'    <att name="direction_bias" type="real" value="{bias:.4f}"/>\n')
            f.write(f'    <att name="node_color_hex" type="string" value="{color}"/>\n')
            f.write(f'    <att name="node_size" type="real" value="{size:.1f}"/>\n')
            f.write(f'    <att name="is_hormone" type="boolean" value="{"true" if n == hormone_full else "false"}"/>\n')
            f.write(f'  </node>\n')

        for _, r in hormone_edges.iterrows():
            src = escape(short_label(r['source']))
            tgt = escape(short_label(r['target']))
            color = to_hex(cmap(edge_norm(r['edge_bias'])))
            width = r['mean_weight'] * 8

            f.write(f'  <edge source="{src}" target="{tgt}" label="{src} - {tgt}">\n')
            f.write(f'    <att name="UP_weight" type="real" value="{r["UP_weight"]:.4f}"/>\n')
            f.write(f'    <att name="DOWN_weight" type="real" value="{r["DOWN_weight"]:.4f}"/>\n')
            f.write(f'    <att name="mean_weight" type="real" value="{r["mean_weight"]:.4f}"/>\n')
            f.write(f'    <att name="edge_bias" type="real" value="{r["edge_bias"]:.4f}"/>\n')
            f.write(f'    <att name="edge_color_hex" type="string" value="{color}"/>\n')
            f.write(f'    <att name="edge_width" type="real" value="{width:.2f}"/>\n')
            f.write(f'    <att name="interaction" type="string" value="co-occurrence"/>\n')
            f.write(f'  </edge>\n')

        f.write('</graph>\n')

    return fpath, len(nodes), len(hormone_edges)


# ═══════════════════════════════════════════════════════════════════════════
# Generate all hormone x stress networks
# ═══════════════════════════════════════════════════════════════════════════
print("Loading per-stress L1 networks...")
stress_edges = {}
for stress in STRESSES:
    stress_edges[stress] = load_stress_edges(stress)
    print(f"  {stress}: {len(stress_edges[stress])} edges")

print("\nGenerating hormone x stress networks...")
TOP_K = 10  # top edges per hormone

summary_rows = []
for hormone_full, hormone_short in HORMONES.items():
    for stress in STRESSES:
        edf = stress_edges[stress]
        result = write_hormone_stress_xgmml(hormone_full, hormone_short, stress, edf, top_k=TOP_K)
        if result:
            fpath, n_nodes, n_edges = result
            print(f"  {hormone_short:>20} x {stress:<12} -> {n_nodes:>3} nodes, {n_edges:>3} edges")
            summary_rows.append({
                'hormone': hormone_short, 'stress': stress,
                'n_nodes': n_nodes, 'n_edges': n_edges,
                'file': os.path.basename(fpath),
            })
        else:
            print(f"  {hormone_short:>20} x {stress:<12} -> no edges")

summary_df = pd.DataFrame(summary_rows)
summary_df.to_csv(os.path.join(OUT, "hormone_network_summary.csv"), index=False)

# Also create a combined XGMML per hormone (all stresses merged)
print("\nGenerating combined per-hormone networks (all stresses)...")
# Merge all stress edges
all_edges = pd.concat(stress_edges.values())
# Average across stresses for same edge
all_edges['edge_key'] = all_edges.apply(lambda r: tuple(sorted([r['source'], r['target']])), axis=1)
merged = all_edges.groupby('edge_key').agg(
    UP_weight=('UP_weight', 'mean'),
    DOWN_weight=('DOWN_weight', 'mean'),
    mean_weight=('mean_weight', 'mean'),
    edge_bias=('edge_bias', 'mean'),
).reset_index()
merged['source'] = merged['edge_key'].apply(lambda x: x[0])
merged['target'] = merged['edge_key'].apply(lambda x: x[1])

for hormone_full, hormone_short in HORMONES.items():
    result = write_hormone_stress_xgmml(hormone_full, hormone_short, 'All_stresses', merged, top_k=TOP_K)
    if result:
        fpath, n_nodes, n_edges = result
        print(f"  {hormone_short:>20} (all stresses) -> {n_nodes:>3} nodes, {n_edges:>3} edges")

# Copy to Google Drive
import shutil
GDRIVE_OUT = "/Users/vjx443/Library/CloudStorage/GoogleDrive-mutwil@plant.tools/My Drive/Projects/2026_KingdomStress/CC-Fig 4 data/outputs/hormone_networks"
os.makedirs(GDRIVE_OUT, exist_ok=True)
for f in os.listdir(OUT):
    shutil.copy2(os.path.join(OUT, f), os.path.join(GDRIVE_OUT, f))

print(f"\n{'=' * 60}")
print(f"HORMONE NETWORKS COMPLETE")
print(f"  {len(os.listdir(OUT))} files in {OUT}")
print(f"  Copied to {GDRIVE_OUT}")
print(f"{'=' * 60}")
