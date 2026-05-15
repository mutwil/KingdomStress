"""
Export as XGMML format -- embeds node and edge attributes directly.
No separate table import needed.
"""

import os
import math
import pandas as pd
import numpy as np
from xml.sax.saxutils import escape

OUT = "/tmp/mercator_outputs"

def short_label(name):
    """Part after last dot, capitalize first letter."""
    s = str(name).split('.')[-1].strip()
    return s[0].upper() + s[1:] if s else str(name)


def compute_concentric_positions(nodes_df, n_rings=4, base_radius=200, ring_spacing=180):
    """
    Concentric layout: most DOWN-biased in center, most UP-biased on outside.
    Returns dict of {name: (x, y)}.
    """
    # Sort by direction_bias: most negative first (center)
    sorted_nodes = nodes_df.sort_values('direction_bias').reset_index(drop=True)

    nodes_per_ring = len(sorted_nodes) / n_rings
    pos = {}

    for i, (_, r) in enumerate(sorted_nodes.iterrows()):
        ring = min(int(i / nodes_per_ring), n_rings - 1)
        # Collect nodes per ring first
        pos[str(r['name'])] = ring

    # Now assign angular positions within each ring
    ring_nodes = {}
    for name, ring in pos.items():
        ring_nodes.setdefault(ring, []).append(name)

    positions = {}
    for ring in range(n_rings):
        nodes_in_ring = ring_nodes.get(ring, [])
        radius = base_radius + ring * ring_spacing
        for j, name in enumerate(nodes_in_ring):
            angle = 2 * math.pi * j / max(len(nodes_in_ring), 1) - math.pi / 2
            x = radius * math.cos(angle)
            y = radius * math.sin(angle)
            positions[name] = (x, y)

    return positions


def write_xgmml(nodes_csv, edges_csv, output_path, title):
    nodes = pd.read_csv(nodes_csv)
    edges = pd.read_csv(edges_csv)

    # Build full name -> short name map for consistent IDs
    name_map = {}
    for _, r in nodes.iterrows():
        name_map[str(r['name'])] = short_label(r['name'])

    # Compute concentric positions
    positions = compute_concentric_positions(nodes)

    with open(output_path, 'w') as f:
        f.write('<?xml version="1.0" encoding="UTF-8"?>\n')
        f.write('<graph label="{}" xmlns="http://www.cs.rpi.edu/XGMML" directed="0">\n'.format(escape(title)))

        # Nodes -- use short name as both id and label, embed x/y
        for _, r in nodes.iterrows():
            full_name = escape(str(r['name']))
            short = escape(short_label(r['name']))
            x, y = positions.get(str(r['name']), (0, 0))
            f.write(f'  <node id="{short}" label="{short}">\n')
            f.write(f'    <graphics x="{x:.1f}" y="{y:.1f}"/>\n')
            f.write(f'    <att name="name" type="string" value="{short}"/>\n')
            f.write(f'    <att name="shared name" type="string" value="{short}"/>\n')
            f.write(f'    <att name="full_name" type="string" value="{full_name}"/>\n')
            f.write(f'    <att name="parent_bin" type="string" value="{escape(str(r["parent_bin"]))}"/>\n')
            f.write(f'    <att name="direction_bias" type="real" value="{r["direction_bias"]}"/>\n')
            f.write(f'    <att name="mean_UP" type="real" value="{r["mean_UP"]}"/>\n')
            f.write(f'    <att name="mean_DOWN" type="real" value="{r["mean_DOWN"]}"/>\n')
            f.write(f'    <att name="total_responsive" type="real" value="{r["total_responsive"]}"/>\n')
            f.write(f'    <att name="node_color_hex" type="string" value="{r["node_color_hex"]}"/>\n')
            f.write(f'    <att name="node_size" type="real" value="{r["node_size"]}"/>\n')
            f.write(f'  </node>\n')

        # Edges -- use short names matching node IDs
        for _, r in edges.iterrows():
            src = escape(name_map.get(str(r['source']), short_label(r['source'])))
            tgt = escape(name_map.get(str(r['target']), short_label(r['target'])))
            f.write(f'  <edge source="{src}" target="{tgt}" label="{src} (co-occurrence) {tgt}">\n')
            f.write(f'    <graphics cy:edgeBend="" cy:curved="STRAIGHT_LINES" xmlns:cy="http://www.cytoscape.org"/>\n')
            f.write(f'    <att name="UP_weight" type="real" value="{r["UP_weight"]}"/>\n')
            f.write(f'    <att name="DOWN_weight" type="real" value="{r["DOWN_weight"]}"/>\n')
            f.write(f'    <att name="mean_weight" type="real" value="{r["mean_weight"]}"/>\n')
            f.write(f'    <att name="edge_bias" type="real" value="{r["edge_bias"]}"/>\n')
            f.write(f'    <att name="edge_color_hex" type="string" value="{r["edge_color_hex"]}"/>\n')
            f.write(f'    <att name="edge_width" type="real" value="{r["edge_width"]}"/>\n')
            f.write(f'    <att name="interaction" type="string" value="co-occurrence"/>\n')
            f.write(f'  </edge>\n')

        f.write('</graph>\n')

    print(f"  {output_path} ({os.path.getsize(output_path)/1024:.1f} KB)")


# Export all versions
for level in ['Level0', 'Level0_full', 'Level1_filtered', 'Level1_full']:
    nodes_csv = os.path.join(OUT, f"cytoscape_nodes_{level}.csv")
    edges_csv = os.path.join(OUT, f"cytoscape_edges_{level}.csv")
    out_path = os.path.join(OUT, f"network_{level}.xgmml")
    if os.path.exists(nodes_csv) and os.path.exists(edges_csv):
        write_xgmml(nodes_csv, edges_csv, out_path, f"MapMan {level} co-occurrence")

# Copy to Google Drive
import shutil
GDRIVE_OUT = "/Users/vjx443/Library/CloudStorage/GoogleDrive-mutwil@plant.tools/My Drive/Projects/2026_KingdomStress/CC-Fig 4 data/outputs"
for f in os.listdir(OUT):
    if f.endswith('.xgmml'):
        shutil.copy2(os.path.join(OUT, f), os.path.join(GDRIVE_OUT, f))

print("\nImport in Cytoscape: File > Import > Network from File > select .xgmml")
print("All node/edge attributes are embedded. Then just set Style mappings.")
