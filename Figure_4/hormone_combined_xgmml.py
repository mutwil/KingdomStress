"""
Single XGMML with all hormone x stress networks as a grid layout.
Hormones as columns, stresses as rows.
Each sub-network is offset in x,y to form a grid.
"""

import os
import math
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from xml.sax.saxutils import escape

OUT = "/tmp/mercator_outputs"
BASE = "/tmp/mercator_data"

HORMONES = [
    ('Phytohormone action.abscisic acid', 'Abscisic acid'),
    ('Phytohormone action.salicylic acid', 'Salicylic acid'),
    ('Phytohormone action.brassinosteroid', 'Brassinosteroid'),
    ('Phytohormone action.gibberellin', 'Gibberellin'),
    ('Phytohormone action.jasmonic acid', 'Jasmonic acid'),
    ('Phytohormone action.ethylene', 'Ethylene'),
    ('Phytohormone action.auxin', 'Auxin'),
    ('Phytohormone action.cytokinin', 'Cytokinin'),
]

STRESSES = ['Heat', 'Cold', 'Drought', 'Salt', 'Pathogen', 'Heavy metal']

# Grid spacing
CELL_W = 700
CELL_H = 700
RADIUS = 220
TOP_K = 10

# Load summaries
stress_df = pd.read_csv(os.path.join(OUT, "bin_per_stress_L1.csv"))
bin_resp = dict(zip(
    pd.read_csv(os.path.join(OUT, "bin_summary_L1.csv"))['bin_name'],
    pd.read_csv(os.path.join(OUT, "bin_summary_L1.csv"))['total_responsive']
))

cmap = plt.cm.RdBu_r


def get_stress_bias(stress, bin_name):
    row = stress_df[(stress_df['stress'] == stress) & (stress_df['bin_name'] == bin_name)]
    return row.iloc[0]['bias'] if not row.empty else 0


def short_label(name):
    s = str(name).split('.')[-1].strip()
    return s[0].upper() + s[1:] if s else str(name)


def to_hex(rgba):
    return '#{:02x}{:02x}{:02x}'.format(int(rgba[0]*255), int(rgba[1]*255), int(rgba[2]*255))


def load_stress_edges(stress):
    edges = {}
    for fname, direction in [
        (f"Mercator_network_UP_Level1 (All_organ).csv", 'up'),
        (f"Mercator_network_DOWN_Level1 (All_organ).csv", 'down'),
    ]:
        path = os.path.join(BASE, "Stresses", stress, fname)
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


# ── Load all stress edges ─────────────────────────────────────────────────
print("Loading per-stress L1 networks...")
all_stress_edges = {}
for stress in STRESSES:
    all_stress_edges[stress] = load_stress_edges(stress)
    print(f"  {stress}: {len(all_stress_edges[stress])} edges")

# ── Collect global bias range for consistent coloring ─────────────────────
all_node_biases = []
all_edge_biases = []
for stress in STRESSES:
    for hfull, hshort in HORMONES:
        edf = all_stress_edges[stress]
        mask = (edf['source'] == hfull) | (edf['target'] == hfull)
        h_edges = edf[mask].nlargest(TOP_K, 'mean_weight')
        nodes = set(h_edges['source'].tolist() + h_edges['target'].tolist())
        for n in nodes:
            all_node_biases.append(get_stress_bias(stress, n))
        all_edge_biases.extend(h_edges['edge_bias'].tolist())

max_nb = max(abs(min(all_node_biases)), abs(max(all_node_biases))) if all_node_biases else 0.3
max_eb = max(abs(min(all_edge_biases)), abs(max(all_edge_biases))) if all_edge_biases else 0.3
node_norm = Normalize(vmin=-max_nb, vmax=max_nb)
edge_norm = Normalize(vmin=-max_eb, vmax=max_eb)

# ── Write combined XGMML ─────────────────────────────────────────────────
print("\nWriting combined XGMML...")

# Track unique node IDs (since same bin name appears in multiple sub-networks)
# Use hormone_stress prefix to make unique
fpath = os.path.join(OUT, "hormone_stress_grid.xgmml")

with open(fpath, 'w') as f:
    f.write('<?xml version="1.0" encoding="UTF-8"?>\n')
    f.write('<graph label="Hormone x Stress networks" xmlns="http://www.cs.rpi.edu/XGMML" directed="0">\n')

    # Add label nodes for row/column headers
    for col, (hfull, hshort) in enumerate(HORMONES):
        cx = col * CELL_W
        cy = -CELL_H  # above the first row
        node_id = f"HEADER_{hshort}"
        f.write(f'  <node id="{escape(node_id)}" label="{escape(hshort)}">\n')
        f.write(f'    <graphics x="{cx:.0f}" y="{cy:.0f}" w="10" h="10" fill="#FFFFFF" type="RECTANGLE" outline="#FFFFFF"/>\n')
        f.write(f'    <att name="name" type="string" value="{escape(node_id)}"/>\n')
        f.write(f'    <att name="shared name" type="string" value="{escape(node_id)}"/>\n')
        f.write(f'    <att name="is_header" type="boolean" value="true"/>\n')
        f.write(f'    <att name="header_type" type="string" value="hormone"/>\n')
        f.write(f'  </node>\n')

    for row, stress in enumerate(STRESSES):
        cx = -CELL_W
        cy = row * CELL_H
        node_id = f"HEADER_{stress}"
        f.write(f'  <node id="{escape(node_id)}" label="{escape(stress)}">\n')
        f.write(f'    <graphics x="{cx:.0f}" y="{cy:.0f}" w="10" h="10" fill="#FFFFFF" type="RECTANGLE" outline="#FFFFFF"/>\n')
        f.write(f'    <att name="name" type="string" value="{escape(node_id)}"/>\n')
        f.write(f'    <att name="shared name" type="string" value="{escape(node_id)}"/>\n')
        f.write(f'    <att name="is_header" type="boolean" value="true"/>\n')
        f.write(f'    <att name="header_type" type="string" value="stress"/>\n')
        f.write(f'  </node>\n')

    # Generate each sub-network
    for col, (hfull, hshort) in enumerate(HORMONES):
        for row, stress in enumerate(STRESSES):
            # Grid offset
            ox = col * CELL_W
            oy = row * CELL_H

            edf = all_stress_edges[stress]
            mask = (edf['source'] == hfull) | (edf['target'] == hfull)
            h_edges = edf[mask].nlargest(TOP_K, 'mean_weight')

            if h_edges.empty:
                continue

            nodes = set(h_edges['source'].tolist() + h_edges['target'].tolist())

            # Also add interconnections between neighbors
            non_hormone = [n for n in nodes if n != hfull]
            inter = edf[edf['source'].isin(non_hormone) & edf['target'].isin(non_hormone)]
            if not inter.empty:
                inter_top = inter.nlargest(min(TOP_K * 2, len(inter)), 'mean_weight')
                h_edges = pd.concat([h_edges, inter_top]).drop_duplicates(subset=['source', 'target'])
                nodes = set(h_edges['source'].tolist() + h_edges['target'].tolist())

            # Positions: hormone center, others in circle
            positions = {}
            positions[hfull] = (ox, oy)
            others = sorted(nodes - {hfull})
            for j, n in enumerate(others):
                angle = 2 * math.pi * j / max(len(others), 1) - math.pi / 2
                positions[n] = (ox + RADIUS * math.cos(angle), oy + RADIUS * math.sin(angle))

            # Write nodes (unique IDs per sub-network)
            prefix = f"{hshort}_{stress}_"
            for n in nodes:
                uid = escape(prefix + short_label(n))
                short = escape(short_label(n))
                x, y = positions[n]
                bias = get_stress_bias(stress, n)
                resp = bin_resp.get(n, 0)
                color = to_hex(cmap(node_norm(bias)))
                size = resp * 60 + 15

                f.write(f'  <node id="{uid}" label="{short}">\n')
                f.write(f'    <graphics x="{x:.0f}" y="{y:.0f}" w="{size:.0f}" h="{size:.0f}" fill="{color}"/>\n')
                f.write(f'    <att name="name" type="string" value="{uid}"/>\n')
                f.write(f'    <att name="shared name" type="string" value="{uid}"/>\n')
                f.write(f'    <att name="short_name" type="string" value="{short}"/>\n')
                f.write(f'    <att name="full_name" type="string" value="{escape(n)}"/>\n')
                f.write(f'    <att name="hormone" type="string" value="{escape(hshort)}"/>\n')
                f.write(f'    <att name="stress" type="string" value="{escape(stress)}"/>\n')
                f.write(f'    <att name="direction_bias" type="real" value="{bias:.4f}"/>\n')
                f.write(f'    <att name="node_color_hex" type="string" value="{color}"/>\n')
                f.write(f'    <att name="node_size" type="real" value="{size:.1f}"/>\n')
                f.write(f'    <att name="is_hormone" type="boolean" value="{"true" if n == hfull else "false"}"/>\n')
                f.write(f'  </node>\n')

            # Write edges
            for _, r in h_edges.iterrows():
                src_uid = escape(prefix + short_label(r['source']))
                tgt_uid = escape(prefix + short_label(r['target']))
                color = to_hex(cmap(edge_norm(r['edge_bias'])))
                width = r['mean_weight'] * 6

                f.write(f'  <edge source="{src_uid}" target="{tgt_uid}">\n')
                f.write(f'    <att name="UP_weight" type="real" value="{r["UP_weight"]:.4f}"/>\n')
                f.write(f'    <att name="DOWN_weight" type="real" value="{r["DOWN_weight"]:.4f}"/>\n')
                f.write(f'    <att name="mean_weight" type="real" value="{r["mean_weight"]:.4f}"/>\n')
                f.write(f'    <att name="edge_bias" type="real" value="{r["edge_bias"]:.4f}"/>\n')
                f.write(f'    <att name="edge_color_hex" type="string" value="{color}"/>\n')
                f.write(f'    <att name="edge_width" type="real" value="{width:.2f}"/>\n')
                f.write(f'    <att name="hormone" type="string" value="{escape(hshort)}"/>\n')
                f.write(f'    <att name="stress" type="string" value="{escape(stress)}"/>\n')
                f.write(f'    <att name="interaction" type="string" value="co-occurrence"/>\n')
                f.write(f'  </edge>\n')

    f.write('</graph>\n')

size_kb = os.path.getsize(fpath) / 1024
print(f"\nSaved: {fpath} ({size_kb:.0f} KB)")

# Copy to Google Drive
import shutil
GDRIVE_OUT = "/Users/vjx443/Library/CloudStorage/GoogleDrive-mutwil@plant.tools/My Drive/Projects/2026_KingdomStress/CC-Fig 4 data/outputs"
shutil.copy2(fpath, os.path.join(GDRIVE_OUT, "hormone_stress_grid.xgmml"))
print(f"Copied to {GDRIVE_OUT}")
print(f"\nGrid layout: {len(HORMONES)} hormones (columns) x {len(STRESSES)} stresses (rows)")
print("Import in Cytoscape, then set Style passthrough on node_color_hex and edge_color_hex")
