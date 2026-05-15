"""
Hormone networks: L1 hormone bins connected to L0 pathway bins.
Uses L1 network data, aggregates non-hormone bins to their L0 parent.
Per-stress, with activity filtering.
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

HORMONES = {
    'Phytohormone action.abscisic acid': 'Abscisic acid',
    'Phytohormone action.salicylic acid': 'Salicylic acid',
    'Phytohormone action.brassinosteroid': 'Brassinosteroid',
    'Phytohormone action.gibberellin': 'Gibberellin',
    'Phytohormone action.jasmonic acid': 'Jasmonic acid',
    'Phytohormone action.ethylene': 'Ethylene',
    'Phytohormone action.auxin': 'Auxin',
}

STRESSES = ['Heat', 'Cold', 'Drought', 'Salt', 'Pathogen', 'Heavy metal']
RESPONSIVENESS_THRESHOLD = 0.20

# Load L0 bin summaries for node coloring
summary_L0 = pd.read_csv(os.path.join(OUT, "bin_summary_L0.csv"))
bin_bias_L0 = dict(zip(summary_L0['bin_name'], summary_L0['direction_bias']))
bin_resp_L0 = dict(zip(summary_L0['bin_name'], summary_L0['total_responsive']))

# Load per-stress L0 bias
stress_L0 = pd.read_csv(os.path.join(OUT, "bin_per_stress_L0.csv"))
# Load per-stress L1 bias (for hormone nodes)
stress_L1 = pd.read_csv(os.path.join(OUT, "bin_per_stress_L1.csv"))

# Load PEA for hormone responsiveness
pea = pd.read_csv('/tmp/Mercator_pathway_analysis_summary_level1.csv')
pea['PARENT_BINCODE'] = pea['PARENT_BINCODE'].astype(str)
pea['UP'] = pea['US'] + pea['UDS']
pea['DOWN'] = pea['DS'] + pea['UDS']

proc = pd.read_csv('/tmp/mercator_process_list.csv', index_col=0)
proc['Bincode'] = proc['Bincode'].astype(str)
name2bc = dict(zip(proc['Bincode name'], proc['Bincode']))

# Compute hormone activity
hormone_activity = {}
for hfull, hshort in HORMONES.items():
    bc = name2bc.get(hfull)
    if not bc:
        continue
    hdf = pea[pea['PARENT_BINCODE'] == bc]
    for stress in STRESSES:
        sdf = hdf[hdf['stress'] == stress]
        if sdf.empty:
            continue
        up = sdf['UP'].mean()
        down = sdf['DOWN'].mean()
        hormone_activity[(hshort, stress)] = {
            'up': up, 'down': down, 'total': up + down,
            'bias': up - down,
            'active': (up + down) >= RESPONSIVENESS_THRESHOLD,
        }


def get_node_bias(stress, node_name, is_hormone=False):
    """Get per-stress bias for a node."""
    df = stress_L1 if is_hormone else stress_L0
    row = df[(df['stress'] == stress) & (df['bin_name'] == node_name)]
    return row.iloc[0]['bias'] if not row.empty else 0


def short_label(name):
    s = str(name).split('.')[-1].strip()
    return s[0].upper() + s[1:] if s else str(name)


def to_hex(rgba):
    return '#{:02x}{:02x}{:02x}'.format(int(rgba[0]*255), int(rgba[1]*255), int(rgba[2]*255))


cmap = plt.cm.RdBu_r


def load_and_aggregate_stress(stress):
    """
    Load L1 edges for a stress, then for edges involving a hormone bin:
    aggregate the other node to its L0 parent.
    Returns DataFrame with hormone (L1) -> pathway (L0) edges.
    """
    edges = {}
    for fname, direction in [
        ("Mercator_network_UP_Level1 (All_organ).csv", 'up'),
        ("Mercator_network_DOWN_Level1 (All_organ).csv", 'down'),
    ]:
        path = os.path.join(BASE, "Stresses", stress, fname)
        if not os.path.exists(path):
            continue
        df = pd.read_csv(path, index_col=0)
        for _, r in df.iterrows():
            src, tgt = r['source'], r['target']
            # Check if either is a hormone bin
            src_is_h = src in HORMONES
            tgt_is_h = tgt in HORMONES
            if not src_is_h and not tgt_is_h:
                continue  # skip non-hormone edges
            # Identify hormone and other
            if src_is_h:
                hormone = src
                other_l1 = tgt
            else:
                hormone = tgt
                other_l1 = src
            # Map other to L0
            other_l0 = other_l1.split('.')[0]
            # Skip non-biological-process bins
            EXCLUDE_L0 = {
                'Phytohormone action',   # hormone-hormone edges
                'Enzyme classification',  # generic EC categories, not biological processes
                'not assigned',           # unannotated
                'Protein modification',   # mostly kinases, overlaps with EC
                'Protein biosynthesis',   # housekeeping
            }
            if other_l0 in EXCLUDE_L0:
                continue
            key = (hormone, other_l0)
            edges.setdefault(key, {'up': [], 'down': []})
            edges[key][direction].append(r['weight'])

    # Aggregate: mean of L1 weights per hormone-L0 pair
    rows = []
    for (hormone, l0_bin), vals in edges.items():
        up_mean = np.mean(vals['up']) if vals['up'] else 0
        down_mean = np.mean(vals['down']) if vals['down'] else 0
        rows.append({
            'hormone': hormone,
            'target_L0': l0_bin,
            'UP_weight': up_mean,
            'DOWN_weight': down_mean,
            'mean_weight': (up_mean + down_mean) / 2,
            'edge_bias': up_mean - down_mean,
            'n_l1_edges_up': len(vals['up']),
            'n_l1_edges_down': len(vals['down']),
        })
    return pd.DataFrame(rows)


# ── Load and aggregate all stresses ──────────────────────────────────────
print("Loading and aggregating L1 -> L0 edges per stress...")
stress_agg = {}
for stress in STRESSES:
    adf = load_and_aggregate_stress(stress)
    stress_agg[stress] = adf
    print(f"  {stress}: {len(adf)} hormone-L0 edges")

# ── Collect global ranges ────────────────────────────────────────────────
all_nb = []
all_eb = []
for stress in STRESSES:
    adf = stress_agg[stress]
    for _, r in adf.iterrows():
        all_nb.append(get_node_bias(stress, r['hormone'], is_hormone=True))
        all_nb.append(get_node_bias(stress, r['target_L0'], is_hormone=False))
    all_eb.extend(adf['edge_bias'].tolist())

max_nb = max(abs(min(all_nb)), abs(max(all_nb))) if all_nb else 0.3
max_eb = max(abs(min(all_eb)), abs(max(all_eb))) if all_eb else 0.3
node_norm = Normalize(vmin=-max_nb, vmax=max_nb)
edge_norm = Normalize(vmin=-max_eb, vmax=max_eb)

# ── Determine active hormones ───────────────────────────────────────────
active_hormones = []
for hfull, hshort in HORMONES.items():
    if any(hormone_activity.get((hshort, s), {}).get('active', False) for s in STRESSES):
        active_hormones.append((hfull, hshort))
print(f"\nActive hormones: {[h[1] for h in active_hormones]}")

# ── Write combined XGMML ────────────────────────────────────────────────
CELL_W = 700
CELL_H = 700
RADIUS = 250
TOP_K = 5  # top UP + top DOWN edges per hormone-stress

fpath = os.path.join(OUT, "hormone_L0_grid.xgmml")
print(f"\nWriting {fpath}...")

with open(fpath, 'w') as f:
    f.write('<?xml version="1.0" encoding="UTF-8"?>\n')
    f.write('<graph label="Hormone (L1) x Pathway (L0) networks" xmlns="http://www.cs.rpi.edu/XGMML" directed="0">\n')

    # Column headers
    for col, (hfull, hshort) in enumerate(active_hormones):
        cx = col * CELL_W
        cy = -CELL_H * 0.6
        uid = f"HDR_{hshort}"
        f.write(f'  <node id="{escape(uid)}" label="{escape(hshort)}">\n')
        f.write(f'    <graphics x="{cx:.0f}" y="{cy:.0f}" w="5" h="5" fill="#FFFFFF" outline="#FFFFFF"/>\n')
        f.write(f'    <att name="shared name" type="string" value="{escape(uid)}"/>\n')
        f.write(f'    <att name="is_header" type="boolean" value="true"/>\n')
        f.write(f'  </node>\n')

    # Row headers
    for row, stress in enumerate(STRESSES):
        cx = -CELL_W * 0.6
        cy = row * CELL_H
        uid = f"HDR_{stress}"
        f.write(f'  <node id="{escape(uid)}" label="{escape(stress)}">\n')
        f.write(f'    <graphics x="{cx:.0f}" y="{cy:.0f}" w="5" h="5" fill="#FFFFFF" outline="#FFFFFF"/>\n')
        f.write(f'    <att name="shared name" type="string" value="{escape(uid)}"/>\n')
        f.write(f'    <att name="is_header" type="boolean" value="true"/>\n')
        f.write(f'  </node>\n')

    # Sub-networks
    for col, (hfull, hshort) in enumerate(active_hormones):
        for row, stress in enumerate(STRESSES):
            ox = col * CELL_W
            oy = row * CELL_H
            prefix = f"{hshort}_{stress}_"

            if not hormone_activity.get((hshort, stress), {}).get('active', False):
                # Inactive placeholder
                uid = f"{prefix}inactive"
                f.write(f'  <node id="{escape(uid)}" label="(inactive)">\n')
                f.write(f'    <graphics x="{ox:.0f}" y="{oy:.0f}" w="12" h="12" fill="#EEEEEE" outline="#CCCCCC"/>\n')
                f.write(f'    <att name="shared name" type="string" value="{escape(uid)}"/>\n')
                f.write(f'    <att name="is_header" type="boolean" value="false"/>\n')
                f.write(f'  </node>\n')
                continue

            adf = stress_agg[stress]
            h_edges = adf[adf['hormone'] == hfull].copy()
            if h_edges.empty:
                continue

            # Select top edges by mean weight
            sel = h_edges.nlargest(TOP_K, 'mean_weight')

            # Nodes: hormone center + L0 targets
            nodes = set(sel['target_L0'].tolist())
            positions = {hfull: (ox, oy)}
            others = sorted(nodes)
            for j, n in enumerate(others):
                angle = 2 * math.pi * j / max(len(others), 1) - math.pi / 2
                positions[n] = (ox + RADIUS * math.cos(angle), oy + RADIUS * math.sin(angle))

            # Write hormone node
            hbias = get_node_bias(stress, hfull, is_hormone=True)
            hcolor = to_hex(cmap(node_norm(hbias)))
            h_act = hormone_activity.get((hshort, stress), {})
            hsize = h_act.get('total', 0.2) * 80 + 20
            uid = escape(prefix + hshort)
            f.write(f'  <node id="{uid}" label="{escape(hshort)}">\n')
            f.write(f'    <graphics x="{ox:.0f}" y="{oy:.0f}" w="{hsize:.0f}" h="{hsize:.0f}" fill="{hcolor}"/>\n')
            f.write(f'    <att name="shared name" type="string" value="{uid}"/>\n')
            f.write(f'    <att name="short_name" type="string" value="{escape(hshort)}"/>\n')
            f.write(f'    <att name="full_name" type="string" value="{escape(hfull)}"/>\n')
            f.write(f'    <att name="hormone" type="string" value="{escape(hshort)}"/>\n')
            f.write(f'    <att name="stress" type="string" value="{escape(stress)}"/>\n')
            f.write(f'    <att name="direction_bias" type="real" value="{hbias:.4f}"/>\n')
            f.write(f'    <att name="node_color_hex" type="string" value="{hcolor}"/>\n')
            f.write(f'    <att name="node_size" type="real" value="{hsize:.1f}"/>\n')
            f.write(f'    <att name="is_hormone" type="boolean" value="true"/>\n')
            f.write(f'    <att name="bin_level" type="string" value="L1"/>\n')
            f.write(f'  </node>\n')

            # Write L0 target nodes
            for n in others:
                x, y = positions[n]
                bias = get_node_bias(stress, n, is_hormone=False)
                resp = bin_resp_L0.get(n, 0)
                color = to_hex(cmap(node_norm(bias)))
                size = resp * 50 + 15
                uid = escape(prefix + n)
                short = escape(short_label(n))
                f.write(f'  <node id="{uid}" label="{short}">\n')
                f.write(f'    <graphics x="{x:.0f}" y="{y:.0f}" w="{size:.0f}" h="{size:.0f}" fill="{color}"/>\n')
                f.write(f'    <att name="shared name" type="string" value="{uid}"/>\n')
                f.write(f'    <att name="short_name" type="string" value="{short}"/>\n')
                f.write(f'    <att name="full_name" type="string" value="{escape(n)}"/>\n')
                f.write(f'    <att name="hormone" type="string" value="{escape(hshort)}"/>\n')
                f.write(f'    <att name="stress" type="string" value="{escape(stress)}"/>\n')
                f.write(f'    <att name="direction_bias" type="real" value="{bias:.4f}"/>\n')
                f.write(f'    <att name="node_color_hex" type="string" value="{color}"/>\n')
                f.write(f'    <att name="node_size" type="real" value="{size:.1f}"/>\n')
                f.write(f'    <att name="is_hormone" type="boolean" value="false"/>\n')
                f.write(f'    <att name="bin_level" type="string" value="L0"/>\n')
                f.write(f'  </node>\n')

            # Write edges
            for _, r in sel.iterrows():
                src_uid = escape(prefix + hshort)
                tgt_uid = escape(prefix + r['target_L0'])
                color = to_hex(cmap(edge_norm(r['edge_bias'])))
                width = r['mean_weight'] * 8

                f.write(f'  <edge source="{src_uid}" target="{tgt_uid}">\n')
                f.write(f'    <att name="UP_weight" type="real" value="{r["UP_weight"]:.4f}"/>\n')
                f.write(f'    <att name="DOWN_weight" type="real" value="{r["DOWN_weight"]:.4f}"/>\n')
                f.write(f'    <att name="mean_weight" type="real" value="{r["mean_weight"]:.4f}"/>\n')
                f.write(f'    <att name="edge_bias" type="real" value="{r["edge_bias"]:.4f}"/>\n')
                f.write(f'    <att name="edge_color_hex" type="string" value="{color}"/>\n')
                f.write(f'    <att name="edge_width" type="real" value="{width:.2f}"/>\n')
                n_l1 = r['n_l1_edges_up'] + r['n_l1_edges_down']
                f.write(f'    <att name="n_l1_edges" type="integer" value="{n_l1}"/>\n')
                f.write(f'    <att name="hormone" type="string" value="{escape(hshort)}"/>\n')
                f.write(f'    <att name="stress" type="string" value="{escape(stress)}"/>\n')
                f.write(f'    <att name="interaction" type="string" value="co-occurrence"/>\n')
                f.write(f'  </edge>\n')

    f.write('</graph>\n')

size_kb = os.path.getsize(fpath) / 1024
print(f"Saved: {fpath} ({size_kb:.0f} KB)")

# Also save the aggregated edge data as CSV
all_agg = []
for stress in STRESSES:
    adf = stress_agg[stress].copy()
    adf['stress'] = stress
    adf['hormone_short'] = adf['hormone'].map(HORMONES)
    all_agg.append(adf)
agg_df = pd.concat(all_agg)
agg_df.to_csv(os.path.join(OUT, "hormone_L0_edges.csv"), index=False)

# Copy to Google Drive
import shutil
GDRIVE_OUT = "/Users/vjx443/Library/CloudStorage/GoogleDrive-mutwil@plant.tools/My Drive/Projects/2026_KingdomStress/CC-Fig 4 data/outputs"
shutil.copy2(fpath, os.path.join(GDRIVE_OUT, "hormone_L0_grid.xgmml"))
shutil.copy2(os.path.join(OUT, "hormone_L0_edges.csv"), os.path.join(GDRIVE_OUT, "hormone_L0_edges.csv"))

# Print summary
print(f"\nGrid: {len(active_hormones)} hormones x {len(STRESSES)} stresses")
print(f"Hormone nodes at Level 1, pathway nodes at Level 0")
print(f"Top {TOP_K} UP + {TOP_K} DOWN edges per hormone-stress combo")
