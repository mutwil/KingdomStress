"""
Hormone x stress grid v2:
- Only show hormone-stress combos where the hormone is significantly responsive
- Select top connected bins by stress-specific edge strength (deviation from cross-stress mean)
- Separate top UP-biased and DOWN-biased edges
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

HORMONES_FULL = {
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

# ── Load PEA data to determine hormone responsiveness ─────────────────────
pea = pd.read_csv('/tmp/Mercator_pathway_analysis_summary_level1.csv')
pea['PARENT_BINCODE'] = pea['PARENT_BINCODE'].astype(str)
pea['UP'] = pea['US'] + pea['UDS']
pea['DOWN'] = pea['DS'] + pea['UDS']

proc = pd.read_csv('/tmp/mercator_process_list.csv', index_col=0)
proc['Bincode'] = proc['Bincode'].astype(str)
name2bc = dict(zip(proc['Bincode name'], proc['Bincode']))

# Compute hormone responsiveness per stress
RESPONSIVENESS_THRESHOLD = 0.20  # UP+DOWN must exceed this

hormone_activity = {}
for hfull, hshort in HORMONES_FULL.items():
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
        total = up + down
        bias = up - down
        hormone_activity[(hshort, stress)] = {
            'up': up, 'down': down, 'total': total, 'bias': bias,
            'active': total >= RESPONSIVENESS_THRESHOLD,
        }

print("Hormone-stress activity (threshold = {:.2f}):".format(RESPONSIVENESS_THRESHOLD))
for hfull, hshort in HORMONES_FULL.items():
    active_stresses = [s for s in STRESSES if hormone_activity.get((hshort, s), {}).get('active', False)]
    inactive = [s for s in STRESSES if not hormone_activity.get((hshort, s), {}).get('active', False)]
    print(f"  {hshort:<18} active: {', '.join(active_stresses)}")
    if inactive:
        print(f"  {'':18} inactive: {', '.join(inactive)}")

# ── Load per-stress edges ─────────────────────────────────────────────────
print("\nLoading per-stress L1 networks...")
stress_edge_dfs = {}
for stress in STRESSES:
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
    stress_edge_dfs[stress] = pd.DataFrame(rows)

# ── Compute cross-stress mean for each hormone edge ──────────────────────
# For each hormone, collect edge weights across stresses
print("Computing stress-specific edge scores...")

hormone_edge_means = {}
for hfull in HORMONES_FULL:
    all_hw = {}
    for stress in STRESSES:
        edf = stress_edge_dfs[stress]
        mask = (edf['source'] == hfull) | (edf['target'] == hfull)
        for _, r in edf[mask].iterrows():
            other = r['target'] if r['source'] == hfull else r['source']
            all_hw.setdefault(other, {})
            all_hw[other][stress] = {
                'up': r['UP_weight'], 'down': r['DOWN_weight'], 'mean': r['mean_weight'],
                'bias': r['edge_bias'],
            }
    hormone_edge_means[hfull] = all_hw

# ── Select top edges per hormone-stress ──────────────────────────────────
TOP_UP = 5
TOP_DOWN = 5

def select_edges(hfull, stress):
    """Select top edges for a hormone-stress combo by stress-specificity."""
    hw = hormone_edge_means.get(hfull, {})
    if not hw:
        return pd.DataFrame()

    rows = []
    for other, stress_vals in hw.items():
        if stress not in stress_vals:
            continue
        this_w = stress_vals[stress]

        # Cross-stress mean for this edge
        all_means = [v['mean'] for v in stress_vals.values()]
        cross_mean = np.mean(all_means)
        cross_std = np.std(all_means) if len(all_means) > 1 else 0.01

        # Stress-specificity: how much stronger is this edge in this stress?
        specificity = (this_w['mean'] - cross_mean) / max(cross_std, 0.01)

        rows.append({
            'source': hfull, 'target': other,
            'UP_weight': this_w['up'], 'DOWN_weight': this_w['down'],
            'mean_weight': this_w['mean'], 'edge_bias': this_w['bias'],
            'specificity': specificity,
            'cross_stress_mean': cross_mean,
        })

    rdf = pd.DataFrame(rows)
    if rdf.empty:
        return rdf

    # Select: top UP-biased + top DOWN-biased edges (by weight in this stress, not just specificity)
    # This ensures we see the strongest connections, not just the most specific ones
    up_edges = rdf[rdf['edge_bias'] > 0].nlargest(TOP_UP, 'UP_weight')
    down_edges = rdf[rdf['edge_bias'] < 0].nlargest(TOP_DOWN, 'DOWN_weight')

    # If not enough UP or DOWN, fill from the other
    selected = pd.concat([up_edges, down_edges])
    if len(selected) < TOP_UP + TOP_DOWN:
        remaining = rdf[~rdf['target'].isin(selected['target'])]
        extra = remaining.nlargest(TOP_UP + TOP_DOWN - len(selected), 'mean_weight')
        selected = pd.concat([selected, extra])

    return selected


# ── Load per-stress node bias ────────────────────────────────────────────
stress_bias_df = pd.read_csv(os.path.join(OUT, "bin_per_stress_L1.csv"))
bin_resp = dict(zip(
    pd.read_csv(os.path.join(OUT, "bin_summary_L1.csv"))['bin_name'],
    pd.read_csv(os.path.join(OUT, "bin_summary_L1.csv"))['total_responsive']
))

def get_stress_bias(stress, bin_name):
    row = stress_bias_df[(stress_bias_df['stress'] == stress) & (stress_bias_df['bin_name'] == bin_name)]
    return row.iloc[0]['bias'] if not row.empty else 0

def short_label(name):
    s = str(name).split('.')[-1].strip()
    return s[0].upper() + s[1:] if s else str(name)

def to_hex(rgba):
    return '#{:02x}{:02x}{:02x}'.format(int(rgba[0]*255), int(rgba[1]*255), int(rgba[2]*255))

cmap = plt.cm.RdBu_r

# ── Collect global ranges ────────────────────────────────────────────────
all_nb = []
all_eb = []
active_combos = []
for hfull, hshort in HORMONES_FULL.items():
    for stress in STRESSES:
        if not hormone_activity.get((hshort, stress), {}).get('active', False):
            continue
        active_combos.append((hfull, hshort, stress))
        sel = select_edges(hfull, stress)
        if sel.empty:
            continue
        nodes = set(sel['source'].tolist() + sel['target'].tolist())
        for n in nodes:
            all_nb.append(get_stress_bias(stress, n))
        all_eb.extend(sel['edge_bias'].tolist())

max_nb = max(abs(min(all_nb)), abs(max(all_nb))) if all_nb else 0.3
max_eb = max(abs(min(all_eb)), abs(max(all_eb))) if all_eb else 0.3
node_norm = Normalize(vmin=-max_nb, vmax=max_nb)
edge_norm = Normalize(vmin=-max_eb, vmax=max_eb)

# ── Determine grid layout ────────────────────────────────────────────────
# Only include active hormone-stress combos
# Columns = hormones that are active in at least 1 stress
# Rows = stresses
active_hormones = []
for hfull, hshort in HORMONES_FULL.items():
    if any(hormone_activity.get((hshort, s), {}).get('active', False) for s in STRESSES):
        active_hormones.append((hfull, hshort))

print(f"\nActive combos: {len(active_combos)}")
print(f"Active hormones: {[h[1] for h in active_hormones]}")

CELL_W = 700
CELL_H = 700
RADIUS = 220

# ── Write XGMML ──────────────────────────────────────────────────────────
fpath = os.path.join(OUT, "hormone_stress_grid_v2.xgmml")

with open(fpath, 'w') as f:
    f.write('<?xml version="1.0" encoding="UTF-8"?>\n')
    f.write('<graph label="Hormone x Stress networks (active only)" xmlns="http://www.cs.rpi.edu/XGMML" directed="0">\n')

    # Column headers (hormones)
    for col, (hfull, hshort) in enumerate(active_hormones):
        cx = col * CELL_W
        cy = -CELL_H * 0.7
        uid = f"HDR_H_{hshort}"
        f.write(f'  <node id="{escape(uid)}" label="{escape(hshort)}">\n')
        f.write(f'    <graphics x="{cx:.0f}" y="{cy:.0f}" w="5" h="5" fill="#FFFFFF" type="RECTANGLE" outline="#FFFFFF"/>\n')
        f.write(f'    <att name="shared name" type="string" value="{escape(uid)}"/>\n')
        f.write(f'    <att name="is_header" type="boolean" value="true"/>\n')
        f.write(f'  </node>\n')

    # Row headers (stresses)
    for row, stress in enumerate(STRESSES):
        cx = -CELL_W * 0.7
        cy = row * CELL_H
        uid = f"HDR_S_{stress}"
        f.write(f'  <node id="{escape(uid)}" label="{escape(stress)}">\n')
        f.write(f'    <graphics x="{cx:.0f}" y="{cy:.0f}" w="5" h="5" fill="#FFFFFF" type="RECTANGLE" outline="#FFFFFF"/>\n')
        f.write(f'    <att name="shared name" type="string" value="{escape(uid)}"/>\n')
        f.write(f'    <att name="is_header" type="boolean" value="true"/>\n')
        f.write(f'  </node>\n')

    # Sub-networks
    for col, (hfull, hshort) in enumerate(active_hormones):
        for row, stress in enumerate(STRESSES):
            if not hormone_activity.get((hshort, stress), {}).get('active', False):
                # Add a placeholder "inactive" node
                ox = col * CELL_W
                oy = row * CELL_H
                uid = f"{hshort}_{stress}_inactive"
                f.write(f'  <node id="{escape(uid)}" label="(inactive)">\n')
                f.write(f'    <graphics x="{ox:.0f}" y="{oy:.0f}" w="15" h="15" fill="#EEEEEE" type="ELLIPSE" outline="#CCCCCC"/>\n')
                f.write(f'    <att name="shared name" type="string" value="{escape(uid)}"/>\n')
                f.write(f'    <att name="is_header" type="boolean" value="false"/>\n')
                f.write(f'    <att name="hormone" type="string" value="{escape(hshort)}"/>\n')
                f.write(f'    <att name="stress" type="string" value="{escape(stress)}"/>\n')
                f.write(f'  </node>\n')
                continue

            sel = select_edges(hfull, stress)
            if sel.empty:
                continue

            ox = col * CELL_W
            oy = row * CELL_H

            # Add interconnections among neighbors
            neighbors = set(sel['target'].tolist())
            edf = stress_edge_dfs[stress]
            inter = edf[edf['source'].isin(neighbors) & edf['target'].isin(neighbors)]
            if not inter.empty:
                inter_top = inter.nlargest(min(10, len(inter)), 'mean_weight')
                sel = pd.concat([sel, inter_top]).drop_duplicates(subset=['source', 'target'])

            nodes = set(sel['source'].tolist() + sel['target'].tolist())

            # Layout: hormone center, others in circle
            positions = {hfull: (ox, oy)}
            others = sorted(nodes - {hfull})
            for j, n in enumerate(others):
                angle = 2 * math.pi * j / max(len(others), 1) - math.pi / 2
                positions[n] = (ox + RADIUS * math.cos(angle), oy + RADIUS * math.sin(angle))

            prefix = f"{hshort}_{stress}_"

            # Nodes
            for n in nodes:
                uid = escape(prefix + short_label(n))
                short = escape(short_label(n))
                x, y = positions.get(n, (ox, oy))
                bias = get_stress_bias(stress, n)
                resp = bin_resp.get(n, 0)
                color = to_hex(cmap(node_norm(bias)))
                size = resp * 60 + 15

                # Mark hormone node
                is_h = n == hfull
                if is_h:
                    size = max(size, 40)  # ensure hormone node is visible

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
                f.write(f'    <att name="is_hormone" type="boolean" value="{"true" if is_h else "false"}"/>\n')
                f.write(f'  </node>\n')

            # Edges
            for _, r in sel.iterrows():
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

# Copy
import shutil
GDRIVE_OUT = "/Users/vjx443/Library/CloudStorage/GoogleDrive-mutwil@plant.tools/My Drive/Projects/2026_KingdomStress/CC-Fig 4 data/outputs"
shutil.copy2(fpath, os.path.join(GDRIVE_OUT, "hormone_stress_grid_v2.xgmml"))

# Print summary table
print(f"\n{'Hormone':<18} {'Stress':<14} {'UP':>5} {'DOWN':>6} {'bias':>7} {'active':>7}")
print('-' * 60)
for hfull, hshort in HORMONES_FULL.items():
    for stress in STRESSES:
        act = hormone_activity.get((hshort, stress), {})
        if act:
            mark = 'YES' if act['active'] else 'no'
            print(f"  {hshort:<16} {stress:<14} {act['up']:>5.2f} {act['down']:>6.2f} {act['bias']:>+7.3f} {mark:>7}")
