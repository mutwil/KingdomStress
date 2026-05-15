"""
Stress-signature network: L1 bins as nodes, edges colored by the stress
they are most specific to. Only top stress-specific edges included.
"""

import os
import math
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
import matplotlib.patches as mpatches
import networkx as nx
from xml.sax.saxutils import escape
import warnings
warnings.filterwarnings('ignore')

OUT = "/tmp/mercator_outputs"
BASE = "/tmp/mercator_data"

plt.rcParams.update({
    'font.family': 'Arial', 'font.size': 10,
    'figure.dpi': 300, 'savefig.dpi': 300,
    'savefig.bbox': 'tight', 'figure.facecolor': 'white',
})

STRESSES = ['Heat', 'Cold', 'Drought', 'Salt', 'Pathogen', 'Heavy metal']
STRESS_COLORS = {
    'Heat': '#E63946',
    'Cold': '#457B9D',
    'Drought': '#E9C46A',
    'Salt': '#2A9D8F',
    'Pathogen': '#8338EC',
    'Heavy metal': '#6D6875',
}

EXCLUDE_L0 = {'Enzyme classification', 'not assigned', 'Protein modification',
              'Protein biosynthesis'}

TOP_PER_STRESS = 8  # top edges per stress (UP + DOWN specific)


def short_label(name):
    s = str(name).split('.')[-1].strip()
    return s[0].upper() + s[1:] if s else str(name)


# ── Load per-stress L1 edges ─────────────────────────────────────────────
print("Loading per-stress L1 edges...")
stress_edge_dfs = {}
for stress in STRESSES:
    edges = {}
    for fname, d in [('Mercator_network_UP_Level1 (All_organ).csv', 'up'),
                      ('Mercator_network_DOWN_Level1 (All_organ).csv', 'down')]:
        path = os.path.join(BASE, 'Stresses', stress, fname)
        if not os.path.exists(path):
            continue
        df = pd.read_csv(path, index_col=0)
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
            'bias': vals['up'] - vals['down'],
            'mean_weight': (vals['up'] + vals['down']) / 2,
        })
    stress_edge_dfs[stress] = pd.DataFrame(rows)

# ── Build deviation matrix ───────────────────────────────────────────────
all_edges = set()
for df in stress_edge_dfs.values():
    all_edges.update(df['edge'].tolist())

bias_mat = pd.DataFrame(0.0, index=sorted(all_edges), columns=STRESSES)
for stress in STRESSES:
    edf = stress_edge_dfs[stress].set_index('edge')
    common = bias_mat.index.intersection(edf.index)
    bias_mat.loc[common, stress] = edf.loc[common, 'bias']

cross_mean = bias_mat.mean(axis=1)
dev_mat = bias_mat.sub(cross_mean, axis=0)

# ── Select top stress-specific edges ─────────────────────────────────────
selected_edges = []
for stress in STRESSES:
    dev = dev_mat[stress]
    # Top UP-specific
    for edge in dev.nlargest(TOP_PER_STRESS).index:
        selected_edges.append({
            'edge': edge, 'stress': stress,
            'deviation': dev[edge], 'bias': bias_mat.loc[edge, stress],
            'direction': 'UP',
        })
    # Top DOWN-specific
    for edge in dev.nsmallest(TOP_PER_STRESS).index:
        selected_edges.append({
            'edge': edge, 'stress': stress,
            'deviation': dev[edge], 'bias': bias_mat.loc[edge, stress],
            'direction': 'DOWN',
        })

sel_df = pd.DataFrame(selected_edges)

# For edges claimed by multiple stresses, keep the one with strongest deviation
sel_df['abs_dev'] = sel_df['deviation'].abs()
sel_df = sel_df.sort_values('abs_dev', ascending=False).drop_duplicates('edge', keep='first')
print(f"Selected {len(sel_df)} unique stress-specific edges")

# ── Build node + edge data for network ───────────────────────────────────
# Load node bias (overall mean across stresses)
bin_summary = pd.read_csv(os.path.join(OUT, "bin_summary_L1.csv"))
bin_bias = dict(zip(bin_summary['bin_name'], bin_summary['direction_bias']))

nodes = set()
for _, r in sel_df.iterrows():
    parts = r['edge'].split(' -- ')
    nodes.update(parts)

print(f"Total nodes: {len(nodes)}")

# ── Matplotlib figure ────────────────────────────────────────────────────
G = nx.Graph()
for n in nodes:
    G.add_node(n)

for _, r in sel_df.iterrows():
    parts = r['edge'].split(' -- ')
    if len(parts) == 2:
        G.add_edge(parts[0], parts[1], stress=r['stress'], deviation=r['deviation'],
                    bias=r['bias'], direction=r['direction'])

pos = nx.spring_layout(G, k=2.0, seed=42, iterations=200)

cmap_nodes = plt.cm.RdBu_r
max_nb = max(abs(bin_bias.get(n, 0)) for n in G.nodes) if G.nodes else 0.3
node_norm = Normalize(vmin=-max_nb, vmax=max_nb)

fig, ax = plt.subplots(figsize=(22, 18))

# Draw edges colored by stress
for (u, v, data) in G.edges(data=True):
    color = STRESS_COLORS[data['stress']]
    width = abs(data['deviation']) * 12
    style = '-' if data['direction'] == 'UP' else '--'
    ax.plot([pos[u][0], pos[v][0]], [pos[u][1], pos[v][1]],
            color=color, linewidth=width, alpha=0.6, linestyle=style,
            solid_capstyle='round', dash_capstyle='round')

# Draw nodes
node_list = list(G.nodes)
node_colors = [cmap_nodes(node_norm(bin_bias.get(n, 0))) for n in node_list]
wdeg = dict(G.degree())
max_deg = max(wdeg.values()) if wdeg else 1
node_sizes = [wdeg[n] / max_deg * 1500 + 200 for n in node_list]

nx.draw_networkx_nodes(G, pos, ax=ax, nodelist=node_list,
                       node_color=node_colors, node_size=node_sizes,
                       alpha=0.9, edgecolors='black', linewidths=0.8)

# Labels
labels = {}
for n in node_list:
    if wdeg[n] >= 2:  # only label nodes with 2+ edges
        s = short_label(n)
        # Wrap long labels
        if len(s) > 20:
            mid = s.rfind(' ', 0, 20)
            if mid > 5:
                s = s[:mid] + '\n' + s[mid+1:]
        labels[n] = s

nx.draw_networkx_labels(G, pos, labels=labels, ax=ax, font_size=7, font_weight='bold')

# Legend: stress colors
legend_elements = [mpatches.Patch(facecolor=STRESS_COLORS[s], alpha=0.6, label=s)
                   for s in STRESSES]
legend_elements.append(plt.Line2D([0], [0], color='grey', linewidth=2, linestyle='-',
                                   label='UP-specific'))
legend_elements.append(plt.Line2D([0], [0], color='grey', linewidth=2, linestyle='--',
                                   label='DOWN-specific'))
ax.legend(handles=legend_elements, loc='lower left', fontsize=10, title='Stress / Direction',
          title_fontsize=11, framealpha=0.9)

# Node colorbar
sm = ScalarMappable(cmap=cmap_nodes, norm=node_norm)
sm.set_array([])
cbar = plt.colorbar(sm, ax=ax, shrink=0.3, aspect=20, pad=0.02)
cbar.set_label('Node direction bias (UP - DOWN)', fontsize=11)

ax.set_title("Stress-signature co-occurrence network (L1)\n"
             "Edge color = stress where edge is most specific | "
             "Solid = UP-specific, Dashed = DOWN-specific\n"
             "Node color = overall UP (red) / DOWN (blue) bias | Node size = degree",
             fontsize=14, fontweight='bold')
ax.axis('off')
plt.tight_layout()
fig.savefig(os.path.join(OUT, "stress_signature_network_L1.png"))
fig.savefig(os.path.join(OUT, "stress_signature_network_L1.pdf"))
plt.close(fig)

# ── Export XGMML ─────────────────────────────────────────────────────────
def to_hex(rgba):
    return '#{:02x}{:02x}{:02x}'.format(int(rgba[0]*255), int(rgba[1]*255), int(rgba[2]*255))

xgmml_path = os.path.join(OUT, "stress_signature_network_L1.xgmml")
with open(xgmml_path, 'w') as f:
    f.write('<?xml version="1.0" encoding="UTF-8"?>\n')
    f.write('<graph label="Stress-signature L1 network" xmlns="http://www.cs.rpi.edu/XGMML" directed="0">\n')

    for n in G.nodes:
        short = escape(short_label(n))
        x, y = pos[n][0] * 500, pos[n][1] * 500
        bias = bin_bias.get(n, 0)
        color = to_hex(cmap_nodes(node_norm(bias)))
        size = wdeg[n] / max_deg * 60 + 15
        deg = wdeg[n]

        f.write(f'  <node id="{escape(short)}" label="{escape(short)}">\n')
        f.write(f'    <graphics x="{x:.0f}" y="{y:.0f}" w="{size:.0f}" h="{size:.0f}" fill="{color}"/>\n')
        f.write(f'    <att name="shared name" type="string" value="{escape(short)}"/>\n')
        f.write(f'    <att name="full_name" type="string" value="{escape(n)}"/>\n')
        f.write(f'    <att name="direction_bias" type="real" value="{bias:.4f}"/>\n')
        f.write(f'    <att name="node_color_hex" type="string" value="{color}"/>\n')
        f.write(f'    <att name="degree" type="integer" value="{deg}"/>\n')
        f.write(f'  </node>\n')

    for (u, v, data) in G.edges(data=True):
        src = escape(short_label(u))
        tgt = escape(short_label(v))
        stress = data['stress']
        color = STRESS_COLORS[stress]

        f.write(f'  <edge source="{src}" target="{tgt}">\n')
        f.write(f'    <att name="stress" type="string" value="{escape(stress)}"/>\n')
        f.write(f'    <att name="deviation" type="real" value="{data["deviation"]:.4f}"/>\n')
        f.write(f'    <att name="bias" type="real" value="{data["bias"]:.4f}"/>\n')
        f.write(f'    <att name="direction" type="string" value="{data["direction"]}"/>\n')
        f.write(f'    <att name="edge_color_hex" type="string" value="{color}"/>\n')
        ew = abs(data['deviation']) * 8
        f.write(f'    <att name="edge_width" type="real" value="{ew:.2f}"/>\n')
        f.write(f'    <att name="interaction" type="string" value="stress-specific"/>\n')
        f.write(f'  </edge>\n')

    f.write('</graph>\n')

print(f"XGMML: {xgmml_path} ({os.path.getsize(xgmml_path)/1024:.0f} KB)")

# Copy
import shutil
GDRIVE_OUT = "/Users/vjx443/Library/CloudStorage/GoogleDrive-mutwil@plant.tools/My Drive/Projects/2026_KingdomStress/CC-Fig 4 data/outputs"
for f in ['stress_signature_network_L1.png', 'stress_signature_network_L1.pdf',
          'stress_signature_network_L1.xgmml']:
    shutil.copy2(os.path.join(OUT, f), os.path.join(GDRIVE_OUT, f))

print("Saved: stress_signature_network_L1.png/pdf/xgmml")
