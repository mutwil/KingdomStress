"""
Stress-specific EDGES at L0: which bin-bin co-occurrences are unique to each stress?
Shows deviation of edge bias from cross-stress mean.
"""

import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

OUT = "/tmp/mercator_outputs"
BASE = "/tmp/mercator_data"

plt.rcParams.update({
    'font.family': 'Arial', 'font.size': 11,
    'figure.dpi': 300, 'savefig.dpi': 300,
    'savefig.bbox': 'tight', 'figure.facecolor': 'white',
})

STRESSES = ['Heat', 'Cold', 'Drought', 'Salt', 'Pathogen', 'Heavy metal']

# ── Load per-stress edges ─────────────────────────────────────────────────
stress_edges = {}
for stress in STRESSES:
    edges = {}
    for fname, d in [('Mercator_network_UP_Level0 (All_organ).csv', 'up'),
                      ('Mercator_network_DOWN_Level0 (All_organ).csv', 'down')]:
        path = os.path.join(BASE, 'Stresses', stress, fname)
        if not os.path.exists(path):
            continue
        df = pd.read_csv(path, index_col=0)
        for _, r in df.iterrows():
            key = tuple(sorted([r['source'], r['target']]))
            edges.setdefault(key, {'up': 0, 'down': 0})
            edges[key][d] = r['weight']

    rows = []
    for (src, tgt), vals in edges.items():
        if 'not assigned' in (src, tgt):
            continue
        rows.append({
            'source': src, 'target': tgt,
            'edge': f'{src} -- {tgt}',
            'UP': vals['up'], 'DOWN': vals['down'],
            'mean_weight': (vals['up'] + vals['down']) / 2,
            'bias': vals['up'] - vals['down'],
        })
    stress_edges[stress] = pd.DataFrame(rows)

# Build edge x stress bias matrix
all_edges = set()
for df in stress_edges.values():
    all_edges.update(df['edge'].tolist())

bias_mat = pd.DataFrame(0.0, index=sorted(all_edges), columns=STRESSES)
for stress in STRESSES:
    for _, r in stress_edges[stress].iterrows():
        bias_mat.loc[r['edge'], stress] = r['bias']

cross_mean = bias_mat.mean(axis=1)
dev_mat = bias_mat.sub(cross_mean, axis=0)


def short_edge(edge, n=22):
    """Shorten edge label: take last word of each bin."""
    parts = edge.split(' -- ')
    if len(parts) == 2:
        a = parts[0].split()[-1] if len(parts[0]) > n else parts[0]
        b = parts[1].split()[-1] if len(parts[1]) > n else parts[1]
        return f"{a} -- {b}"
    return edge[:n*2]


# ── Figure 1: Per-stress top specific edges (diverging bars) ──────────────
fig, axes = plt.subplots(2, 3, figsize=(30, 20))

for i, stress in enumerate(STRESSES):
    ax = axes[i // 3, i % 3]

    dev = dev_mat[stress]
    # Top 8 UP-specific + top 8 DOWN-specific
    top_up = dev.nlargest(8)
    top_dn = dev.nsmallest(8)
    combined = pd.concat([top_dn, top_up])

    colors = ['#2A9D8F' if v > 0 else '#8B3A62' for v in combined]
    ax.barh(range(len(combined)), combined.values, color=colors, alpha=0.85)
    ax.set_yticks(range(len(combined)))
    ax.set_yticklabels([short_edge(e) for e in combined.index], fontsize=9)
    ax.axvline(0, color='black', lw=0.5)
    ax.set_xlabel("Deviation from cross-stress mean", fontsize=11)
    ax.set_title(stress, fontsize=16, fontweight='bold')

fig.suptitle("Stress-specific edges: which bin-bin co-occurrences are unique to each stress?\n"
             "(Green = more UP here than average | Purple = more DOWN here than average)",
             fontsize=18, fontweight='bold', y=1.02)
plt.tight_layout()
fig.savefig(os.path.join(OUT, "stress_specific_edges_bars.png"))
fig.savefig(os.path.join(OUT, "stress_specific_edges_bars.pdf"))
plt.close(fig)


# ── Figure 2: Heatmap of top stress-specific edges across all stresses ───
# Select the most variable edges overall
edge_var = dev_mat.var(axis=1).nlargest(40)
top_edges = edge_var.index

fig, ax = plt.subplots(figsize=(14, 16))
plot_data = bias_mat.loc[top_edges]

# Sort by the stress with max absolute deviation
plot_data['_max_stress'] = dev_mat.loc[top_edges].abs().idxmax(axis=1)
plot_data['_max_dev'] = dev_mat.loc[top_edges].abs().max(axis=1)
plot_data = plot_data.sort_values(['_max_stress', '_max_dev'], ascending=[True, False])
plot_data = plot_data.drop(columns=['_max_stress', '_max_dev'])

max_b = max(abs(plot_data.min().min()), abs(plot_data.max().max()))
sns.heatmap(plot_data, cmap='RdBu_r', center=0, vmin=-max_b, vmax=max_b,
            ax=ax, linewidths=0.4, linecolor='white',
            yticklabels=[short_edge(e) for e in plot_data.index],
            annot=True, fmt='+.2f', annot_kws={'size': 8},
            cbar_kws={'label': 'UP - DOWN bias', 'shrink': 0.4})
ax.set_title("Top 40 most stress-variable edges\n(sorted by which stress they are most specific to)",
             fontsize=15, fontweight='bold')
ax.set_xlabel("Stress", fontsize=13)
ax.tick_params(axis='x', labelsize=12, rotation=45)
ax.tick_params(axis='y', labelsize=9)
plt.tight_layout()
fig.savefig(os.path.join(OUT, "stress_specific_edges_heatmap.png"))
fig.savefig(os.path.join(OUT, "stress_specific_edges_heatmap.pdf"))
plt.close(fig)


# ── Figure 3: Summary -- what defines each stress ────────────────────────
fig, ax = plt.subplots(figsize=(16, 10))

# For each stress, its top 3 UP-specific and top 3 DOWN-specific edges
rows_data = []
for stress in STRESSES:
    dev = dev_mat[stress]
    for edge in dev.nlargest(3).index:
        rows_data.append({'stress': stress, 'edge': short_edge(edge),
                          'deviation': dev[edge], 'direction': 'UP-specific'})
    for edge in dev.nsmallest(3).index:
        rows_data.append({'stress': stress, 'edge': short_edge(edge),
                          'deviation': dev[edge], 'direction': 'DOWN-specific'})

summary = pd.DataFrame(rows_data)
summary_pivot = summary.pivot_table(index='edge', columns='stress', values='deviation',
                                      aggfunc='first')
summary_pivot = summary_pivot.reindex(columns=STRESSES)

# Sort by stress assignment
summary_pivot['_stress'] = summary_pivot.abs().idxmax(axis=1)
summary_pivot['_val'] = summary_pivot[STRESSES].max(axis=1)
summary_pivot = summary_pivot.sort_values(['_stress', '_val'], ascending=[True, False])
summary_pivot = summary_pivot.drop(columns=['_stress', '_val'])

max_d = max(abs(summary_pivot.min().min()), abs(summary_pivot.max().max()))
sns.heatmap(summary_pivot, cmap='PiYG', center=0, vmin=-max_d, vmax=max_d,
            ax=ax, linewidths=0.5, linecolor='white',
            annot=True, fmt='+.2f', annot_kws={'size': 9},
            cbar_kws={'label': 'Deviation from cross-stress mean', 'shrink': 0.5})
ax.set_title("Signature edges per stress\n"
             "(Top 3 UP-specific + top 3 DOWN-specific edges per stress)",
             fontsize=15, fontweight='bold')
ax.set_xlabel("Stress", fontsize=13)
ax.tick_params(axis='x', labelsize=12, rotation=45)
ax.tick_params(axis='y', labelsize=10)
plt.tight_layout()
fig.savefig(os.path.join(OUT, "stress_signature_edges.png"))
fig.savefig(os.path.join(OUT, "stress_signature_edges.pdf"))
plt.close(fig)

# Copy
import shutil
GDRIVE_OUT = "/Users/vjx443/Library/CloudStorage/GoogleDrive-mutwil@plant.tools/My Drive/Projects/2026_KingdomStress/CC-Fig 4 data/outputs"
for f in os.listdir(OUT):
    if 'stress_specific_edges' in f or 'stress_signature' in f:
        shutil.copy2(os.path.join(OUT, f), os.path.join(GDRIVE_OUT, f))

print("Saved: stress_specific_edges_bars.png/pdf")
print("Saved: stress_specific_edges_heatmap.png/pdf")
print("Saved: stress_signature_edges.png/pdf")
