"""
Stress-specific EDGES at L1: which sub-bin co-occurrences are unique to each stress?
Exclude Enzyme classification, not assigned, Protein modification, Protein biosynthesis.
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
EXCLUDE_L0 = {'Enzyme classification', 'not assigned', 'Protein modification',
              'Protein biosynthesis'}


def short_label(name):
    s = str(name).split('.')[-1].strip()
    return s[0].upper() + s[1:] if s else str(name)


def short_edge(edge):
    parts = edge.split(' -- ')
    if len(parts) == 2:
        return f"{short_label(parts[0])} -- {short_label(parts[1])}"
    return edge


# ── Load per-stress L1 edges ─────────────────────────────────────────────
print("Loading per-stress L1 edges...")
stress_edges = {}
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
            'UP': vals['up'], 'DOWN': vals['down'],
            'mean_weight': (vals['up'] + vals['down']) / 2,
            'bias': vals['up'] - vals['down'],
        })
    stress_edges[stress] = pd.DataFrame(rows)
    print(f"  {stress}: {len(stress_edges[stress])} edges")

# ── Build edge x stress bias matrix ──────────────────────────────────────
print("Building bias matrix...")
all_edges = set()
for df in stress_edges.values():
    all_edges.update(df['edge'].tolist())

bias_mat = pd.DataFrame(0.0, index=sorted(all_edges), columns=STRESSES)
weight_mat = pd.DataFrame(0.0, index=sorted(all_edges), columns=STRESSES)

for stress in STRESSES:
    edf = stress_edges[stress].set_index('edge')
    common = bias_mat.index.intersection(edf.index)
    bias_mat.loc[common, stress] = edf.loc[common, 'bias']
    weight_mat.loc[common, stress] = edf.loc[common, 'mean_weight']

cross_mean = bias_mat.mean(axis=1)
dev_mat = bias_mat.sub(cross_mean, axis=0)

print(f"Total unique L1 edges: {len(all_edges)}")

# ── Figure 1: Per-stress top specific edges (bar charts) ─────────────────
print("Generating bar charts...")
fig, axes = plt.subplots(2, 3, figsize=(30, 22))

for i, stress in enumerate(STRESSES):
    ax = axes[i // 3, i % 3]
    dev = dev_mat[stress]

    top_up = dev.nlargest(10)
    top_dn = dev.nsmallest(10)
    combined = pd.concat([top_dn, top_up])

    colors = ['#2A9D8F' if v > 0 else '#8B3A62' for v in combined]
    ax.barh(range(len(combined)), combined.values, color=colors, alpha=0.85)
    ax.set_yticks(range(len(combined)))
    ax.set_yticklabels([short_edge(e) for e in combined.index], fontsize=8)
    ax.axvline(0, color='black', lw=0.5)
    ax.set_xlabel("Deviation from cross-stress mean", fontsize=11)
    ax.set_title(stress, fontsize=16, fontweight='bold')

fig.suptitle("Stress-specific L1 edges: which sub-bin co-occurrences are unique to each stress?\n"
             "(Green = more UP here | Purple = more DOWN here)",
             fontsize=18, fontweight='bold', y=1.02)
plt.tight_layout()
fig.savefig(os.path.join(OUT, "stress_specific_edges_L1_bars.png"))
fig.savefig(os.path.join(OUT, "stress_specific_edges_L1_bars.pdf"))
plt.close(fig)

# ── Figure 2: Signature heatmap (top 5 UP + 5 DOWN per stress) ──────────
print("Generating signature heatmap...")
rows_data = []
for stress in STRESSES:
    dev = dev_mat[stress]
    for edge in dev.nlargest(5).index:
        rows_data.append({'stress': stress, 'edge': edge, 'dev': dev[edge]})
    for edge in dev.nsmallest(5).index:
        rows_data.append({'stress': stress, 'edge': edge, 'dev': dev[edge]})

sig_edges = list(set(r['edge'] for r in rows_data))

sig_bias = bias_mat.loc[sig_edges][STRESSES]
sig_dev = dev_mat.loc[sig_edges][STRESSES]

# Sort: group by which stress they're most specific to
sig_dev['_stress'] = sig_dev.abs().idxmax(axis=1)
sig_dev['_val'] = sig_dev[STRESSES].abs().max(axis=1)
sig_dev = sig_dev.sort_values(['_stress', '_val'], ascending=[True, False])
order = sig_dev.index
sig_dev = sig_dev.drop(columns=['_stress', '_val'])

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 20), sharey=True,
                                 gridspec_kw={'width_ratios': [1, 1], 'wspace': 0.05})

# Left: actual bias
max_b = max(abs(sig_bias.loc[order].min().min()), abs(sig_bias.loc[order].max().max()))
sns.heatmap(sig_bias.loc[order], cmap='RdBu_r', center=0, vmin=-max_b, vmax=max_b,
            ax=ax1, linewidths=0.3, linecolor='white',
            yticklabels=[short_edge(e) for e in order],
            cbar_kws={'label': 'UP - DOWN bias', 'shrink': 0.3})
ax1.set_title("Direction bias", fontweight='bold', fontsize=14)
ax1.tick_params(axis='x', labelsize=11, rotation=45)
ax1.tick_params(axis='y', labelsize=8)

# Right: deviation from mean
max_d = max(abs(sig_dev.loc[order].min().min()), abs(sig_dev.loc[order].max().max()))
sns.heatmap(sig_dev.loc[order], cmap='PiYG', center=0, vmin=-max_d, vmax=max_d,
            ax=ax2, linewidths=0.3, linecolor='white',
            yticklabels=False,
            cbar_kws={'label': 'Deviation from mean', 'shrink': 0.3})
ax2.set_title("Stress-specificity", fontweight='bold', fontsize=14)
ax2.tick_params(axis='x', labelsize=11, rotation=45)

fig.suptitle("L1 signature edges per stress\n"
             "(Left: actual bias | Right: how specific to this stress)",
             fontsize=16, fontweight='bold', y=1.01)

fig.savefig(os.path.join(OUT, "stress_signature_edges_L1.png"))
fig.savefig(os.path.join(OUT, "stress_signature_edges_L1.pdf"))
plt.close(fig)

# ── Print summary ────────────────────────────────────────────────────────
for stress in STRESSES:
    dev = dev_mat[stress]
    print(f"\n{stress.upper()} -- top 5 UP-specific L1 edges:")
    for edge in dev.nlargest(5).index:
        print(f"  {short_edge(edge):<55} dev={dev[edge]:+.3f}  bias={bias_mat.loc[edge, stress]:+.3f}")
    print(f"{stress.upper()} -- top 5 DOWN-specific L1 edges:")
    for edge in dev.nsmallest(5).index:
        print(f"  {short_edge(edge):<55} dev={dev[edge]:+.3f}  bias={bias_mat.loc[edge, stress]:+.3f}")

# Copy
import shutil
GDRIVE_OUT = "/Users/vjx443/Library/CloudStorage/GoogleDrive-mutwil@plant.tools/My Drive/Projects/2026_KingdomStress/CC-Fig 4 data/outputs"
for f in os.listdir(OUT):
    if 'edges_L1' in f or 'signature_edges_L1' in f:
        shutil.copy2(os.path.join(OUT, f), os.path.join(GDRIVE_OUT, f))

print("\nSaved: stress_specific_edges_L1_bars.png/pdf, stress_signature_edges_L1.png/pdf")
