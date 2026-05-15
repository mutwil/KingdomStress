"""
Phylogenetic conservation of stress responses across plant clades.
Which L1 bins and edges are conserved from chlorophytes to angiosperms?
"""

import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.cluster.hierarchy import linkage, dendrogram
from scipy.spatial.distance import pdist
import warnings
warnings.filterwarnings('ignore')

OUT = "/tmp/mercator_outputs"
BASE = "/tmp/mercator_data"

plt.rcParams.update({
    'font.family': 'Arial', 'font.size': 11,
    'figure.dpi': 300, 'savefig.dpi': 300,
    'savefig.bbox': 'tight', 'figure.facecolor': 'white',
})

# Phylogenetic order (basal to derived)
CLADES = ['Chlorophyte', 'Charophyte', 'Bryophyte', 'Lycophyte', 'Gymnosperm', 'Dicot', 'Monocot']
STRESSES = ['Heat', 'Cold', 'Drought', 'Salt', 'Pathogen', 'Heavy metal']

EXCLUDE_L0 = {'Enzyme classification', 'not assigned', 'Protein modification',
              'Protein biosynthesis'}


def short_label(name):
    s = str(name).split('.')[-1].strip()
    return s[0].upper() + s[1:] if s else str(name)


def load_clade_edges(clade, stress=None):
    """Load UP and DOWN L1 edges for a clade (optionally per stress)."""
    if stress:
        base_dir = os.path.join(BASE, 'Clades', clade, stress)
    else:
        base_dir = os.path.join(BASE, 'Clades', clade)

    edges = {}
    for fname, d in [('Mercator_network_UP_Level1 (Normalised).csv', 'up'),
                      ('Mercator_network_DOWN_Level1 (Normalised).csv', 'down'),
                      ('Mercator_network_UP_level1 (Normalised).csv', 'up'),
                      ('Mercator_network_DOWN_level1 (Normalised).csv', 'down')]:
        path = os.path.join(base_dir, fname)
        if not os.path.exists(path):
            continue
        try:
            df = pd.read_csv(path, index_col=0)
        except Exception:
            continue
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
    return pd.DataFrame(rows)


# ═══════════════════════════════════════════════════════════════════════════
# 1. Overall clade L1 bin direction bias (all stresses combined)
# ═══════════════════════════════════════════════════════════════════════════
print("Loading clade-level L1 data (all stresses combined)...")

clade_bin_bias = {}
for clade in CLADES:
    edf = load_clade_edges(clade)
    if edf.empty:
        continue

    # Compute per-bin bias: average edge bias for edges involving each bin
    bin_biases = {}
    for _, r in edf.iterrows():
        for node in [r['source'], r['target']]:
            bin_biases.setdefault(node, []).append(r['bias'])

    clade_bin_bias[clade] = {b: np.mean(v) for b, v in bin_biases.items()}
    print(f"  {clade}: {len(edf)} edges, {len(bin_biases)} bins")

# Build bin x clade matrix
all_bins = set()
for v in clade_bin_bias.values():
    all_bins.update(v.keys())

bin_clade_mat = pd.DataFrame(0.0, index=sorted(all_bins), columns=CLADES)
for clade in CLADES:
    for b, bias in clade_bin_bias.get(clade, {}).items():
        bin_clade_mat.loc[b, clade] = bias

# Filter to bins present in at least 4 clades
presence = (bin_clade_mat != 0).sum(axis=1)
common_bins = presence[presence >= 4].index
bin_clade_mat = bin_clade_mat.loc[common_bins]

# Sort by mean bias
bin_clade_mat['_mean'] = bin_clade_mat[CLADES].mean(axis=1)
bin_clade_mat = bin_clade_mat.sort_values('_mean')
top_bottom = pd.concat([bin_clade_mat.head(20), bin_clade_mat.tail(20)])
top_bottom = top_bottom.drop(columns='_mean')


# ═══════════════════════════════════════════════════════════════════════════
# 2. Per-stress clade comparison
# ═══════════════════════════════════════════════════════════════════════════
print("\nLoading per-stress clade data...")

# For each stress, compute bin-level bias per clade
stress_clade_bias = {}
for stress in STRESSES:
    stress_clade_bias[stress] = {}
    for clade in CLADES:
        edf = load_clade_edges(clade, stress)
        if edf.empty:
            continue
        bin_biases = {}
        for _, r in edf.iterrows():
            for node in [r['source'], r['target']]:
                bin_biases.setdefault(node, []).append(r['bias'])
        stress_clade_bias[stress][clade] = {b: np.mean(v) for b, v in bin_biases.items()}
    print(f"  {stress}: {len(stress_clade_bias[stress])} clades with data")


# ═══════════════════════════════════════════════════════════════════════════
# 3. Conservation score: how consistent is each bin's direction across clades?
# ═══════════════════════════════════════════════════════════════════════════
print("\nComputing conservation scores...")

# For each bin: count how many clades agree on direction
conservation = []
for b in common_bins:
    vals = bin_clade_mat.loc[b, CLADES]
    n_up = (vals > 0.01).sum()
    n_down = (vals < -0.01).sum()
    n_present = (vals != 0).sum()
    consistency = max(n_up, n_down) / max(n_present, 1)
    dominant_dir = 'UP' if n_up > n_down else 'DOWN'
    conservation.append({
        'bin': b, 'short': short_label(b),
        'n_clades': n_present,
        'consistency': consistency,
        'dominant_direction': dominant_dir,
        'mean_bias': vals.mean(),
    })

cons_df = pd.DataFrame(conservation).sort_values('consistency', ascending=False)
cons_df.to_csv(os.path.join(OUT, "phylo_conservation_L1.csv"), index=False)


# ═══════════════════════════════════════════════════════════════════════════
# FIGURES
# ═══════════════════════════════════════════════════════════════════════════

# ── Figure 1: Bin direction bias across clades (phylogenetic order) ──────
fig, ax = plt.subplots(figsize=(12, 16))

max_b = max(abs(top_bottom.min().min()), abs(top_bottom.max().max()))
sns.heatmap(top_bottom, cmap='RdBu_r', center=0, vmin=-max_b, vmax=max_b,
            ax=ax, linewidths=0.4, linecolor='white',
            annot=True, fmt='+.2f', annot_kws={'size': 8},
            yticklabels=[short_label(b) for b in top_bottom.index],
            cbar_kws={'label': 'Direction bias (UP - DOWN)', 'shrink': 0.3})
ax.set_title("L1 bin direction bias across plant clades\n"
             "(phylogenetic order: basal left, derived right)\n"
             "Top 20 UP-biased + top 20 DOWN-biased bins",
             fontsize=14, fontweight='bold')
ax.tick_params(axis='x', labelsize=12, rotation=45)
ax.tick_params(axis='y', labelsize=9)

plt.tight_layout()
fig.savefig(os.path.join(OUT, "phylo_bin_bias_L1.png"))
fig.savefig(os.path.join(OUT, "phylo_bin_bias_L1.pdf"))
plt.close(fig)


# ── Figure 2: Conservation score -- which bins are universally conserved? ─
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))

# Most conserved UP bins
up_cons = cons_df[(cons_df['dominant_direction'] == 'UP') & (cons_df['n_clades'] >= 5)]
up_cons = up_cons.nlargest(15, 'consistency')

ax1.barh(range(len(up_cons)), up_cons['mean_bias'], color='#E63946', alpha=0.85)
ax1.set_yticks(range(len(up_cons)))
ax1.set_yticklabels(up_cons['short'], fontsize=10)
# Add clade count
for i, (_, r) in enumerate(up_cons.iterrows()):
    ax1.text(r['mean_bias'] + 0.005, i, f"{int(r['n_clades'])}/{len(CLADES)} clades",
             fontsize=8, va='center')
ax1.set_xlabel("Mean direction bias", fontsize=12)
ax1.set_title("Most conserved UP-biased bins\n(consistent across clades)", fontweight='bold', fontsize=14)
ax1.invert_yaxis()

# Most conserved DOWN bins
dn_cons = cons_df[(cons_df['dominant_direction'] == 'DOWN') & (cons_df['n_clades'] >= 5)]
dn_cons = dn_cons.nlargest(15, 'consistency')

ax2.barh(range(len(dn_cons)), dn_cons['mean_bias'], color='#457B9D', alpha=0.85)
ax2.set_yticks(range(len(dn_cons)))
ax2.set_yticklabels(dn_cons['short'], fontsize=10)
for i, (_, r) in enumerate(dn_cons.iterrows()):
    ax2.text(r['mean_bias'] - 0.005, i, f"{int(r['n_clades'])}/{len(CLADES)} clades",
             fontsize=8, va='center', ha='right')
ax2.set_xlabel("Mean direction bias", fontsize=12)
ax2.set_title("Most conserved DOWN-biased bins\n(consistent across clades)", fontweight='bold', fontsize=14)
ax2.invert_yaxis()

fig.suptitle("Phylogenetically conserved stress responses (L1 bins, all stresses)",
             fontsize=16, fontweight='bold', y=1.02)
plt.tight_layout()
fig.savefig(os.path.join(OUT, "phylo_conservation_bars.png"))
fig.savefig(os.path.join(OUT, "phylo_conservation_bars.pdf"))
plt.close(fig)


# ── Figure 3: Per-stress conservation heatmap ────────────────────────────
# For key bins, show their direction bias across clades x stresses

key_up = ['Protein homeostasis.protein quality control',
          'Redox homeostasis.glutathione-based redox regulation',
          'Carbohydrate metabolism.galactose metabolism',
          'Phytohormone action.abscisic acid',
          'Multi-process regulation.retrograde signalling',
          'Multi-process regulation.circadian clock system']

key_down = ['Cell wall organisation.pectin',
            'Cell wall organisation.cell wall proteins',
            'Photosynthesis.photophosphorylation',
            'Cytoskeleton organisation.microtubular network',
            'Phytohormone action.brassinosteroid',
            'Chromatin organisation.chromatin structure']

key_bins = key_up + key_down

fig, axes = plt.subplots(2, 3, figsize=(24, 14))

for i, stress in enumerate(STRESSES):
    ax = axes[i // 3, i % 3]

    data = pd.DataFrame(0.0, index=key_bins, columns=CLADES)
    for clade in CLADES:
        clade_data = stress_clade_bias.get(stress, {}).get(clade, {})
        for b in key_bins:
            if b in clade_data:
                data.loc[b, clade] = clade_data[b]

    max_b = 0.4
    sns.heatmap(data, cmap='RdBu_r', center=0, vmin=-max_b, vmax=max_b,
                ax=ax, linewidths=0.4, linecolor='white',
                annot=True, fmt='+.2f', annot_kws={'size': 8},
                yticklabels=[short_label(b) for b in data.index],
                cbar=i == 0,
                cbar_kws={'label': 'Bias', 'shrink': 0.5} if i == 0 else {})
    ax.set_title(stress, fontsize=16, fontweight='bold')
    ax.tick_params(axis='x', labelsize=10, rotation=45)
    ax.tick_params(axis='y', labelsize=9)

fig.suptitle("Key L1 bins: direction bias per clade per stress\n"
             "(Clades in phylogenetic order: basal left, derived right)",
             fontsize=16, fontweight='bold', y=1.02)
plt.tight_layout()
fig.savefig(os.path.join(OUT, "phylo_per_stress_key_bins.png"))
fig.savefig(os.path.join(OUT, "phylo_per_stress_key_bins.pdf"))
plt.close(fig)


# ── Figure 4: Clade similarity dendrogram based on L1 bias profiles ──────
fig, ax = plt.subplots(figsize=(10, 6))

# Use bin_clade_mat transposed: clades as observations, bins as features
clade_profiles = bin_clade_mat[CLADES].T.fillna(0)
dist = pdist(clade_profiles.values, metric='correlation')
Z = linkage(dist, method='average')
dendrogram(Z, labels=CLADES, ax=ax, leaf_rotation=45, leaf_font_size=12)
ax.set_ylabel("Correlation distance", fontsize=12)
ax.set_title("Clade similarity based on L1 stress response profiles\n"
             "(average linkage, correlation distance)",
             fontsize=14, fontweight='bold')
plt.tight_layout()
fig.savefig(os.path.join(OUT, "phylo_clade_dendrogram.png"))
fig.savefig(os.path.join(OUT, "phylo_clade_dendrogram.pdf"))
plt.close(fig)


# Copy
import shutil
GDRIVE_OUT = "/Users/vjx443/Library/CloudStorage/GoogleDrive-mutwil@plant.tools/My Drive/Projects/2026_KingdomStress/CC-Fig 4 data/outputs"
for f in os.listdir(OUT):
    if f.startswith("phylo_"):
        shutil.copy2(os.path.join(OUT, f), os.path.join(GDRIVE_OUT, f))

print("\nSaved: phylo_bin_bias_L1, phylo_conservation_bars, phylo_per_stress_key_bins, phylo_clade_dendrogram")
