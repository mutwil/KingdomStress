"""
Supplementary figure:
A) Species cutoff curve
B) Jaccard clustermap UP edges
C) Jaccard clustermap DOWN edges
D) Phylogenetic conservation: key bins across clades
E) Clade dendrogram
"""

import os
import pickle
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.cluster.hierarchy import linkage, leaves_list, dendrogram
from scipy.spatial.distance import squareform, pdist
from collections import defaultdict
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
CLADES = ['Chlorophyte', 'Charophyte', 'Bryophyte', 'Lycophyte', 'Gymnosperm', 'Dicot', 'Monocot']
STRESS_COLORS = {
    'Heat': '#E63946', 'Cold': '#457B9D', 'Drought': '#E9C46A',
    'Salt': '#2A9D8F', 'Pathogen': '#8338EC', 'Heavy metal': '#6D6875',
}
EXCLUDE_L0 = {'Enzyme classification', 'not assigned', 'Protein modification',
              'Protein biosynthesis'}
STRESS_N = {'Heat': 96, 'Cold': 70, 'Drought': 92, 'Salt': 68, 'Pathogen': 61, 'Heavy metal': 46}
STRESS_N_SPECIES = {'Heat': 28, 'Cold': 33, 'Drought': 31, 'Salt': 26, 'Pathogen': 27, 'Heavy metal': 22}
MIN_EXP = 2
MIN_SPECIES = 5


def short_label(name):
    s = str(name).split('.')[-1].strip()
    return s[0].upper() + s[1:] if s else str(name)


# ═══════════════════════════════════════════════════════════════════════════
# Load data
# ═══════════════════════════════════════════════════════════════════════════
print("Loading data...")

# Species cutoff curve
cutoff_df = pd.read_csv(os.path.join(OUT, "species_cutoff_curve.csv"))

# Jaccard data (recompute from stress edges)
stress_edge_bias = {}
for stress in STRESSES:
    edges = {}
    for fname, d in [('Mercator_network_UP_Level1 (All_organ).csv', 'up'),
                      ('Mercator_network_DOWN_Level1 (All_organ).csv', 'down')]:
        path = os.path.join(BASE, 'Stresses', stress, fname)
        if not os.path.exists(path):
            continue
        df = pd.read_csv(path, index_col=0)
        for _, r in df.iterrows():
            if r['source'].split('.')[0] in EXCLUDE_L0 or r['target'].split('.')[0] in EXCLUDE_L0:
                continue
            key = tuple(sorted([r['source'], r['target']]))
            edges.setdefault(key, {'up': 0, 'down': 0})
            edges[key][d] = r['weight']
    stress_edge_bias[stress] = edges

with open(os.path.join(OUT, "species_cooccurrence_L1.pkl"), 'rb') as f:
    species_cooc = pickle.load(f)

stress_up, stress_down = {}, {}
for stress in STRESSES:
    n = STRESS_N[stress]
    bias_thresh = MIN_EXP / n
    sp_data = species_cooc.get(stress, {})
    up_set, down_set = set(), set()
    for k, v in stress_edge_bias[stress].items():
        bias = v['up'] - v['down']
        sp = sp_data.get(k, {'n_species_up': 0, 'n_species_down': 0})
        if bias > bias_thresh and sp['n_species_up'] >= MIN_SPECIES:
            up_set.add(k)
        if bias < -bias_thresh and sp['n_species_down'] >= MIN_SPECIES:
            down_set.add(k)
    stress_up[stress] = up_set
    stress_down[stress] = down_set

universal_up = set.intersection(*stress_up.values())
universal_down = set.intersection(*stress_down.values())
for stress in STRESSES:
    stress_up[stress] -= universal_up
    stress_down[stress] -= universal_down

def jaccard(set_dict):
    mat = pd.DataFrame(1.0, index=STRESSES, columns=STRESSES)
    for s1 in STRESSES:
        for s2 in STRESSES:
            if s1 == s2:
                continue
            a, b = set_dict[s1], set_dict[s2]
            union = len(a | b)
            mat.loc[s1, s2] = len(a & b) / union if union > 0 else 0
    return mat

jac_up = jaccard(stress_up)
jac_down = jaccard(stress_down)
jac_combined = (jac_up + jac_down) / 2

# Phylogenetic data
def load_clade_edges(clade):
    edges = {}
    for fname, d in [('Mercator_network_UP_Level1 (Normalised).csv', 'up'),
                      ('Mercator_network_DOWN_Level1 (Normalised).csv', 'down'),
                      ('Mercator_network_UP_level1 (Normalised).csv', 'up'),
                      ('Mercator_network_DOWN_level1 (Normalised).csv', 'down')]:
        path = os.path.join(BASE, 'Clades', clade, fname)
        if not os.path.exists(path):
            continue
        try:
            df = pd.read_csv(path, index_col=0)
        except Exception:
            continue
        for _, r in df.iterrows():
            if r['source'].split('.')[0] in EXCLUDE_L0 or r['target'].split('.')[0] in EXCLUDE_L0:
                continue
            key = tuple(sorted([r['source'], r['target']]))
            edges.setdefault(key, {'up': 0, 'down': 0})
            edges[key][d] = r['weight']
    rows = []
    for (src, tgt), vals in edges.items():
        rows.append({'source': src, 'target': tgt, 'bias': vals['up'] - vals['down']})
    return pd.DataFrame(rows)

print("Loading clade data...")
clade_bin_bias = {}
for clade in CLADES:
    edf = load_clade_edges(clade)
    if edf.empty:
        continue
    bin_biases = {}
    for _, r in edf.iterrows():
        for node in [r['source'], r['target']]:
            bin_biases.setdefault(node, []).append(r['bias'])
    clade_bin_bias[clade] = {b: np.mean(v) for b, v in bin_biases.items()}

# Build bin x clade matrix
all_bins = set()
for v in clade_bin_bias.values():
    all_bins.update(v.keys())
bin_clade_mat = pd.DataFrame(0.0, index=sorted(all_bins), columns=CLADES)
for clade in CLADES:
    for b, bias in clade_bin_bias.get(clade, {}).items():
        bin_clade_mat.loc[b, clade] = bias

# Key bins for phylo panel
key_bins = [
    'Protein homeostasis.protein quality control',
    'Redox homeostasis.glutathione-based redox regulation',
    'Carbohydrate metabolism.galactose metabolism',
    'Phytohormone action.abscisic acid',
    'Multi-process regulation.retrograde signalling',
    'Multi-process regulation.circadian clock system',
    'Amino acid metabolism.amino acid degradation',
    'Solute transport.carrier-mediated transport',
    'Cell wall organisation.pectin',
    'Cell wall organisation.cell wall proteins',
    'Photosynthesis.photophosphorylation',
    'Cytoskeleton organisation.microtubular network',
    'Phytohormone action.brassinosteroid',
    'Chromatin organisation.chromatin structure',
    'Cell division.cell cycle organisation',
    'Secondary metabolism.phenolics biosynthesis',
]
key_bins = [b for b in key_bins if b in bin_clade_mat.index]

# ═══════════════════════════════════════════════════════════════════════════
# Build figure
# ═══════════════════════════════════════════════════════════════════════════
print("Building supplementary figure...")

fig = plt.figure(figsize=(24, 22))
gs = fig.add_gridspec(3, 2, hspace=0.35, wspace=0.3,
                       height_ratios=[1.2, 1, 1.2])

# ── Panel A: Phylogenetic conservation heatmap (referenced by main Fig A) ─
ax_a = fig.add_subplot(gs[0, :])

phylo_data_a = bin_clade_mat.loc[key_bins, CLADES]
phylo_data_a['_mean'] = phylo_data_a.mean(axis=1)
phylo_data_a = phylo_data_a.sort_values('_mean')
phylo_data_a = phylo_data_a.drop('_mean', axis=1)

max_b_a = max(abs(phylo_data_a.min().min()), abs(phylo_data_a.max().max()))
sns.heatmap(phylo_data_a, cmap='RdBu_r', center=0, vmin=-max_b_a, vmax=max_b_a,
            ax=ax_a, linewidths=0.5, linecolor='white',
            annot=True, fmt='+.2f', annot_kws={'size': 9},
            yticklabels=[short_label(b) for b in phylo_data_a.index],
            cbar_kws={'label': 'Direction bias (UP - DOWN)', 'shrink': 0.4})
ax_a.set_title("A) Phylogenetic conservation of key L1 bin responses\n"
               "(clades in phylogenetic order: basal left, derived right)",
               fontweight='bold', fontsize=15)
ax_a.tick_params(axis='x', labelsize=12, rotation=45)
ax_a.tick_params(axis='y', labelsize=10)

# ── Panel B: Clade dendrogram (referenced by main Fig A) ─────────────────
ax_b = fig.add_subplot(gs[1, 0])

clade_profiles = bin_clade_mat[CLADES].T.fillna(0)
dist_clade = pdist(clade_profiles.values, metric='correlation')
Z_clade = linkage(dist_clade, method='average')
dendrogram(Z_clade, labels=CLADES, ax=ax_b, leaf_rotation=45, leaf_font_size=12)
ax_b.set_ylabel("Correlation distance", fontsize=13)
ax_b.set_title("B) Clade similarity\n(L1 stress response profiles)", fontweight='bold', fontsize=15)

# ── Panel C: Species cutoff curve (referenced by main Fig C) ─────────────
ax_c = fig.add_subplot(gs[1, 1])

for stress in STRESSES:
    sdf = cutoff_df[cutoff_df['stress'] == stress]
    ax_c.plot(sdf['cutoff'], sdf['total'], '-o', color=STRESS_COLORS[stress],
              label=f"{stress} ({STRESS_N_SPECIES[stress]} spp.)",
              linewidth=2, markersize=4)

ax_c.axvline(5, color='black', ls='--', alpha=0.5, lw=1.5)
ax_c.annotate('cutoff = 5', xy=(5.3, ax_c.get_ylim()[1]*0.85), fontsize=11,
              fontweight='bold', ha='left')
ax_c.set_xlabel('Minimum species cutoff', fontsize=13)
ax_c.set_ylabel('Number of directional L1 edges', fontsize=13)
ax_c.set_title('C) Edge retention vs species cutoff', fontweight='bold', fontsize=15)
ax_c.legend(fontsize=9, loc='upper right')
ax_c.set_xlim(1, 20)

# ── Panel D: Jaccard UP (referenced by main Fig C) ───────────────────────
ax_d = fig.add_subplot(gs[2, 0])

jac_dist = 1 - jac_combined.values
np.fill_diagonal(jac_dist, 0)
Z_jac = linkage(squareform(jac_dist), method='average')
leaf_order = leaves_list(Z_jac)
jac_order = [STRESSES[i] for i in leaf_order]

jac_up_ordered = jac_up.loc[jac_order, jac_order]
sns.heatmap(jac_up_ordered, cmap='Reds', vmin=0, vmax=0.45,
            ax=ax_d, linewidths=1, linecolor='white',
            annot=True, fmt='.2f', annot_kws={'size': 14, 'fontweight': 'bold'},
            square=True,
            cbar_kws={'label': 'Jaccard index', 'shrink': 0.6})
ax_d.set_title("D) UP edges: Jaccard similarity\n(universal removed, >=5 species)",
               fontweight='bold', fontsize=15)
ax_d.tick_params(axis='x', labelsize=12, rotation=45)
ax_d.tick_params(axis='y', labelsize=12, rotation=0)
for lbl in ax_d.get_xticklabels():
    lbl.set_color(STRESS_COLORS.get(lbl.get_text(), 'black'))
    lbl.set_fontweight('bold')
for lbl in ax_d.get_yticklabels():
    lbl.set_color(STRESS_COLORS.get(lbl.get_text(), 'black'))
    lbl.set_fontweight('bold')

# ── Panel E: Jaccard DOWN (referenced by main Fig C) ─────────────────────
ax_e = fig.add_subplot(gs[2, 1])

jac_down_ordered = jac_down.loc[jac_order, jac_order]
sns.heatmap(jac_down_ordered, cmap='Blues', vmin=0, vmax=0.45,
            ax=ax_e, linewidths=1, linecolor='white',
            annot=True, fmt='.2f', annot_kws={'size': 14, 'fontweight': 'bold'},
            square=True,
            cbar_kws={'label': 'Jaccard index', 'shrink': 0.6})
ax_e.set_title("E) DOWN edges: Jaccard similarity\n(universal removed, >=5 species)",
               fontweight='bold', fontsize=15)
ax_e.tick_params(axis='x', labelsize=12, rotation=45)
ax_e.tick_params(axis='y', labelsize=12, rotation=0)
for lbl in ax_e.get_xticklabels():
    lbl.set_color(STRESS_COLORS.get(lbl.get_text(), 'black'))
    lbl.set_fontweight('bold')
for lbl in ax_e.get_yticklabels():
    lbl.set_color(STRESS_COLORS.get(lbl.get_text(), 'black'))
    lbl.set_fontweight('bold')

# (old phylo panel removed -- now Panel A)
ax_e.tick_params(axis='x', labelsize=12, rotation=45)
ax_e.tick_params(axis='y', labelsize=10)

fig.suptitle("Supplementary Figure S4: Phylogenetic conservation and stress similarity of co-occurrence edges",
             fontsize=20, fontweight='bold', y=1.01)

fig.savefig(os.path.join(OUT, "supplementary_figure.png"))
fig.savefig(os.path.join(OUT, "supplementary_figure.pdf"))
plt.close(fig)

# Copy
import shutil
GDRIVE_OUT = "/Users/vjx443/Library/CloudStorage/GoogleDrive-mutwil@plant.tools/My Drive/Projects/2026_KingdomStress/CC-Fig 4 data/outputs"
shutil.copy2(os.path.join(OUT, "supplementary_figure.png"), os.path.join(GDRIVE_OUT, "supplementary_figure.png"))
shutil.copy2(os.path.join(OUT, "supplementary_figure.pdf"), os.path.join(GDRIVE_OUT, "supplementary_figure.pdf"))
print("Saved: supplementary_figure.png/pdf")
