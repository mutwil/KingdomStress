"""
Heatmap: KG entries per hormone (rows) x stress (columns).
Shows literature bias vs actual transcriptomic responsiveness.
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

plt.rcParams.update({
    'font.family': 'Arial', 'font.size': 12,
    'axes.titlesize': 16, 'axes.labelsize': 14,
    'figure.dpi': 300, 'savefig.dpi': 300,
    'savefig.bbox': 'tight', 'figure.facecolor': 'white',
})

# Load KG
hdf = pd.read_csv('/tmp/kg/hormone_kg_entries.csv', low_memory=False)

HORMONES = ['Abscisic acid', 'Salicylic acid', 'Brassinosteroid', 'Gibberellin',
            'Jasmonic acid', 'Ethylene', 'Auxin', 'Cytokinin']
H_SHORT = ['ABA', 'SA', 'BR', 'GA', 'JA', 'Ethylene', 'Auxin', 'Cytokinin']

HORMONE_TERMS = {
    'Abscisic acid': ['abscisic acid'],
    'Salicylic acid': ['salicyl'],
    'Brassinosteroid': ['brassinosteroid', 'brassinolide'],
    'Gibberellin': ['gibberellin'],
    'Jasmonic acid': ['jasmonic', 'jasmonate'],
    'Ethylene': ['ethylene'],
    'Auxin': ['auxin'],
    'Cytokinin': ['cytokinin'],
}

STRESSES = ['Heat', 'Cold', 'Drought', 'Salt', 'Pathogen', 'Heavy metal']
STRESS_TERMS = {
    'Heat': ['heat', 'thermotolerance', 'high temperature', 'heat shock', 'HSP', 'HSF'],
    'Cold': ['cold', 'freezing', 'chilling', 'low temperature', 'frost'],
    'Drought': ['drought', 'water deficit', 'desiccation', 'dehydration', 'osmotic stress'],
    'Salt': ['salt', 'salinity', 'NaCl', 'sodium chloride', 'ionic stress'],
    'Pathogen': ['pathogen', 'disease', 'resistance', 'immunity', 'defense', 'infection'],
    'Heavy metal': ['heavy metal', 'cadmium', 'arsenic', 'zinc toxicity', 'copper toxicity', 'lead'],
}

# Count KG entries per hormone x stress
print("Counting KG entries per hormone x stress...")
kg_counts = pd.DataFrame(0, index=HORMONES, columns=STRESSES)

for hormone, h_terms in HORMONE_TERMS.items():
    h_mask = pd.Series(False, index=hdf.index)
    for t in h_terms:
        h_mask |= (hdf['source resolved'].str.contains(t, case=False, na=False) |
                    hdf['target resolved'].str.contains(t, case=False, na=False))
    h_entries = hdf[h_mask]

    for stress, s_terms in STRESS_TERMS.items():
        s_mask = pd.Series(False, index=h_entries.index)
        for t in s_terms:
            s_mask |= (h_entries['source resolved'].str.contains(t, case=False, na=False) |
                       h_entries['target resolved'].str.contains(t, case=False, na=False))
        kg_counts.loc[hormone, stress] = s_mask.sum()

print(kg_counts)

# Also load PEA-based responsiveness (UP+DOWN) for comparison
pea = pd.read_csv('/tmp/Mercator_pathway_analysis_summary_level1.csv')
pea['PARENT_BINCODE'] = pea['PARENT_BINCODE'].astype(str)
pea['UP'] = pea['US'] + pea['UDS']
pea['DOWN'] = pea['DS'] + pea['UDS']

proc = pd.read_csv('/tmp/mercator_process_list.csv', index_col=0)
proc['Bincode'] = proc['Bincode'].astype(str)
name2bc = dict(zip(proc['Bincode name'], proc['Bincode']))

pea_resp = pd.DataFrame(0.0, index=HORMONES, columns=STRESSES)
pea_bias = pd.DataFrame(0.0, index=HORMONES, columns=STRESSES)

for hormone in HORMONES:
    hfull = f"Phytohormone action.{hormone.lower()}"
    bc = name2bc.get(hfull)
    if not bc:
        continue
    hdf_pea = pea[pea['PARENT_BINCODE'] == bc]
    for stress in STRESSES:
        sdf = hdf_pea[hdf_pea['stress'] == stress]
        if sdf.empty:
            continue
        pea_resp.loc[hormone, stress] = sdf['UP'].mean() + sdf['DOWN'].mean()
        pea_bias.loc[hormone, stress] = sdf['UP'].mean() - sdf['DOWN'].mean()

# ── Figure: 3 panels side by side ────────────────────────────────────────
fig = plt.figure(figsize=(22, 8))
gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 1], wspace=0.3)

# Panel A: KG literature counts (log scale)
ax1 = fig.add_subplot(gs[0])
kg_log = np.log10(kg_counts.replace(0, np.nan))
sns.heatmap(kg_log, cmap='Greens', ax=ax1, linewidths=0.5, linecolor='white',
            annot=kg_counts.values.astype(int), fmt='d', annot_kws={'size': 11},
            yticklabels=H_SHORT, mask=kg_counts == 0,
            cbar_kws={'label': 'KG entries (log10)', 'shrink': 0.6})
ax1.set_title("A) Knowledge graph\n(literature)", fontweight='bold')
ax1.set_xlabel("")
ax1.tick_params(axis='x', rotation=45, labelsize=12)
ax1.tick_params(axis='y', labelsize=13)

# Panel B: Transcriptomic responsiveness (UP+DOWN)
ax2 = fig.add_subplot(gs[1])
sns.heatmap(pea_resp, cmap='YlOrRd', ax=ax2, linewidths=0.5, linecolor='white',
            annot=True, fmt='.2f', annot_kws={'size': 11},
            yticklabels=H_SHORT,
            cbar_kws={'label': 'Mean UP + DOWN score', 'shrink': 0.6})
ax2.set_title("B) Transcriptomic\nresponsiveness", fontweight='bold')
ax2.set_xlabel("")
ax2.tick_params(axis='x', rotation=45, labelsize=12)
ax2.tick_params(axis='y', labelsize=13)

# Panel C: Direction bias
ax3 = fig.add_subplot(gs[2])
max_b = max(abs(pea_bias.min().min()), abs(pea_bias.max().max()))
sns.heatmap(pea_bias, cmap='RdBu_r', center=0, vmin=-max_b, vmax=max_b,
            ax=ax3, linewidths=0.5, linecolor='white',
            annot=True, fmt='+.2f', annot_kws={'size': 11},
            yticklabels=H_SHORT,
            cbar_kws={'label': 'UP - DOWN bias', 'shrink': 0.6})
ax3.set_title("C) Direction bias\n(stress network)", fontweight='bold')
ax3.set_xlabel("")
ax3.tick_params(axis='x', rotation=45, labelsize=12)
ax3.tick_params(axis='y', labelsize=13)

fig.suptitle("Hormone-stress associations: literature (KG) vs transcriptomics (36 species)",
             fontsize=17, fontweight='bold', y=1.03)

fig.savefig(os.path.join(OUT, "kg_vs_transcriptome_hormones.png"))
fig.savefig(os.path.join(OUT, "kg_vs_transcriptome_hormones.pdf"))
plt.close(fig)

# Copy
import shutil
GDRIVE_OUT = "/Users/vjx443/Library/CloudStorage/GoogleDrive-mutwil@plant.tools/My Drive/Projects/2026_KingdomStress/CC-Fig 4 data/outputs"
shutil.copy2(os.path.join(OUT, "kg_vs_transcriptome_hormones.png"),
             os.path.join(GDRIVE_OUT, "kg_vs_transcriptome_hormones.png"))
shutil.copy2(os.path.join(OUT, "kg_vs_transcriptome_hormones.pdf"),
             os.path.join(GDRIVE_OUT, "kg_vs_transcriptome_hormones.pdf"))

print("\nSaved: kg_vs_transcriptome_hormones.png/pdf")
