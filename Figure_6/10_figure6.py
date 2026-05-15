#!/usr/bin/env python3
"""
Figure 5 v22: All 8 models in panel B. UP/DOWN shown as faded/solid bars
with triangle markers (no error bars). Panels A-H.

Layout:
  Row 1-2 left: A - Model sketches (spanning 2 rows)
  Row 1 right:  B - Prediction performance bars (8 models, 5 stresses)
  Row 2 right:  C - DNA vs RNA scatter, D - Fusion gain, E - Window scan
  Row 3:        F - Cis-element heatmap, G - Cohen's d, H - GBM importance
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
import matplotlib.gridspec as gridspec
from matplotlib import rcParams
from scipy import stats as sp_stats

rcParams['font.family'] = 'Arial'
rcParams['font.size'] = 7
rcParams['axes.linewidth'] = 0.5
rcParams['xtick.major.width'] = 0.5
rcParams['ytick.major.width'] = 0.5

# ---- Nature vibrant colors ----
C_CNN = '#BBBBBB'
C_CAD2 = '#009988'
C_RNA = '#0077BB'
C_FUSED = '#000000'
C_DINUC = '#33BBEE'
C_MOTIF = '#EE7733'
C_STRUCT = '#EE3377'
C_COMBINED = '#CC3311'

STRESSES = ['Heat', 'Cold', 'Drought', 'Salt', 'Pathogen']
STRESS_COLORS = {'Heat': '#CC3311', 'Cold': '#0077BB', 'Drought': '#EE7733',
                 'Salt': '#009988', 'Pathogen': '#EE3377'}

FUSION_DIR = '/Users/vjx443/Library/CloudStorage/GoogleDrive-mutwil@plant.tools/My Drive/Projects/2026_KingdomStress/CC-Fig 5 data_CNN/fusion'

# ---- Load data ----
fusion = pd.read_csv('/Users/vjx443/Downloads/fusion/fusion_results_final.csv')
fusion = fusion[~fusion['stress'].isin(['Flooding'])]

cnn = pd.read_csv('/Users/vjx443/Downloads/cnn_results/all_results.csv')
cnn = cnn[~cnn['stress'].isin(['Flooding'])]

v3 = pd.read_csv('/Users/vjx443/Downloads/motif_model_v3_results/v3_results.csv')
v3 = v3[~v3['stress'].isin(['Flooding'])]

stats = pd.read_csv('/Users/vjx443/Library/CloudStorage/GoogleDrive-mutwil@plant.tools/My Drive/Projects/2026_KingdomStress/CC-Fig 5 data_CNN/stats_results/all_feature_stats.csv')
stats = stats[~stats['stress'].isin(['Flooding'])]

window = pd.read_csv('/Users/vjx443/Downloads/rna_window_scan/lumi_window_results.csv')
window = window[~window['stress'].isin(['Flooding'])]

enrichment = pd.read_csv('/Users/vjx443/Downloads/cnn_results/element_enrichment.csv')
enrichment = enrichment[~enrichment['stress'].isin(['Flooding'])]

ablation = pd.read_csv('/Users/vjx443/Downloads/motif_model_v3_results/ablation_results.csv')
ablation = ablation[~ablation['stress'].isin(['Flooding'])]

# ---- Figure setup ----
fig = plt.figure(figsize=(7.2, 8.5))

# Main grid: 3 rows
outer = gridspec.GridSpec(3, 1, figure=fig, height_ratios=[1.0, 0.85, 0.95],
                          hspace=0.28, top=0.97, bottom=0.04, left=0.06, right=0.98)

# Row 1: Models (left) + B (right)
row1 = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=outer[0], width_ratios=[0.22, 0.78], wspace=0.08)

# Row 2: (models continue) + C, D, E
row2 = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=outer[1], width_ratios=[0.22, 0.78], wspace=0.08)

# Row 3: F, G, H
row3 = gridspec.GridSpecFromSubplotSpec(1, 3, subplot_spec=outer[2], width_ratios=[0.25, 0.37, 0.38], wspace=0.3)


# ===================== PANEL A: MODEL SKETCHES (left column, rows 1-2) =====================
ax_models = fig.add_axes([0.01, 0.36, 0.17, 0.61])
ax_models.set_xlim(0, 10)
ax_models.set_ylim(-3, 30)
ax_models.axis('off')
ax_models.text(0.5, 29.8, 'A', fontsize=8, fontweight='bold', transform=ax_models.transData)

def draw_model_box(ax, y_center, width, height, color, text, alpha=0.85):
    x = 5 - width/2
    box = FancyBboxPatch((x, y_center - height/2), width, height,
                          boxstyle="round,pad=0.15", facecolor=color, edgecolor='#333333',
                          linewidth=0.5, alpha=alpha, zorder=2)
    ax.add_patch(box)
    ax.text(5, y_center, text, ha='center', va='center', fontsize=5,
            fontweight='normal', zorder=3, color='black')

def draw_arrow(ax, y1, y2):
    ax.annotate('', xy=(5, y2 + 0.25), xytext=(5, y1 - 0.25),
                arrowprops=dict(arrowstyle='->', color='#333333', lw=0.7))

# CNN model (top)
y = 28.0
ax_models.text(5, y + 0.8, 'CNN', ha='center', va='center', fontsize=6.5, fontweight='bold')
draw_model_box(ax_models, y - 0.5, 7, 1.0, '#DDDDDD', '600bp DNA\none-hot')
draw_arrow(ax_models, y - 1.0, y - 1.8)
draw_model_box(ax_models, y - 2.3, 7, 1.0, '#CCCCCC', 'Conv1D x3\n(64>128>64)')
draw_arrow(ax_models, y - 2.8, y - 3.6)
draw_model_box(ax_models, y - 4.1, 7, 1.0, '#BBBBBB', 'Flatten + Dense\n128 > 64')
draw_arrow(ax_models, y - 4.6, y - 5.4)
draw_model_box(ax_models, y - 5.9, 7, 1.0, '#AAAAAA', 'Binary classifier')
draw_arrow(ax_models, y - 6.4, y - 7.2)
draw_model_box(ax_models, y - 7.7, 7, 1.0, '#999999', 'auROC: 0.612', alpha=0.6)

# PlantCAD2 (middle)
y = 17.5
ax_models.text(5, y + 0.8, 'PlantCAD2', ha='center', va='center', fontsize=6.5, fontweight='bold', color=C_CAD2)
ax_models.text(5, y + 0.1, '(DNA LLM)', ha='center', va='center', fontsize=5, color=C_CAD2)
draw_model_box(ax_models, y - 1.0, 7, 1.0, '#B2E0D9', '600bp DNA\ntokenized')
draw_arrow(ax_models, y - 1.5, y - 2.3)
draw_model_box(ax_models, y - 2.8, 7, 1.0, '#80D0C4', 'Mamba/Caduceus\n65 genomes, 88M')
draw_arrow(ax_models, y - 3.3, y - 4.1)
draw_model_box(ax_models, y - 4.6, 7, 1.0, '#4DC0AE', 'LoRA r=8\n2.4M trainable')
draw_arrow(ax_models, y - 5.1, y - 5.9)
draw_model_box(ax_models, y - 6.4, 7, 1.0, '#1AB098', 'Binary classifier')
draw_arrow(ax_models, y - 6.9, y - 7.7)
draw_model_box(ax_models, y - 8.2, 7, 1.0, '#009988', 'auROC: 0.743', alpha=0.6)

# PlantRNA-FM (bottom)
y = 6.5
ax_models.text(5, y + 0.8, 'PlantRNA-FM', ha='center', va='center', fontsize=6.5, fontweight='bold', color=C_RNA)
ax_models.text(5, y + 0.1, '(RNA LLM)', ha='center', va='center', fontsize=5, color=C_RNA)
draw_model_box(ax_models, y - 1.0, 7, 1.0, '#B3D9EE', '1024bp mRNA\ntokenized')
draw_arrow(ax_models, y - 1.5, y - 2.3)
draw_model_box(ax_models, y - 2.8, 7, 1.0, '#80C0DD', 'ESM transformer\n1124 spp, 33M')
draw_arrow(ax_models, y - 3.3, y - 4.1)
draw_model_box(ax_models, y - 4.6, 7, 1.0, '#4DA6CC', 'LoRA r=16\n369K trainable')
draw_arrow(ax_models, y - 5.1, y - 5.9)
draw_model_box(ax_models, y - 6.4, 7, 1.0, '#1A8CBB', 'Binary classifier')
draw_arrow(ax_models, y - 6.9, y - 7.7)
draw_model_box(ax_models, y - 8.2, 7, 1.0, '#0077BB', 'auROC: 0.721', alpha=0.6)


# ===================== PANEL B: Prediction performance bars (8 models) =====================
ax_b = fig.add_subplot(row1[0, 1])
ax_b.set_title('B  Prediction performance (25 species)', fontsize=7, fontweight='bold', loc='left')

# 8 models: 4 deep learning + 4 interpretable
all_models = ['CNN', 'PlantCAD2', 'PlantRNA-FM', 'Fused', 'DinucFreq', 'Motif', 'Structure', 'Combined']
all_colors = [C_CNN, C_CAD2, C_RNA, C_FUSED, C_DINUC, C_MOTIF, C_STRUCT, C_COMBINED]
n_models = len(all_models)
bar_w = 0.09
group_width = n_models * bar_w

for si, stress in enumerate(STRESSES):
    sf = fusion[fusion['stress'] == stress]
    sc = cnn[cnn['stress'] == stress]
    sa = ablation[ablation['stress'] == stress]

    for mi, (model, color) in enumerate(zip(all_models, all_colors)):
        vals_up, vals_down = [], []

        if model == 'CNN':
            for d in ['UP', 'DOWN']:
                cr = sc[sc['direction'] == d]
                if len(cr) > 0:
                    if d == 'UP': vals_up.append(cr['test_auROC'].values[0])
                    else: vals_down.append(cr['test_auROC'].values[0])
        elif model == 'PlantCAD2':
            for _, r in sf.iterrows():
                if r['direction'] == 'UP': vals_up.append(r['DNA'])
                else: vals_down.append(r['DNA'])
        elif model == 'PlantRNA-FM':
            for _, r in sf.iterrows():
                if r['direction'] == 'UP': vals_up.append(r['RNA'])
                else: vals_down.append(r['RNA'])
        elif model == 'Fused':
            for _, r in sf.iterrows():
                if r['direction'] == 'UP': vals_up.append(r['Avg'])
                else: vals_down.append(r['Avg'])
        elif model in ['DinucFreq', 'Motif', 'Structure', 'Combined']:
            sm = sa[sa['model'] == model]
            for _, r in sm.iterrows():
                if r['direction'] == 'UP': vals_up.append(r['auROC'])
                else: vals_down.append(r['auROC'])

        up_val = np.mean(vals_up) if vals_up else 0
        down_val = np.mean(vals_down) if vals_down else 0

        x = si + (mi - (n_models - 1) / 2) * bar_w
        taller = max(up_val, down_val)
        shorter = min(up_val, down_val)
        # Taller bar faded behind, shorter bar solid in front
        ax_b.bar(x, taller, bar_w * 0.85, color=color, alpha=0.35, zorder=2)
        ax_b.bar(x, shorter, bar_w * 0.85, color=color, alpha=0.9, zorder=3)
        # Triangle on taller bar: ^ means UP is taller, v means DOWN is taller
        if up_val >= down_val:
            ax_b.plot(x, taller + 0.005, marker='^', markersize=2.5, color=color,
                     markeredgecolor='black', markeredgewidth=0.3, zorder=5)
        else:
            ax_b.plot(x, taller + 0.005, marker='v', markersize=2.5, color=color,
                     markeredgecolor='black', markeredgewidth=0.3, zorder=5)

    # Dotted separator between deep learning (4) and interpretable (4) models
    sep_x = si + (3.5 - (n_models - 1) / 2) * bar_w
    ax_b.axvline(sep_x, color='grey', linestyle=':', linewidth=0.5, alpha=0.6, ymin=0, ymax=0.95)

ax_b.set_ylabel('Test auROC', fontsize=6)
ax_b.set_xticks(range(len(STRESSES)))
ax_b.set_xticklabels(STRESSES, fontsize=6)
ax_b.set_ylim(0.40, 0.85)
ax_b.axhline(0.5, color='grey', linestyle='--', linewidth=0.5, alpha=0.5)
legend_patches = [mpatches.Patch(color=c, label=m) for m, c in zip(all_models, all_colors)]
ax_b.legend(handles=legend_patches, fontsize=4.5, loc='upper left', framealpha=0.8, ncol=4)
ax_b.tick_params(axis='both', labelsize=5.5)


# ===================== PANEL C: DNA vs RNA scatter =====================
row2_right = gridspec.GridSpecFromSubplotSpec(1, 3, subplot_spec=row2[0, 1], wspace=0.35)
ax_c = fig.add_subplot(row2_right[0])
ax_c.set_title('C  DNA vs RNA', fontsize=7, fontweight='bold', loc='left')

all_dna_pts, all_rna_pts = [], []
for stress in STRESSES:
    for direction in ['UP', 'DOWN']:
        tag = f"{stress}_{direction}"
        dna_file = f"{FUSION_DIR}/dna/{tag}_predictions.csv"
        rna_file = f"{FUSION_DIR}/rna/{tag}_predictions.csv"
        try:
            dna_df = pd.read_csv(dna_file)
            rna_df = pd.read_csv(rna_file)
            merged = dna_df.merge(rna_df, on='gene_id', suffixes=('_dna', '_rna'))
            n_plot = min(800, len(merged))
            sub = merged.sample(n_plot, random_state=42) if len(merged) > n_plot else merged
            ax_c.scatter(sub['prediction_dna'], sub['prediction_rna'],
                        c=STRESS_COLORS[stress], alpha=0.06, s=1.5, marker='.', rasterized=True)
            all_dna_pts.extend(sub['prediction_dna'].tolist())
            all_rna_pts.extend(sub['prediction_rna'].tolist())
        except FileNotFoundError:
            pass

r = np.corrcoef(all_dna_pts, all_rna_pts)[0, 1] if all_dna_pts else 0
ax_c.plot([0, 1], [0, 1], '--', color='grey', linewidth=0.5, alpha=0.5)
ax_c.set_xlabel('PlantCAD2', fontsize=5.5)
ax_c.set_ylabel('PlantRNA-FM', fontsize=5.5)
ax_c.set_xlim(0.15, 0.95)
ax_c.set_ylim(0.15, 0.95)
ax_c.set_aspect('equal')
ax_c.text(0.05, 0.95, f'r={r:.2f}', transform=ax_c.transAxes, fontsize=5, va='top')
ax_c.tick_params(axis='both', labelsize=5)


# ===================== PANEL D: Fusion gain =====================
ax_d = fig.add_subplot(row2_right[1])
ax_d.set_title('D  Fusion gain', fontsize=7, fontweight='bold', loc='left')

for _, row in fusion.iterrows():
    stress = row['stress']
    direction = row['direction']
    marker = '^' if direction == 'UP' else 'v'
    ax_d.scatter(row['corr'], row['gain'], c=STRESS_COLORS[stress],
                marker=marker, s=25, edgecolors='black', linewidth=0.3, zorder=3)

ax_d.set_xlabel('DNA-RNA corr.', fontsize=5.5)
ax_d.set_ylabel('auROC gain', fontsize=5.5)
ax_d.tick_params(axis='both', labelsize=5)

corrs = fusion['corr'].values
gains = fusion['gain'].values
z = np.polyfit(corrs, gains, 1)
p = np.poly1d(z)
x_line = np.linspace(corrs.min(), corrs.max(), 50)
ax_d.plot(x_line, p(x_line), '--', color='grey', linewidth=0.7)
r_val, p_val = sp_stats.pearsonr(corrs, gains)
ax_d.text(0.05, 0.05, f'r={r_val:.2f}\nP={p_val:.2g}', transform=ax_d.transAxes, fontsize=5, va='bottom')

handles = []
handles.append(ax_d.scatter([], [], marker='^', c='grey', s=15, label='UP'))
handles.append(ax_d.scatter([], [], marker='v', c='grey', s=15, label='DOWN'))
for s, c in STRESS_COLORS.items():
    handles.append(ax_d.scatter([], [], marker='o', c=c, s=12, label=s))
ax_d.legend(fontsize=3.5, loc='upper right', framealpha=0.7, ncol=2, handletextpad=0.2)


# ===================== PANEL E: Window scan =====================
ax_e = fig.add_subplot(row2_right[2])
ax_e.set_title('E  Window scan', fontsize=7, fontweight='bold', loc='left')

window_labels = ['0-1kb', '+0.5-1.5kb', '+1-2kb', 'TTS-1kb']
window_display = ["0 to\n+1kb", '+0.5 to\n+1.5kb', '+1 to\n+2kb', "TTS\n-1kb"]
window_colors = ['#EE7733', '#33BBEE', '#0077BB', '#009988']

for si, stress in enumerate(STRESSES):
    sw = window[window['stress'] == stress]
    for wi, wname in enumerate(window_labels):
        wdata = sw[sw['window'] == wname]
        if len(wdata) > 0:
            up_val = wdata[wdata['direction'] == 'UP']['auROC'].values
            down_val = wdata[wdata['direction'] == 'DOWN']['auROC'].values
            mean_val = np.mean(np.concatenate([up_val, down_val]))
            x = si + wi * 0.17 - 0.26
            ax_e.bar(x, mean_val, 0.13, color=window_colors[wi], alpha=0.85, zorder=2)
            if len(up_val) > 0 and len(down_val) > 0:
                err_lo = mean_val - min(up_val[0], down_val[0])
                err_hi = max(up_val[0], down_val[0]) - mean_val
                ax_e.errorbar(x, mean_val, yerr=[[err_lo], [err_hi]],
                             fmt='none', ecolor='black', elinewidth=0.4, capsize=1, capthick=0.4, zorder=3)

ax_e.set_ylabel('Test auROC', fontsize=5.5)
ax_e.set_xticks(range(len(STRESSES)))
ax_e.set_xticklabels(STRESSES, fontsize=5, rotation=0)
ax_e.set_ylim(0.45, 0.85)
ax_e.axhline(0.5, color='grey', linestyle='--', linewidth=0.5, alpha=0.5)
ax_e.tick_params(axis='both', labelsize=5)

w_patches = [mpatches.Patch(color=c, label=l) for c, l in zip(window_colors, window_display)]
ax_e.legend(handles=w_patches, fontsize=3.5, loc='upper right', framealpha=0.7, ncol=1,
           handlelength=1, handleheight=0.8)


# ===================== PANEL F: Cis-element enrichment heatmap (3 models) =====================
ax_f = fig.add_subplot(row3[0])
ax_f.set_title('F  Known cis-element enrichment', fontsize=7, fontweight='bold', loc='left')

# Known motifs to search for in discovered motif sequences
KNOWN_ELEMENTS = {
    'ABRE': 'ACGTG', 'DRE/CRT': 'CCGAC', 'W-box': 'TTGAC', 'GCC-box': 'GCCGCC',
    'G-box': 'CACGTG', 'TATA': 'TATAAAT', 'CAAT': 'CCAAT', 'MBS': 'CAACTG',
    'as-1': 'TGACG', 'MYC': 'CATGTG', 'LTR': 'CCGAAA', 'ARE': 'AAACCA',
    'AU-rich': 'ATTTA', 'PUF': 'TGTA', 'DST': 'ATAGAT', 'Poly-A': 'AATAAA',
}

def reverse_complement(seq):
    comp = {'A':'T','T':'A','C':'G','G':'C','N':'N'}
    return ''.join(comp.get(c,'N') for c in reversed(seq))

def count_element_in_motifs(motif_dir, stresses, element_seq, n_seqs=200):
    """Count how many top motif sequences contain a known element."""
    total = 0
    for stress in stresses:
        for direction in ['UP', 'DOWN']:
            mfile = motif_dir / f"{stress}_{direction}_motifs.txt"
            if not mfile.exists():
                continue
            try:
                df = pd.read_csv(mfile, sep='\t')
                seqs = df['sequence'].head(n_seqs).values
                for seq in seqs:
                    seq = seq.upper().replace('U', 'T')
                    rc = reverse_complement(element_seq)
                    if element_seq in seq or rc in seq:
                        total += 1
            except:
                pass
    return total

from pathlib import Path
CNN_MOTIF_DIR = Path('/Users/vjx443/Downloads/cnn_results/motifs')
CAD2_MOTIF_DIR = Path('/Users/vjx443/Library/CloudStorage/GoogleDrive-mutwil@plant.tools/My Drive/Projects/2026_KingdomStress/CC-Fig 5 data_CNN/plantcad2_interpretability/motifs')
RNA_MOTIF_DIR2 = Path('/Users/vjx443/Downloads/rna_interpretability/motifs')

elements_list = list(KNOWN_ELEMENTS.keys())
model_dirs = [('CNN', CNN_MOTIF_DIR), ('CAD2', CAD2_MOTIF_DIR), ('RNA-FM', RNA_MOTIF_DIR2)]

heatmap_data = np.zeros((len(elements_list), len(model_dirs)))
for mi, (mname, mdir) in enumerate(model_dirs):
    for ei, elem in enumerate(elements_list):
        heatmap_data[ei, mi] = count_element_in_motifs(mdir, STRESSES, KNOWN_ELEMENTS[elem])

im = ax_f.imshow(heatmap_data, aspect='auto', cmap='YlOrRd', interpolation='nearest')
ax_f.set_xticks(range(len(model_dirs)))
ax_f.set_xticklabels([m[0] for m in model_dirs], fontsize=5.5)
ax_f.set_yticks(range(len(elements_list)))
ax_f.set_yticklabels(elements_list, fontsize=4.5)
ax_f.tick_params(axis='both', labelsize=5)

for i in range(len(elements_list)):
    for j in range(len(model_dirs)):
        val = int(heatmap_data[i, j])
        if val > 0:
            color = 'white' if val > heatmap_data.max() * 0.6 else 'black'
            ax_f.text(j, i, str(val), ha='center', va='center', fontsize=4.5, color=color)


# ===================== PANEL G: Cohen's d effect sizes (v19 style) =====================
ax_g = fig.add_subplot(row3[1])
ax_g.set_title("G  Effect sizes\n(all p<0.001)", fontsize=7, fontweight='bold', loc='left')

# All features, ordered by absolute Cohen's d, with category colors
# Feature display names and category assignment
feat_info = {
    'mrna_CG_dinuc': ('mRNA CG', 'comp'), 'prom_CG_dinuc': ('Prom CG', 'comp'),
    'mrna_GC_content': ('mRNA GC%', 'comp'), 'prom_GC_content': ('Prom GC%', 'comp'),
    'mrna_AA_dinuc': ('mRNA AA', 'comp'), 'prom_TG_dinuc': ('Prom TG', 'comp'),
    'struct_frac_paired': ('Frac paired', 'struct'), 'struct_MFE_per_nt': ('RNA MFE', 'struct'),
    'struct_g_quad': ('G-quad', 'struct'), 'struct_au_rich_runs': ('AU-rich', 'struct'),
    'prom_ABRE': ('Prom ABRE', 'dna_motif'), 'prom_TATA_box': ('Prom TATA', 'dna_motif'),
    'prom_GCC_box': ('Prom GCC', 'dna_motif'), 'prom_W_box': ('Prom W-box', 'dna_motif'),
    'prom_DRE_CRT': ('Prom DRE', 'dna_motif'),
    'mrna_DRE_CRT': ('mRNA DRE', 'rna_motif'), 'mrna_DST_element': ('mRNA DST', 'rna_motif'),
    'mrna_PUF_binding': ('mRNA PUF', 'rna_motif'), 'mrna_ARE_aurich': ('mRNA ARE', 'rna_motif'),
}

cat_colors = {'comp': '#33BBEE', 'struct': '#EE3377', 'dna_motif': '#009988', 'rna_motif': '#0077BB'}

# Compute mean Cohen's d across stresses for each feature
feat_d = {}
for feat, (disp, cat) in feat_info.items():
    sf = stats[stats['feature'] == feat]
    if len(sf) > 0:
        feat_d[feat] = sf['cohens_d'].mean()

# Sort by absolute d
sorted_feats = sorted(feat_d.keys(), key=lambda f: abs(feat_d[f]))
# Plot bottom-to-top (largest at top)
for i, feat in enumerate(sorted_feats):
    disp, cat = feat_info[feat]
    ax_g.barh(i, feat_d[feat], color=cat_colors[cat], alpha=0.85, height=0.7)

ax_g.set_yticks(range(len(sorted_feats)))
ax_g.set_yticklabels([feat_info[f][0] for f in sorted_feats], fontsize=4.5)
for i, feat in enumerate(sorted_feats):
    ax_g.get_yticklabels()[i].set_color(cat_colors[feat_info[feat][1]])
ax_g.set_xlabel("Cohen's d", fontsize=5.5)
ax_g.axvline(0, color='black', linewidth=0.5)
ax_g.tick_params(axis='both', labelsize=5)

cat_patches = [
    mpatches.Patch(color='#33BBEE', label='Comp.'),
    mpatches.Patch(color='#EE3377', label='Struct.'),
    mpatches.Patch(color='#009988', label='DNA motif'),
    mpatches.Patch(color='#0077BB', label='RNA motif'),
]
ax_g.legend(handles=cat_patches, fontsize=3.5, loc='lower right', framealpha=0.7)


# ===================== PANEL H: GBM feature importance (v19 style) =====================
ax_h = fig.add_subplot(row3[2])
ax_h.set_title('H  Predictive\nimportance', fontsize=7, fontweight='bold', loc='left')

all_imp = []
for stress in STRESSES:
    for direction in ['UP', 'DOWN']:
        tag = f"{stress}_{direction}"
        imp_file = f"/Users/vjx443/Downloads/motif_model_v3_results/{tag}_feature_importance.csv"
        try:
            imp = pd.read_csv(imp_file, index_col=0, header=None)
            imp.columns = ['importance']
            all_imp.append(imp)
        except FileNotFoundError:
            pass

if all_imp:
    combined = pd.concat(all_imp, axis=1).mean(axis=1).sort_values(ascending=False)
    top20 = combined.head(20)

    def get_feat_color(name):
        if 'struct_' in name: return '#EE3377'
        if 'dn_' in name: return '#33BBEE'
        if 'gc' in name.lower(): return '#33BBEE'
        # Check if it's a known DNA promoter motif or RNA motif
        if 'prom_' in name and any(m in name for m in ['ABRE','DRE','W_box','GCC','TATA','G_box','MBS','as1','MYC','LTR','CAAT']):
            return '#009988'
        if 'mrna_' in name and any(m in name for m in ['ARE','PUF','DST','poly_A','DRE','W_box','TATA','ABRE','GCC','CAAT','MBS','as1','MYC','G_box','LTR']):
            return '#0077BB'
        if 'prom_' in name: return '#009988'
        if 'mrna_' in name: return '#0077BB'
        return 'grey'

    # Display names matching v19 style
    display_names = []
    for name in top20.index:
        dn = name
        dn = dn.replace('prom_dn_', 'Prom ').replace('mrna_dn_', 'mRNA ')
        dn = dn.replace('prom_', 'Prom ').replace('mrna_', 'mRNA ')
        dn = dn.replace('struct_', '')
        dn = dn.replace('_', ' ').replace('MFE per nt', 'RNA MFE')
        dn = dn.replace('frac paired', 'Frac paired')
        dn = dn.replace('g quad', 'G-quad').replace('au rich runs', 'AU-rich')
        dn = dn.replace('GC content', 'GC%')
        display_names.append(dn)

    colors = [get_feat_color(f) for f in top20.index]
    y_pos = range(len(top20))
    bars = ax_h.barh(y_pos, top20.values, color=colors, alpha=0.85, height=0.7)
    ax_h.set_yticks(y_pos)
    ax_h.set_yticklabels(display_names, fontsize=4, fontweight='bold')
    for i, feat in enumerate(top20.index):
        ax_h.get_yticklabels()[i].set_color(get_feat_color(feat))
    ax_h.set_xlabel('GBM importance', fontsize=5.5)
    ax_h.tick_params(axis='both', labelsize=5)
    ax_h.invert_yaxis()

    # Annotate values
    for i, (val, name) in enumerate(zip(top20.values, top20.index)):
        ax_h.text(val + 0.001, i, f'{val:.3f}', va='center', fontsize=4, color='#444444')

    cat_patches = [
        mpatches.Patch(color='#33BBEE', label='Dinucleotide'),
        mpatches.Patch(color='#EE3377', label='Structure'),
        mpatches.Patch(color='#009988', label='DNA motif'),
        mpatches.Patch(color='#0077BB', label='RNA motif'),
    ]
    ax_h.legend(handles=cat_patches, fontsize=3.5, loc='lower right', framealpha=0.7)


# ---- Save ----
out = '/Users/vjx443/Downloads/fusion/Figure5_v28'
fig.savefig(f'{out}.png', dpi=300, bbox_inches='tight')
fig.savefig(f'{out}.pdf', bbox_inches='tight')
print(f"Saved {out}.png and .pdf")
plt.close()
