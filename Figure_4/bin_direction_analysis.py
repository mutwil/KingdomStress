"""
Which MapMan bins are most consistently up- or downregulated across stresses?
Uses PEA summary data (fraction of experiments where each bin is enriched).
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
os.makedirs(OUT, exist_ok=True)

plt.rcParams.update({
    'font.family': 'Arial', 'font.size': 10, 'axes.titlesize': 12,
    'axes.labelsize': 10, 'figure.dpi': 300, 'savefig.dpi': 300,
    'savefig.bbox': 'tight', 'figure.facecolor': 'white',
})

STRESSES = ['Heat', 'Cold', 'Drought', 'Salt', 'Pathogen', 'Heavy metal',
            'High light', 'Flooding', 'Nitrogen', 'Herbivory']
STRESS_COLORS = {
    'Heat': '#E63946', 'Cold': '#457B9D', 'Drought': '#E9C46A',
    'Salt': '#2A9D8F', 'Pathogen': '#8338EC', 'Heavy metal': '#6D6875',
    'High light': '#F4A261', 'Flooding': '#264653', 'Nitrogen': '#A8DADC',
    'Herbivory': '#D62828'
}

def shorten(s, n=35):
    return s if len(s) <= n else s[:n-2] + '..'

# ── Load data ─────────────────────────────────────────────────────────────
print("Loading PEA summaries and bin names...")
proc = pd.read_csv("/tmp/mercator_process_list.csv", index_col=0)
proc['Bincode'] = proc['Bincode'].astype(str)
bincode_to_name = dict(zip(proc['Bincode'], proc['Bincode name']))

results = {}
for level in [0, 1]:
    df = pd.read_csv(f"/tmp/Mercator_pathway_analysis_summary_level{level}.csv")
    df['PARENT_BINCODE'] = df['PARENT_BINCODE'].astype(str)
    df['bin_name'] = df['PARENT_BINCODE'].map(bincode_to_name)
    df['UP_score'] = df['US'] + df['UDS']
    df['DOWN_score'] = df['DS'] + df['UDS']
    df['direction_bias'] = df['UP_score'] - df['DOWN_score']  # positive = UP-biased

    # Normalize stress names
    df['stress'] = df['stress'].str.replace('High Light', 'High light')
    results[level] = df
    print(f"  Level{level}: {len(df)} rows, {df['bin_name'].nunique()} bins, "
          f"{df['species'].nunique()} species, stresses: {sorted(df['stress'].unique())}")


# ═══════════════════════════════════════════════════════════════════════════
# For each bin: compute mean UP and DOWN scores across all experiments
# ═══════════════════════════════════════════════════════════════════════════

def summarize_bins(df, level_label):
    """Compute per-bin summary across all experiments."""
    # Mean across all species/stress/organ combinations
    bin_summary = df.groupby('bin_name').agg(
        mean_UP=('UP_score', 'mean'),
        mean_DOWN=('DOWN_score', 'mean'),
        mean_NS=('NS', 'mean'),
        n_experiments=('UP_score', 'count'),
    ).reset_index()

    bin_summary['direction_bias'] = bin_summary['mean_UP'] - bin_summary['mean_DOWN']
    bin_summary['total_responsive'] = bin_summary['mean_UP'] + bin_summary['mean_DOWN']
    bin_summary = bin_summary.sort_values('direction_bias', ascending=False)

    print(f"\n{'=' * 70}")
    print(f"{level_label}: BINS RANKED BY DIRECTION BIAS (UP - DOWN)")
    print(f"{'=' * 70}")

    # Top upregulated
    print(f"\nTop 15 UP-biased bins:")
    for _, r in bin_summary.head(15).iterrows():
        print(f"  {shorten(r['bin_name'], 50):<52} UP={r['mean_UP']:.3f}  DOWN={r['mean_DOWN']:.3f}  bias={r['direction_bias']:+.3f}")

    # Top downregulated
    print(f"\nTop 15 DOWN-biased bins:")
    for _, r in bin_summary.tail(15).iloc[::-1].iterrows():
        print(f"  {shorten(r['bin_name'], 50):<52} UP={r['mean_UP']:.3f}  DOWN={r['mean_DOWN']:.3f}  bias={r['direction_bias']:+.3f}")

    return bin_summary


summary_L0 = summarize_bins(results[0], "Level 0")
summary_L1 = summarize_bins(results[1], "Level 1")


# ═══════════════════════════════════════════════════════════════════════════
# Per-stress breakdown: which bins are UP/DOWN in each stress?
# ═══════════════════════════════════════════════════════════════════════════

def per_stress_summary(df, level_label):
    """Mean UP and DOWN per bin per stress."""
    stress_bin = df.groupby(['stress', 'bin_name']).agg(
        mean_UP=('UP_score', 'mean'),
        mean_DOWN=('DOWN_score', 'mean'),
    ).reset_index()
    stress_bin['bias'] = stress_bin['mean_UP'] - stress_bin['mean_DOWN']
    return stress_bin


stress_L0 = per_stress_summary(results[0], "Level 0")
stress_L1 = per_stress_summary(results[1], "Level 1")


# ═══════════════════════════════════════════════════════════════════════════
# PLOTS
# ═══════════════════════════════════════════════════════════════════════════

# ── Plot 1: Diverging bar chart of direction bias (Level 0) ──────────────
for level, summary in [('Level0', summary_L0), ('Level1', summary_L1)]:
    data = summary.dropna(subset=['bin_name']).sort_values('direction_bias')

    if level == 'Level1':
        # Too many bins, show top/bottom 25
        top = data.tail(25)
        bottom = data.head(25)
        data = pd.concat([bottom, top])

    fig, ax = plt.subplots(figsize=(12, max(8, len(data) * 0.3)))
    colors = ['#E63946' if b > 0 else '#457B9D' for b in data['direction_bias']]
    ax.barh(range(len(data)), data['direction_bias'], color=colors, alpha=0.85)
    ax.set_yticks(range(len(data)))
    ax.set_yticklabels([shorten(n, 45) for n in data['bin_name']], fontsize=7)
    ax.axvline(0, color='black', lw=0.5)
    ax.set_xlabel("Direction bias (mean UP - mean DOWN score)")
    ax.set_title(f"MapMan bin direction bias across all stresses ({level})\n"
                 f"Red = UP-biased, Blue = DOWN-biased")
    plt.tight_layout()
    fig.savefig(os.path.join(OUT, f"bin_direction_bias_{level}.png"))
    fig.savefig(os.path.join(OUT, f"bin_direction_bias_{level}.pdf"))
    plt.close(fig)


# ── Plot 2: Heatmap of UP score per bin x stress (Level 0) ──────────────
for level, sdf in [('Level0', stress_L0), ('Level1', stress_L1)]:
    # Pivot: bin x stress
    for score, label, cmap in [('mean_UP', 'UP', 'Reds'), ('mean_DOWN', 'DOWN', 'Blues'), ('bias', 'Bias', 'RdBu_r')]:
        pivot = sdf.pivot_table(index='bin_name', columns='stress', values=score, aggfunc='first')
        pivot = pivot.dropna(how='all')

        if level == 'Level1':
            # Show top 30 by absolute mean
            row_means = pivot.abs().mean(axis=1).nlargest(30)
            pivot = pivot.loc[row_means.index]

        # Sort by mean value
        pivot['_mean'] = pivot.mean(axis=1)
        pivot = pivot.sort_values('_mean', ascending=True)
        pivot = pivot.drop('_mean', axis=1)

        # Reorder columns
        stress_order = [s for s in STRESSES if s in pivot.columns]
        pivot = pivot[stress_order]

        fig, ax = plt.subplots(figsize=(12, max(8, len(pivot) * 0.35)))
        vmin = None
        vmax = None
        center = None
        if score == 'bias':
            max_abs = max(abs(pivot.min().min()), abs(pivot.max().max()))
            vmin, vmax, center = -max_abs, max_abs, 0

        sns.heatmap(pivot, cmap=cmap, ax=ax, linewidths=0.3, linecolor='white',
                    vmin=vmin, vmax=vmax, center=center,
                    yticklabels=[shorten(n, 45) for n in pivot.index],
                    cbar_kws={'label': f'Mean {label} score', 'shrink': 0.5})
        ax.set_title(f"MapMan bin {label} scores per stress ({level})")
        ax.set_xlabel("Stress")
        plt.tight_layout()
        fig.savefig(os.path.join(OUT, f"bin_{label.lower()}_heatmap_{level}.png"))
        fig.savefig(os.path.join(OUT, f"bin_{label.lower()}_heatmap_{level}.pdf"))
        plt.close(fig)


# ── Plot 3: Scatter UP vs DOWN for each bin (Level 0), colored by stress ─
for level, sdf in [('Level0', stress_L0)]:
    fig, ax = plt.subplots(figsize=(10, 10))

    for stress in STRESSES:
        subset = sdf[sdf['stress'] == stress]
        if subset.empty:
            continue
        ax.scatter(subset['mean_DOWN'], subset['mean_UP'],
                   alpha=0.5, s=20, c=STRESS_COLORS.get(stress, 'grey'),
                   label=stress, edgecolors='none')

    max_val = max(sdf['mean_UP'].max(), sdf['mean_DOWN'].max()) * 1.05
    ax.plot([0, max_val], [0, max_val], 'k--', alpha=0.3, lw=1)
    ax.set_xlabel("Mean DOWN score")
    ax.set_ylabel("Mean UP score")
    ax.set_title(f"UP vs DOWN enrichment per bin and stress ({level})")
    ax.legend(fontsize=8)
    plt.tight_layout()
    fig.savefig(os.path.join(OUT, f"bin_up_vs_down_{level}.png"))
    fig.savefig(os.path.join(OUT, f"bin_up_vs_down_{level}.pdf"))
    plt.close(fig)


# ── Plot 4: Consistently UP or DOWN across ALL stresses ──────────────────
for level, sdf in [('Level0', stress_L0), ('Level1', stress_L1)]:
    bias_pivot = sdf.pivot_table(index='bin_name', columns='stress', values='bias', aggfunc='first')
    bias_pivot = bias_pivot.dropna(how='all')

    # Fill missing with 0
    bias_pivot = bias_pivot.fillna(0)

    # Consistently UP: positive bias in all stresses
    all_up = bias_pivot[(bias_pivot > 0).all(axis=1)]
    all_down = bias_pivot[(bias_pivot < 0).all(axis=1)]

    mean_bias = bias_pivot.mean(axis=1)
    all_up_sorted = mean_bias.loc[all_up.index].sort_values(ascending=False)
    all_down_sorted = mean_bias.loc[all_down.index].sort_values(ascending=True)

    print(f"\n{'=' * 70}")
    print(f"{level}: BINS CONSISTENTLY UP-BIASED IN ALL STRESSES")
    print(f"{'=' * 70}")
    for name in all_up_sorted.head(20).index:
        print(f"  {shorten(name, 55):<57} mean bias={all_up_sorted[name]:+.3f}")

    print(f"\n{level}: BINS CONSISTENTLY DOWN-BIASED IN ALL STRESSES")
    for name in all_down_sorted.head(20).index:
        print(f"  {shorten(name, 55):<57} mean bias={all_down_sorted[name]:+.3f}")

    # Save
    consistent = pd.DataFrame({
        'bin': list(all_up_sorted.index) + list(all_down_sorted.index),
        'mean_bias': list(all_up_sorted.values) + list(all_down_sorted.values),
        'direction': ['UP'] * len(all_up_sorted) + ['DOWN'] * len(all_down_sorted),
    })
    consistent.to_csv(os.path.join(OUT, f"consistent_direction_{level}.csv"), index=False)


# ── Save all summaries ───────────────────────────────────────────────────
summary_L0.to_csv(os.path.join(OUT, "bin_summary_L0.csv"), index=False)
summary_L1.to_csv(os.path.join(OUT, "bin_summary_L1.csv"), index=False)
stress_L0.to_csv(os.path.join(OUT, "bin_per_stress_L0.csv"), index=False)
stress_L1.to_csv(os.path.join(OUT, "bin_per_stress_L1.csv"), index=False)

# ── Copy to Google Drive ─────────────────────────────────────────────────
import shutil
GDRIVE_OUT = "/Users/vjx443/Library/CloudStorage/GoogleDrive-mutwil@plant.tools/My Drive/Projects/2026_KingdomStress/CC-Fig 4 data/outputs"
for f in os.listdir(OUT):
    if f.startswith("bin_") or f.startswith("consistent_"):
        shutil.copy2(os.path.join(OUT, f), os.path.join(GDRIVE_OUT, f))

print("\n" + "=" * 60)
print("BIN DIRECTION ANALYSIS COMPLETE")
print("=" * 60)
for f in sorted(os.listdir(OUT)):
    if f.startswith("bin_") or f.startswith("consistent_"):
        size = os.path.getsize(os.path.join(OUT, f))
        print(f"  {f} ({size/1024:.1f} KB)")
