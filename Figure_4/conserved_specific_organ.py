"""
Conserved vs stress-specific MapMan bin associations, and Leaf vs Root comparison.
Levels 0 and 1, MERGE direction.
"""

import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import zscore
import warnings
warnings.filterwarnings('ignore')

BASE = "/tmp/mercator_data"
OUT = "/tmp/mercator_outputs"
os.makedirs(OUT, exist_ok=True)

plt.rcParams.update({
    'font.family': 'Arial', 'font.size': 10, 'axes.titlesize': 12,
    'axes.labelsize': 10, 'figure.dpi': 300, 'savefig.dpi': 300,
    'savefig.bbox': 'tight', 'figure.facecolor': 'white',
})

STRESSES = ['Heat', 'Cold', 'Drought', 'Salt', 'Pathogen', 'Heavy metal']
STRESS_COLORS = {'Heat': '#E63946', 'Cold': '#457B9D', 'Drought': '#E9C46A',
                 'Salt': '#2A9D8F', 'Pathogen': '#8338EC', 'Heavy metal': '#6D6875'}


def shorten(label, n=30):
    return label if len(label) <= n else label[:n-2] + '..'


def load_edge_dict(path):
    """Load CSV, return {sorted(src,tgt): weight} dict."""
    if not os.path.exists(path):
        return {}
    try:
        df = pd.read_csv(path, index_col=0)
    except Exception:
        return {}
    out = {}
    for _, r in df.iterrows():
        k = tuple(sorted([r['source'], r['target']]))
        out[k] = r['weight']
    return out


def build_stress_matrix(level):
    """Build edge x stress weight matrix for a given level."""
    all_edges = set()
    stress_dicts = {}
    for stress in STRESSES:
        path = os.path.join(BASE, "Stresses", stress,
                            f"Mercator_network_MERGE_Level{level} (All_organ).csv")
        ew = load_edge_dict(path)
        stress_dicts[stress] = ew
        all_edges.update(ew.keys())

    edge_list = sorted(all_edges)
    labels = [f"{e[0]} -- {e[1]}" for e in edge_list]
    mat = pd.DataFrame(0.0, index=labels, columns=STRESSES)
    for i, e in enumerate(edge_list):
        for stress in STRESSES:
            mat.iloc[i, mat.columns.get_loc(stress)] = stress_dicts[stress].get(e, 0.0)
    return mat


def build_organ_matrix(level, organ):
    """Load All_stress Normalised organ-specific network as edge dict."""
    suffix = f"({organ})" if organ != 'All' else ""
    path = os.path.join(BASE, "All_stress",
                        f"Mercator_network_MERGE_Level{level} (Normalised){suffix}.csv")
    return load_edge_dict(path)


# ═══════════════════════════════════════════════════════════════════════════
# Load data
# ═══════════════════════════════════════════════════════════════════════════
print("Loading data...")
mat_L0 = build_stress_matrix(0)
mat_L1 = build_stress_matrix(1)
leaf_L0 = build_organ_matrix(0, 'Leaf')
root_L0 = build_organ_matrix(0, 'Root')
leaf_L1 = build_organ_matrix(1, 'Leaf')
root_L1 = build_organ_matrix(1, 'Root')
print(f"  Level0: {len(mat_L0)} edges")
print(f"  Level1: {len(mat_L1)} edges")
print(f"  Leaf L0: {len(leaf_L0)}, Root L0: {len(root_L0)} edges")
print(f"  Leaf L1: {len(leaf_L1)}, Root L1: {len(root_L1)} edges")


# ═══════════════════════════════════════════════════════════════════════════
# PART 1: Conserved vs stress-specific (per level)
# ═══════════════════════════════════════════════════════════════════════════

def classify_edges(mat, level_label):
    """
    Classify edges as conserved or stress-enriched.
    Conserved: high mean AND low CV (consistent across stresses).
    Stress-enriched: z-score > 2 in one stress relative to its mean across stresses.
    """
    mat = mat[(mat > 0).any(axis=1)].copy()

    mat['mean'] = mat[STRESSES].mean(axis=1)
    mat['std'] = mat[STRESSES].std(axis=1)
    mat['cv'] = mat['std'] / mat['mean'].replace(0, np.nan)
    mat['max_stress'] = mat[STRESSES].idxmax(axis=1)
    mat['max_weight'] = mat[STRESSES].max(axis=1)
    mat['presence'] = (mat[STRESSES] > 0).sum(axis=1)

    # Z-scores per edge (across stresses)
    zscores = mat[STRESSES].apply(zscore, axis=1, result_type='broadcast')
    mat['max_zscore'] = zscores.max(axis=1)
    mat['max_z_stress'] = zscores.idxmax(axis=1)

    # Specificity score: max_weight / mean (how much the top stress exceeds average)
    mat['specificity'] = mat['max_weight'] / mat['mean'].replace(0, np.nan)

    # Classifications
    # Conserved: present in all 6 stresses, CV < median CV, mean > median mean
    all_present = mat['presence'] == len(STRESSES)
    median_cv = mat.loc[all_present, 'cv'].median()
    median_mean = mat.loc[all_present, 'mean'].median()

    mat['category'] = 'other'
    mat.loc[all_present & (mat['cv'] < median_cv) & (mat['mean'] > median_mean), 'category'] = 'conserved'
    mat.loc[mat['max_zscore'] > 1.5, 'category'] = 'stress-enriched'

    # For stress-enriched, which stress?
    mat['enriched_in'] = ''
    mask_enriched = mat['category'] == 'stress-enriched'
    mat.loc[mask_enriched, 'enriched_in'] = mat.loc[mask_enriched, 'max_z_stress']

    n_conserved = (mat['category'] == 'conserved').sum()
    n_enriched = (mat['category'] == 'stress-enriched').sum()
    n_other = (mat['category'] == 'other').sum()
    print(f"\n  {level_label}: conserved={n_conserved}, stress-enriched={n_enriched}, other={n_other}")

    return mat


print("\n=== Part 1: Conserved vs stress-specific ===")
classified_L0 = classify_edges(mat_L0, "Level0")
classified_L1 = classify_edges(mat_L1, "Level1")

# Save tables
classified_L0.to_csv(os.path.join(OUT, "conserved_specific_L0.csv"))
classified_L1.to_csv(os.path.join(OUT, "conserved_specific_L1.csv"))


# ── Plot 1A: Scatterplot mean vs CV, colored by category ─────────────────
for level, clf in [('Level0', classified_L0), ('Level1', classified_L1)]:
    fig, ax = plt.subplots(figsize=(10, 8))

    for cat, color, marker, alpha in [
        ('other', '#CCCCCC', '.', 0.3),
        ('conserved', '#2A9D8F', 'o', 0.7),
        ('stress-enriched', '#E63946', '^', 0.7),
    ]:
        subset = clf[clf['category'] == cat]
        ax.scatter(subset['mean'], subset['cv'], c=color, marker=marker,
                   alpha=alpha, s=15 if cat == 'other' else 30, label=f"{cat} (n={len(subset)})",
                   edgecolors='none')

    # Label top conserved
    top_cons = clf[clf['category'] == 'conserved'].nlargest(5, 'mean')
    for idx in top_cons.index:
        ax.annotate(shorten(idx, 35), (top_cons.loc[idx, 'mean'], top_cons.loc[idx, 'cv']),
                    fontsize=5, alpha=0.8)

    # Label top stress-enriched
    top_enr = clf[clf['category'] == 'stress-enriched'].nlargest(5, 'specificity')
    for idx in top_enr.index:
        ax.annotate(shorten(idx, 35), (top_enr.loc[idx, 'mean'], top_enr.loc[idx, 'cv']),
                    fontsize=5, alpha=0.8)

    ax.set_xlabel("Mean co-occurrence frequency across stresses")
    ax.set_ylabel("Coefficient of variation across stresses")
    ax.set_title(f"Conserved vs stress-specific associations ({level} MERGE)")
    ax.legend(fontsize=9)
    plt.tight_layout()
    fig.savefig(os.path.join(OUT, f"conserved_vs_specific_{level}.png"))
    fig.savefig(os.path.join(OUT, f"conserved_vs_specific_{level}.pdf"))
    plt.close(fig)


# ── Plot 1B: Top conserved edges - heatmap across stresses ───────────────
for level, clf in [('Level0', classified_L0), ('Level1', classified_L1)]:
    conserved = clf[clf['category'] == 'conserved'].nlargest(30, 'mean')
    if conserved.empty:
        continue

    fig, ax = plt.subplots(figsize=(10, 10))
    hm_data = conserved[STRESSES]
    sns.heatmap(hm_data, cmap='YlOrRd', ax=ax, linewidths=0.3, linecolor='white',
                yticklabels=[shorten(l, 50) for l in hm_data.index],
                vmin=0, cbar_kws={'label': 'Co-occurrence frequency', 'shrink': 0.5})
    ax.set_title(f"Top 30 conserved associations ({level} MERGE)\nHigh mean, low variation across stresses")
    ax.set_xlabel("Stress")
    plt.tight_layout()
    fig.savefig(os.path.join(OUT, f"conserved_top30_{level}.png"))
    fig.savefig(os.path.join(OUT, f"conserved_top30_{level}.pdf"))
    plt.close(fig)


# ── Plot 1C: Stress-enriched edges - which stress enriches which edges ───
for level, clf in [('Level0', classified_L0), ('Level1', classified_L1)]:
    enriched = clf[clf['category'] == 'stress-enriched'].copy()
    if enriched.empty:
        continue

    # Count how many edges each stress enriches
    enr_counts = enriched['enriched_in'].value_counts()
    print(f"\n  {level} stress-enriched edge counts:")
    for stress in STRESSES:
        print(f"    {stress}: {enr_counts.get(stress, 0)} edges")

    # Top enriched edges per stress
    fig, axes = plt.subplots(2, 3, figsize=(22, 12))
    for i, stress in enumerate(STRESSES):
        ax = axes[i // 3, i % 3]
        stress_enr = enriched[enriched['enriched_in'] == stress].nlargest(15, 'max_zscore')
        if stress_enr.empty:
            ax.set_title(f"{stress}\n(no enriched edges)", fontweight='bold')
            ax.axis('off')
            continue

        # Show the weight profile across stresses
        plot_data = stress_enr[STRESSES]
        bars_y = range(len(plot_data))
        for j, s in enumerate(STRESSES):
            offset = j * 0.12
            ax.barh([y + offset for y in bars_y], plot_data[s],
                    height=0.11, color=STRESS_COLORS[s], alpha=0.8,
                    label=s if i == 0 else '')

        ax.set_yticks([y + 0.3 for y in bars_y])
        ax.set_yticklabels([shorten(idx, 40) for idx in plot_data.index], fontsize=6)
        ax.set_xlabel("Co-occurrence frequency")
        ax.set_title(f"{stress}-enriched\n({len(stress_enr)} shown)", fontweight='bold',
                     color=STRESS_COLORS[stress])
        ax.invert_yaxis()

    # Single legend
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=6, fontsize=9,
              bbox_to_anchor=(0.5, -0.02))
    fig.suptitle(f"Top stress-enriched associations ({level} MERGE)", fontsize=14, y=1.01)
    plt.tight_layout()
    fig.savefig(os.path.join(OUT, f"stress_enriched_{level}.png"))
    fig.savefig(os.path.join(OUT, f"stress_enriched_{level}.pdf"))
    plt.close(fig)


# ── Plot 1D: Summary bar - conserved vs enriched per stress ──────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
for i, (level, clf) in enumerate([('Level0', classified_L0), ('Level1', classified_L1)]):
    ax = axes[i]
    cats = clf['category'].value_counts()
    # Enriched breakdown by stress
    enr = clf[clf['category'] == 'stress-enriched']['enriched_in'].value_counts()

    bar_labels = ['conserved'] + [f'{s}-enriched' for s in STRESSES if s in enr.index] + ['other']
    bar_values = [cats.get('conserved', 0)]
    bar_colors = ['#2A9D8F']
    for s in STRESSES:
        if s in enr.index:
            bar_values.append(enr[s])
            bar_colors.append(STRESS_COLORS[s])
    bar_values.append(cats.get('other', 0))
    bar_colors.append('#CCCCCC')

    ax.bar(range(len(bar_labels)), bar_values, color=bar_colors, alpha=0.85)
    ax.set_xticks(range(len(bar_labels)))
    ax.set_xticklabels(bar_labels, rotation=45, ha='right', fontsize=8)
    ax.set_ylabel("Number of edges")
    ax.set_title(f"{level} MERGE", fontweight='bold')

    for j, v in enumerate(bar_values):
        ax.text(j, v + 1, str(v), ha='center', fontsize=7)

fig.suptitle("Edge classification: conserved vs stress-enriched", fontsize=13, y=1.02)
plt.tight_layout()
fig.savefig(os.path.join(OUT, "conserved_vs_enriched_summary.png"))
fig.savefig(os.path.join(OUT, "conserved_vs_enriched_summary.pdf"))
plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════════
# PART 2: Leaf vs Root comparison
# ═══════════════════════════════════════════════════════════════════════════
print("\n=== Part 2: Leaf vs Root comparison ===")


def organ_comparison(leaf_dict, root_dict, level_label):
    """Compare leaf and root networks, return DataFrame."""
    all_edges = set(leaf_dict.keys()) | set(root_dict.keys())
    rows = []
    for e in sorted(all_edges):
        lw = leaf_dict.get(e, 0.0)
        rw = root_dict.get(e, 0.0)
        rows.append({
            'edge': f"{e[0]} -- {e[1]}",
            'Leaf': lw, 'Root': rw,
            'diff': lw - rw,
            'abs_diff': abs(lw - rw),
            'mean': (lw + rw) / 2,
            'organ_bias': 'Leaf' if lw > rw else ('Root' if rw > lw else 'equal'),
        })
    df = pd.DataFrame(rows)

    # Classify
    # Leaf-specific: present in leaf but absent/very low in root (or vice versa)
    df['leaf_specific'] = (df['Leaf'] > 0) & (df['Root'] == 0)
    df['root_specific'] = (df['Root'] > 0) & (df['Leaf'] == 0)
    df['shared'] = (df['Leaf'] > 0) & (df['Root'] > 0)

    n_leaf_only = df['leaf_specific'].sum()
    n_root_only = df['root_specific'].sum()
    n_shared = df['shared'].sum()
    r = df['Leaf'].corr(df['Root'])
    print(f"\n  {level_label}: Leaf-only={n_leaf_only}, Root-only={n_root_only}, "
          f"shared={n_shared}, Pearson r={r:.3f}")

    return df


organ_L0 = organ_comparison(leaf_L0, root_L0, "Level0")
organ_L1 = organ_comparison(leaf_L1, root_L1, "Level1")
organ_L0.to_csv(os.path.join(OUT, "organ_comparison_L0.csv"), index=False)
organ_L1.to_csv(os.path.join(OUT, "organ_comparison_L1.csv"), index=False)


# ── Plot 2A: Scatter Leaf vs Root with marginal histograms ───────────────
for level, odf in [('Level0', organ_L0), ('Level1', organ_L1)]:
    fig = plt.figure(figsize=(10, 10))
    gs = fig.add_gridspec(2, 2, width_ratios=[4, 1], height_ratios=[1, 4],
                          hspace=0.05, wspace=0.05)

    ax_main = fig.add_subplot(gs[1, 0])
    ax_top = fig.add_subplot(gs[0, 0], sharex=ax_main)
    ax_right = fig.add_subplot(gs[1, 1], sharey=ax_main)

    shared = odf[odf['shared']]
    leaf_only = odf[odf['leaf_specific']]
    root_only = odf[odf['root_specific']]

    ax_main.scatter(shared['Root'], shared['Leaf'], alpha=0.3, s=8, c='#457B9D',
                    edgecolors='none', label=f'Shared ({len(shared)})')
    ax_main.scatter(leaf_only['Root'], leaf_only['Leaf'], alpha=0.7, s=20, c='#E63946',
                    marker='^', edgecolors='none', label=f'Leaf-only ({len(leaf_only)})')
    ax_main.scatter(root_only['Root'], root_only['Leaf'], alpha=0.7, s=20, c='#2A9D8F',
                    marker='v', edgecolors='none', label=f'Root-only ({len(root_only)})')

    max_w = max(odf['Leaf'].max(), odf['Root'].max()) * 1.05
    ax_main.plot([0, max_w], [0, max_w], 'k--', alpha=0.3, lw=1)
    ax_main.set_xlabel("Root co-occurrence frequency")
    ax_main.set_ylabel("Leaf co-occurrence frequency")
    ax_main.legend(fontsize=8, loc='upper left')
    r = odf['Leaf'].corr(odf['Root'])
    ax_main.text(0.95, 0.05, f'r = {r:.3f}', transform=ax_main.transAxes,
                 ha='right', fontsize=10)

    # Annotate most divergent shared edges
    top_div = shared.nlargest(8, 'abs_diff')
    for _, row in top_div.iterrows():
        ax_main.annotate(shorten(row['edge'], 30), (row['Root'], row['Leaf']),
                         fontsize=4.5, alpha=0.7)

    # Marginals
    ax_top.hist(odf['Root'], bins=50, color='#457B9D', alpha=0.5, density=True)
    ax_top.set_ylabel("Density")
    ax_top.tick_params(labelbottom=False)

    ax_right.hist(odf['Leaf'], bins=50, color='#E63946', alpha=0.5, density=True,
                  orientation='horizontal')
    ax_right.set_xlabel("Density")
    ax_right.tick_params(labelleft=False)

    fig.suptitle(f"Leaf vs Root co-occurrence networks ({level} MERGE)", fontsize=13, y=0.95)
    fig.savefig(os.path.join(OUT, f"leaf_vs_root_scatter_{level}.png"))
    fig.savefig(os.path.join(OUT, f"leaf_vs_root_scatter_{level}.pdf"))
    plt.close(fig)


# ── Plot 2B: Top leaf-biased and root-biased edges ──────────────────────
for level, odf in [('Level0', organ_L0), ('Level1', organ_L1)]:
    top_leaf = odf[odf['shared']].nlargest(15, 'diff')
    top_root = odf[odf['shared']].nsmallest(15, 'diff')
    combined = pd.concat([top_root, top_leaf]).sort_values('diff')

    fig, ax = plt.subplots(figsize=(12, 10))
    colors = ['#2A9D8F' if d < 0 else '#E63946' for d in combined['diff']]
    ax.barh(range(len(combined)), combined['diff'], color=colors, alpha=0.85)
    ax.set_yticks(range(len(combined)))
    ax.set_yticklabels([shorten(e, 50) for e in combined['edge']], fontsize=7)
    ax.set_xlabel("Leaf - Root co-occurrence frequency")
    ax.axvline(0, color='black', lw=0.5)
    ax.set_title(f"Most organ-biased associations ({level} MERGE)\n"
                 f"Red = leaf-enriched, teal = root-enriched")
    plt.tight_layout()
    fig.savefig(os.path.join(OUT, f"organ_biased_edges_{level}.png"))
    fig.savefig(os.path.join(OUT, f"organ_biased_edges_{level}.pdf"))
    plt.close(fig)


# ── Plot 2C: Are conserved edges also organ-conserved? ───────────────────
# Cross-reference conserved/enriched classification with organ bias
for level, clf, odf in [('Level0', classified_L0, organ_L0), ('Level1', classified_L1, organ_L1)]:
    # Merge on edge label
    merged = clf[['category', 'mean', 'cv', 'enriched_in']].merge(
        odf[['edge', 'Leaf', 'Root', 'diff', 'abs_diff']],
        left_index=True, right_on='edge', how='inner'
    )

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

    # Left: organ bias by category
    for cat, color in [('conserved', '#2A9D8F'), ('stress-enriched', '#E63946'), ('other', '#CCCCCC')]:
        subset = merged[merged['category'] == cat]
        if subset.empty:
            continue
        ax1.scatter(subset['abs_diff'], subset['mean'], c=color, alpha=0.4, s=15,
                    label=f"{cat} (n={len(subset)})", edgecolors='none')

    ax1.set_xlabel("|Leaf - Root| difference")
    ax1.set_ylabel("Mean co-occurrence frequency")
    ax1.set_title(f"Organ divergence by conservation category ({level})")
    ax1.legend(fontsize=8)

    # Right: violin of abs_diff per category
    cat_data = []
    cat_labels = []
    for cat in ['conserved', 'stress-enriched', 'other']:
        vals = merged.loc[merged['category'] == cat, 'abs_diff'].values
        if len(vals) > 0:
            cat_data.append(vals)
            cat_labels.append(cat)

    if cat_data:
        parts = ax2.violinplot(cat_data, showmeans=True, showmedians=True)
        ax2.set_xticks(range(1, len(cat_labels) + 1))
        ax2.set_xticklabels(cat_labels, fontsize=9)
        ax2.set_ylabel("|Leaf - Root| difference")
        ax2.set_title(f"Organ divergence distribution by category ({level})")

        # Print means
        for i, (cat, vals) in enumerate(zip(cat_labels, cat_data)):
            print(f"  {level} {cat}: mean |Leaf-Root| = {np.mean(vals):.4f}, "
                  f"median = {np.median(vals):.4f}")

    plt.tight_layout()
    fig.savefig(os.path.join(OUT, f"organ_by_category_{level}.png"))
    fig.savefig(os.path.join(OUT, f"organ_by_category_{level}.pdf"))
    plt.close(fig)


# ── Plot 2D: Leaf-only and root-only edges - what are they? ─────────────
for level, odf in [('Level0', organ_L0), ('Level1', organ_L1)]:
    leaf_only_edges = odf[odf['leaf_specific']].nlargest(20, 'Leaf')
    root_only_edges = odf[odf['root_specific']].nlargest(20, 'Root')

    if leaf_only_edges.empty and root_only_edges.empty:
        print(f"  {level}: No organ-specific edges")
        continue

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

    if not leaf_only_edges.empty:
        ax1.barh(range(len(leaf_only_edges)), leaf_only_edges['Leaf'].values,
                 color='#E63946', alpha=0.8)
        ax1.set_yticks(range(len(leaf_only_edges)))
        ax1.set_yticklabels([shorten(e, 45) for e in leaf_only_edges['edge']], fontsize=6)
        ax1.set_xlabel("Co-occurrence frequency")
        ax1.set_title(f"Leaf-only edges ({level})", fontweight='bold', color='#E63946')
        ax1.invert_yaxis()
    else:
        ax1.text(0.5, 0.5, "No leaf-only edges", ha='center', va='center',
                 transform=ax1.transAxes)
        ax1.set_title(f"Leaf-only edges ({level})")

    if not root_only_edges.empty:
        ax2.barh(range(len(root_only_edges)), root_only_edges['Root'].values,
                 color='#2A9D8F', alpha=0.8)
        ax2.set_yticks(range(len(root_only_edges)))
        ax2.set_yticklabels([shorten(e, 45) for e in root_only_edges['edge']], fontsize=6)
        ax2.set_xlabel("Co-occurrence frequency")
        ax2.set_title(f"Root-only edges ({level})", fontweight='bold', color='#2A9D8F')
        ax2.invert_yaxis()
    else:
        ax2.text(0.5, 0.5, "No root-only edges", ha='center', va='center',
                 transform=ax2.transAxes)
        ax2.set_title(f"Root-only edges ({level})")

    plt.tight_layout()
    fig.savefig(os.path.join(OUT, f"organ_specific_edges_{level}.png"))
    fig.savefig(os.path.join(OUT, f"organ_specific_edges_{level}.pdf"))
    plt.close(fig)


# ── Plot 2E: Side-by-side adjacency matrices Leaf vs Root (Level0) ──────
print("\n=== Side-by-side Leaf vs Root matrices ===")

# Get all Level0 bins
all_bins_L0 = set()
for e in list(leaf_L0.keys()) + list(root_L0.keys()):
    all_bins_L0.update(e)
all_bins_L0 = sorted(all_bins_L0)

leaf_mat = pd.DataFrame(0.0, index=all_bins_L0, columns=all_bins_L0)
root_mat = pd.DataFrame(0.0, index=all_bins_L0, columns=all_bins_L0)
for e, w in leaf_L0.items():
    leaf_mat.loc[e[0], e[1]] = w
    leaf_mat.loc[e[1], e[0]] = w
for e, w in root_L0.items():
    root_mat.loc[e[0], e[1]] = w
    root_mat.loc[e[1], e[0]] = w

# Difference matrix
diff_mat = leaf_mat - root_mat

fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(30, 9))
short_labels = [shorten(l, 18) for l in all_bins_L0]

sns.heatmap(leaf_mat, ax=ax1, cmap='YlOrRd', vmin=0, vmax=0.6,
            xticklabels=short_labels, yticklabels=short_labels,
            linewidths=0.3, linecolor='white', cbar_kws={'shrink': 0.5, 'label': 'Co-occ. freq.'})
ax1.set_title("Leaf", fontsize=14, fontweight='bold', color='#E63946')
ax1.tick_params(labelsize=6)
plt.setp(ax1.xaxis.get_majorticklabels(), rotation=90, ha='right')

sns.heatmap(root_mat, ax=ax2, cmap='YlOrRd', vmin=0, vmax=0.6,
            xticklabels=short_labels, yticklabels=short_labels,
            linewidths=0.3, linecolor='white', cbar_kws={'shrink': 0.5, 'label': 'Co-occ. freq.'})
ax2.set_title("Root", fontsize=14, fontweight='bold', color='#2A9D8F')
ax2.tick_params(labelsize=6)
plt.setp(ax2.xaxis.get_majorticklabels(), rotation=90, ha='right')

sns.heatmap(diff_mat, ax=ax3, cmap='RdBu_r', center=0, vmin=-0.15, vmax=0.15,
            xticklabels=short_labels, yticklabels=short_labels,
            linewidths=0.3, linecolor='white', cbar_kws={'shrink': 0.5, 'label': 'Leaf - Root'})
ax3.set_title("Difference (Leaf - Root)", fontsize=14, fontweight='bold')
ax3.tick_params(labelsize=6)
plt.setp(ax3.xaxis.get_majorticklabels(), rotation=90, ha='right')

fig.suptitle("Level0 MapMan bin co-occurrence: Leaf vs Root (all stresses, normalised)", fontsize=14, y=1.02)
plt.tight_layout()
fig.savefig(os.path.join(OUT, "leaf_root_matrices_L0.png"))
fig.savefig(os.path.join(OUT, "leaf_root_matrices_L0.pdf"))
plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════════
# Copy to Google Drive
# ═══════════════════════════════════════════════════════════════════════════
import shutil
GDRIVE_OUT = "/Users/vjx443/Library/CloudStorage/GoogleDrive-mutwil@plant.tools/My Drive/Projects/2026_KingdomStress/CC-Fig 4 data/outputs"
os.makedirs(GDRIVE_OUT, exist_ok=True)
new_files = [f for f in os.listdir(OUT)
             if any(f.startswith(p) for p in ['conserved_', 'stress_enriched_', 'organ_', 'leaf_'])]
for f in new_files:
    shutil.copy2(os.path.join(OUT, f), os.path.join(GDRIVE_OUT, f))

print("\n" + "=" * 60)
print("CONSERVED / SPECIFIC / ORGAN ANALYSIS COMPLETE")
print("=" * 60)
for f in sorted(new_files):
    size = os.path.getsize(os.path.join(OUT, f))
    print(f"  {f} ({size/1024:.1f} KB)")
