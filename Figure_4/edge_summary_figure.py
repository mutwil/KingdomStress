"""
Summary figure: how many L1 edges are universal/stress-specific/organ-shared.
Stacked bars + Venn-style breakdown.
"""

import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

OUT = "/tmp/mercator_outputs"

plt.rcParams.update({
    'font.family': 'Arial', 'font.size': 13,
    'figure.dpi': 300, 'savefig.dpi': 300,
    'savefig.bbox': 'tight', 'figure.facecolor': 'white',
})

# ── Data ──────────────────────────────────────────────────────────────────
total = 19929

# Stress conservation
stress_counts = {1: 1402, 2: 1405, 3: 1804, 4: 2196, 5: 3160, 6: 9962}
stress_unique = {'Heat': 313, 'Cold': 89, 'Drought': 522, 'Salt': 314,
                 'Pathogen': 33, 'Heavy metal': 131}

# Organ
leaf_total = 19168
root_total = 12094
both_organs = 12058
leaf_only = 7110
root_only = 36
same_dir = 4659
flips = 1827
neutral = both_organs - same_dir - flips  # neither clearly up nor down in both

# ── Figure ────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(20, 14))
gs = fig.add_gridspec(2, 2, hspace=0.35, wspace=0.3)

# ── Panel A: Stress conservation (stacked bar) ───────────────────────────
ax1 = fig.add_subplot(gs[0, 0])

colors_stress = ['#264653', '#2A9D8F', '#E9C46A', '#F4A261', '#E76F51', '#E63946']
labels_stress = ['6/6 (universal)', '5/6', '4/6', '3/6', '2/6', '1/6 (unique)']
values_stress = [stress_counts[6], stress_counts[5], stress_counts[4],
                 stress_counts[3], stress_counts[2], stress_counts[1]]

bars = ax1.bar(0, values_stress[0], color=colors_stress[0], width=0.6)
bottom = values_stress[0]
for i in range(1, len(values_stress)):
    ax1.bar(0, values_stress[i], bottom=bottom, color=colors_stress[i], width=0.6)
    bottom += values_stress[i]

# Labels on bars
bottom = 0
for i, (val, label) in enumerate(zip(values_stress, labels_stress)):
    if val > 400:
        ax1.text(0, bottom + val/2, f'{label}\n{val:,} ({val/total*100:.0f}%)',
                 ha='center', va='center', fontsize=10, fontweight='bold',
                 color='white' if i < 2 else 'black')
    bottom += val

ax1.set_xlim(-0.8, 0.8)
ax1.set_xticks([])
ax1.set_ylabel("Number of L1 edges", fontsize=14)
ax1.set_title("A) Stress conservation\n(in how many stresses?)", fontweight='bold', fontsize=16)

# ── Panel B: Stress-unique breakdown ──────────────────────────────────────
ax2 = fig.add_subplot(gs[0, 1])

stress_colors = {'Heat': '#E63946', 'Cold': '#457B9D', 'Drought': '#E9C46A',
                 'Salt': '#2A9D8F', 'Pathogen': '#8338EC', 'Heavy metal': '#6D6875'}

stresses = list(stress_unique.keys())
vals = [stress_unique[s] for s in stresses]
colors = [stress_colors[s] for s in stresses]

bars = ax2.barh(range(len(stresses)), vals, color=colors, alpha=0.85)
ax2.set_yticks(range(len(stresses)))
ax2.set_yticklabels(stresses, fontsize=12)
for i, v in enumerate(vals):
    ax2.text(v + 10, i, str(v), va='center', fontsize=11)
ax2.set_xlabel("Number of unique edges", fontsize=13)
ax2.set_title("B) Stress-unique edges\n(present in only 1 stress)", fontweight='bold', fontsize=16)
ax2.invert_yaxis()

# ── Panel C: Organ breakdown (nested donut) ──────────────────────────────
ax3 = fig.add_subplot(gs[1, 0])

# Outer ring: leaf-only, shared, root-only
outer_sizes = [leaf_only, both_organs, root_only]
outer_colors = ['#E63946', '#888888', '#2A9D8F']
outer_labels = [f'Leaf-only\n{leaf_only:,}', f'Both organs\n{both_organs:,}', f'Root-only\n{root_only}']

wedges1, texts1 = ax3.pie(outer_sizes, colors=outer_colors, radius=1.0,
                            startangle=90, counterclock=False,
                            wedgeprops=dict(width=0.35, edgecolor='white', linewidth=2))

# Inner ring: breakdown of "both organs"
inner_sizes = [same_dir, flips, neutral]
inner_colors = ['#6CA6C1', '#D4A373', '#CCCCCC']
inner_labels = [f'Same dir.\n{same_dir:,}', f'Flips\n{flips:,}', f'Neutral\n{neutral:,}']

wedges2, texts2 = ax3.pie(inner_sizes, colors=inner_colors, radius=0.65,
                            startangle=90, counterclock=False,
                            wedgeprops=dict(width=0.35, edgecolor='white', linewidth=2))

# Center text
ax3.text(0, 0, f'{total:,}\ntotal', ha='center', va='center', fontsize=14, fontweight='bold')

# Legend
legend_elements = [
    mpatches.Patch(facecolor='#E63946', label=f'Leaf-only: {leaf_only:,} ({leaf_only/total*100:.0f}%)'),
    mpatches.Patch(facecolor='#2A9D8F', label=f'Root-only: {root_only} ({root_only/total*100:.1f}%)'),
    mpatches.Patch(facecolor='#6CA6C1', label=f'Both, same direction: {same_dir:,} ({same_dir/total*100:.0f}%)'),
    mpatches.Patch(facecolor='#D4A373', label=f'Both, direction flips: {flips:,} ({flips/total*100:.0f}%)'),
    mpatches.Patch(facecolor='#CCCCCC', label=f'Both, neutral: {neutral:,} ({neutral/total*100:.0f}%)'),
]
ax3.legend(handles=legend_elements, loc='center left', bbox_to_anchor=(1.0, 0.5),
           fontsize=10, frameon=False)

ax3.set_title("C) Organ distribution", fontweight='bold', fontsize=16)

# ── Panel D: Summary text ─────────────────────────────────────────────────
ax4 = fig.add_subplot(gs[1, 1])
ax4.axis('off')

summary_text = (
    "Key numbers (L1, excl. Enzyme classification):\n\n"
    f"Total edges: {total:,}\n\n"
    "Stress conservation:\n"
    "  - 50% universal (all 6 stresses)\n"
    "  - 43% shared (2-5 stresses)\n"
    "  - 7% stress-unique (1 stress only)\n"
    "  - Drought has most unique edges (522)\n"
    "  - Pathogen has fewest unique (33)\n\n"
    "Organ specificity:\n"
    "  - 63% present in both leaf and root\n"
    "  - 37% leaf-only, <1% root-only\n"
    "  - Of shared edges:\n"
    f"      39% same direction\n"
    f"      15% flip direction between organs\n"
    f"      46% neutral in one or both organs\n\n"
    "  Leaves have 60% more edges than roots,\n"
    "  suggesting a richer co-regulatory network\n"
    "  in the primary site of stress perception."
)

ax4.text(0.05, 0.95, summary_text, transform=ax4.transAxes,
         fontsize=12, verticalalignment='top', fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor='#F5F5F5', alpha=0.8))
ax4.set_title("D) Summary", fontweight='bold', fontsize=16)

fig.suptitle("MapMan L1 co-occurrence edge landscape: stress and organ dimensions",
             fontsize=18, fontweight='bold', y=1.01)

fig.savefig(os.path.join(OUT, "edge_summary.png"))
fig.savefig(os.path.join(OUT, "edge_summary.pdf"))
plt.close(fig)

# Copy
import shutil
GDRIVE_OUT = "/Users/vjx443/Library/CloudStorage/GoogleDrive-mutwil@plant.tools/My Drive/Projects/2026_KingdomStress/CC-Fig 4 data/outputs"
shutil.copy2(os.path.join(OUT, "edge_summary.png"), os.path.join(GDRIVE_OUT, "edge_summary.png"))
shutil.copy2(os.path.join(OUT, "edge_summary.pdf"), os.path.join(GDRIVE_OUT, "edge_summary.pdf"))
print("Saved: edge_summary.png/pdf")
