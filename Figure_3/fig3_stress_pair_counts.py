#!/usr/bin/env python3
"""
Figure 3: Significant stress pair counts (horizontal bar chart).
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os

OUT_DIR = os.path.dirname(os.path.abspath(__file__))

# Data from the original figure (sorted descending)
pairs = [
    ("Drought-Salt", 10),
    ("Drought-Pathogen", 7),
    ("Cold-Drought", 6),
    ("Cold-Heat", 4),
    ("Cold-Pathogen", 4),
    ("Cold-Salt", 4),
    ("Drought-Heat", 4),
    ("Heat-Salt", 4),
    ("Heavy metal-Pathogen", 3),
    ("Heavy metal-Salt", 3),
    ("Drought-Heavy metal", 2),
    ("Heat-Heavy metal", 2),
    ("Pathogen-Salt", 2),
    ("Cold-Heavy metal", 1),
    ("Heat-Pathogen", 1),
]

labels = [p[0] for p in pairs]
counts = [p[1] for p in pairs]

# Reverse so highest is at top
labels = labels[::-1]
counts = counts[::-1]

fig, ax = plt.subplots(figsize=(6, 6))

ax.barh(range(len(labels)), counts, color="#7fb3d8", edgecolor="white", linewidth=0.5)

ax.set_yticks(range(len(labels)))
ax.set_yticklabels(labels, fontsize=10)
ax.set_xlabel("Stress pair counts", fontsize=11)
ax.set_title("Significant stress pair counts (P_adj < 0.05)",
             fontsize=12, fontweight="bold", pad=10)

# Add count labels at end of bars
for i, c in enumerate(counts):
    ax.text(c + 0.15, i, str(c), ha="left", va="center", fontsize=9, color="#333333")

ax.set_xlim(0, max(counts) + 1.5)
ax.tick_params(labelsize=9)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

plt.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "fig3_stress_pair_counts.pdf"), dpi=300, bbox_inches="tight")
fig.savefig(os.path.join(OUT_DIR, "fig3_stress_pair_counts.png"), dpi=300, bbox_inches="tight")
print("Saved: fig3_stress_pair_counts.pdf/.png")
plt.close(fig)
