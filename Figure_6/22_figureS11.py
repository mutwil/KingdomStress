#!/usr/bin/env python3
"""
Assemble Supplementary Figure S11 for the Figure 6 revision.

Panels
  A  Label prevalence tracks assay depth (the confound behind the pooled auROC)
  B  Pooled vs within-species auROC per model (how much of the headline is
     between-species prevalence)
  C  Lineage-restricted training vs a size-matched control (the reviewer's question)
  D  Prevalence balancing: pooled falls, within-species rises
  E  Label-noise ceiling vs achieved performance
  F  ChIP-seq binding enrichment, stress TFs vs developmental TFs (specificity control)
  G  Organ-restricted labels vs all-organ labels

Reads only the intermediate CSVs, so the figure regenerates without rerunning any
model. Style matches 10_figure6.py (Arial, 7 pt base, Nature vibrant palette).

Usage
  python 22_figureS11.py --revision_dir ".../revision" --output_dir ".../figures"

Kingdom Stress Atlas, Mutwil lab.
"""

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
from matplotlib import rcParams

rcParams["font.family"] = "Arial"
rcParams["font.size"] = 7
rcParams["axes.linewidth"] = 0.5
rcParams["xtick.major.width"] = 0.5
rcParams["ytick.major.width"] = 0.5

C_CAD2, C_RNA, C_FUSED = "#009988", "#0077BB", "#000000"
C_STRESS, C_DEV = "#CC3311", "#BBBBBB"
C_LIN = {"Monocot": "#EE7733", "Rosid": "#009988", "Asterid": "#33BBEE"}
STRESS_ORDER = ["Heat", "Cold", "Drought", "Salt", "Pathogen"]


def panel_A(ax, R):
    f = os.path.join(R, "depth_vs_prevalence.csv")
    if not os.path.exists(f):
        ax.set_axis_off(); return
    d = pd.read_csv(f)
    ax.scatter(d["n_experiments"], d["mean_pos_rate"], s=18, color=C_CAD2,
               edgecolor="black", linewidth=0.3, zorder=3)
    if len(d) > 2:
        z = np.polyfit(d["n_experiments"], d["mean_pos_rate"], 1)
        xs = np.linspace(d["n_experiments"].min(), d["n_experiments"].max(), 50)
        ax.plot(xs, np.poly1d(z)(xs), "--", color="grey", lw=0.7)
        from scipy.stats import spearmanr
        rho, p = spearmanr(d["n_experiments"], d["mean_pos_rate"])
        ax.text(0.03, 0.95, f"Spearman $\\rho$={rho:.2f}\nP={p:.3g}",
                transform=ax.transAxes, va="top", fontsize=5)
    for _, r in d.nlargest(3, "n_experiments").iterrows():
        ax.annotate(r["species"].split("_")[0], (r["n_experiments"], r["mean_pos_rate"]),
                    fontsize=4, xytext=(3, 2), textcoords="offset points")
    ax.set_xlabel("Experiments per species", fontsize=6)
    ax.set_ylabel("Positive rate", fontsize=6)
    ax.set_title("A  Label prevalence tracks assay depth", fontsize=7,
                 fontweight="bold", loc="left")
    ax.tick_params(labelsize=5)


def panel_B(ax, R):
    f = os.path.join(R, "decomposition_summary.csv")
    if not os.path.exists(f):
        ax.set_axis_off(); return
    d = pd.read_csv(f)
    models, colors = ["DNA", "RNA", "Fused"], [C_CAD2, C_RNA, C_FUSED]
    labels = ["PlantCAD2", "PlantRNA-FM", "Fused"]
    x = np.arange(len(models))
    pooled = [d[d.model == m]["pooled_auroc"].mean() for m in models]
    within = [d[d.model == m]["within_species_auroc_weighted"].mean() for m in models]
    w = 0.34
    ax.bar(x - w/2, pooled, w, color=colors, alpha=0.45, label="pooled")
    ax.bar(x + w/2, within, w, color=colors, alpha=0.95, label="within-species")
    for i, (p, wi) in enumerate(zip(pooled, within)):
        ax.text(i, max(p, wi) + 0.004, f"$-${p-wi:.3f}", ha="center", fontsize=4.5)
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=5.5)
    ax.set_ylim(0.6, 0.79)
    ax.set_ylabel("Test auROC", fontsize=6)
    ax.set_title("B  Pooled vs within-species", fontsize=7, fontweight="bold", loc="left")
    ax.legend(fontsize=4.5, loc="upper left", framealpha=0.8)
    ax.tick_params(labelsize=5)


def panel_C(ax, R):
    f = os.path.join(R, "sweep", "sweep_results.csv")
    if not os.path.exists(f):
        ax.set_axis_off(); return
    d = pd.read_csv(f)
    piv = (d[d.arm.isin(["stratum", "pan_matched"])]
           .pivot_table(index=["stress", "direction", "stratum"], columns="arm",
                        values="pooled_auroc").dropna())
    piv["delta"] = piv["stratum"] - piv["pan_matched"]
    lins = [l for l in ["Monocot", "Rosid", "Asterid"]
            if l in piv.index.get_level_values("stratum")]
    for i, L in enumerate(lins):
        vals = piv.xs(L, level="stratum")["delta"].values
        ax.scatter(np.full(len(vals), i) + np.random.RandomState(0).uniform(-.09, .09, len(vals)),
                   vals, s=12, color=C_LIN.get(L, "grey"), edgecolor="black",
                   linewidth=0.25, zorder=3)
        ax.hlines(vals.mean(), i - 0.28, i + 0.28, color="black", lw=1.2, zorder=4)
        ax.text(i - 0.34, vals.mean(), f"{vals.mean():+.3f}", ha="right",
                va="center", fontsize=4.5)
    ax.axhline(0, color="grey", ls="--", lw=0.6)
    ax.set_xticks(range(len(lins))); ax.set_xticklabels(lins, fontsize=5.5)
    ax.set_ylabel("$\\Delta$ auROC vs size-matched", fontsize=6)
    ax.set_title("C  Lineage-restricted training", fontsize=7, fontweight="bold", loc="left")
    ax.tick_params(labelsize=5)
    from scipy.stats import wilcoxon
    try:
        _, p = wilcoxon(piv["stratum"], piv["pan_matched"])
        ax.text(0.97, 0.05, f"P={p:.1e}\nn={len(piv)}", transform=ax.transAxes,
                ha="right", va="bottom", fontsize=5)
    except ValueError:
        pass


def panel_D(ax, R):
    f = os.path.join(R, "sweep", "sweep_results.csv")
    if not os.path.exists(f):
        ax.set_axis_off(); return
    d = pd.read_csv(f)
    b = d[d.arm.isin(["baseline", "balanced"])].pivot_table(
        index=["stress", "direction"], columns="arm",
        values=["pooled_auroc", "within_species_auroc"])
    dp = (b[("pooled_auroc", "balanced")] - b[("pooled_auroc", "baseline")]).dropna()
    dw = (b[("within_species_auroc", "balanced")]
          - b[("within_species_auroc", "baseline")]).dropna()
    for i, (vals, lab, col) in enumerate([(dp, "pooled", "#EE3377"),
                                          (dw, "within-species", C_CAD2)]):
        ax.scatter(np.full(len(vals), i) + np.random.RandomState(1).uniform(-.08, .08, len(vals)),
                   vals, s=12, color=col, edgecolor="black", linewidth=0.25, zorder=3)
        ax.hlines(vals.mean(), i - 0.28, i + 0.28, color="black", lw=1.2, zorder=4)
        ax.text(i, vals.mean() + 0.002, f"{vals.mean():+.3f}", ha="center", fontsize=5)
    ax.axhline(0, color="grey", ls="--", lw=0.6)
    ax.set_xticks([0, 1]); ax.set_xticklabels(["pooled", "within-\nspecies"], fontsize=5.5)
    ax.set_ylabel("$\\Delta$ auROC (balanced $-$ baseline)", fontsize=6)
    ax.set_title("D  Prevalence balancing", fontsize=7, fontweight="bold", loc="left")
    ax.tick_params(labelsize=5)


def panel_E(ax, R):
    fc = os.path.join(R, "ceiling", "ceiling_summary.csv")
    fd = os.path.join(R, "decomposition_summary.csv")
    if not (os.path.exists(fc) and os.path.exists(fd)):
        ax.set_axis_off(); return
    c = pd.read_csv(fc); c = c[c.organ == "all"]
    d = pd.read_csv(fd)
    x = np.arange(len(STRESS_ORDER))
    ceil = [c[c.stress == s]["within_mean"].mean() for s in STRESS_ORDER]
    ceil_sd = [c[c.stress == s]["within_sd"].mean() for s in STRESS_ORDER]
    mod = [d[(d.stress == s) & (d.model == "Fused")]["within_species_auroc_weighted"].mean()
           for s in STRESS_ORDER]
    ax.bar(x, ceil, 0.6, yerr=ceil_sd, color="#DDDDDD", edgecolor="black",
           linewidth=0.4, error_kw=dict(elinewidth=0.5, capsize=1.5), label="label ceiling")
    ax.plot(x, mod, "o-", color=C_FUSED, ms=3.5, lw=1.0, label="fused model")
    for i, (m, cc) in enumerate(zip(mod, ceil)):
        if np.isfinite(m) and np.isfinite(cc) and cc > 0.5:
            ax.text(i + 0.30, m, f"{100*(m-0.5)/(cc-0.5):.0f}%", ha="left",
                    va="center", fontsize=4.5)
    ax.set_xticks(x); ax.set_xticklabels(STRESS_ORDER, fontsize=5, rotation=20)
    ax.set_ylim(0.5, 0.95)
    ax.set_ylabel("Within-species auROC", fontsize=6)
    ax.set_title("E  Achieved vs attainable", fontsize=7, fontweight="bold", loc="left")
    ax.legend(fontsize=4.5, loc="lower right", framealpha=0.8)
    ax.tick_params(labelsize=5)


def panel_F(ax, R):
    f = None
    for cand in ["tf_binding_enrichment_top2000.csv", "tf_binding_enrichment_all_peaks.csv",
                 "tf_binding_enrichment.csv"]:
        p = os.path.join(R, "tfbinding", cand)
        if os.path.exists(p):
            f = p; break
    if f is None:
        ax.set_axis_off(); return
    d = pd.read_csv(f)
    d = d[d.label_tier == "held_out"]
    groups = [("stress", C_STRESS, "stress TFs"), ("developmental", C_DEV, "developmental TFs")]
    data, colors, labels = [], [], []
    for cls, col, lab in groups:
        v = d[d.tf_class == cls]["log2_or"].dropna().values
        if len(v):
            data.append(v); colors.append(col); labels.append(f"{lab}\n(n={len(v)})")
    if not data:
        ax.set_axis_off(); return
    bp = ax.boxplot(data, widths=0.55, patch_artist=True, showfliers=False)
    for patch, col in zip(bp["boxes"], colors):
        patch.set_facecolor(col); patch.set_alpha(0.8); patch.set_linewidth(0.5)
    for med in bp["medians"]:
        med.set_color("black"); med.set_linewidth(1.0)
    ax.axhline(0, color="grey", ls="--", lw=0.6)
    ax.set_xticklabels(labels, fontsize=5)
    ax.set_ylabel("log$_2$ odds ratio", fontsize=6)
    ax.set_title("F  ChIP-seq binding enrichment", fontsize=7, fontweight="bold", loc="left")
    if len(data) == 2:
        from scipy.stats import mannwhitneyu
        _, p = mannwhitneyu(data[0], data[1], alternative="greater")
        ax.text(0.97, 0.95, f"P={p:.1e}", transform=ax.transAxes, ha="right",
                va="top", fontsize=5)
    ax.tick_params(labelsize=5)


def panel_G(ax, R):
    base = os.path.join(R, "sweep_organ")
    if not os.path.isdir(base):
        ax.set_axis_off(); return
    rows = []
    for scheme in ["all_organ_rebuild", "Leaf", "Root"]:
        p = os.path.join(base, scheme, "sweep_results.csv")
        if not os.path.exists(p):
            continue
        d = pd.read_csv(p)
        d = d[d.arm == "baseline"]
        for _, r in d.iterrows():
            rows.append(dict(scheme=scheme, stress=r["stress"], direction=r["direction"],
                             within=r["within_species_auroc"]))
    if not rows:
        ax.set_axis_off(); return
    df = pd.DataFrame(rows)
    schemes = [s for s in ["all_organ_rebuild", "Leaf", "Root"] if s in set(df.scheme)]
    names = {"all_organ_rebuild": "all organs", "Leaf": "Leaf only", "Root": "Root only"}
    cols = {"all_organ_rebuild": "#BBBBBB", "Leaf": "#009988", "Root": "#EE7733"}
    x = np.arange(len(STRESS_ORDER)); w = 0.8 / max(len(schemes), 1)
    for i, s in enumerate(schemes):
        vals = [df[(df.scheme == s) & (df.stress == st)]["within"].mean()
                for st in STRESS_ORDER]
        ax.bar(x + i*w - 0.4 + w/2, vals, w*0.9, color=cols[s], alpha=0.9,
               label=names[s])
    ax.set_xticks(x); ax.set_xticklabels(STRESS_ORDER, fontsize=5, rotation=20)
    ax.set_ylim(0.55, 0.80)
    ax.set_ylabel("Within-species auROC", fontsize=6)
    ax.set_title("G  Organ-restricted labels", fontsize=7, fontweight="bold", loc="left")
    ax.legend(fontsize=4.5, loc="upper left", framealpha=0.8, ncol=1)
    ax.tick_params(labelsize=5)


def main():
    ap = argparse.ArgumentParser(description="Assemble Supplementary Figure S11")
    ap.add_argument("--revision_dir", required=True)
    ap.add_argument("--output_dir", required=True)
    args = ap.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    R = args.revision_dir

    fig = plt.figure(figsize=(7.2, 8.0))
    gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.42, wspace=0.42,
                           top=0.96, bottom=0.06, left=0.08, right=0.97)
    panel_A(fig.add_subplot(gs[0, 0]), R)
    panel_B(fig.add_subplot(gs[0, 1]), R)
    panel_C(fig.add_subplot(gs[0, 2]), R)
    panel_D(fig.add_subplot(gs[1, 0]), R)
    panel_E(fig.add_subplot(gs[1, 1]), R)
    panel_F(fig.add_subplot(gs[1, 2]), R)
    panel_G(fig.add_subplot(gs[2, 0:2]), R)
    fig.add_subplot(gs[2, 2]).set_axis_off()

    out = os.path.join(args.output_dir, "Figure S11")
    fig.savefig(out + ".png", dpi=300, bbox_inches="tight")
    fig.savefig(out + ".pdf", bbox_inches="tight")
    print(f"written: {out}.png / .pdf")


if __name__ == "__main__":
    main()
