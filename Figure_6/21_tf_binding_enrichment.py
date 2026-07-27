#!/usr/bin/env python3
"""
Phase C2 for the Figure 6 revision - the reviewer's explicit request.

"perform a targeted analysis using existing plant DAP-seq or ChIP-seq data
 (e.g., from PlantRegMap) to check whether genes containing these motifs in their
 promoters are more likely to be stress-responsive in independent datasets from
 the same species."

Design
  Binding evidence is EXPERIMENTAL ChIP-seq (PlantRegMap section 2.5), not
  motif-scan predictions. PlantRegMap also distributes TFBS_from_motif_* files;
  using those would validate our motifs against someone else's motif scan, which
  is circular. They are excluded here.

  PlantRegMap provides 21 Arabidopsis TFs assayed BOTH under ABA treatment and in
  the untreated control. ABA is the master abiotic-stress hormone, so the ABA-vs-
  normal contrast is a built-in specificity control: if binding-based enrichment
  reflects stress regulation rather than generic promoter accessibility, the ABA
  condition should enrich more strongly.

  For each TF x condition:
      bound genes    = genes whose promoter window contains a peak summit
      contingency    = bound/unbound  x  stress-responsive/non-responsive
      test           = Fisher exact, odds ratio, BH-corrected across all tests

Label independence (three tiers, reported separately)
  held_out    labels from the published test split (orthogroup-independent)
  all         all labelled Arabidopsis genes
  (bioproject-holdout labels are produced by a separate script and can be passed
   via --extra_labels to add a fully experiment-independent tier)

Peak-to-gene assignment uses the peak SUMMIT, not the peak span: some PlantRegMap
peaks are >10 kb, and span-based assignment would call almost every gene bound.

Usage
  python 21_tf_binding_enrichment.py \
      --bed_dir ".../tfdb/beds" \
      --gtf ".../Arabidopsis_thaliana.TAIR10.59.gtf" \
      --dataset_dir ".../plantrna_fm_results" \
      --output_dir ".../revision/tfbinding" \
      --promoter_up 1000 --promoter_down 200

Kingdom Stress Atlas, Mutwil lab.
"""

import argparse
import os
import pickle
import re
import sys
from collections import defaultdict

import numpy as np
import pandas as pd
from scipy.stats import fisher_exact

STRESSES = ["Heat", "Cold", "Drought", "Salt", "Pathogen"]
DIRECTIONS = ["UP", "DOWN"]

# Stress relevance of the assayed TFs, for interpretation only (never for filtering).
TF_ANNOTATION = {
    "AT1G49720": "ABF1 (ABA bZIP)",
    "AT3G19290": "ABF4/AREB2 (ABA bZIP)",
    "AT4G34000": "ABF3 (ABA bZIP)",
    "AT5G05410": "DREB2A (drought/heat AP2)",
    "AT5G43840": "HSFA6A (heat shock factor)",
    "AT4G27410": "RD26 (dehydration NAC)",
    "AT2G46680": "ATHB7 (drought HD-Zip)",
    "AT2G22430": "ATHB6 (ABA HD-Zip)",
    "AT5G65310": "ATHB5 (HD-Zip)",
    "AT2G46270": "GBF3 (G-box bZIP)",
    "AT4G01120": "GBF2 (G-box bZIP)",
    "AT5G04340": "ZAT6 (stress C2H2)",
    "AT1G77450": "NAC032",
    "AT5G63790": "NAC102",
    "AT5G67300": "MYB44",
    "AT1G22640": "MYB3",
    "AT5G47640": "NF-YB2",
    "AT4G37790": "HAT22",
    "AT1G51140": "bHLH",
    "AT5G04760": "MYB-like",
    "AT1G56170": "NF-YC2",
}

# Specificity control. If enrichment reflects stress regulation rather than generic
# promoter accessibility, stress-associated TFs should enrich more than TFs whose
# primary role is developmental or circadian.
STRESS_TFS = {
    "AT1G49720", "AT3G19290", "AT4G34000",   # ABF1, ABF4/AREB2, ABF3
    "AT5G05410", "AT5G43840",                 # DREB2A, HSFA6A
    "AT4G27410", "AT2G46680", "AT2G22430",    # RD26, ATHB7, ATHB6
    "AT5G04340", "AT1G77450", "AT5G63790",    # ZAT6, NAC032, NAC102
    "AT5G67300",                              # MYB44
}
DEVELOPMENTAL_TFS = {
    "AT1G24260",  # SEP3
    "AT4G18960",  # AG
    "AT5G61850",  # LFY
    "AT2G01570",  # RGA
    "AT2G46830",  # CCA1 (circadian)
    "AT5G17430",  # BBM
    "AT3G48430",  # REF6
    "AT1G49950",  # TRB1
}


def parse_gtf_tss(gtf_path):
    """gene_id -> (chrom, tss, strand) for protein-coding genes, from an Ensembl GTF."""
    tss = {}
    gid_re = re.compile(r'gene_id "([^"]+)"')
    bt_re = re.compile(r'gene_biotype "([^"]+)"')
    with open(gtf_path) as fh:
        for line in fh:
            if line.startswith("#"):
                continue
            f = line.rstrip("\n").split("\t")
            if len(f) < 9 or f[2] != "gene":
                continue
            bt = bt_re.search(f[8])
            if bt and bt.group(1) != "protein_coding":
                continue
            g = gid_re.search(f[8])
            if not g:
                continue
            chrom, start, end, strand = f[0], int(f[3]), int(f[4]), f[6]
            tss[g.group(1)] = (chrom, start if strand == "+" else end, strand)
    return tss


def build_promoter_index(tss, up, down, binsize=10000):
    """Bin promoter intervals for fast summit lookup."""
    index = defaultdict(list)
    for gid, (chrom, t, strand) in tss.items():
        if strand == "+":
            lo, hi = t - up, t + down
        else:
            lo, hi = t - down, t + up
        for b in range(lo // binsize, hi // binsize + 1):
            index[(chrom, b)].append((lo, hi, gid))
    return index, binsize


def norm_chrom(c):
    """PlantRegMap uses Chr1..Chr5/ChrC/ChrM; Ensembl GTF uses 1..5/Pt/Mt."""
    c = c.strip()
    m = re.match(r"^(?:Chr|chr)?(\w+)$", c)
    if not m:
        return c
    v = m.group(1)
    return {"C": "Pt", "M": "Mt"}.get(v, v)


def genes_bound(bed_path, index, binsize, top_peaks=0, max_width=0):
    """Genes whose promoter window contains a peak summit.

    top_peaks : keep only the strongest N peaks by signalValue (column 7). Peak sets
                here cover ~30% of the genome, so a high-confidence restriction is
                needed to test whether enrichment is binding-driven rather than a
                generic open-promoter effect.
    max_width : drop peaks wider than this (some PlantRegMap peaks exceed 10 kb).
    """
    peaks = []
    with open(bed_path) as fh:
        for line in fh:
            f = line.rstrip("\n").split("\t")
            if len(f) < 3:
                continue
            start, end = int(f[1]), int(f[2])
            if max_width and (end - start) > max_width:
                continue
            summit = start + int(f[9]) if len(f) >= 10 and f[9].lstrip("-").isdigit() \
                else (start + end) // 2
            try:
                signal = float(f[6]) if len(f) >= 7 else 0.0
            except ValueError:
                signal = 0.0
            peaks.append((signal, norm_chrom(f[0]), summit))

    n_total = len(peaks)
    if top_peaks and n_total > top_peaks:
        peaks.sort(key=lambda x: -x[0])
        peaks = peaks[:top_peaks]

    bound = set()
    for _, chrom, summit in peaks:
        for lo, hi, gid in index.get((chrom, summit // binsize), ()):
            if lo <= summit <= hi:
                bound.add(gid)
    return bound, n_total


def load_labels(dataset_dir, ath_prefix="AT"):
    """(stress, direction) -> {'all': {g: y}, 'held_out': {g: y}} for Arabidopsis."""
    out = {}
    for s in STRESSES:
        for d in DIRECTIONS:
            p = os.path.join(dataset_dir, f"dataset_{s}_{d}.pkl")
            if not os.path.exists(p):
                continue
            with open(p, "rb") as fh:
                splits = pickle.load(fh)
            all_l, held = {}, {}
            for split, (gids, labs) in splits.items():
                for g, y in zip(gids, labs):
                    if not g.startswith(ath_prefix):
                        continue
                    all_l[g] = int(y)
                    if split == "test":
                        held[g] = int(y)
            out[(s, d)] = {"all": all_l, "held_out": held}
    return out


def enrich(bound, labels):
    """Fisher exact for bound vs unbound against responsive vs non-responsive."""
    universe = set(labels)
    b = bound & universe
    if len(b) < 20 or len(universe) - len(b) < 20:
        return None
    a11 = sum(labels[g] for g in b)                       # bound & responsive
    a12 = len(b) - a11                                    # bound & non-responsive
    a21 = sum(labels[g] for g in universe) - a11          # unbound & responsive
    a22 = (len(universe) - len(b)) - a21                  # unbound & non-responsive
    if min(a11, a12, a21, a22) < 5:
        return None
    orr, p = fisher_exact([[a11, a12], [a21, a22]], alternative="two-sided")
    return dict(n_bound=len(b), n_universe=len(universe),
                bound_responsive=a11, bound_nonresponsive=a12,
                unbound_responsive=a21, unbound_nonresponsive=a22,
                frac_resp_bound=a11 / len(b),
                frac_resp_unbound=a21 / max(a21 + a22, 1),
                odds_ratio=float(orr), p_value=float(p))


def bh(pvals):
    p = np.asarray(pvals, dtype=float)
    n = len(p)
    order = np.argsort(p)
    q = np.empty(n)
    prev = 1.0
    for rank, i in enumerate(order[::-1]):
        k = n - rank
        prev = min(prev, p[i] * n / k)
        q[i] = prev
    return q


def main():
    ap = argparse.ArgumentParser(description="TF-binding enrichment vs stress labels")
    ap.add_argument("--bed_dir", required=True)
    ap.add_argument("--gtf", required=True)
    ap.add_argument("--dataset_dir", required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--promoter_up", type=int, default=1000)
    ap.add_argument("--promoter_down", type=int, default=200)
    ap.add_argument("--top_peaks", type=int, default=0,
                    help="keep only the N strongest peaks per TF (0 = all)")
    ap.add_argument("--max_peak_width", type=int, default=0,
                    help="drop peaks wider than this many bp (0 = no filter)")
    ap.add_argument("--tag", default="all_peaks",
                    help="suffix for output files, so thresholds do not overwrite")
    ap.add_argument("--extra_labels", default=None,
                    help="optional CSV gene_id,stress,direction,label for an "
                         "experiment-independent tier")
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print("[1/4] TAIR10 gene models ...")
    tss = parse_gtf_tss(args.gtf)
    print(f"      {len(tss):,} protein-coding genes")
    index, binsize = build_promoter_index(tss, args.promoter_up, args.promoter_down)

    print("[2/4] Peak -> promoter assignment ...")
    beds = sorted(f for f in os.listdir(args.bed_dir) if f.endswith(".bed"))
    bound_sets, meta = {}, []
    for b in beds:
        m = re.match(r"TFBS_(AT[0-9A-Za-z]+)_Ath_(.+)\.bed$", b)
        if not m:
            continue
        tf, cond = m.group(1), m.group(2)
        genes, n_peaks = genes_bound(os.path.join(args.bed_dir, b), index, binsize,
                                     args.top_peaks, args.max_peak_width)
        key = (tf, cond)
        bound_sets[key] = genes
        meta.append({"tf": tf, "condition": cond, "n_peaks": n_peaks,
                     "n_bound_genes": len(genes),
                     "annotation": TF_ANNOTATION.get(tf, "")})
        print(f"      {tf} {cond[:38]:38s} peaks={n_peaks:6d} bound={len(genes):6d}")
    pd.DataFrame(meta).to_csv(
        os.path.join(args.output_dir, f"tf_peak_summary_{args.tag}.csv"), index=False)

    print("[3/4] Labels ...")
    labels = load_labels(args.dataset_dir)
    extra = None
    if args.extra_labels and os.path.exists(args.extra_labels):
        ex = pd.read_csv(args.extra_labels)
        extra = {(s, d): dict(zip(g.gene_id, g.label))
                 for (s, d), g in ex.groupby(["stress", "direction"])}
        print(f"      extra label tier loaded for {len(extra)} sets")

    print("[4/4] Enrichment tests ...")
    rows = []
    for (tf, cond), genes in bound_sets.items():
        for (stress, direction), tiers in labels.items():
            todo = {"all": tiers["all"], "held_out": tiers["held_out"]}
            if extra and (stress, direction) in extra:
                todo["bioproject_holdout"] = extra[(stress, direction)]
            for tier, lab in todo.items():
                r = enrich(genes, lab)
                if r is None:
                    continue
                rows.append(dict(tf=tf, annotation=TF_ANNOTATION.get(tf, ""),
                                 condition=cond,
                                 aba="ABA" if "ABA" in cond else "normal",
                                 stress=stress, direction=direction,
                                 label_tier=tier, **r))
    if not rows:
        print("no testable comparisons")
        return

    df = pd.DataFrame(rows)
    df["q_value"] = np.nan
    for tier, g in df.groupby("label_tier"):
        df.loc[g.index, "q_value"] = bh(g["p_value"].values)
    df["log2_or"] = np.log2(df["odds_ratio"].replace(0, np.nan))
    df["tf_class"] = np.where(df["tf"].isin(STRESS_TFS), "stress",
                              np.where(df["tf"].isin(DEVELOPMENTAL_TFS),
                                       "developmental", "other"))
    df.sort_values("q_value").to_csv(
        os.path.join(args.output_dir, f"tf_binding_enrichment_{args.tag}.csv"),
        index=False)

    ho = df[df.label_tier == "held_out"]
    print("\n" + "=" * 78)
    print("HELD-OUT TIER: strongest enrichments")
    print("=" * 78)
    top = ho.sort_values("q_value").head(20)
    for _, r in top.iterrows():
        print(f"  {r.tf} {r.annotation[:26]:26s} {r.aba:6s} {r.stress:9s} {r.direction:4s} "
              f"OR={r.odds_ratio:5.2f} q={r.q_value:.2e} "
              f"({r.frac_resp_bound:.3f} vs {r.frac_resp_unbound:.3f})")

    sig = ho[ho.q_value < 0.05]
    print(f"\nsignificant (q<0.05): {len(sig)}/{len(ho)} tests; "
          f"{(sig.odds_ratio > 1).sum()} enriched, {(sig.odds_ratio < 1).sum()} depleted")

    # ---- specificity control: stress TFs vs developmental TFs ------------
    print("\n" + "=" * 78)
    print(f"SPECIFICITY CONTROL ({args.tag}): stress vs developmental TFs")
    print("=" * 78)
    print(f"  median bound genes per TF: "
          f"{pd.DataFrame(meta)['n_bound_genes'].median():.0f} of "
          f"{ho['n_universe'].max()} labelled Arabidopsis genes")
    cls = ho.groupby("tf_class")["log2_or"].agg(["mean", "median", "count"])
    print(cls.round(3).to_string())
    a = ho[ho.tf_class == "stress"]["log2_or"].dropna()
    b_ = ho[ho.tf_class == "developmental"]["log2_or"].dropna()
    if len(a) > 5 and len(b_) > 5:
        from scipy.stats import mannwhitneyu
        u, pmw = mannwhitneyu(a, b_, alternative="greater")
        print(f"  stress > developmental: Mann-Whitney U P = {pmw:.3g} "
              f"(mean log2OR {a.mean():.3f} vs {b_.mean():.3f})")
        print("  NOTE: if developmental TFs enrich nearly as strongly, the signal is "
              "largely promoter accessibility, not stress-specific regulation.")
    else:
        print("  too few developmental-TF tests for a contrast")

    # ABA vs normal specificity control, paired within TF x stress x direction.
    print("\n" + "=" * 78)
    print("ABA vs NORMAL (paired within TF x stress x direction, held-out tier)")
    print("=" * 78)
    piv = (ho[ho.condition.str.contains("ABA_treatment|^seedling_normal$", regex=True)]
             .pivot_table(index=["tf", "stress", "direction"], columns="aba",
                          values="log2_or"))
    piv = piv.dropna()
    if len(piv):
        piv["aba_minus_normal"] = piv["ABA"] - piv["normal"]
        piv.to_csv(os.path.join(args.output_dir, "aba_vs_normal.csv"))
        from scipy.stats import wilcoxon
        try:
            stat, p = wilcoxon(piv["ABA"], piv["normal"])
            print(f"  n pairs = {len(piv)}, mean delta log2(OR) = "
                  f"{piv['aba_minus_normal'].mean():+.3f}, Wilcoxon P = {p:.3g}")
        except ValueError as e:
            print(f"  Wilcoxon not computable: {e}")
        print(piv.groupby("stress")["aba_minus_normal"].mean().round(3).to_string())
    else:
        print("  no paired ABA/normal comparisons available")

    print(f"\nWritten to {args.output_dir}")


if __name__ == "__main__":
    main()
