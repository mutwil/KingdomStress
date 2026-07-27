#!/usr/bin/env python3
"""
Phase B for the Figure 6 revision: how much of the label is reproducible at all?

The reviewer reads auROC < 0.8 as a deficiency. The labels are DEG-frequency
quartiles aggregated over hundreds of heterogeneous experiments, so part of the
gap to 1.0 is label noise that no sequence model can cross. This measures it.

Design (split-half reliability over bioprojects)
  For each stress x direction, repeatedly:
    1. split the bioprojects into two disjoint halves A and B
    2. recompute DEG frequency independently within each half, per species,
       exactly as the original recipe does
    3. label half B by the same quartile rule
    4. score auROC of half-A's continuous frequency predicting half-B's binary label
  The mean over splits is an empirical ceiling: the best a *perfect* predictor of
  the underlying signal could score against labels this noisy.

What it does and does not bound
  It bounds agreement with THESE labels, derived this way, at this experiment depth.
  It is not a bound on the underlying biology, and a model could in principle exceed
  it by exploiting signal shared between halves that the frequency statistic misses.
  Reported as a reference line, not a hard limit.

Also computed
  - per-organ ceilings (Leaf vs Root vs all), linking Phase B to A2
  - per-species ceilings, linking Phase B to the assay-depth finding in Phase 0.1

Usage
  python 20_label_noise_ceiling.py \
      --deg_table ".../deg_v3.csv" \
      --feature_table ".../feature_table.csv.gz" \
      --id_mapping_dir ".../id_mappings" \
      --dataset_dir ".../plantrna_fm_results" \
      --output_dir ".../revision/ceiling" \
      --n_splits 20

Kingdom Stress Atlas, Mutwil lab.
"""

import argparse
import gzip
import os
import pickle
import re
import sys
from collections import defaultdict

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

STRESSES = ["Heat", "Cold", "Drought", "Salt", "Pathogen"]
DIRECTIONS = ["UP", "DOWN"]
SEED = 42
MIN_EVAL = 200          # min genes to score a species
MIN_CLASS = 20
MIN_BIOPROJ = 6         # need enough bioprojects to split meaningfully

ORGAN_MAP = {
    "Leaf": "Leaf", "Rosette": "Leaf", "Leaf Sheath": "Leaf",
    "Root": "Root", "Root tips": "Root", "Root Cortex": "Root",
    "Root Stele": "Root", "Root Rhizodermis": "Root", "Root Endodermis": "Root",
    "Seedling": "Seedling", "seedling": "Seedling",
    "Shoot": "Shoot", "Shoot tip": "Shoot",
}


def strip_transcript_suffix(g):
    g = re.sub(r"_T\d+$", "", g)
    return re.sub(r"\.\d+$", "", g)


def load_species_genes(feature_table):
    s2g = defaultdict(set)
    with gzip.open(feature_table, "rt") as fh:
        fh.readline()
        for line in fh:
            i = line.index(",")
            j = line.index(",", i + 1)
            s2g[line[i + 1:j]].add(line[:i])
    return s2g


def load_eval_genes(dataset_dir):
    """(stress, direction) -> set of test-split genes, for comparability with model auROC."""
    out = {}
    for s in STRESSES:
        for d in DIRECTIONS:
            p = os.path.join(dataset_dir, f"dataset_{s}_{d}.pkl")
            if os.path.exists(p):
                with open(p, "rb") as fh:
                    sp = pickle.load(fh)
                out[(s, d)] = set(sp["test"][0])
    return out


def load_id_mappings(d):
    m = {}
    for fn in sorted(os.listdir(d)):
        if fn.endswith(".tsv") and " 2.tsv" not in fn:
            df = pd.read_csv(os.path.join(d, fn), sep="\t")
            if {"deg_id", "gtf_id"}.issubset(df.columns):
                m[fn[:-4]] = dict(zip(df["deg_id"], df["gtf_id"]))
    return m


def load_deg(deg_table, id_mappings):
    cols = ["species", "stress", "bioproject", "organ", "experiment", "direction", "gene"]
    print(f"  reading {deg_table} ...")
    df = pd.read_csv(deg_table, usecols=cols)
    df = df[df["stress"].isin(STRESSES)]
    print(f"  {len(df):,} rows across the five focal stresses")
    df["organ_harmonised"] = df["organ"].map(ORGAN_MAP).fillna("Other")
    df["gene_locus"] = df["gene"].map(strip_transcript_suffix)
    for species, mapping in id_mappings.items():
        m = df["species"] == species
        if m.any():
            df.loc[m, "gene_locus"] = df.loc[m, "gene_locus"].map(
                lambda x: mapping.get(x, x))
    return df


def freq_for(sub, all_genes, direction):
    """DEG frequency over all_genes using only the rows given."""
    n_exp = sub["experiment"].nunique()
    if n_exp < 1:
        return None
    d = sub[sub["direction"] == direction]
    if len(d) == 0:
        return None
    counts = d.groupby("gene_locus")["experiment"].nunique()
    freq = pd.Series(0.0, index=sorted(all_genes), dtype=float)
    common = counts.index.intersection(freq.index)
    freq.loc[common] = (counts.loc[common] / n_exp).astype(float)
    return freq


def quartile_labels(freq):
    p25, p75 = np.percentile(freq.values, 25), np.percentile(freq.values, 75)
    if p25 == p75:
        return None
    lab = pd.Series(np.nan, index=freq.index)
    lab[freq <= p25] = 0.0
    lab[freq >= p75] = 1.0
    return lab.dropna()


def ceiling_for(sub_stress, s2g, eval_genes, direction, rng, n_splits, tag):
    """Split-half reliability for one stress x direction (optionally organ-filtered)."""
    bioprojects = sub_stress["bioproject"].unique()
    if len(bioprojects) < MIN_BIOPROJ:
        return []

    rows = []
    for it in range(n_splits):
        shuffled = rng.permutation(bioprojects)
        half = len(shuffled) // 2
        A, B = set(shuffled[:half]), set(shuffled[half:])
        subA = sub_stress[sub_stress["bioproject"].isin(A)]
        subB = sub_stress[sub_stress["bioproject"].isin(B)]

        aur, wts, pooled_s, pooled_y = [], [], [], []
        for species, genes in s2g.items():
            sa = subA[subA["species"] == species]
            sb = subB[subB["species"] == species]
            if len(sa) == 0 or len(sb) == 0:
                continue
            fa = freq_for(sa, genes, direction)
            fb = freq_for(sb, genes, direction)
            if fa is None or fb is None:
                continue
            lb = quartile_labels(fb)
            if lb is None or lb.empty:
                continue
            common = lb.index
            if eval_genes is not None:
                common = common.intersection(pd.Index(list(eval_genes)))
            if len(common) < MIN_EVAL:
                continue
            y = lb.loc[common].values
            s = fa.loc[common].values
            npos = int(y.sum())
            if npos < MIN_CLASS or (len(y) - npos) < MIN_CLASS:
                continue
            aur.append(roc_auc_score(y, s))
            wts.append(len(y))
            pooled_s.append(s)
            pooled_y.append(y)

        if not aur:
            continue
        py = np.concatenate(pooled_y)
        ps = np.concatenate(pooled_s)
        rows.append({
            "tag": tag, "split": it,
            "n_bioprojects": len(bioprojects),
            "within_species_ceiling": float(np.average(aur, weights=wts)),
            "pooled_ceiling": float(roc_auc_score(py, ps)) if len(np.unique(py)) > 1 else np.nan,
            "n_species": len(aur), "n_genes": int(np.sum(wts)),
        })
    return rows


def main():
    ap = argparse.ArgumentParser(description="Label-noise ceiling via bioproject split-half")
    ap.add_argument("--deg_table", required=True)
    ap.add_argument("--feature_table", required=True)
    ap.add_argument("--id_mapping_dir", required=True)
    ap.add_argument("--dataset_dir", required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--n_splits", type=int, default=20)
    ap.add_argument("--organs", default="all,Leaf,Root")
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    rng = np.random.RandomState(SEED)

    print("[1/4] Gene universe ...")
    s2g = load_species_genes(args.feature_table)
    print(f"      {len(s2g)} species")

    print("[2/4] Test-split genes ...")
    eval_sets = load_eval_genes(args.dataset_dir)

    print("[3/4] DEG table ...")
    deg = load_deg(args.deg_table, load_id_mappings(args.id_mapping_dir))

    print("[4/4] Split-half reliability ...")
    organs = [o.strip() for o in args.organs.split(",") if o.strip()]
    all_rows = []
    for stress in STRESSES:
        base = deg[deg["stress"] == stress]
        for organ in organs:
            sub = base if organ == "all" else base[base["organ_harmonised"] == organ]
            if len(sub) == 0:
                continue
            for direction in DIRECTIONS:
                tag = f"{stress}_{direction}_{organ}"
                ev = eval_sets.get((stress, direction)) if organ == "all" else None
                rows = ceiling_for(sub, s2g, ev, direction, rng, args.n_splits, tag)
                for r in rows:
                    r.update({"stress": stress, "direction": direction, "organ": organ})
                all_rows.extend(rows)
                if rows:
                    w = np.mean([r["within_species_ceiling"] for r in rows])
                    p = np.nanmean([r["pooled_ceiling"] for r in rows])
                    print(f"  {stress:9s} {direction:4s} {organ:6s} "
                          f"within={w:.4f} pooled={p:.4f} "
                          f"({len(rows)} splits, {rows[0]['n_bioprojects']} bioprojects)")

    if not all_rows:
        print("no ceilings computed")
        return
    df = pd.DataFrame(all_rows)
    df.to_csv(os.path.join(args.output_dir, "ceiling_splits.csv"), index=False)

    summ = (df.groupby(["stress", "direction", "organ"])
              .agg(within_mean=("within_species_ceiling", "mean"),
                   within_sd=("within_species_ceiling", "std"),
                   pooled_mean=("pooled_ceiling", "mean"),
                   pooled_sd=("pooled_ceiling", "std"),
                   n_splits=("split", "count"),
                   n_bioprojects=("n_bioprojects", "first"))
              .reset_index())
    summ.to_csv(os.path.join(args.output_dir, "ceiling_summary.csv"), index=False)

    print("\n" + "=" * 78)
    print("LABEL-NOISE CEILING")
    print("=" * 78)
    print(summ.round(4).to_string(index=False))
    for organ, g in summ.groupby("organ"):
        print(f"\n  organ={organ:6s} mean within-species ceiling = {g['within_mean'].mean():.4f}"
              f"   mean pooled ceiling = {g['pooled_mean'].mean():.4f}")
    print(f"\nWritten to {args.output_dir}")


if __name__ == "__main__":
    main()
