#!/usr/bin/env python3
"""
Phase 0.1 + A0 for the Figure 6 revision (Molecular Plant MOLECULAR-PLANT-D-26-00539).

Answers two questions with NO model retraining:

  0.1  Is the DEG-frequency label's positive rate confounded with how deeply each
       species was assayed?  If species with more experiments carry systematically
       more positives, then a model that merely infers species from sequence
       composition can score above chance without learning stress biology.

  A0   How much of the published pooled auROC is *within-species* discrimination
       (the biology of interest) versus *between-species* prevalence differences?
       Decomposes each published number into pooled / within-species / per-lineage.

Inputs
  --feature_table   feature_table_all_genes.csv.gz  (gene_id, species, ... )
  --dataset_dir     directory of dataset_{stress}_{direction}.pkl
  --fusion_dir      directory containing dna/ and rna/ *_predictions.csv
  --deg_table       (optional) DEG table CSV for per-species experiment counts
  --output_dir      where CSVs are written

Outputs
  per_species_label_rates.csv    stress, direction, species, n, n_pos, pos_rate, (test-only too)
  per_species_auroc.csv          stress, direction, model, species, n_test, pos_rate, auroc
  decomposition_summary.csv      stress, direction, model, pooled, within_weighted,
                                 within_unweighted, gap, n_species_scored
  species_experiment_counts.csv  species, n_rows, n_experiments, n_bioprojects  (if --deg_table)

Reference values for the verification gate (published Figure 6, Heat_UP):
  DNA 0.6777, RNA 0.6856, Fused 0.6995

Usage
  python 17_label_audit_and_decomposition.py \
      --feature_table ".../motif_model_v3_results/feature_table_all_genes.csv.gz" \
      --dataset_dir   ".../plantrna_fm_results" \
      --fusion_dir    ".../fusion" \
      --deg_table     ".../ms/Supplementary Dataset 2.csv" \
      --output_dir    ".../Figure 6/outputs/data_summaries/revision"

Kingdom Stress Atlas, Mutwil lab.
"""

import argparse
import csv
import gzip
import json
import os
import pickle
import sys
from collections import Counter, defaultdict

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

STRESSES = ["Heat", "Cold", "Drought", "Salt", "Pathogen"]
DIRECTIONS = ["UP", "DOWN"]

# Minimum test genes and minimum count of each class for a per-species auROC to be
# considered stable enough to report.
MIN_TEST_GENES = 200
MIN_CLASS_COUNT = 20

# Published pooled auROCs used as the correctness gate (from fusion_results_final.csv).
REFERENCE = {
    ("Heat", "UP"): {"DNA": 0.6777, "RNA": 0.6856, "Fused": 0.6995},
    ("Heat", "DOWN"): {"DNA": 0.7114, "RNA": 0.7176, "Fused": 0.7351},
}
REF_TOL = 0.01


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------
def load_gene_species(feature_table):
    """gene_id -> species. Streams only the first two fields (fast)."""
    g2s = {}
    with gzip.open(feature_table, "rt") as fh:
        header = fh.readline().rstrip("\n").split(",")
        gi, si = header.index("gene_id"), header.index("species")
        assert gi == 0 and si == 1, (
            f"expected gene_id,species as first two columns, got {header[:3]}"
        )
        for line in fh:
            # Only the first two fields are needed; split with a bound for speed.
            parts = line.split(",", 2)
            g2s[parts[0]] = parts[1]
    return g2s


def load_predictions(fusion_dir, stress, direction):
    """Merge DNA and RNA per-gene predictions on gene_id; add a fused column."""
    tag = f"{stress}_{direction}"
    dna_p = os.path.join(fusion_dir, "dna", f"{tag}_predictions.csv")
    rna_p = os.path.join(fusion_dir, "rna", f"{tag}_predictions.csv")
    if not (os.path.exists(dna_p) and os.path.exists(rna_p)):
        return None

    dna = pd.read_csv(dna_p)
    rna = pd.read_csv(rna_p)
    merged = dna.merge(rna, on="gene_id", suffixes=("_dna", "_rna"))

    # Labels must agree between the two files; they were derived from the same split.
    disagree = int((merged["label_dna"].astype(float) != merged["label_rna"].astype(float)).sum())
    merged["label"] = merged["label_dna"].astype(float)
    merged["DNA"] = merged["prediction_dna"].astype(float)
    merged["RNA"] = merged["prediction_rna"].astype(float)
    merged["Fused"] = (merged["DNA"] + merged["RNA"]) / 2.0
    return merged[["gene_id", "label", "DNA", "RNA", "Fused"]], disagree


def safe_auroc(y, s):
    """auROC, or NaN when only one class is present."""
    y = np.asarray(y)
    if len(np.unique(y)) < 2:
        return float("nan")
    return roc_auc_score(y, np.asarray(s))


# ---------------------------------------------------------------------------
# 0.1  Label prevalence audit
# ---------------------------------------------------------------------------
def audit_label_rates(dataset_dir, g2s):
    rows = []
    for stress in STRESSES:
        for direction in DIRECTIONS:
            path = os.path.join(dataset_dir, f"dataset_{stress}_{direction}.pkl")
            if not os.path.exists(path):
                print(f"  [skip] missing {os.path.basename(path)}")
                continue
            with open(path, "rb") as fh:
                splits = pickle.load(fh)

            n_all, pos_all = Counter(), Counter()
            n_test, pos_test = Counter(), Counter()
            for split_name, (gids, labs) in splits.items():
                for g, l in zip(gids, labs):
                    sp = g2s.get(g, "UNMAPPED")
                    n_all[sp] += 1
                    pos_all[sp] += int(l)
                    if split_name == "test":
                        n_test[sp] += 1
                        pos_test[sp] += int(l)

            for sp, n in n_all.items():
                rows.append({
                    "stress": stress,
                    "direction": direction,
                    "species": sp,
                    "n": n,
                    "n_pos": pos_all[sp],
                    "pos_rate": pos_all[sp] / n if n else float("nan"),
                    "n_test": n_test.get(sp, 0),
                    "n_pos_test": pos_test.get(sp, 0),
                    "pos_rate_test": (pos_test[sp] / n_test[sp]) if n_test.get(sp) else float("nan"),
                })
            print(f"  {stress}_{direction}: {len(n_all)} species, {sum(n_all.values())} genes, "
                  f"pos_rate={sum(pos_all.values())/sum(n_all.values()):.3f}")
    return pd.DataFrame(rows)


def count_experiments_per_species(deg_table):
    """Per-species experiment and bioproject counts, to test the depth hypothesis."""
    rows_n = Counter()
    experiments = defaultdict(set)
    bioprojects = defaultdict(set)
    with open(deg_table, newline="") as fh:
        rdr = csv.DictReader(fh)
        cols = rdr.fieldnames
        need = {"species", "experiment", "bioproject"}
        if not need.issubset(cols):
            print(f"  [warn] DEG table lacks {need - set(cols)}; skipping experiment counts")
            return None
        for row in rdr:
            sp = row["species"]
            rows_n[sp] += 1
            experiments[sp].add(row["experiment"])
            bioprojects[sp].add(row["bioproject"])
    return pd.DataFrame([
        {"species": sp,
         "n_deg_rows": rows_n[sp],
         "n_experiments": len(experiments[sp]),
         "n_bioprojects": len(bioprojects[sp])}
        for sp in sorted(rows_n)
    ])


# ---------------------------------------------------------------------------
# A0  auROC decomposition
# ---------------------------------------------------------------------------
def decompose(fusion_dir, g2s, lineage_map=None):
    per_species_rows = []
    summary_rows = []
    gate_failures = []

    for stress in STRESSES:
        for direction in DIRECTIONS:
            loaded = load_predictions(fusion_dir, stress, direction)
            if loaded is None:
                print(f"  [skip] no predictions for {stress}_{direction}")
                continue
            preds, disagree = loaded
            preds["species"] = preds["gene_id"].map(g2s).fillna("UNMAPPED")
            if disagree:
                print(f"  [warn] {stress}_{direction}: {disagree} label mismatches DNA vs RNA")

            for model in ["DNA", "RNA", "Fused"]:
                pooled = safe_auroc(preds["label"], preds[model])

                # Verification gate against the published numbers.
                ref = REFERENCE.get((stress, direction), {}).get(model)
                if ref is not None:
                    delta = abs(pooled - ref)
                    status = "OK" if delta <= REF_TOL else "MISMATCH"
                    print(f"    gate {stress}_{direction} {model}: "
                          f"pooled={pooled:.4f} published={ref:.4f} d={delta:.4f} [{status}]")
                    if delta > REF_TOL:
                        gate_failures.append((stress, direction, model, pooled, ref))

                # Per-species auROC.
                aurocs, weights = [], []
                for sp, grp in preds.groupby("species"):
                    n = len(grp)
                    n_pos = int(grp["label"].sum())
                    n_neg = n - n_pos
                    au = float("nan")
                    if n >= MIN_TEST_GENES and n_pos >= MIN_CLASS_COUNT and n_neg >= MIN_CLASS_COUNT:
                        au = safe_auroc(grp["label"], grp[model])
                        aurocs.append(au)
                        weights.append(n)
                    per_species_rows.append({
                        "stress": stress, "direction": direction, "model": model,
                        "species": sp, "n_test": n, "n_pos": n_pos,
                        "pos_rate": n_pos / n if n else float("nan"),
                        "auroc": au,
                        "scored": bool(np.isfinite(au)),
                    })

                if aurocs:
                    w = np.asarray(weights, dtype=float)
                    within_w = float(np.average(aurocs, weights=w))
                    within_u = float(np.mean(aurocs))
                else:
                    within_w = within_u = float("nan")

                summary_rows.append({
                    "stress": stress, "direction": direction, "model": model,
                    "pooled_auroc": pooled,
                    "within_species_auroc_weighted": within_w,
                    "within_species_auroc_unweighted": within_u,
                    "gap_pooled_minus_within_weighted": pooled - within_w,
                    "n_species_scored": len(aurocs),
                    "n_genes": len(preds),
                })

    return (pd.DataFrame(per_species_rows),
            pd.DataFrame(summary_rows),
            gate_failures)


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="Label prevalence audit + auROC decomposition")
    ap.add_argument("--feature_table", required=True)
    ap.add_argument("--dataset_dir", required=True)
    ap.add_argument("--fusion_dir", required=True)
    ap.add_argument("--deg_table", default=None,
                    help="Optional DEG table for per-species experiment counts")
    ap.add_argument("--output_dir", required=True)
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print("[1/4] Loading gene -> species map ...")
    g2s = load_gene_species(args.feature_table)
    print(f"      {len(g2s):,} genes mapped to {len(set(g2s.values()))} species")

    print("[2/4] Auditing per-species label rates ...")
    rates = audit_label_rates(args.dataset_dir, g2s)
    rates.to_csv(os.path.join(args.output_dir, "per_species_label_rates.csv"), index=False)

    print("[3/4] Decomposing auROC (pooled vs within-species) ...")
    per_species, summary, gate_failures = decompose(args.fusion_dir, g2s)
    per_species.to_csv(os.path.join(args.output_dir, "per_species_auroc.csv"), index=False)
    summary.to_csv(os.path.join(args.output_dir, "decomposition_summary.csv"), index=False)

    exp_counts = None
    if args.deg_table:
        print("[4/4] Counting experiments per species (streaming the DEG table) ...")
        exp_counts = count_experiments_per_species(args.deg_table)
        if exp_counts is not None:
            exp_counts.to_csv(
                os.path.join(args.output_dir, "species_experiment_counts.csv"), index=False)
    else:
        print("[4/4] --deg_table not given; skipping experiment counts")

    # ---- Console report -------------------------------------------------
    print("\n" + "=" * 78)
    print("PREVALENCE SPREAD PER STRESS-DIRECTION")
    print("=" * 78)
    for (s, d), grp in rates.groupby(["stress", "direction"], sort=False):
        g = grp[grp["n"] >= 500]
        if g.empty:
            continue
        lo, hi = g["pos_rate"].min(), g["pos_rate"].max()
        print(f"{s:9s} {d:4s}  species={len(g):3d}  overall={grp['n_pos'].sum()/grp['n'].sum():.3f}  "
              f"min={lo:.3f} ({g.loc[g['pos_rate'].idxmin(),'species'][:22]})  "
              f"max={hi:.3f} ({g.loc[g['pos_rate'].idxmax(),'species'][:22]})  "
              f"spread={hi/lo if lo > 0 else float('inf'):.1f}x")

    if exp_counts is not None:
        merged = (rates.groupby("species", as_index=False)
                       .apply(lambda x: pd.Series({"mean_pos_rate": x["n_pos"].sum() / x["n"].sum()}),
                              include_groups=False)
                       .merge(exp_counts, on="species", how="inner"))
        if len(merged) >= 4:
            from scipy.stats import spearmanr
            rho, p = spearmanr(merged["n_experiments"], merged["mean_pos_rate"])
            print(f"\nDEPTH HYPOTHESIS: Spearman(n_experiments, mean pos_rate) "
                  f"rho={rho:.3f}  P={p:.3g}  (n={len(merged)} species)")
            merged.to_csv(os.path.join(args.output_dir, "depth_vs_prevalence.csv"), index=False)

    print("\n" + "=" * 78)
    print("auROC DECOMPOSITION  (pooled vs within-species)")
    print("=" * 78)
    print(f"{'stress':9s} {'dir':5s} {'model':6s} {'pooled':>7s} {'within_w':>9s} "
          f"{'within_u':>9s} {'gap':>7s} {'nsp':>4s}")
    for _, r in summary.iterrows():
        print(f"{r['stress']:9s} {r['direction']:5s} {r['model']:6s} "
              f"{r['pooled_auroc']:7.4f} {r['within_species_auroc_weighted']:9.4f} "
              f"{r['within_species_auroc_unweighted']:9.4f} "
              f"{r['gap_pooled_minus_within_weighted']:7.4f} {int(r['n_species_scored']):4d}")

    print("\nMEAN ACROSS ALL STRESS-DIRECTIONS")
    for model, grp in summary.groupby("model"):
        print(f"  {model:6s} pooled={grp['pooled_auroc'].mean():.4f}  "
              f"within_species={grp['within_species_auroc_weighted'].mean():.4f}  "
              f"gap={grp['gap_pooled_minus_within_weighted'].mean():+.4f}")

    if gate_failures:
        print("\n[GATE] pooled auROC did NOT reproduce published values:")
        for s, d, m, got, ref in gate_failures:
            print(f"  {s}_{d} {m}: got {got:.4f}, published {ref:.4f}")
        print("  Investigate before trusting the decomposition.")
    else:
        print("\n[GATE] pooled auROC reproduces all checked published values.")

    print(f"\nWritten to {args.output_dir}")


if __name__ == "__main__":
    main()
