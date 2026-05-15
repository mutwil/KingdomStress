#!/usr/bin/env python3
"""
Interpretable feature-based models for stress-responsive gene prediction.
Replication script for Figure 5B (right side: DinucFreq, Motif, Structure, Combined).

Trains gradient-boosted machines (GBMs) on four feature groups:
  1. DinucFreq  -- 16 dinucleotide frequencies (prom + mRNA) + GC content = 34 features
  2. Motif      -- 16 known + ~30 discovered cis-element counts (prom + mRNA) = ~92 features
  3. Structure  -- RNA secondary structure descriptors from ViennaRNA = 8 features
  4. Combined   -- all of the above = ~134 features

Input:
  - feature_table_all_genes.csv.gz  (precomputed features for 1.7M genes)
  - dataset_{stress}_{direction}.pkl (orthogroup-based train/val/test splits)

Output:
  - ablation_results.csv  (auROC per stress x direction x model)
  - feature_importance_{stress}_{direction}.csv  (GBM Gini importance for Combined model)

Usage:
  python 07c_interpretable_model_replication.py --feature_table <path> --dataset_dir <path> --output_dir <path>

Reference:
  Mutwil et al., Kingdom Stress Atlas, 2026.
"""

import argparse
import pickle
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import roc_auc_score, f1_score
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

STRESSES = ["Heat", "Cold", "Drought", "Salt", "Pathogen"]
DIRECTIONS = ["UP", "DOWN"]

# GBM hyperparameters (same as used in paper)
GBM_PARAMS = dict(
    n_estimators=300,
    max_depth=5,
    learning_rate=0.1,
    min_samples_leaf=10,
    subsample=0.8,
    random_state=RANDOM_SEED,
)

# Feature column definitions
DINUC_COLS = (
    [f"prom_dn_{a}{b}" for a in "ACGT" for b in "ACGT"]
    + [f"mrna_dn_{a}{b}" for a in "ACGT" for b in "ACGT"]
    + ["prom_gc", "mrna_gc"]
)  # 34 features

KNOWN_MOTIFS = [
    "ABRE", "DRE_CRT", "W_box", "GCC_box", "G_box", "TATA_box", "CAAT_box",
    "MBS", "as1", "MYC", "LTR", "ARE_anaerobic", "ARE_aurich", "PUF_binding",
    "DST_element", "poly_A_signal",
]

STRUCT_COLS = [
    "struct_mfe_per_nt", "struct_frac_paired", "struct_n_stems",
    "struct_n_hairpins", "struct_max_stem_len", "struct_g_quad_count",
    "struct_c_quad_count", "struct_au_rich_runs",
]  # 8 features

MAX_TRAIN = 50000  # max training genes per model (for speed; balanced afterward)
MAX_TEST = 20000   # max test genes per model


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------
def get_motif_cols(all_cols):
    """Identify motif columns (known + discovered) from the feature table."""
    motif_cols = []
    for col in all_cols:
        # Known motifs: prom_ABRE, mrna_ABRE, etc.
        for m in KNOWN_MOTIFS:
            if col == f"prom_{m}" or col == f"mrna_{m}":
                motif_cols.append(col)
        # Discovered motifs: prom_cnn_km_*, mrna_cnn_km_*, prom_dna_km_*, etc.
        if "_km_" in col:
            motif_cols.append(col)
    return sorted(set(motif_cols))


def balance_classes(y, seed=RANDOM_SEED):
    """Undersample majority class. Returns indices."""
    rng = np.random.RandomState(seed)
    pos = np.where(y == 1)[0]
    neg = np.where(y == 0)[0]
    n_min = min(len(pos), len(neg))
    sel = np.concatenate([
        rng.choice(pos, n_min, replace=False),
        rng.choice(neg, n_min, replace=False),
    ])
    return sel


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Train interpretable models (Figure 5B)")
    parser.add_argument("--feature_table", type=str, required=True,
                        help="Path to feature_table_all_genes.csv.gz")
    parser.add_argument("--dataset_dir", type=str, required=True,
                        help="Directory containing dataset_{stress}_{direction}.pkl files")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Output directory for results")
    args = parser.parse_args()

    feature_table_path = Path(args.feature_table)
    dataset_dir = Path(args.dataset_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ---- Load feature table ----
    print(f"Loading feature table from {feature_table_path} ...")
    ft = pd.read_csv(feature_table_path, index_col="gene_id")
    print(f"  {ft.shape[0]} genes x {ft.shape[1]} columns")

    # Identify feature groups
    all_cols = ft.columns.tolist()
    dinuc_cols = [c for c in DINUC_COLS if c in all_cols]
    motif_cols = get_motif_cols(all_cols)
    struct_cols = [c for c in STRUCT_COLS if c in all_cols]

    print(f"  DinucFreq:  {len(dinuc_cols)} features")
    print(f"  Motif:      {len(motif_cols)} features")
    print(f"  Structure:  {len(struct_cols)} features")
    print(f"  Combined:   {len(dinuc_cols) + len(motif_cols) + len(struct_cols)} features")
    print()

    # ---- Train models ----
    all_results = []

    for stress in STRESSES:
        for direction in DIRECTIONS:
            tag = f"{stress}_{direction}"
            dataset_file = dataset_dir / f"dataset_{tag}.pkl"

            if not dataset_file.exists():
                print(f"  SKIP {tag}: dataset not found")
                continue

            print(f"{'='*60}")
            print(f"{tag}")

            with open(dataset_file, "rb") as f:
                splits = pickle.load(f)

            # Get gene IDs and labels for train and test
            train_gids, train_labs = splits["train"]
            test_gids, test_labs = splits["test"]

            # Filter to genes present in feature table
            train_mask = [g in ft.index for g in train_gids]
            test_mask = [g in ft.index for g in test_gids]

            train_gids_filt = [g for g, m in zip(train_gids, train_mask) if m][:MAX_TRAIN]
            train_labs_filt = [l for l, m in zip(train_labs, train_mask) if m][:MAX_TRAIN]
            test_gids_filt = [g for g, m in zip(test_gids, test_mask) if m][:MAX_TEST]
            test_labs_filt = [l for l, m in zip(test_labs, test_mask) if m][:MAX_TEST]

            y_train = np.array(train_labs_filt, dtype=np.float32)
            y_test = np.array(test_labs_filt, dtype=np.float32)

            print(f"  Train: {len(y_train)} genes ({y_train.sum():.0f} pos, {(1-y_train).sum():.0f} neg)")
            print(f"  Test:  {len(y_test)} genes ({y_test.sum():.0f} pos, {(1-y_test).sum():.0f} neg)")

            # Balance training set
            sel = balance_classes(y_train)
            y_train_bal = y_train[sel]

            # Define feature groups
            feature_groups = {
                "DinucFreq": dinuc_cols,
                "Motif": motif_cols,
                "Structure": struct_cols,
                "Combined": dinuc_cols + motif_cols + struct_cols,
            }

            for group_name, cols in feature_groups.items():
                # Extract features
                X_train_full = ft.loc[train_gids_filt, cols].fillna(0).values
                X_test = ft.loc[test_gids_filt, cols].fillna(0).values

                X_train_bal = X_train_full[sel]

                # Standardize
                scaler = StandardScaler()
                X_tr = scaler.fit_transform(X_train_bal)
                X_te = scaler.transform(X_test)

                # Train GBM
                gbm = GradientBoostingClassifier(**GBM_PARAMS)
                gbm.fit(X_tr, y_train_bal)

                # Evaluate
                probs = gbm.predict_proba(X_te)[:, 1]
                auroc = roc_auc_score(y_test, probs)
                preds = (probs >= 0.5).astype(int)
                f1 = f1_score(y_test, preds)

                print(f"  {group_name:12s}: auROC={auroc:.4f}  F1={f1:.4f}  ({len(cols)} features)")

                all_results.append({
                    "stress": stress,
                    "direction": direction,
                    "model": group_name,
                    "auROC": auroc,
                    "F1": f1,
                    "n_features": len(cols),
                    "n_train": len(y_train_bal),
                    "n_test": len(y_test),
                })

                # Save feature importance for Combined model
                if group_name == "Combined":
                    imp = pd.Series(gbm.feature_importances_, index=cols)
                    imp = imp.sort_values(ascending=False)
                    imp.to_csv(output_dir / f"feature_importance_{tag}.csv")

    # ---- Save results ----
    results_df = pd.DataFrame(all_results)
    results_path = output_dir / "ablation_results.csv"
    results_df.to_csv(results_path, index=False)

    print(f"\n{'='*60}")
    print("SUMMARY (mean auROC per model):")
    print(results_df.groupby("model")["auROC"].agg(["mean", "std"]).sort_values("mean", ascending=False).to_string())
    print(f"\nResults saved to {results_path}")
    print(f"Feature importances saved to {output_dir}/feature_importance_*.csv")


if __name__ == "__main__":
    main()
