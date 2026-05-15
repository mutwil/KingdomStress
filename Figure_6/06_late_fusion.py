#!/usr/bin/env python3
"""
Late fusion of PlantCAD2 (DNA) and PlantRNA-FM (RNA) predictions.
Both prediction CSVs must be aligned (same gene order from same dataset pickle).
"""
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.metrics import roc_auc_score, f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
import json

RESULTS_DIR = Path("/scratch/project_465002754/kingdom_stress/results")
STRESSES = ["Heat", "Cold", "Drought", "Salt", "Pathogen", "Flooding"]
DIRECTIONS = ["UP", "DOWN"]

all_results = []

for stress in STRESSES:
    for direction in DIRECTIONS:
        tag = f"{stress}_{direction}"

        dna_file = RESULTS_DIR / f"{tag}_dna_predictions.csv"
        rna_file = RESULTS_DIR / f"{tag}_predictions.csv"

        if not dna_file.exists() or not rna_file.exists():
            print(f"{tag}: missing predictions")
            continue

        dna = pd.read_csv(dna_file)
        rna = pd.read_csv(rna_file)

        # Merge on gene_id
        merged = pd.merge(dna, rna, on="gene_id", suffixes=("_dna", "_rna"))

        labels = merged["label_dna"].values
        dna_p = merged["prediction_dna"].values
        rna_p = merged["prediction_rna"].values
        n = len(merged)

        dna_auroc = roc_auc_score(labels, dna_p)
        rna_auroc = roc_auc_score(labels, rna_p)
        avg_auroc = roc_auc_score(labels, (dna_p + rna_p) / 2)

        # LR 5-fold CV
        X = np.column_stack([dna_p, rna_p])
        cv_preds = np.zeros(n)
        for tr, te in StratifiedKFold(5, shuffle=True, random_state=42).split(X, labels):
            lr = LogisticRegression()
            lr.fit(X[tr], labels[tr])
            cv_preds[te] = lr.predict_proba(X[te])[:, 1]
        lr_auroc = roc_auc_score(labels, cv_preds)
        corr = np.corrcoef(dna_p, rna_p)[0, 1]

        best_single = max(dna_auroc, rna_auroc)
        gain = lr_auroc - best_single

        print(f"{tag:20s}  DNA={dna_auroc:.4f}  RNA={rna_auroc:.4f}  "
              f"Avg={avg_auroc:.4f}  LR={lr_auroc:.4f}  "
              f"corr={corr:.3f}  gain={gain:+.4f}  n={n}")

        all_results.append({
            "stress": stress, "direction": direction,
            "DNA": dna_auroc, "RNA": rna_auroc,
            "Avg": avg_auroc, "LR": lr_auroc,
            "corr": corr, "gain": gain, "n": n,
        })

if all_results:
    df = pd.DataFrame(all_results)
    print(f"\n{'='*70}")
    print(f"  DNA only:     {df['DNA'].mean():.4f}")
    print(f"  RNA only:     {df['RNA'].mean():.4f}")
    print(f"  Simple avg:   {df['Avg'].mean():.4f}")
    print(f"  LR fusion:    {df['LR'].mean():.4f}")
    print(f"  Mean gain:    {df['gain'].mean():+.4f}")
    print(f"  Correlation:  {df['corr'].mean():.3f}")
    df.to_csv(RESULTS_DIR / "fusion_results.csv", index=False)
