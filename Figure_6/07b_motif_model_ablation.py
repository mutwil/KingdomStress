#!/usr/bin/env python3
"""
Feature group ablation: train GBM on individual feature groups
(dinuc-only, motif-only, structure-only, combined) to get auROC for each.
Reuses structure_cache.pkl from motif_model_v3.
"""

import pickle, warnings, re
import numpy as np
import pandas as pd
from pathlib import Path
from collections import Counter
from itertools import product
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler

try:
    import RNA
    HAS_VIENNA = True
except ImportError:
    HAS_VIENNA = False

warnings.filterwarnings("ignore")

GENE_SEQS_DIR = Path("/Users/vjx443/Library/CloudStorage/GoogleDrive-mutwil@plant.tools/My Drive/Projects/2026_KingdomStress/CC-Fig 5 data_CNN/gene_seqs")
DATASET_DIR = Path("/Users/vjx443/Library/CloudStorage/GoogleDrive-mutwil@plant.tools/My Drive/Projects/2026_KingdomStress/CC-Fig 5 data_CNN/plantrna_fm_results")
CNN_MOTIF_DIR = Path("/Users/vjx443/Downloads/cnn_results/motifs")
DNA_LLM_MOTIF_DIR = Path("/Users/vjx443/Library/CloudStorage/GoogleDrive-mutwil@plant.tools/My Drive/Projects/2026_KingdomStress/CC-Fig 5 data_CNN/plantcad2_interpretability/motifs")
RNA_MOTIF_DIR = Path("/Users/vjx443/Downloads/rna_interpretability/motifs")
RESULTS_DIR = Path("/Users/vjx443/Downloads/motif_model_v3_results")

STRESSES = ["Heat", "Cold", "Drought", "Salt", "Pathogen"]
DIRECTIONS = ["UP", "DOWN"]
DINUCS = ["".join(p) for p in product("ACGT", repeat=2)]

KNOWN_MOTIFS = {
    "ABRE": "ACGTG", "DRE_CRT": "CCGAC", "W_box": "TTGAC", "GCC_box": "GCCGCC",
    "G_box": "CACGTG", "TATA_box": "TATAAAT", "CAAT_box": "CCAAT", "MBS": "CAACTG",
    "as1": "TGACG", "MYC": "CATGTG", "LTR": "CCGAAA", "ARE_anaerobic": "AAACCA",
    "ARE_aurich": "ATTTA", "PUF_binding": "TGTA", "DST_element": "ATAGAT",
    "poly_A_signal": "AATAAA",
}

def reverse_complement(seq):
    comp = {"A": "T", "T": "A", "C": "G", "G": "C", "N": "N"}
    return "".join(comp.get(c, "N") for c in reversed(seq))

def count_motif(sequence, motif):
    seq = sequence.upper().replace("U", "T")
    motif = motif.upper().replace("U", "T")
    count = seq.count(motif)
    rc = reverse_complement(motif)
    if rc != motif:
        count += seq.count(rc)
    return count

def dinucleotide_freqs(seq):
    seq = seq.upper().replace("U", "T")
    total = max(len(seq) - 1, 1)
    counts = Counter(seq[i:i+2] for i in range(len(seq)-1))
    return {f"dn_{d}": counts.get(d, 0) / total for d in DINUCS}

def get_top_kmers(motif_file, k=8, top_n=10, n_seqs=200):
    if not motif_file.exists():
        return {}
    df = pd.read_csv(motif_file, sep="\t")
    seqs = df["sequence"].head(n_seqs).values
    kmers = Counter()
    for seq in seqs:
        seq = seq.upper().replace("U", "T")
        for i in range(len(seq) - k + 1):
            kmers[seq[i:i+k]] += 1
    return {f"km_{km}": km for km, _ in kmers.most_common(top_n)}

# Load gene sequences
print("Loading gene sequences...")
gene_promoter, gene_mrna = {}, {}
for fa in sorted(GENE_SEQS_DIR.glob("*.fa")):
    sp = fa.stem
    gid, seq_parts = None, []
    with open(fa) as f:
        for line in f:
            if line.startswith(">"):
                if gid and seq_parts:
                    full = "".join(seq_parts)
                    if len(full) >= 6024:
                        gene_promoter[gid] = full[4700:5300]
                        gene_mrna[gid] = full[5000:6024]
                gid = line[1:].split()[0]
                seq_parts = []
            else:
                seq_parts.append(line.strip())
        if gid and seq_parts:
            full = "".join(seq_parts)
            if len(full) >= 6024:
                gene_promoter[gid] = full[4700:5300]
                gene_mrna[gid] = full[5000:6024]
    print(f"  {sp}: done")
print(f"Total: {len(gene_promoter)} genes\n")

# Load structure cache
struct_cache = RESULTS_DIR / "structure_cache.pkl"
with open(struct_cache, "rb") as f:
    gene_struct = pickle.load(f)
print(f"Structure cache: {len(gene_struct)} genes\n")

# Train per stress x direction, per feature group
all_results = []

for stress in STRESSES:
    for direction in DIRECTIONS:
        tag = f"{stress}_{direction}"
        dataset_file = DATASET_DIR / f"dataset_{tag}.pkl"
        if not dataset_file.exists():
            continue

        print(f"\n{'='*60}")
        print(f"{tag}")

        # Build motif set
        motifs = dict(KNOWN_MOTIFS)
        for prefix, mdir in [("cnn", CNN_MOTIF_DIR), ("dna", DNA_LLM_MOTIF_DIR), ("rna", RNA_MOTIF_DIR)]:
            for name, seq in get_top_kmers(mdir / f"{tag}_motifs.txt").items():
                motifs[f"{prefix}_{name}"] = seq
        seen = {}
        unique_motifs = {}
        for name, seq in motifs.items():
            key = min(seq.upper().replace("U","T"), reverse_complement(seq.upper().replace("U","T")))
            if key not in seen:
                seen[key] = name
                unique_motifs[name] = seq

        with open(dataset_file, "rb") as f:
            splits = pickle.load(f)

        def make_features(gids, labs, max_n=None):
            """Build all feature groups separately."""
            dinuc_rows, motif_rows, struct_rows, comp_rows = [], [], [], []
            y = []
            for gid, lab in zip(gids, labs):
                if gid not in gene_promoter:
                    continue
                prom = gene_promoter[gid]
                mrna = gene_mrna[gid]

                # Dinucleotide features
                df = {}
                for k, v in dinucleotide_freqs(prom).items():
                    df[f"prom_{k}"] = v
                for k, v in dinucleotide_freqs(mrna).items():
                    df[f"mrna_{k}"] = v
                df["prom_gc"] = (prom.count("G") + prom.count("C")) / 600
                df["mrna_gc"] = (mrna.count("G") + mrna.count("C")) / 1024
                dinuc_rows.append(df)

                # Motif features
                mf = {}
                for name, motif in unique_motifs.items():
                    mf[f"prom_{name}"] = count_motif(prom, motif)
                    mf[f"mrna_{name}"] = count_motif(mrna, motif)
                motif_rows.append(mf)

                # Structure features
                sf = {}
                if gid in gene_struct:
                    for k, v in gene_struct[gid].items():
                        sf[f"struct_{k}"] = v
                struct_rows.append(sf)

                y.append(lab)
                if max_n and len(y) >= max_n:
                    break

            return (pd.DataFrame(dinuc_rows), pd.DataFrame(motif_rows),
                    pd.DataFrame(struct_rows), np.array(y, dtype=np.float32))

        train_gids, train_labs = splits["train"]
        test_gids, test_labs = splits["test"]

        din_tr, mot_tr, str_tr, y_train = make_features(train_gids, train_labs, 50000)
        din_te, mot_te, str_te, y_test = make_features(test_gids, test_labs, 20000)

        # Balance training
        pos = np.where(y_train == 1)[0]
        neg = np.where(y_train == 0)[0]
        n_min = min(len(pos), len(neg))
        sel = np.concatenate([np.random.choice(pos, n_min, replace=False),
                              np.random.choice(neg, n_min, replace=False)])

        feature_groups = {
            'DinucFreq': (din_tr, din_te),
            'Motif': (mot_tr, mot_te),
            'Structure': (str_tr, str_te),
            'Combined': (pd.concat([din_tr, mot_tr, str_tr], axis=1),
                        pd.concat([din_te, mot_te, str_te], axis=1)),
        }

        for group_name, (X_train_full, X_test_full) in feature_groups.items():
            X_tr_bal = X_train_full.iloc[sel]
            y_tr_bal = y_train[sel]

            scaler = StandardScaler()
            X_tr = scaler.fit_transform(X_tr_bal)
            X_te = scaler.transform(X_test_full)

            gbm = GradientBoostingClassifier(n_estimators=300, max_depth=5, learning_rate=0.1,
                                              min_samples_leaf=10, subsample=0.8, random_state=42)
            gbm.fit(X_tr, y_tr_bal)
            probs = gbm.predict_proba(X_te)[:, 1]
            auroc = roc_auc_score(y_test, probs)
            print(f"  {group_name:12s}: auROC={auroc:.4f} ({X_tr.shape[1]} features)")

            all_results.append({
                'stress': stress, 'direction': direction,
                'model': group_name, 'auROC': auroc,
                'n_features': X_tr.shape[1],
            })

df = pd.DataFrame(all_results)
df.to_csv(RESULTS_DIR / "ablation_results.csv", index=False)
print(f"\n{'='*60}")
print("Summary (mean auROC per model):")
print(df.groupby('model')['auROC'].mean().sort_values(ascending=False).to_string())
print(f"\nSaved to {RESULTS_DIR / 'ablation_results.csv'}")
