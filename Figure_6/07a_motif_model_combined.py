#!/usr/bin/env python3
"""
Motif model v3: motifs + dinucleotide frequencies + RNA secondary structure features.

Features:
  1. Known + discovered motif counts (promoter + mRNA)
  2. All 16 dinucleotide frequencies (promoter + mRNA) -- inspired by Meng et al. PNAS 2021
  3. RNA secondary structure features (mRNA):
     - MFE (minimum free energy)
     - Fraction paired bases
     - GC content (for comparison)
     - Local structure stability in 5'UTR, CDS start, mid, 3' regions
  4. G-quadruplex potential (GGG runs)
  5. Sequence composition (mono/dinucleotide)

Tests whether RNA structure explains the GC-content signal.
"""

import pickle, json, warnings, re, gc as gc_collect
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
    print("WARNING: ViennaRNA not available, skipping structure features")

try:
    import xgboost as xgb
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

warnings.filterwarnings("ignore")

# ---- Config ----
GENE_SEQS_DIR = Path("/Users/vjx443/Library/CloudStorage/GoogleDrive-mutwil@plant.tools/My Drive/Projects/2026_KingdomStress/CC-Fig 5 data_CNN/gene_seqs")
DATASET_DIR = Path("/Users/vjx443/Library/CloudStorage/GoogleDrive-mutwil@plant.tools/My Drive/Projects/2026_KingdomStress/CC-Fig 5 data_CNN/plantrna_fm_results")
CNN_MOTIF_DIR = Path("/Users/vjx443/Downloads/cnn_results/motifs")
DNA_LLM_MOTIF_DIR = Path("/Users/vjx443/Library/CloudStorage/GoogleDrive-mutwil@plant.tools/My Drive/Projects/2026_KingdomStress/CC-Fig 5 data_CNN/plantcad2_interpretability/motifs")
RNA_MOTIF_DIR = Path("/Users/vjx443/Downloads/rna_interpretability/motifs")
RESULTS_DIR = Path("/Users/vjx443/Downloads/motif_model_v3_results")
RESULTS_DIR.mkdir(exist_ok=True)

STRESSES = ["Heat", "Cold", "Drought", "Salt", "Pathogen", "Flooding"]
DIRECTIONS = ["UP", "DOWN"]

KNOWN_MOTIFS = {
    "ABRE": "ACGTG", "DRE_CRT": "CCGAC", "W_box": "TTGAC", "GCC_box": "GCCGCC",
    "G_box": "CACGTG", "TATA_box": "TATAAAT", "CAAT_box": "CCAAT", "MBS": "CAACTG",
    "as1": "TGACG", "MYC": "CATGTG", "LTR": "CCGAAA", "ARE_anaerobic": "AAACCA",
    "ARE_aurich": "ATTTA", "PUF_binding": "TGTA", "DST_element": "ATAGAT",
    "poly_A_signal": "AATAAA",
}

DINUCS = ["".join(p) for p in product("ACGT", repeat=2)]


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
    """Compute all 16 dinucleotide frequencies."""
    seq = seq.upper().replace("U", "T")
    total = max(len(seq) - 1, 1)
    counts = Counter(seq[i:i+2] for i in range(len(seq)-1))
    return {f"dn_{d}": counts.get(d, 0) / total for d in DINUCS}


def rna_structure_features(seq):
    """Compute RNA secondary structure features using ViennaRNA.
    Fast version: fold only first 200bp (5'UTR/CDS start).
    """
    if not HAS_VIENNA:
        return {}

    rna_seq = seq.upper().replace("T", "U")
    feats = {}

    # Fold first 200bp (5'UTR region -- most structured, most relevant)
    sub = rna_seq[:200]
    ss, mfe = RNA.fold(sub)
    feats["mfe_per_nt"] = mfe / max(len(sub), 1)
    feats["frac_paired"] = ss.count("(") * 2 / max(len(ss), 1)

    # Count structural elements from the dot-bracket notation
    feats["n_stems"] = ss.count("(")
    feats["n_hairpins"] = len(re.findall(r"\(\.+\)", ss))
    feats["max_stem_len"] = max((len(m) for m in re.findall(r"\(+", ss)), default=0)

    # G-quadruplex potential (GGG runs in full sequence)
    feats["g_quad_count"] = len(re.findall(r"G{3,}", seq.upper()))
    feats["c_quad_count"] = len(re.findall(r"C{3,}", seq.upper()))

    # AU-rich stretches
    feats["au_rich_runs"] = len(re.findall(r"[ATU]{5,}", seq.upper()))

    return feats


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


# ---- Load gene sequences ----
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


# ---- Collect genes we actually need (from all dataset pickles) ----
needed_genes = set()
for stress in STRESSES:
    for direction in DIRECTIONS:
        tag = f"{stress}_{direction}"
        dataset_file = DATASET_DIR / f"dataset_{tag}.pkl"
        if not dataset_file.exists():
            continue
        with open(dataset_file, "rb") as f:
            splits = pickle.load(f)
        for split_name in ["train", "val", "test"]:
            gids, _ = splits[split_name]
            for gid in gids:
                if gid in gene_mrna:
                    needed_genes.add(gid)
print(f"Genes needed across all datasets: {len(needed_genes)}")

# ---- Precompute structure features (only for needed genes) ----
struct_cache = RESULTS_DIR / "structure_cache.pkl"
if struct_cache.exists():
    print("Loading cached structure features...")
    with open(struct_cache, "rb") as f:
        gene_struct = pickle.load(f)
    print(f"  {len(gene_struct)} genes cached")
    # Check if we need more
    missing = needed_genes - set(gene_struct.keys())
    if missing:
        print(f"  Computing {len(missing)} missing genes...")
        for i, gid in enumerate(missing):
            gene_struct[gid] = rna_structure_features(gene_mrna[gid])
            if (i+1) % 5000 == 0:
                print(f"    {i+1}/{len(missing)}")
        with open(struct_cache, "wb") as f:
            pickle.dump(gene_struct, f)
else:
    print(f"Computing RNA structure features for {len(needed_genes)} genes...")
    gene_struct = {}
    for i, gid in enumerate(needed_genes):
        gene_struct[gid] = rna_structure_features(gene_mrna[gid])
        if (i+1) % 5000 == 0:
            print(f"  {i+1}/{len(needed_genes)}")
    with open(struct_cache, "wb") as f:
        pickle.dump(gene_struct, f)
    print(f"  {len(gene_struct)} genes computed and cached")


# ---- Main loop ----
all_results = []

for stress in STRESSES:
    for direction in DIRECTIONS:
        tag = f"{stress}_{direction}"
        dataset_file = DATASET_DIR / f"dataset_{tag}.pkl"
        if not dataset_file.exists():
            continue

        print(f"\n{'='*60}")
        print(f"{tag}")
        print(f"{'='*60}")

        # Build motif set
        motifs = dict(KNOWN_MOTIFS)
        for prefix, mdir in [("cnn", CNN_MOTIF_DIR), ("dna", DNA_LLM_MOTIF_DIR), ("rna", RNA_MOTIF_DIR)]:
            for name, seq in get_top_kmers(mdir / f"{tag}_motifs.txt").items():
                motifs[f"{prefix}_{name}"] = seq

        # Deduplicate
        seen = {}
        unique_motifs = {}
        for name, seq in motifs.items():
            key = min(seq.upper().replace("U","T"), reverse_complement(seq.upper().replace("U","T")))
            if key not in seen:
                seen[key] = name
                unique_motifs[name] = seq

        # Build features
        with open(dataset_file, "rb") as f:
            splits = pickle.load(f)

        def make_Xy(gids, labs, max_n=None):
            X_rows, y = [], []
            for gid, lab in zip(gids, labs):
                if gid not in gene_promoter:
                    continue
                feats = {}

                # 1. Motif counts
                for name, motif in unique_motifs.items():
                    feats[f"prom_{name}"] = count_motif(gene_promoter[gid], motif)
                    feats[f"mrna_{name}"] = count_motif(gene_mrna[gid], motif)

                # 2. Dinucleotide frequencies
                for k, v in dinucleotide_freqs(gene_promoter[gid]).items():
                    feats[f"prom_{k}"] = v
                for k, v in dinucleotide_freqs(gene_mrna[gid]).items():
                    feats[f"mrna_{k}"] = v

                # 3. RNA structure features
                if gid in gene_struct:
                    for k, v in gene_struct[gid].items():
                        feats[f"struct_{k}"] = v

                # 4. Basic composition
                prom = gene_promoter[gid]
                mrna = gene_mrna[gid]
                feats["prom_gc"] = (prom.count("G") + prom.count("C")) / 600
                feats["mrna_gc"] = (mrna.count("G") + mrna.count("C")) / 1024
                feats["prom_len_at_rich"] = prom.count("AAAA") + prom.count("TTTT")
                feats["mrna_len_at_rich"] = mrna.count("AAAA") + mrna.count("TTTT")

                X_rows.append(feats)
                y.append(lab)
                if max_n and len(X_rows) >= max_n:
                    break
            return pd.DataFrame(X_rows), np.array(y, dtype=np.float32)

        train_gids, train_labs = splits["train"]
        val_gids, val_labs = splits["val"]
        test_gids, test_labs = splits["test"]

        X_train, y_train = make_Xy(train_gids, train_labs, max_n=50000)
        X_val, y_val = make_Xy(val_gids, val_labs, max_n=20000)
        X_test, y_test = make_Xy(test_gids, test_labs, max_n=20000)

        n_feats = X_train.shape[1]
        print(f"  Train: {len(X_train)}, Test: {len(X_test)}, Features: {n_feats}")

        # Balance training
        pos = np.where(y_train == 1)[0]
        neg = np.where(y_train == 0)[0]
        n_min = min(len(pos), len(neg))
        sel = np.concatenate([np.random.choice(pos, n_min, replace=False),
                              np.random.choice(neg, n_min, replace=False)])
        X_train_bal = X_train.iloc[sel]
        y_train_bal = y_train[sel]

        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X_train_bal)
        X_va = scaler.transform(X_val)
        X_te = scaler.transform(X_test)

        # Train GBM (best from HPO)
        gbm = GradientBoostingClassifier(n_estimators=300, max_depth=5, learning_rate=0.1,
                                          min_samples_leaf=10, subsample=0.8, random_state=42)
        gbm.fit(X_tr, y_train_bal)
        gbm_probs = gbm.predict_proba(X_te)[:, 1]
        gbm_auroc = roc_auc_score(y_test, gbm_probs)
        print(f"  GBM auROC: {gbm_auroc:.4f}")

        # Feature importance
        imp = pd.Series(gbm.feature_importances_, index=X_train.columns).sort_values(ascending=False)

        # Categorize top features
        top20 = imp.head(20)
        struct_feats = [f for f in top20.index if "struct_" in f]
        dinuc_feats = [f for f in top20.index if "dn_" in f]
        motif_feats = [f for f in top20.index if f not in struct_feats and f not in dinuc_feats and "gc" not in f and "at_rich" not in f]
        gc_feats = [f for f in top20.index if "gc" in f]

        print(f"  Top 15 features:")
        for feat, score in imp.head(15).items():
            cat = "STRUCT" if "struct_" in feat else "DINUC" if "dn_" in feat else "GC" if "gc" in feat else "MOTIF"
            print(f"    [{cat:6s}] {feat:40s} {score:.4f}")

        print(f"  Feature category summary (top 20):")
        print(f"    Structure: {len(struct_feats)}, Dinucleotide: {len(dinuc_feats)}, Motif: {len(motif_feats)}, GC: {len(gc_feats)}")

        all_results.append({
            "stress": stress, "direction": direction,
            "GBM_auROC": gbm_auroc, "n_features": n_feats,
            "top_feature": imp.index[0],
            "n_struct_top20": len(struct_feats),
            "n_dinuc_top20": len(dinuc_feats),
            "n_motif_top20": len(motif_feats),
        })

        imp.to_csv(RESULTS_DIR / f"{tag}_feature_importance.csv")

# Summary
print(f"\n{'='*60}")
print("Summary")
print(f"{'='*60}")
df = pd.DataFrame(all_results)
print(df[["stress", "direction", "GBM_auROC", "top_feature", "n_struct_top20", "n_dinuc_top20"]].to_string(index=False))
print(f"\nMean GBM: {df['GBM_auROC'].mean():.4f}")
print(f"(v1 known motifs only: 0.687)")
print(f"(v2 + discovered motifs: 0.680)")
print(f"\nMean structure features in top 20: {df['n_struct_top20'].mean():.1f}")
print(f"Mean dinucleotide features in top 20: {df['n_dinuc_top20'].mean():.1f}")
df.to_csv(RESULTS_DIR / "v3_results.csv", index=False)
