#!/usr/bin/env python3
"""
CNN baseline for all 6 stresses x 2 directions.
Uses the same dataset pickles as PlantRNA-FM (same train/test split).
Trains on DNA sequence (600bp centered on TSS, one-hot encoded).
"""

import os, re, json, pickle, gc
import numpy as np
import pandas as pd
from pathlib import Path

os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"

import tensorflow as tf
from tensorflow.keras import Sequential, optimizers, backend, models
from tensorflow.keras.layers import Conv1D, Dense, MaxPool1D, Dropout, Flatten
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.metrics import roc_auc_score, f1_score
from sklearn.utils import shuffle as sk_shuffle

# ---- Config ----
GENE_SEQS_DIR = Path("/Users/vjx443/Library/CloudStorage/GoogleDrive-mutwil@plant.tools/My Drive/Projects/2026_KingdomStress/CC-Fig 5 data_CNN/gene_seqs")
DATASET_DIR = Path("/Users/vjx443/Library/CloudStorage/GoogleDrive-mutwil@plant.tools/My Drive/Projects/2026_KingdomStress/CC-Fig 5 data_CNN/plantrna_fm_results")
RESULTS_DIR = Path("/Users/vjx443/Downloads/cnn_results")
RESULTS_DIR.mkdir(exist_ok=True)

SEQ_LEN = 600  # same as PlantCAD2
TRIM_START = 4700  # center of 10kb window
TRIM_END = 5300
BATCH_SIZE = 64
EPOCHS = 20
PATIENCE = 5

STRESSES = ["Heat", "Cold", "Drought", "Salt", "Pathogen", "Flooding"]
DIRECTIONS = ["UP", "DOWN"]


def onehot(seq):
    code = {"A": [1,0,0,0], "C": [0,1,0,0], "G": [0,0,1,0], "T": [0,0,0,1]}
    enc = np.zeros((len(seq), 4), dtype=np.float32)
    for i, nt in enumerate(seq):
        if nt in code:
            enc[i] = code[nt]
    return enc


def build_cnn(input_shape=(SEQ_LEN, 4)):
    model = Sequential([
        Conv1D(64, 8, activation="relu", padding="same", input_shape=input_shape),
        Conv1D(64, 8, activation="relu", padding="same"),
        MaxPool1D(8, padding="same"), Dropout(0.25),
        Conv1D(128, 8, activation="relu", padding="same"),
        Conv1D(128, 8, activation="relu", padding="same"),
        MaxPool1D(8, padding="same"), Dropout(0.25),
        Conv1D(64, 8, activation="relu", padding="same"),
        Conv1D(64, 8, activation="relu", padding="same"),
        MaxPool1D(8, padding="same"), Dropout(0.25),
        Flatten(),
        Dense(128, activation="relu"), Dropout(0.25),
        Dense(64, activation="relu"),
        Dense(1, activation="sigmoid")
    ])
    model.compile(loss="binary_crossentropy", optimizer=optimizers.Adam(0.0001), metrics=["accuracy"])
    return model


# ---- Load gene sequences (600bp DNA) ----
print("Loading gene sequences...")
gene_dna = {}  # gene_id -> one-hot encoded array

for fa in sorted(GENE_SEQS_DIR.glob("*.fa")):
    sp = fa.stem
    gid = None
    seq_parts = []
    with open(fa) as f:
        for line in f:
            if line.startswith(">"):
                if gid and seq_parts:
                    full_seq = "".join(seq_parts)
                    if len(full_seq) >= TRIM_END:
                        dna = full_seq[TRIM_START:TRIM_END]
                        gene_dna[gid] = onehot(dna)
                gid = line[1:].split()[0]
                seq_parts = []
            else:
                seq_parts.append(line.strip())
        if gid and seq_parts:
            full_seq = "".join(seq_parts)
            if len(full_seq) >= TRIM_END:
                gene_dna[gid] = onehot(full_seq[TRIM_START:TRIM_END])
    print(f"  {sp}: done")

print(f"Total: {len(gene_dna)} genes\n")


# ---- Train CNN for each stress x direction ----
all_results = []

for stress in STRESSES:
    for direction in DIRECTIONS:
        tag = f"{stress}_{direction}"
        result_file = RESULTS_DIR / f"{tag}_result.json"

        if result_file.exists():
            with open(result_file) as f:
                r = json.load(f)
            print(f"{tag}: already done (auROC={r['test_auROC']:.4f})")
            all_results.append(r)
            continue

        dataset_file = DATASET_DIR / f"dataset_{tag}.pkl"
        if not dataset_file.exists():
            print(f"{tag}: no dataset, skipping")
            continue

        with open(dataset_file, "rb") as f:
            splits = pickle.load(f)

        print(f"\n{'='*60}")
        print(f"CNN: {tag}")
        print(f"{'='*60}")

        # Build arrays from gene IDs
        def build_arrays(gids, labs, max_n=20000):
            X, y = [], []
            for gid, lab in zip(gids, labs):
                if gid in gene_dna:
                    X.append(gene_dna[gid])
                    y.append(lab)
                if max_n and len(X) >= max_n:
                    break
            return np.array(X), np.array(y, dtype=np.float32)

        train_gids, train_labs = splits["train"]
        val_gids, val_labs = splits["val"]
        test_gids, test_labs = splits["test"]

        X_train, y_train = build_arrays(train_gids, train_labs, max_n=20000)
        X_val, y_val = build_arrays(val_gids, val_labs, max_n=10000)
        X_test, y_test = build_arrays(test_gids, test_labs, max_n=10000)

        print(f"  Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")

        # Balance training
        neg = np.where(y_train == 0)[0]
        pos = np.where(y_train == 1)[0]
        n_min = min(len(neg), len(pos))
        if n_min < 50:
            print(f"  Too few samples, skipping")
            continue
        sel = np.concatenate([np.random.choice(neg, n_min, replace=False),
                              np.random.choice(pos, n_min, replace=False)])
        X_train, y_train = X_train[sel], y_train[sel]
        X_train, y_train = sk_shuffle(X_train, y_train, random_state=42)
        print(f"  Balanced: {len(X_train)} ({n_min}/class)")

        # Train
        backend.clear_session()
        model = build_cnn()
        callbacks = [
            EarlyStopping(patience=PATIENCE, restore_best_weights=True, monitor="val_loss"),
            ReduceLROnPlateau(patience=3, factor=0.1, verbose=0),
        ]
        model.fit(X_train, y_train, batch_size=BATCH_SIZE, epochs=EPOCHS,
                  validation_data=(X_val, y_val), callbacks=callbacks, verbose=1)

        # Evaluate
        preds = model.predict(X_test, verbose=0).flatten()
        auroc = roc_auc_score(y_test, preds)
        f1 = f1_score(y_test, (preds > 0.5).astype(int))
        print(f"  TEST: auROC={auroc:.4f}, F1={f1:.4f}")

        result = {
            "stress": stress, "direction": direction,
            "test_auROC": auroc, "test_F1": f1,
            "n_train": len(X_train), "n_test": len(X_test),
        }
        all_results.append(result)
        with open(result_file, "w") as f:
            json.dump(result, f, indent=2)

        # Save predictions with gene IDs for fusion
        test_gene_ids = [gid for gid in test_gids[:10000] if gid in gene_dna][:len(preds)]
        pd.DataFrame({"gene_id": test_gene_ids, "label": y_test, "prediction": preds}).to_csv(
            RESULTS_DIR / f"{tag}_predictions.csv", index=False)

        del model
        gc.collect()

# Summary
print(f"\n{'='*60}")
print("CNN Results Summary")
print(f"{'='*60}")
df = pd.DataFrame(all_results)
print(df[["stress", "direction", "test_auROC", "test_F1"]].to_string(index=False))
print(f"\nMean auROC: {df['test_auROC'].mean():.4f}")
df.to_csv(RESULTS_DIR / "all_results.csv", index=False)
