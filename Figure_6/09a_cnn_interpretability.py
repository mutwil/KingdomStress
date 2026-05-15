#!/usr/bin/env python3
"""
CNN interpretability pipeline:
1. Train CNN (save model)
2. Compute integrated gradients per nucleotide
3. Run TF-MoDISco to discover motifs
4. Compare to JASPAR plant motifs

Runs on all 6 stresses x 2 directions.
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
MODELS_DIR = RESULTS_DIR / "saved_models"
MODELS_DIR.mkdir(exist_ok=True)
ATTR_DIR = RESULTS_DIR / "attributions"
ATTR_DIR.mkdir(exist_ok=True)
MOTIF_DIR = RESULTS_DIR / "motifs"
MOTIF_DIR.mkdir(exist_ok=True)

SEQ_LEN = 600
TRIM_START = 4700
TRIM_END = 5300
BATCH_SIZE = 64
EPOCHS = 20
PATIENCE = 5

STRESSES = ["Heat", "Cold", "Drought", "Salt", "Pathogen", "Flooding"]
DIRECTIONS = ["UP", "DOWN"]

NT_MAP = {"A": 0, "C": 1, "G": 2, "T": 3}
NT_LIST = ["A", "C", "G", "T"]


def onehot(seq):
    enc = np.zeros((len(seq), 4), dtype=np.float32)
    for i, nt in enumerate(seq):
        if nt in NT_MAP:
            enc[i, NT_MAP[nt]] = 1.0
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


def integrated_gradients(model, input_seq, baseline=None, steps=50):
    """Compute integrated gradients for a single input sequence.

    Args:
        model: trained Keras model
        input_seq: (seq_len, 4) one-hot encoded
        baseline: reference input (default: all zeros)
        steps: number of interpolation steps

    Returns:
        attributions: (seq_len, 4) attribution scores
    """
    if baseline is None:
        baseline = np.zeros_like(input_seq)

    # Generate interpolated inputs
    alphas = np.linspace(0, 1, steps + 1)
    interpolated = np.array([baseline + alpha * (input_seq - baseline) for alpha in alphas])
    interpolated = tf.convert_to_tensor(interpolated, dtype=tf.float32)

    # Compute gradients
    with tf.GradientTape() as tape:
        tape.watch(interpolated)
        predictions = model(interpolated)
    grads = tape.gradient(predictions, interpolated).numpy()

    # Average gradients and multiply by (input - baseline)
    avg_grads = np.mean(grads, axis=0)
    attributions = (input_seq - baseline) * avg_grads

    return attributions


def batch_integrated_gradients(model, sequences, batch_size=100, steps=50):
    """Compute integrated gradients for a batch of sequences."""
    all_attrs = []
    baseline = np.zeros((SEQ_LEN, 4), dtype=np.float32)

    for i in range(0, len(sequences), batch_size):
        batch = sequences[i:i+batch_size]
        batch_attrs = []

        # Interpolate
        alphas = np.linspace(0, 1, steps + 1)

        for seq in batch:
            interpolated = np.array([baseline + alpha * (seq - baseline) for alpha in alphas])
            interpolated = tf.convert_to_tensor(interpolated, dtype=tf.float32)

            with tf.GradientTape() as tape:
                tape.watch(interpolated)
                preds = model(interpolated)
            grads = tape.gradient(preds, interpolated).numpy()
            avg_grads = np.mean(grads, axis=0)
            attr = (seq - baseline) * avg_grads
            batch_attrs.append(attr)

        all_attrs.extend(batch_attrs)
        if (i // batch_size) % 10 == 0:
            print(f"    {i+len(batch)}/{len(sequences)} sequences processed")

    return np.array(all_attrs)


def extract_motifs_from_attributions(attributions, sequences, gene_ids, labels,
                                     window_size=20, top_n=1000):
    """Extract high-attribution windows as candidate motifs.

    For responsive genes (label=1), find the top windows by attribution score.
    Returns a list of (sequence, score, gene_id, position) tuples.
    """
    motifs = []

    # Only look at responsive genes
    responsive_idx = np.where(labels == 1)[0]

    for idx in responsive_idx:
        attr = attributions[idx]  # (seq_len, 4)
        seq = sequences[idx]
        gid = gene_ids[idx]

        # Per-position importance = sum of absolute attributions across nucleotides
        importance = np.sum(np.abs(attr), axis=1)  # (seq_len,)

        # Sliding window scores
        for pos in range(0, len(importance) - window_size):
            score = np.mean(importance[pos:pos+window_size])
            if score > 0:
                # Decode sequence
                seq_str = ""
                for j in range(pos, pos + window_size):
                    nt_idx = np.argmax(seq[j])
                    seq_str += NT_LIST[nt_idx] if seq[j].sum() > 0 else "N"
                motifs.append((seq_str, score, gid, pos))

    # Sort by score, take top N
    motifs.sort(key=lambda x: -x[1])
    return motifs[:top_n]


def save_motifs_as_meme(motifs, output_file, top_n=50):
    """Save top motifs in MEME format for comparison with JASPAR."""
    from collections import Counter

    # Cluster similar motifs by simple k-mer counting
    # Group motifs by their most common 8-mer
    kmer_groups = {}
    for seq, score, gid, pos in motifs[:top_n * 10]:
        for k_start in range(0, len(seq) - 7):
            kmer = seq[k_start:k_start+8]
            if kmer not in kmer_groups:
                kmer_groups[kmer] = []
            kmer_groups[kmer].append((seq, score))

    # Take top kmers by total score
    kmer_scores = [(k, sum(s for _, s in v), len(v), v) for k, v in kmer_groups.items()]
    kmer_scores.sort(key=lambda x: -x[1])

    with open(output_file, "w") as f:
        f.write("MEME version 5\n\n")
        f.write("ALPHABET= ACGT\n\n")
        f.write("strands: + -\n\n")

        for i, (kmer, total_score, count, seqs) in enumerate(kmer_scores[:top_n]):
            f.write(f"MOTIF motif_{i+1} {kmer}\n")
            f.write(f"letter-probability matrix: alength= 4 w= {len(kmer)} nsites= {count}\n")

            # Build position frequency matrix from aligned sequences
            for pos in range(len(kmer)):
                counts = Counter()
                for seq, _ in seqs[:100]:
                    if pos < len(seq):
                        counts[seq[pos]] += 1
                total = sum(counts.values())
                if total == 0:
                    total = 1
                f.write(f" {counts.get('A',0)/total:.4f} {counts.get('C',0)/total:.4f} "
                       f"{counts.get('G',0)/total:.4f} {counts.get('T',0)/total:.4f}\n")
            f.write("\n")

    print(f"    Saved {min(top_n, len(kmer_scores))} motifs to {output_file}")


# ---- Load gene sequences ----
print("Loading gene sequences...")
gene_dna = {}  # gene_id -> one-hot
gene_seq_str = {}  # gene_id -> string

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
                        gene_seq_str[gid] = dna
                gid = line[1:].split()[0]
                seq_parts = []
            else:
                seq_parts.append(line.strip())
        if gid and seq_parts:
            full_seq = "".join(seq_parts)
            if len(full_seq) >= TRIM_END:
                gene_dna[gid] = onehot(full_seq[TRIM_START:TRIM_END])
                gene_seq_str[gid] = full_seq[TRIM_START:TRIM_END]
    print(f"  {sp}: done")

print(f"Total: {len(gene_dna)} genes\n")


# ---- Main loop ----
all_results = []

for stress in STRESSES:
    for direction in DIRECTIONS:
        tag = f"{stress}_{direction}"
        model_path = MODELS_DIR / f"{tag}.keras"
        attr_path = ATTR_DIR / f"{tag}_attributions.npz"
        motif_path = MOTIF_DIR / f"{tag}_motifs.txt"
        meme_path = MOTIF_DIR / f"{tag}_motifs.meme"

        dataset_file = DATASET_DIR / f"dataset_{tag}.pkl"
        if not dataset_file.exists():
            print(f"{tag}: no dataset, skipping")
            continue

        with open(dataset_file, "rb") as f:
            splits = pickle.load(f)

        print(f"\n{'='*60}")
        print(f"{tag}")
        print(f"{'='*60}")

        # Build arrays
        def build_arrays(gids, labs, max_n=None):
            X, y, ids = [], [], []
            for gid, lab in zip(gids, labs):
                if gid in gene_dna:
                    X.append(gene_dna[gid])
                    y.append(lab)
                    ids.append(gid)
                if max_n and len(X) >= max_n:
                    break
            return np.array(X), np.array(y, dtype=np.float32), ids

        train_gids, train_labs = splits["train"]
        val_gids, val_labs = splits["val"]
        test_gids, test_labs = splits["test"]

        X_train, y_train, _ = build_arrays(train_gids, train_labs, max_n=20000)
        X_val, y_val, _ = build_arrays(val_gids, val_labs, max_n=10000)
        X_test, y_test, test_ids = build_arrays(test_gids, test_labs, max_n=5000)

        print(f"  Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")

        # ---- Step 1: Train (or load) model ----
        if model_path.exists():
            print(f"  Loading saved model...")
            model = models.load_model(str(model_path))
            preds = model.predict(X_test, verbose=0).flatten()
            auroc = roc_auc_score(y_test, preds)
            print(f"  Loaded: auROC={auroc:.4f}")
        else:
            # Balance training
            neg = np.where(y_train == 0)[0]
            pos = np.where(y_train == 1)[0]
            n_min = min(len(neg), len(pos))
            if n_min < 50:
                print(f"  Too few samples, skipping")
                continue
            sel = np.concatenate([np.random.choice(neg, n_min, replace=False),
                                  np.random.choice(pos, n_min, replace=False)])
            X_train_bal, y_train_bal = X_train[sel], y_train[sel]
            X_train_bal, y_train_bal = sk_shuffle(X_train_bal, y_train_bal, random_state=42)

            backend.clear_session()
            model = build_cnn()
            callbacks = [
                EarlyStopping(patience=PATIENCE, restore_best_weights=True, monitor="val_loss"),
                ReduceLROnPlateau(patience=3, factor=0.1, verbose=0),
            ]
            model.fit(X_train_bal, y_train_bal, batch_size=BATCH_SIZE, epochs=EPOCHS,
                      validation_data=(X_val, y_val), callbacks=callbacks, verbose=1)

            preds = model.predict(X_test, verbose=0).flatten()
            auroc = roc_auc_score(y_test, preds)
            print(f"  Trained: auROC={auroc:.4f}")

            model.save(str(model_path))
            print(f"  Saved model to {model_path}")

        # Save predictions
        pd.DataFrame({"gene_id": test_ids, "label": y_test, "prediction": preds}).to_csv(
            RESULTS_DIR / f"{tag}_predictions.csv", index=False)

        result = {"stress": stress, "direction": direction, "test_auROC": auroc}

        # ---- Step 2: Integrated gradients ----
        if attr_path.exists():
            print(f"  Loading saved attributions...")
            data = np.load(attr_path)
            attributions = data["attributions"]
        else:
            print(f"  Computing integrated gradients (5000 genes, 50 steps)...")
            attributions = batch_integrated_gradients(model, X_test, batch_size=50, steps=50)
            np.savez_compressed(attr_path, attributions=attributions,
                               gene_ids=test_ids, labels=y_test)
            print(f"  Saved attributions to {attr_path}")

        # ---- Step 3: Extract motifs ----
        print(f"  Extracting motifs...")
        motifs = extract_motifs_from_attributions(
            attributions, X_test, test_ids, y_test, window_size=15, top_n=5000)

        # Save top motifs as text
        with open(motif_path, "w") as f:
            f.write("rank\tsequence\tscore\tgene_id\tposition\n")
            for i, (seq, score, gid, pos) in enumerate(motifs[:500]):
                f.write(f"{i+1}\t{seq}\t{score:.6f}\t{gid}\t{pos}\n")

        # Save as MEME format
        save_motifs_as_meme(motifs, meme_path, top_n=30)

        # ---- Summary stats ----
        if motifs:
            top_seqs = [m[0] for m in motifs[:100]]
            # Find most common 8-mers in top motifs
            from collections import Counter
            kmers = Counter()
            for seq in top_seqs:
                for k in range(len(seq) - 7):
                    kmers[seq[k:k+8]] += 1
            top_kmers = kmers.most_common(10)
            print(f"  Top 8-mers in responsive genes:")
            for kmer, count in top_kmers:
                print(f"    {kmer}: {count}")
            result["top_kmer"] = top_kmers[0][0] if top_kmers else ""

        all_results.append(result)

        del model
        gc.collect()

# Summary
print(f"\n{'='*60}")
print("Summary")
print(f"{'='*60}")
df = pd.DataFrame(all_results)
print(df.to_string(index=False))
df.to_csv(RESULTS_DIR / "interpretability_results.csv", index=False)
print(f"\nAll results saved to {RESULTS_DIR}")
