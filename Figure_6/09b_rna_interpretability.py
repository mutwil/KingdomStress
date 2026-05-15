#!/usr/bin/env python3
"""
PlantRNA-FM interpretability pipeline:
1. Load trained models (from LUMI results on Drive)
2. Compute integrated gradients per nucleotide
3. Extract top motifs
4. Save in MEME format

Uses the same dataset pickles and test genes as the CNN pipeline.
"""

import os, re, json, pickle, gc
import numpy as np
import pandas as pd
from pathlib import Path
from collections import Counter

import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer
from peft import LoraConfig, get_peft_model
from sklearn.metrics import roc_auc_score

# ---- Config ----
RESULTS_DIR = Path("/Users/vjx443/Downloads/rna_interpretability")
RESULTS_DIR.mkdir(exist_ok=True)
ATTR_DIR = RESULTS_DIR / "attributions"
ATTR_DIR.mkdir(exist_ok=True)
MOTIF_DIR = RESULTS_DIR / "motifs"
MOTIF_DIR.mkdir(exist_ok=True)

LUMI_RESULTS = Path("/Users/vjx443/Downloads/fusion/rna")  # has _best.pt files from LUMI
DATASET_DIR = Path("/Users/vjx443/Library/CloudStorage/GoogleDrive-mutwil@plant.tools/My Drive/Projects/2026_KingdomStress/CC-Fig 5 data_CNN/plantrna_fm_results")
GENE_SEQS_DIR = Path("/Users/vjx443/Library/CloudStorage/GoogleDrive-mutwil@plant.tools/My Drive/Projects/2026_KingdomStress/CC-Fig 5 data_CNN/gene_seqs")

MODEL_NAME = "yangheng/PlantRNA-FM"
MAX_SEQ_LEN = 1024
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

STRESSES = ["Heat", "Cold", "Drought", "Salt", "Pathogen", "Flooding"]
DIRECTIONS = ["UP", "DOWN"]
NT_LIST = ["A", "C", "G", "U"]


class PlantRNAClassifier(nn.Module):
    def __init__(self, base_model, hidden_size):
        super().__init__()
        self.base = base_model
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, 128), nn.ReLU(), nn.Dropout(0.1), nn.Linear(128, 1))

    def forward(self, input_ids, attention_mask):
        outputs = self.base(input_ids=input_ids, attention_mask=attention_mask)
        hidden = outputs.last_hidden_state
        mask = attention_mask.unsqueeze(-1).float()
        pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1)
        return self.classifier(pooled).squeeze(-1)


def integrated_gradients_rna(model, input_ids, attention_mask, tokenizer, steps=30):
    """Compute integrated gradients for RNA model via embedding space.

    Returns: (seq_len,) attribution scores per token position.
    """
    model.eval()

    # Get embeddings
    embedding_layer = model.base.get_base_model().embeddings.word_embeddings

    # Baseline = pad token embedding repeated
    pad_id = tokenizer.pad_token_id or 0
    baseline_ids = torch.full_like(input_ids, pad_id)

    # Get baseline and input embeddings
    with torch.no_grad():
        input_embeds = embedding_layer(input_ids).detach()
        baseline_embeds = embedding_layer(baseline_ids).detach()

    # Interpolate
    alphas = torch.linspace(0, 1, steps + 1, device=DEVICE)

    all_grads = []
    for alpha in alphas:
        interp = baseline_embeds + alpha * (input_embeds - baseline_embeds)
        interp = interp.detach().requires_grad_(True)

        # Forward pass through model using embeddings directly
        # We need to bypass the embedding layer
        outputs = model.base.get_base_model().encoder(interp, attention_mask=attention_mask.unsqueeze(0) if attention_mask.dim() == 1 else attention_mask)
        hidden = outputs.last_hidden_state
        mask = attention_mask.unsqueeze(-1).float()
        if mask.dim() == 2:
            mask = mask.unsqueeze(0)
        pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1)
        logit = model.classifier(pooled)

        logit.backward()
        all_grads.append(interp.grad.detach().clone())
        model.zero_grad()

    # Average gradients
    avg_grads = torch.stack(all_grads).mean(dim=0)

    # Attributions = (input - baseline) * avg_grads, summed over embedding dim
    attrs = ((input_embeds - baseline_embeds) * avg_grads).sum(dim=-1)  # (batch, seq_len)

    return attrs.squeeze().cpu().numpy()


def simple_gradient_attribution(model, input_ids, attention_mask):
    """Simple gradient * input attribution (faster than integrated gradients).
    Returns per-token importance scores.
    """
    model.eval()
    model.zero_grad()

    embedding_layer = model.base.get_base_model().embeddings.word_embeddings
    embeds = embedding_layer(input_ids).detach().requires_grad_(True)

    # Forward through encoder
    encoder_output = model.base.get_base_model().encoder(embeds, attention_mask=attention_mask.float())
    hidden = encoder_output.last_hidden_state
    mask = attention_mask.unsqueeze(-1).float()
    pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1)
    logit = model.classifier(pooled)

    logit.backward()

    # Gradient * embedding, summed over embedding dim
    grad = embeds.grad.detach()
    attr = (grad * embeds.detach()).sum(dim=-1)  # (batch, seq_len)

    return attr.squeeze().cpu().numpy()


def decode_tokens(input_ids, tokenizer):
    """Decode token IDs to nucleotide string."""
    tokens = tokenizer.convert_ids_to_tokens(input_ids.cpu().numpy().flatten())
    # Filter special tokens
    seq = "".join([t for t in tokens if t not in ["[CLS]", "[SEP]", "[PAD]", "<pad>", "<cls>", "<eos>"]])
    return seq


def extract_motifs(attributions, sequences, gene_ids, labels, window_size=15, top_n=5000):
    """Extract high-attribution windows from responsive genes."""
    motifs = []
    responsive_idx = np.where(np.array(labels) == 1)[0]

    for idx in responsive_idx:
        attr = np.abs(attributions[idx])
        seq = sequences[idx]
        gid = gene_ids[idx]

        for pos in range(0, len(attr) - window_size):
            score = np.mean(attr[pos:pos+window_size])
            if score > 0:
                motif_seq = seq[pos:pos+window_size]
                if len(motif_seq) == window_size and "N" not in motif_seq:
                    motifs.append((motif_seq, score, gid, pos))

    motifs.sort(key=lambda x: -x[1])
    return motifs[:top_n]


def save_motifs_meme(motifs, output_file, top_n=30):
    """Save motifs in MEME format."""
    kmer_groups = {}
    for seq, score, gid, pos in motifs[:top_n * 10]:
        for k in range(0, len(seq) - 7):
            kmer = seq[k:k+8]
            if kmer not in kmer_groups:
                kmer_groups[kmer] = []
            kmer_groups[kmer].append((seq, score))

    kmer_scores = sorted([(k, sum(s for _, s in v), len(v), v) for k, v in kmer_groups.items()],
                         key=lambda x: -x[1])

    with open(output_file, "w") as f:
        f.write("MEME version 5\n\nALPHABET= ACGU\n\nstrands: +\n\n")
        for i, (kmer, total, count, seqs) in enumerate(kmer_scores[:top_n]):
            f.write(f"MOTIF motif_{i+1} {kmer}\n")
            f.write(f"letter-probability matrix: alength= 4 w= {len(kmer)} nsites= {count}\n")
            for pos in range(len(kmer)):
                counts = Counter(seq[pos] for seq, _ in seqs[:100] if pos < len(seq))
                total_c = sum(counts.values()) or 1
                f.write(f" {counts.get('A',0)/total_c:.4f} {counts.get('C',0)/total_c:.4f} "
                       f"{counts.get('G',0)/total_c:.4f} {counts.get('U',0)/total_c:.4f}\n")
            f.write("\n")
    print(f"    Saved {min(top_n, len(kmer_scores))} motifs to {output_file}")


# ---- Load tokenizer ----
print(f"Device: {DEVICE}")
print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)

# ---- Load gene sequences as RNA ----
print("Loading gene sequences...")
gene_rna = {}  # gene_id -> RNA string (1024bp)

for fa in sorted(GENE_SEQS_DIR.glob("*.fa")):
    sp = fa.stem
    gid = None
    seq_parts = []
    with open(fa) as f:
        for line in f:
            if line.startswith(">"):
                if gid and seq_parts:
                    full_seq = "".join(seq_parts)
                    if len(full_seq) >= 5000 + MAX_SEQ_LEN:
                        dna = full_seq[5000:5000 + MAX_SEQ_LEN]
                        gene_rna[gid] = dna.replace("T", "U")
                gid = line[1:].split()[0]
                seq_parts = []
            else:
                seq_parts.append(line.strip())
        if gid and seq_parts:
            full_seq = "".join(seq_parts)
            if len(full_seq) >= 5000 + MAX_SEQ_LEN:
                gene_rna[gid] = full_seq[5000:5000 + MAX_SEQ_LEN].replace("T", "U")
    print(f"  {sp}: done")

print(f"Total: {len(gene_rna)} genes\n")

# ---- Check for LUMI model weights ----
# The _best.pt files should be in the LUMI results directory
lumi_model_dir = Path("/Users/vjx443/Downloads/fusion/rna")
print("Available model weights:")
for f in sorted(lumi_model_dir.glob("*_best.pt")):
    print(f"  {f.name}")

# ---- Main loop ----
all_results = []

for stress in STRESSES:
    for direction in DIRECTIONS:
        tag = f"{stress}_{direction}"
        motif_path = MOTIF_DIR / f"{tag}_motifs.txt"
        meme_path = MOTIF_DIR / f"{tag}_motifs.meme"
        attr_path = ATTR_DIR / f"{tag}_attributions.npz"

        # Check for model weights
        model_weights = lumi_model_dir / f"{tag}_best.pt"
        if not model_weights.exists():
            print(f"{tag}: no model weights, skipping")
            continue

        dataset_file = DATASET_DIR / f"dataset_{tag}.pkl"
        if not dataset_file.exists():
            print(f"{tag}: no dataset, skipping")
            continue

        if attr_path.exists():
            print(f"{tag}: attributions already computed, loading...")
            data = np.load(attr_path, allow_pickle=True)
            attributions = data["attributions"]
            test_seqs = list(data["sequences"])
            test_ids = list(data["gene_ids"])
            test_labs = list(data["labels"])
            print(f"  Loaded {len(attributions)} attributions")
        else:
            print(f"\n{'='*60}")
            print(f"PlantRNA-FM: {tag}")
            print(f"{'='*60}")

            with open(dataset_file, "rb") as f:
                splits = pickle.load(f)
            test_gids, test_labs_raw = splits["test"]

            # Get test sequences (limit to 2000 for speed)
            test_ids, test_seqs, test_labs = [], [], []
            for gid, lab in zip(test_gids, test_labs_raw):
                if gid in gene_rna:
                    test_ids.append(gid)
                    test_seqs.append(gene_rna[gid])
                    test_labs.append(lab)
                if len(test_ids) >= 2000:
                    break

            print(f"  Test: {len(test_ids)} genes")

            # Load model
            print(f"  Loading model...")
            base = AutoModel.from_pretrained(MODEL_NAME, trust_remote_code=True)
            lora_config = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.1,
                                     target_modules=["query", "value"], bias="none")
            base = get_peft_model(base, lora_config)
            model = PlantRNAClassifier(base, base.config.hidden_size).to(DEVICE)
            model.load_state_dict(torch.load(model_weights, map_location=DEVICE, weights_only=True), strict=False)
            model.eval()

            # Compute attributions using simple gradient * input
            print(f"  Computing gradient attributions ({len(test_ids)} genes)...")
            attributions = []
            for i, seq in enumerate(test_seqs):
                enc = tokenizer(seq, max_length=MAX_SEQ_LEN, padding="max_length",
                               truncation=True, return_tensors="pt")
                ids = enc["input_ids"].to(DEVICE)
                mask = enc["attention_mask"].to(DEVICE)

                attr = simple_gradient_attribution(model, ids, mask)
                # Only keep non-padding positions
                seq_len = mask.sum().item()
                attributions.append(attr[:int(seq_len)])

                if (i + 1) % 200 == 0:
                    print(f"    {i+1}/{len(test_ids)}")

            # Save
            np.savez_compressed(attr_path,
                               attributions=np.array(attributions, dtype=object),
                               sequences=np.array(test_seqs),
                               gene_ids=np.array(test_ids),
                               labels=np.array(test_labs))
            print(f"  Saved attributions")

            del model, base
            gc.collect()
            if DEVICE.type == "mps":
                torch.mps.empty_cache()

        # Extract motifs
        print(f"  Extracting motifs...")
        motifs = extract_motifs(attributions, test_seqs, test_ids, test_labs,
                               window_size=15, top_n=5000)

        with open(motif_path, "w") as f:
            f.write("rank\tsequence\tscore\tgene_id\tposition\n")
            for i, (seq, score, gid, pos) in enumerate(motifs[:500]):
                f.write(f"{i+1}\t{seq}\t{score:.6f}\t{gid}\t{pos}\n")

        save_motifs_meme(motifs, meme_path, top_n=30)

        # Top 8-mers
        if motifs:
            kmers = Counter()
            for seq, _, _, _ in motifs[:100]:
                for k in range(len(seq) - 7):
                    kmers[seq[k:k+8]] += 1
            top_kmers = kmers.most_common(5)
            print(f"  Top 8-mers:")
            for kmer, count in top_kmers:
                print(f"    {kmer}: {count}")
            top_kmer = top_kmers[0][0] if top_kmers else ""
        else:
            top_kmer = ""

        all_results.append({"stress": stress, "direction": direction, "top_kmer": top_kmer})

# Summary
print(f"\n{'='*60}")
print("Summary")
print(f"{'='*60}")
df = pd.DataFrame(all_results)
print(df.to_string(index=False))
df.to_csv(RESULTS_DIR / "rna_motif_summary.csv", index=False)
print(f"\nSaved to {RESULTS_DIR}")
