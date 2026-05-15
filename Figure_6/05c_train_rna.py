#!/usr/bin/env python3
"""
PlantRNA-FM fine-tuning for stress-responsive gene prediction.
Standalone script for LUMI (AMD MI250X / ROCm).

Usage:
    ROCR_VISIBLE_DEVICES=0 python train_rna.py --stress Heat --direction UP --data_dir /path/to/data --output_dir /path/to/results
"""

import argparse
import json
import gc
import os
import pickle
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoModel, AutoTokenizer
from peft import LoraConfig, get_peft_model
from sklearn.metrics import roc_auc_score, f1_score
from pathlib import Path
from collections import defaultdict


# ===========================================================================
# Model
# ===========================================================================

class PlantRNAClassifier(nn.Module):
    def __init__(self, base_model, hidden_size):
        super().__init__()
        self.base = base_model
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 1),
        )

    def forward(self, input_ids, attention_mask):
        outputs = self.base(input_ids=input_ids, attention_mask=attention_mask)
        hidden = outputs.last_hidden_state
        mask = attention_mask.unsqueeze(-1).float()
        pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1)
        return self.classifier(pooled).squeeze(-1)


def evaluate(model, loader, device):
    model.eval()
    all_probs, all_labels = [], []
    with torch.no_grad():
        for ids, mask, labels in loader:
            ids, mask = ids.to(device), mask.to(device)
            with torch.amp.autocast("cuda"):
                logits = model(ids, mask)
            all_probs.extend(torch.sigmoid(logits).cpu().numpy())
            all_labels.extend(labels.numpy())
    p, l = np.array(all_probs), np.array(all_labels)
    return {"auROC": roc_auc_score(l, p), "F1": f1_score(l, (p > 0.5).astype(int))}


# ===========================================================================
# Data loading
# ===========================================================================

def load_split_tensors(gids, labs, gene_token_ids, tok_cache, max_seq_len, max_samples=None):
    """Load pre-tokenized tensors for a list of gene IDs."""
    if max_samples and len(gids) > max_samples:
        idx = np.random.default_rng(42).choice(len(gids), max_samples, replace=False)
        gids = [gids[i] for i in idx]
        labs = [labs[i] for i in idx]

    sp_cache = {}
    all_ids = torch.zeros(len(gids), max_seq_len, dtype=torch.long)
    all_masks = torch.zeros(len(gids), max_seq_len, dtype=torch.long)

    for i, gid in enumerate(gids):
        if gid not in gene_token_ids:
            continue
        sp, idx = gene_token_ids[gid]
        if sp not in sp_cache:
            sp_cache[sp] = torch.load(tok_cache / f"{sp}.pt", weights_only=False)
        all_ids[i] = sp_cache[sp]["input_ids"][idx]
        all_masks[i] = sp_cache[sp]["attention_mask"][idx]

    del sp_cache
    return all_ids, all_masks, torch.tensor(labs, dtype=torch.float32)


# ===========================================================================
# Main
# ===========================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stress", required=True, help="Stress type (Heat, Cold, etc)")
    parser.add_argument("--direction", required=True, help="UP or DOWN")
    parser.add_argument("--data_dir", required=True, help="Directory with token cache and dataset pickles")
    parser.add_argument("--output_dir", required=True, help="Directory for results")
    parser.add_argument("--model_name", default="yangheng/PlantRNA-FM")
    parser.add_argument("--max_seq_len", type=int, default=1024)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--patience", type=int, default=2)
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--max_train", type=int, default=0, help="Max train samples (0=all)")
    parser.add_argument("--max_val", type=int, default=0, help="Max val samples (0=all)")
    parser.add_argument("--max_test", type=int, default=0, help="Max test samples (0=all)")
    parser.add_argument("--ready_dir", default=None, help="Directory with pre-built tensor files (overrides data_dir/ready)")
    parser.add_argument("--tok_cache", default=None, help="Directory with token cache (overrides data_dir/rna_tokens)")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    tok_cache = Path(args.tok_cache) if args.tok_cache else data_dir / "rna_tokens"

    tag = f"{args.stress}_{args.direction}"
    result_file = output_dir / f"{tag}_result.json"

    # Skip if already done
    if result_file.exists():
        with open(result_file) as f:
            r = json.load(f)
        print(f"{tag}: already done (auROC={r['test_auROC']:.4f})")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name()}")

    # Load pre-built tensors (from preprocess_data.py)
    ready_dir = Path(args.ready_dir) if args.ready_dir else data_dir / "ready"
    ready_file = ready_dir / f"{tag}.pt"

    if ready_file.exists():
        print(f"Loading pre-built tensors from {ready_file}...")
        data = torch.load(ready_file, weights_only=False)
        train_ids, train_mask, train_labels_t = data["train_ids"], data["train_masks"], data["train_labels"]
        val_ids, val_mask, val_labels_t = data["val_ids"], data["val_masks"], data["val_labels"]
        test_ids, test_mask, test_labels_t = data["test_ids"], data["test_masks"], data["test_labels"]
        del data
    else:
        # Fallback: load from token cache + dataset pickle
        print("No pre-built tensors, loading from token cache...")
        dataset_file = data_dir / f"dataset_{tag}.pkl"
        if not dataset_file.exists():
            print(f"{tag}: no dataset file at {dataset_file}")
            return

        with open(dataset_file, "rb") as f:
            splits = pickle.load(f)

        train_gids, train_labs = splits["train"]
        val_gids, val_labs = splits["val"]
        test_gids, test_labs = splits["test"]

        gene_token_ids = {}
        for pt_file in sorted(tok_cache.glob("*.pt")):
            sp = pt_file.stem
            d = torch.load(pt_file, weights_only=False)
            for i, gid in enumerate(d["gene_ids"]):
                gene_token_ids[gid] = (sp, i)
            del d
        print(f"  {len(gene_token_ids)} genes indexed")

        max_train = args.max_train if args.max_train > 0 else None
        max_val = args.max_val if args.max_val > 0 else None
        max_test = args.max_test if args.max_test > 0 else None

        train_ids, train_mask, train_labels_t = load_split_tensors(
            train_gids, train_labs, gene_token_ids, tok_cache, args.max_seq_len, max_train)
        val_ids, val_mask, val_labels_t = load_split_tensors(
            val_gids, val_labs, gene_token_ids, tok_cache, args.max_seq_len, max_val)
        test_ids, test_mask, test_labels_t = load_split_tensors(
            test_gids, test_labs, gene_token_ids, tok_cache, args.max_seq_len, max_test)
        del gene_token_ids

    print(f"  Train: {len(train_ids)}, Val: {len(val_ids)}, Test: {len(test_ids)}")

    train_loader = DataLoader(TensorDataset(train_ids, train_mask, train_labels_t),
                              batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(val_ids, val_mask, val_labels_t),
                            batch_size=args.batch_size)
    test_loader = DataLoader(TensorDataset(test_ids, test_mask, test_labels_t),
                             batch_size=args.batch_size)

    del train_ids, train_mask, val_ids, val_mask, test_ids, test_mask
    gc.collect()

    # Build model
    print(f"Loading {args.model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    base = AutoModel.from_pretrained(args.model_name, trust_remote_code=True)
    hidden_size = base.config.hidden_size

    lora_config = LoraConfig(
        r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=0.1,
        target_modules=["query", "value"], bias="none",
    )
    base = get_peft_model(base, lora_config)
    trainable = sum(p.numel() for p in base.parameters() if p.requires_grad)
    total = sum(p.numel() for p in base.parameters())
    print(f"  Trainable: {trainable:,} / {total:,} ({100*trainable/total:.1f}%)")

    model = PlantRNAClassifier(base, hidden_size).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    criterion = nn.BCEWithLogitsLoss()
    scaler = torch.amp.GradScaler("cuda")

    # Train
    best_val_auroc = 0
    patience_counter = 0
    best_path = output_dir / f"{tag}_best.pt"

    print(f"\nTraining {tag}...")
    for epoch in range(args.epochs):
        model.train()
        total_loss, n = 0, 0
        for ids, mask, labels in train_loader:
            ids, mask, labels = ids.to(device), mask.to(device), labels.to(device)
            optimizer.zero_grad()
            with torch.amp.autocast("cuda"):
                logits = model(ids, mask)
                loss = criterion(logits, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total_loss += loss.item() * len(labels)
            n += len(labels)

        val_res = evaluate(model, val_loader, device)
        print(f"  Epoch {epoch+1}: loss={total_loss/n:.4f}, "
              f"val auROC={val_res['auROC']:.4f}, F1={val_res['F1']:.4f}")

        if val_res["auROC"] > best_val_auroc:
            best_val_auroc = val_res["auROC"]
            torch.save(model.state_dict(), best_path)
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"  Early stopping at epoch {epoch+1}")
                break

    # Test
    model.load_state_dict(torch.load(best_path, weights_only=True))
    test_res = evaluate(model, test_loader, device)
    print(f"\n  TEST: auROC={test_res['auROC']:.4f}, F1={test_res['F1']:.4f}")

    # Save per-gene predictions for late fusion
    model.eval()
    all_probs, all_labels = [], []
    with torch.no_grad():
        for ids, mask, labels in test_loader:
            ids, mask = ids.to(device), mask.to(device)
            with torch.amp.autocast("cuda"):
                logits = model(ids, mask)
            all_probs.extend(torch.sigmoid(logits).cpu().numpy())
            all_labels.extend(labels.numpy())

    import pandas as pd
    pred_file = output_dir / f"{tag}_predictions.csv"
    pd.DataFrame({"label": all_labels, "prediction": all_probs}).to_csv(pred_file, index=False)
    print(f"  Predictions saved: {pred_file}")

    result = {
        "stress": args.stress, "direction": args.direction,
        "test_auROC": test_res["auROC"], "test_F1": test_res["F1"],
        "val_auROC": best_val_auroc,
        "n_train": len(train_labels_t), "n_test": len(test_labels_t),
    }
    with open(result_file, "w") as f:
        json.dump(result, f, indent=2)
    print(f"  Saved: {result_file}")


if __name__ == "__main__":
    main()
