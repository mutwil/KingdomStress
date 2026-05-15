#!/usr/bin/env python3
"""
Pre-tokenize all gene sequences with PlantRNA-FM tokenizer.
Saves per-species .pt files to data/rna_tokens/.

Usage:
    python tokenize_rna.py --gene_seqs_dir data/gene_seqs --output_dir data/rna_tokens
"""

import argparse
import gc
import re
from pathlib import Path
import torch
from transformers import AutoTokenizer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gene_seqs_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--model_name", default="yangheng/PlantRNA-FM")
    parser.add_argument("--max_seq_len", type=int, default=1024)
    args = parser.parse_args()

    gene_seqs_dir = Path(args.gene_seqs_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading tokenizer from {args.model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)

    for fa in sorted(gene_seqs_dir.glob("*.fa")):
        sp = fa.stem
        cache_file = output_dir / f"{sp}.pt"
        if cache_file.exists():
            print(f"  {sp}: cached")
            continue

        # Read sequences, take first 1024bp downstream of TSS
        # Gene seqs are 10kb: 5kb up + 5kb down. Position 5000 = TSS.
        gids, seqs = [], []
        gid = None
        seq_parts = []
        with open(fa) as f:
            for line in f:
                if line.startswith(">"):
                    if gid and seq_parts:
                        full_seq = "".join(seq_parts)
                        if len(full_seq) >= 5000 + args.max_seq_len:
                            dna = full_seq[5000:5000 + args.max_seq_len]
                            rna = dna.replace("T", "U")
                            gids.append(gid)
                            seqs.append(rna)
                    gid = line[1:].split()[0]
                    seq_parts = []
                else:
                    seq_parts.append(line.strip())
            if gid and seq_parts:
                full_seq = "".join(seq_parts)
                if len(full_seq) >= 5000 + args.max_seq_len:
                    dna = full_seq[5000:5000 + args.max_seq_len]
                    gids.append(gid)
                    seqs.append(dna.replace("T", "U"))

        if not seqs:
            print(f"  {sp}: no sequences")
            continue

        # Batch tokenize in chunks
        all_ids, all_masks = [], []
        for start in range(0, len(seqs), 2000):
            end = min(start + 2000, len(seqs))
            enc = tokenizer(seqs[start:end], max_length=args.max_seq_len,
                           padding="max_length", truncation=True, return_tensors="pt")
            all_ids.append(enc["input_ids"])
            all_masks.append(enc["attention_mask"])

        torch.save({
            "gene_ids": gids,
            "input_ids": torch.cat(all_ids),
            "attention_mask": torch.cat(all_masks),
        }, cache_file)

        print(f"  {sp}: {len(gids)} tokenized, saved")
        del all_ids, all_masks
        gc.collect()

    print("\nDone.")
    total = sum(1 for _ in output_dir.glob("*.pt"))
    print(f"{total} species tokenized in {output_dir}")


if __name__ == "__main__":
    main()
