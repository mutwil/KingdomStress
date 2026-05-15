#!/usr/bin/env python3
"""
Tokenize gene sequences for different transcript windows.
Each window produces a separate set of per-species .pt files.

Windows:
  w0_1kb:   TSS to TSS+1024        (positions 5000-6024 in FASTA) [already done]
  w05_15kb: TSS+500 to TSS+1524    (positions 5500-6524)
  w1_2kb:   TSS+1000 to TSS+2024   (positions 6000-7024)
  w3end:    TTS-1024 to TTS         (positions 5000+gene_length-1024 to 5000+gene_length)

Usage:
    python tokenize_windows.py --gene_seqs_dir data/gene_seqs \
        --gene_lengths gene_lengths.csv \
        --output_base data/rna_tokens_windows \
        --window w05_15kb
"""

import argparse
import gc
from pathlib import Path
import pandas as pd
import torch
from transformers import AutoTokenizer


WINDOWS = {
    "w05_15kb": {"start_offset": 500, "end_offset": 1524, "use_gene_end": False},
    "w1_2kb":   {"start_offset": 1000, "end_offset": 2024, "use_gene_end": False},
    "w3end":    {"start_offset": None, "end_offset": None, "use_gene_end": True},
}

TSS_POS = 5000  # TSS position in the 10kb FASTA
MAX_SEQ_LEN = 1024


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gene_seqs_dir", required=True)
    parser.add_argument("--gene_lengths", required=True)
    parser.add_argument("--output_base", required=True)
    parser.add_argument("--window", required=True, choices=list(WINDOWS.keys()))
    parser.add_argument("--model_name", default="yangheng/PlantRNA-FM")
    args = parser.parse_args()

    gene_seqs_dir = Path(args.gene_seqs_dir)
    output_dir = Path(args.output_base) / args.window
    output_dir.mkdir(parents=True, exist_ok=True)
    win = WINDOWS[args.window]

    # Load gene lengths for 3' end window
    gene_lengths = {}
    if win["use_gene_end"]:
        df = pd.read_csv(args.gene_lengths)
        gene_lengths = dict(zip(df["gene_id"], df["gene_length"]))
        print(f"Loaded {len(gene_lengths)} gene lengths")

    print(f"Loading tokenizer from {args.model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)

    for fa in sorted(gene_seqs_dir.glob("*.fa")):
        sp = fa.stem
        cache_file = output_dir / f"{sp}.pt"
        if cache_file.exists():
            print(f"  {sp}: cached")
            continue

        gids, seqs = [], []
        gid = None
        seq_parts = []
        with open(fa) as f:
            for line in f:
                if line.startswith(">"):
                    if gid and seq_parts:
                        full_seq = "".join(seq_parts)
                        extracted = extract_window(full_seq, gid, win, gene_lengths)
                        if extracted:
                            gids.append(gid)
                            seqs.append(extracted)
                    gid = line[1:].split()[0]
                    seq_parts = []
                else:
                    seq_parts.append(line.strip())
            if gid and seq_parts:
                full_seq = "".join(seq_parts)
                extracted = extract_window(full_seq, gid, win, gene_lengths)
                if extracted:
                    gids.append(gid)
                    seqs.append(extracted)

        if not seqs:
            print(f"  {sp}: no valid sequences")
            continue

        # Tokenize in chunks
        all_ids, all_masks = [], []
        for start in range(0, len(seqs), 2000):
            end = min(start + 2000, len(seqs))
            enc = tokenizer(seqs[start:end], max_length=MAX_SEQ_LEN, padding="max_length",
                           truncation=True, return_tensors="pt")
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

    print(f"\nDone. Tokens in {output_dir}")


def extract_window(full_seq, gid, win, gene_lengths):
    """Extract the appropriate window from a 10kb gene sequence."""
    if win["use_gene_end"]:
        # 3' end window: last 1024bp of the gene
        gl = gene_lengths.get(gid)
        if gl is None or gl < MAX_SEQ_LEN:
            return None
        # Gene end position in FASTA = TSS_POS + gene_length
        gene_end_in_fasta = TSS_POS + gl
        start = gene_end_in_fasta - MAX_SEQ_LEN
        end = gene_end_in_fasta
        if start < 0 or end > len(full_seq):
            return None
    else:
        start = TSS_POS + win["start_offset"]
        end = TSS_POS + win["end_offset"]
        if end > len(full_seq):
            return None

    dna = full_seq[start:end]
    if len(dna) != MAX_SEQ_LEN:
        return None
    # Convert to RNA
    return dna.replace("T", "U")


if __name__ == "__main__":
    main()
