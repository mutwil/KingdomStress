#!/usr/bin/env python3
"""
Pre-build training tensors for each stress x direction.
Reads token cache + dataset pickles once, outputs one .pt file per model
containing train/val/test tensors ready for DataLoader.

Run on CPU node before GPU training:
    python preprocess_data.py --data_dir /scratch/.../data --output_dir /scratch/.../data/ready
"""

import argparse
import pickle
import gc
import numpy as np
import torch
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--tok_cache", default=None, help="Token cache dir (overrides data_dir/rna_tokens)")
    parser.add_argument("--max_seq_len", type=int, default=1024)
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    tok_cache = Path(args.tok_cache) if args.tok_cache else data_dir / "rna_tokens"

    # Step 1: Build token index (gene_id -> (species, index))
    print("Building token index...")
    gene_token_ids = {}
    for pt_file in sorted(tok_cache.glob("*.pt")):
        sp = pt_file.stem
        data = torch.load(pt_file, weights_only=False)
        for i, gid in enumerate(data["gene_ids"]):
            gene_token_ids[gid] = (sp, i)
        del data
        gc.collect()
    print(f"  {len(gene_token_ids)} genes indexed")

    # Step 2: For each dataset pickle, extract tensors and save
    for pkl_file in sorted(data_dir.glob("dataset_*.pkl")):
        tag = pkl_file.stem.replace("dataset_", "")
        out_file = output_dir / f"{tag}.pt"

        if out_file.exists():
            print(f"  {tag}: already done")
            continue

        with open(pkl_file, "rb") as f:
            splits = pickle.load(f)

        print(f"\n  {tag}: extracting tensors...")
        sp_cache = {}  # lazy-load species tensors

        result = {}
        for split_name in ["train", "val", "test"]:
            gids, labs = splits[split_name]

            ids_list = []
            masks_list = []
            labs_list = []

            for gid, lab in zip(gids, labs):
                if gid not in gene_token_ids:
                    continue
                sp, idx = gene_token_ids[gid]
                if sp not in sp_cache:
                    sp_cache[sp] = torch.load(tok_cache / f"{sp}.pt", weights_only=False)
                ids_list.append(sp_cache[sp]["input_ids"][idx])
                masks_list.append(sp_cache[sp]["attention_mask"][idx])
                labs_list.append(lab)

            if ids_list:
                result[f"{split_name}_ids"] = torch.stack(ids_list)
                result[f"{split_name}_masks"] = torch.stack(masks_list)
                result[f"{split_name}_labels"] = torch.tensor(labs_list, dtype=torch.float32)
                print(f"    {split_name}: {len(ids_list)} samples")
            else:
                print(f"    {split_name}: 0 samples!")

        torch.save(result, out_file)
        print(f"    Saved: {out_file} ({out_file.stat().st_size / 1e9:.1f} GB)")

        # Free species cache periodically
        del sp_cache, result
        gc.collect()

    print("\nDone.")


if __name__ == "__main__":
    main()
