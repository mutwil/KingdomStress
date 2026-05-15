#!/usr/bin/env python3
"""
Prepare DEG-based labels for DeepCRE CNN from kingdom_stress_dict.

For each species x stress:
- Compute DEG frequency = fraction of experiments where gene is DEG.
- Label genes by percentile: top 25% DEG frequency = "responsive" (1),
  bottom 25% = "non-responsive" (0), middle 50% excluded (2).
- This mirrors DeepCRE's logMaxTPM percentile approach.

Gene IDs are stripped to locus level (remove .1/.2 transcript suffixes).
Full gene list from GTF annotation (protein-coding only).

Usage:
    python 01_prepare_labels.py --deg_file kingdom_stress_dict_v3.csv \
                                --gtf_dir gene_models --output_dir labels
    python 01_prepare_labels.py --species Arabidopsis_thaliana
    python 01_prepare_labels.py --threshold median  # above/below median instead
"""

import argparse
import os
import sys
import re
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, os.path.dirname(__file__))
from species_config import ENSEMBL_SPECIES


def strip_transcript_suffix(gene_id):
    """Strip .1, .2, etc. transcript suffix to get locus ID."""
    gene_id = re.sub(r"_T\d+$", "", gene_id)
    gene_id = re.sub(r"\.\d+$", "", gene_id)
    return gene_id


def get_protein_coding_genes(gtf_path):
    """Extract all protein-coding gene IDs from GTF."""
    import pyranges as pr
    gm = pr.read_gtf(gtf_path, as_df=True)
    gm = gm[gm["Feature"] == "gene"]
    gm = gm[gm["gene_biotype"] == "protein_coding"]
    return set(gm["gene_id"].values)


def load_deg_table(deg_file, species_list=None, organ=None):
    """Load DEG table. Returns DataFrame with species, stress, experiment, gene_locus."""
    print(f"Loading DEG table from {deg_file} ...")
    cols = ["species", "stress", "organ", "experiment", "direction", "gene"]
    df = pd.read_csv(deg_file, usecols=cols)
    print(f"  {len(df)} total DEG entries")

    if species_list:
        df = df[df["species"].isin(species_list)]
        print(f"  {len(df)} entries after species filter")

    if organ:
        df = df[df["organ"].str.lower().str.contains(organ.lower())]
        print(f"  {len(df)} entries after organ filter ({organ})")

    df["stress"] = df["stress"].str.replace("High Light", "High light")
    df["gene_locus"] = df["gene"].map(strip_transcript_suffix)
    return df


def load_id_mapping(mapping_dir, species_name):
    """Load gene ID mapping (deg_id -> gtf_id) for a species."""
    mapping_file = Path(mapping_dir) / f"{species_name}.tsv"
    if not mapping_file.exists():
        return None
    df = pd.read_csv(mapping_file, sep="\t")
    return dict(zip(df["deg_id"], df["gtf_id"]))


def apply_id_mapping(deg_df, species_name, id_mapping):
    """Remap DEG gene IDs to GTF gene IDs for a species."""
    mask = deg_df["species"] == species_name
    mapped = deg_df.loc[mask, "gene_locus"].map(lambda x: id_mapping.get(x, x))
    deg_df.loc[mask, "gene_locus"] = mapped
    n_mapped = mask.sum()
    n_changed = (deg_df.loc[mask, "gene_locus"] != deg_df.loc[mask, "gene"].map(strip_transcript_suffix)).sum()
    return n_changed


def compute_deg_frequency(deg_df, species_name, stress_name, all_genes, direction=None):
    """Compute fraction of experiments where each gene is DEG.

    Args:
        direction: None (any direction), "UP", or "DOWN"

    Returns (freq_series, n_experiments) or None.
    Genes not in any experiment get frequency 0.
    """
    subset = deg_df[(deg_df["species"] == species_name) & (deg_df["stress"] == stress_name)]
    if len(subset) == 0:
        return None

    # Total experiments for this species+stress (before direction filter)
    n_experiments = subset["experiment"].nunique()
    if n_experiments < 2:
        return None

    # Filter by direction if specified
    if direction is not None:
        subset = subset[subset["direction"] == direction]
        if len(subset) == 0:
            return None

    # Count how many experiments each gene is DEG (in this direction) in
    gene_exp_counts = subset.groupby("gene_locus")["experiment"].nunique()

    # Build frequency series for all genes
    freq = pd.Series(0.0, index=sorted(all_genes))
    common = set(gene_exp_counts.index) & all_genes
    for g in common:
        freq[g] = gene_exp_counts[g] / n_experiments

    return freq, n_experiments


def label_by_percentile(freq_series, method="quartile"):
    """Assign labels based on DEG frequency percentiles.

    method='quartile': top 25% = 1, bottom 25% = 0, middle = 2
    method='median': above median = 1, below median = 0

    Returns labels Series, or None if labeling fails (e.g., all values tied).
    """
    n_unique = freq_series.nunique()
    n_nonzero = (freq_series > 0).sum()
    n_total = len(freq_series)

    # Sanity check: if <5% of genes have non-zero frequency, ID mapping likely failed
    if n_nonzero < n_total * 0.05:
        return None

    if method == "quartile":
        p25 = np.percentile(freq_series, 25)
        p75 = np.percentile(freq_series, 75)
        # If percentiles are equal, labeling is meaningless
        if p25 == p75:
            return None
        labels = pd.Series(2, index=freq_series.index)  # intermediate
        labels[freq_series <= p25] = 0  # non-responsive
        labels[freq_series >= p75] = 1  # responsive
    elif method == "median":
        med = np.median(freq_series)
        labels = pd.Series(2, index=freq_series.index)
        labels[freq_series < med] = 0
        labels[freq_series > med] = 1
        if (labels == 2).sum() == n_total:
            return None
    return labels


def process_species(species_name, deg_df, gtf_dir, output_dir, method="quartile"):
    """Create label files for one species."""
    # Find GTF
    gtf_files = list(Path(gtf_dir).glob("*.gtf"))
    gtf_path = None
    ensembl_name = ENSEMBL_SPECIES.get(species_name, {}).get("ensembl", "")
    for gf in gtf_files:
        # Match by Ensembl species name prefix (capitalized)
        if ensembl_name:
            prefix = ensembl_name.replace("_", " ").title().replace(" ", "_")
            if gf.name.startswith(prefix) or gf.name.startswith(ensembl_name.capitalize().split("_")[0]):
                gtf_path = gf
                break
    if gtf_path is None:
        for gf in gtf_files:
            if species_name.split("_")[0] in gf.name:
                gtf_path = gf
                break

    if gtf_path is None:
        print(f"  SKIP {species_name}: no GTF in {gtf_dir}")
        return False

    print(f"  GTF: {gtf_path.name}")
    all_genes = get_protein_coding_genes(str(gtf_path))
    print(f"  {len(all_genes)} protein-coding genes")

    sp_dir = Path(output_dir) / species_name
    sp_dir.mkdir(parents=True, exist_ok=True)

    # Get stresses for this species
    sp_deg = deg_df[deg_df["species"] == species_name]
    stresses = sorted(sp_deg["stress"].unique())
    if not stresses:
        print(f"  SKIP {species_name}: no DEG data")
        return False

    stress_summary = []

    for stress_name in stresses:
        # Generate labels for UP, DOWN, and combined (any direction)
        for direction in ["UP", "DOWN"]:
            result = compute_deg_frequency(deg_df, species_name, stress_name, all_genes,
                                           direction=direction)
            if result is None:
                print(f"    {stress_name}/{direction}: skipped (<2 experiments or no DEGs)")
                continue

            freq, n_exp = result
            labels = label_by_percentile(freq, method=method)

            if labels is None:
                n_nonzero = (freq > 0).sum()
                print(f"    {stress_name}/{direction}: {n_exp} experiments, SKIPPED -- "
                      f"labeling failed ({n_nonzero}/{len(freq)} nonzero)")
                continue

            out_df = pd.DataFrame({"deg_frequency": freq, "true_target": labels})
            out_df.index.name = "gene_id"

            safe_stress = stress_name.replace(" ", "_").replace("/", "_")
            out_df.to_csv(sp_dir / f"{safe_stress}_{direction}.csv")

            n_responsive = (labels == 1).sum()
            n_non_responsive = (labels == 0).sum()
            n_intermediate = (labels == 2).sum()
            print(f"    {stress_name}/{direction}: {n_exp} experiments, "
                  f"responsive={n_responsive}, non-responsive={n_non_responsive}, "
                  f"intermediate={n_intermediate}")

            stress_summary.append({
                "stress": stress_name,
                "direction": direction,
                "n_experiments": n_exp,
                "n_responsive": int(n_responsive),
                "n_non_responsive": int(n_non_responsive),
                "n_intermediate": int(n_intermediate),
                "n_total": len(labels),
                "mean_deg_freq": round(freq.mean(), 4),
                "p25_freq": round(np.percentile(freq, 25), 4),
                "p75_freq": round(np.percentile(freq, 75), 4),
            })

    # "any_stress" combined: max DEG frequency across all stresses, per direction
    for direction in ["UP", "DOWN"]:
        all_freqs = []
        for stress_name in stresses:
            result = compute_deg_frequency(deg_df, species_name, stress_name, all_genes,
                                           direction=direction)
            if result is not None:
                all_freqs.append(result[0])
        if all_freqs:
            max_freq = pd.concat(all_freqs, axis=1).max(axis=1)
            labels = label_by_percentile(max_freq, method=method)
            if labels is not None:
                out_df = pd.DataFrame({"deg_frequency": max_freq, "true_target": labels})
                out_df.index.name = "gene_id"
                out_df.to_csv(sp_dir / f"any_stress_{direction}.csv")
                n_r = (labels == 1).sum()
                print(f"    any_stress/{direction}: responsive={n_r}, "
                      f"non-responsive={(labels==0).sum()}")
            else:
                print(f"    any_stress/{direction}: SKIPPED -- labeling failed")

    if stress_summary:
        pd.DataFrame(stress_summary).to_csv(sp_dir / "summary.csv", index=False)
    return True


def main():
    parser = argparse.ArgumentParser(description="Prepare DEG-frequency labels for DeepCRE")
    parser.add_argument("--deg_file", required=True, help="Path to kingdom_stress_dict v3.csv")
    parser.add_argument("--gtf_dir", required=True, help="Directory with GTF files")
    parser.add_argument("--output_dir", default="labels", help="Output directory")
    parser.add_argument("--species", nargs="*", default=None, help="Species subset")
    parser.add_argument("--threshold", choices=["quartile", "median"], default="quartile",
                        help="Labeling method: quartile (25/75 pctl) or median")
    parser.add_argument("--organ", default=None,
                        help="Filter DEGs by organ (e.g., Leaf, Root). Case-insensitive substring match.")
    parser.add_argument("--id_mapping_dir", default=None,
                        help="Directory with gene ID mapping files from 00_build_id_mapping.py")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    species_list = args.species if args.species else list(ENSEMBL_SPECIES.keys())

    deg_df = load_deg_table(args.deg_file, species_list, organ=args.organ)
    print(f"DEG data for {deg_df['species'].nunique()} species")

    # Apply ID mappings if provided
    if args.id_mapping_dir:
        print(f"\nApplying ID mappings from {args.id_mapping_dir} ...")
        for sp in species_list:
            id_map = load_id_mapping(args.id_mapping_dir, sp)
            if id_map:
                n = apply_id_mapping(deg_df, sp, id_map)
                print(f"  {sp}: {n} IDs remapped ({len(id_map)} in mapping)")

    success, fail = 0, 0
    for sp in species_list:
        print(f"\nProcessing {sp} ...")
        if process_species(sp, deg_df, args.gtf_dir, args.output_dir, args.threshold):
            success += 1
        else:
            fail += 1

    print(f"\nDone: {success} processed, {fail} skipped")


if __name__ == "__main__":
    main()
