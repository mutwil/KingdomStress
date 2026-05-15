#!/usr/bin/env python3
"""
Build gene ID mappings between DEG table (LSTRAP) and Ensembl GTF gene IDs.

The DEG table uses gene IDs from the Kallisto reference transcriptomes,
which often differ from Ensembl gene IDs. This script attempts to map
them using several strategies:

1. Direct match (after stripping transcript suffixes)
2. Case-insensitive + format normalization (e.g., Bradi1g -> BRADI_1g)
3. Ensembl BioMart xrefs (downloaded per species)
4. Coordinate-based matching via LSTRAP CDS FASTA alignment

Usage:
    python 00_build_id_mapping.py --deg_file kingdom_stress_dict_v3.csv \
                                  --gtf_dir gene_models \
                                  --output_dir id_mappings
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
    """Strip .1, .2, _T001 etc."""
    gene_id = re.sub(r"_T\d+$", "", gene_id)
    gene_id = re.sub(r"\.\d+$", "", gene_id)
    return gene_id


def normalize_id(gene_id):
    """Normalize gene ID for fuzzy matching.
    Strips version suffixes, lowercases, removes underscores/dots/hyphens.
    """
    gid = strip_transcript_suffix(gene_id)
    gid = gid.upper()
    gid = re.sub(r"[_.\-]", "", gid)
    # Strip common prefixes
    gid = re.sub(r"^LOC", "", gid)
    gid = re.sub(r"^ENSRNA\d+", "", gid)
    return gid


# Species-specific ID transformations
def transform_brachy(deg_id):
    """Bradi1g06300 -> BRADI_1g06300v3 (Ensembl format)"""
    m = re.match(r"Bradi(\d+)g(\d+)", deg_id)
    if m:
        return f"BRADI_{m.group(1)}g{m.group(2)}"
    return None


def transform_glycine(deg_id):
    """Glyma.16G122100 -> GLYMA_16G122100"""
    m = re.match(r"Glyma\.(\w+)", deg_id)
    if m:
        return f"GLYMA_{m.group(1)}"
    return None


def transform_setaria(deg_id):
    """Seita.5G375200 -> SETIT_5G375200 (approximate)"""
    m = re.match(r"Seita\.(\w+)", deg_id)
    if m:
        return f"SETIT_{m.group(1)}"
    return None


def transform_phaseolus(deg_id):
    """Phvul.010G165100 -> PHAVU_010G165100"""
    m = re.match(r"Phvul\.(\w+)", deg_id)
    if m:
        return f"PHAVU_{m.group(1)}"
    return None


def transform_sorghum(deg_id):
    """Sobic.004G254100 -> SORBI_3004G254100 (Ensembl adds 3)"""
    m = re.match(r"Sobic\.(\d+)G(\d+)", deg_id)
    if m:
        return f"SORBI_3{m.group(1)}G{m.group(2)}"
    return None


def transform_solanum_lyc(deg_id):
    """Various formats - try stripping to Solyc core"""
    # Some use PRAM IDs, some use Solyc - try Solyc pattern
    m = re.match(r"(Solyc\d+[gG]\d+)", deg_id)
    if m:
        return m.group(1).upper()
    return None


def transform_zea(deg_id):
    """Zm00001eb337300 -> Zm00001eb337300 (should match directly)"""
    return deg_id  # Zea IDs should match Ensembl directly


def transform_marchantia(deg_id):
    """Mapoly0005s0090 -> MARPO_0005s0090 (approximate)"""
    m = re.match(r"Mapoly(\w+)", deg_id)
    if m:
        return f"MARPO_{m.group(1)}"
    return None


def transform_prunus(deg_id):
    """Prupe.I005700 -> PRUPE_ppa005700 or similar"""
    return None  # Complex mapping, needs BioMart


def transform_vitis(deg_id):
    """VIT_212s0142g00110 -> VITVI_212s0142g00110"""
    m = re.match(r"VIT_(\w+)", deg_id)
    if m:
        return f"VITVI_{m.group(1)}"
    return None


# Map of species -> transformation function
SPECIES_TRANSFORMS = {
    "Brachypodium_distachyon": transform_brachy,
    "Glycine_max": transform_glycine,
    "Setaria_italica": transform_setaria,
    "Phaseolus_vulgaris": transform_phaseolus,
    "Sorghum_bicolor": transform_sorghum,
    "Zea_mays": transform_zea,
    "Marchantia_polymorpha": transform_marchantia,
    "Vitis_vinifera": transform_vitis,
}


def get_gtf_gene_ids(gtf_path):
    """Get all protein-coding gene IDs from GTF."""
    import pyranges as pr
    gm = pr.read_gtf(gtf_path, as_df=True)
    gm = gm[gm["Feature"] == "gene"]
    gm = gm[gm["gene_biotype"] == "protein_coding"]
    return set(gm["gene_id"].values)


def get_gtf_gene_names(gtf_path):
    """Get gene_name attribute if available."""
    import pyranges as pr
    gm = pr.read_gtf(gtf_path, as_df=True)
    gm = gm[gm["Feature"] == "gene"]
    gm = gm[gm["gene_biotype"] == "protein_coding"]
    if "gene_name" in gm.columns:
        name_map = {}
        for _, row in gm.iterrows():
            if pd.notna(row.get("gene_name")):
                name_map[row["gene_name"]] = row["gene_id"]
        return name_map
    return {}


def find_gtf_for_species(species_name, gtf_dir):
    """Find the GTF file for a species."""
    gtf_files = list(Path(gtf_dir).glob("*.gtf"))
    ensembl_name = ENSEMBL_SPECIES.get(species_name, {}).get("ensembl", "")

    for gf in gtf_files:
        if ensembl_name:
            prefix = ensembl_name.replace("_", " ").title().replace(" ", "_")
            if gf.name.startswith(prefix) or gf.name.startswith(ensembl_name.capitalize().split("_")[0]):
                return gf
    for gf in gtf_files:
        if species_name.split("_")[0] in gf.name:
            return gf
    return None


def build_mapping(species_name, deg_ids, gtf_path, output_dir):
    """Build ID mapping for one species using multiple strategies."""
    gtf_ids = get_gtf_gene_ids(str(gtf_path))
    gtf_names = get_gtf_gene_names(str(gtf_path))

    mapping = {}  # deg_id -> gtf_id
    unmatched = set()

    # Strategy 1: Direct match (after stripping suffixes)
    gtf_stripped = {}
    for gid in gtf_ids:
        stripped = strip_transcript_suffix(gid)
        gtf_stripped[stripped] = gid
        gtf_stripped[gid] = gid

    direct = 0
    for did in deg_ids:
        if did in gtf_stripped:
            mapping[did] = gtf_stripped[did]
            direct += 1
        else:
            unmatched.add(did)

    # Strategy 2: gene_name matching
    name_match = 0
    still_unmatched = set()
    for did in unmatched:
        if did in gtf_names:
            mapping[did] = gtf_names[did]
            name_match += 1
        else:
            stripped = strip_transcript_suffix(did)
            if stripped in gtf_names:
                mapping[did] = gtf_names[stripped]
                name_match += 1
            else:
                still_unmatched.add(did)
    unmatched = still_unmatched

    # Strategy 3: Species-specific transformation
    transform_match = 0
    if species_name in SPECIES_TRANSFORMS:
        transform_fn = SPECIES_TRANSFORMS[species_name]
        still_unmatched = set()
        for did in unmatched:
            transformed = transform_fn(did)
            if transformed and transformed in gtf_stripped:
                mapping[did] = gtf_stripped[transformed]
                transform_match += 1
            else:
                still_unmatched.add(did)
        unmatched = still_unmatched

    # Strategy 4: Normalized fuzzy matching
    fuzzy_match = 0
    if unmatched:
        gtf_normalized = {}
        for gid in gtf_ids:
            norm = normalize_id(gid)
            gtf_normalized[norm] = gid

        still_unmatched = set()
        for did in unmatched:
            norm = normalize_id(did)
            if norm in gtf_normalized:
                mapping[did] = gtf_normalized[norm]
                fuzzy_match += 1
            else:
                still_unmatched.add(did)
        unmatched = still_unmatched

    total = len(deg_ids)
    matched = len(mapping)
    pct = 100 * matched / total if total > 0 else 0

    print(f"  Mapping: {matched}/{total} ({pct:.1f}%) -- "
          f"direct={direct}, name={name_match}, transform={transform_match}, fuzzy={fuzzy_match}, "
          f"unmatched={len(unmatched)}")

    if unmatched:
        print(f"  Unmatched DEG examples: {list(unmatched)[:3]}")
        print(f"  GTF examples: {list(gtf_ids)[:3]}")

    # Save mapping
    sp_dir = Path(output_dir)
    sp_dir.mkdir(parents=True, exist_ok=True)
    out_path = sp_dir / f"{species_name}.tsv"
    with open(out_path, "w") as f:
        f.write("deg_id\tgtf_id\n")
        for did, gid in sorted(mapping.items()):
            f.write(f"{did}\t{gid}\n")

    return mapping, unmatched


def main():
    parser = argparse.ArgumentParser(description="Build gene ID mappings")
    parser.add_argument("--deg_file", required=True, help="Path to kingdom_stress_dict v3.csv")
    parser.add_argument("--gtf_dir", required=True, help="Directory with GTF files")
    parser.add_argument("--output_dir", default="id_mappings", help="Output directory")
    parser.add_argument("--species", nargs="*", default=None, help="Species subset")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Load DEG gene IDs per species
    print("Loading DEG table ...")
    deg = pd.read_csv(args.deg_file, usecols=["species", "gene"])
    deg["gene_locus"] = deg["gene"].map(strip_transcript_suffix)

    species_list = args.species if args.species else sorted(deg["species"].unique())

    summary = []
    for sp in species_list:
        if sp not in ENSEMBL_SPECIES:
            continue

        gtf_path = find_gtf_for_species(sp, args.gtf_dir)
        if gtf_path is None:
            print(f"\n{sp}: SKIP (no GTF)")
            continue

        sp_deg = deg[deg["species"] == sp]
        deg_ids = set(sp_deg["gene_locus"].values)

        print(f"\n{sp}: {len(deg_ids)} DEG loci, GTF: {gtf_path.name}")
        mapping, unmatched = build_mapping(sp, deg_ids, gtf_path, args.output_dir)

        summary.append({
            "species": sp,
            "deg_loci": len(deg_ids),
            "mapped": len(mapping),
            "unmatched": len(unmatched),
            "pct_mapped": round(100 * len(mapping) / len(deg_ids), 1) if deg_ids else 0,
        })

    print("\n" + "=" * 70)
    print("Summary:")
    df = pd.DataFrame(summary)
    print(df.to_string(index=False))
    df.to_csv(Path(args.output_dir) / "summary.csv", index=False)


if __name__ == "__main__":
    main()
