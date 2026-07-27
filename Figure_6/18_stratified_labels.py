#!/usr/bin/env python3
"""
Phase 0.2-0.4 + A2/A3 label construction for the Figure 6 revision.

Builds the scaffolding needed by the stratification sweep:

  0.2  species -> lineage map (Monocot / Rosid / Asterid / Bryophyta / Chlorophyta),
       persisted so it is no longer read from an ephemeral /tmp path.

  0.3  organ vocabulary harmonisation (the raw field has case variants, multi-tissue
       strings and at least one junk value), persisted as a supplementary table.

  A2   organ-restricted labels: DEG frequency and quartile thresholds recomputed using
       only experiments from one organ. This changes the LABEL, not just the training
       subset, which is what the reviewer's "training set from specific tissues" means.

  A3   balanced per-species labels: the original recipe already computes quartiles
       within species, but zero-inflation makes the realised positive rate depend on
       assay depth (Spearman rho = 0.771 with experiment count; spread up to 20.9x).
       Rank-based labelling with deterministic tie-breaking forces an equal positive
       rate in every species, removing that artefact by construction.

Faithful reimplementation of the original recipe in 02_prepare_labels.py:
  gene_locus   = strip_transcript_suffix(gene), then id_mappings/{species}.tsv
  n_experiments= distinct experiments for species+stress, BEFORE the direction filter
  freq         = distinct experiments where the gene is DEG (after direction filter)
                 / n_experiments; genes never DEG get 0
  quartile     = freq <= p25 -> 0, freq >= p75 -> 1, otherwise dropped

Two deliberate deviations, both documented and gated:
  * The gene universe comes from the feature table (genes that actually have
    sequence features) rather than a GTF via pyranges, which is not installed.
  * Split membership (train/val/test) is inherited from the existing pickles rather
    than re-deriving orthogroup splits, so every arm of the sweep is scored on an
    identical test set.

The all-organ quartile rebuild is compared against the original pickle labels; that
agreement is the gate that licenses trusting the organ-restricted rebuilds.

Usage
  python 18_stratified_labels.py \
      --deg_table    ".../kingdom_stress_dict v3_.csv" \
      --feature_table ".../feature_table_all_genes.csv.gz" \
      --id_mapping_dir ".../id_mappings" \
      --dataset_dir  ".../plantrna_fm_results" \
      --output_dir   ".../revision/stratified" \
      --schemes all_organ_rebuild,balanced,Leaf,Root

Kingdom Stress Atlas, Mutwil lab.
"""

import argparse
import csv
import gzip
import os
import pickle
import re
import sys
from collections import Counter, defaultdict

import numpy as np
import pandas as pd

STRESSES = ["Heat", "Cold", "Drought", "Salt", "Pathogen"]
DIRECTIONS = ["UP", "DOWN"]

# --------------------------------------------------------------------------
# 0.2  Species -> lineage.  Matches the 5-way scheme fixed in Part 1 of the
# revision plan (finer than the paper's 7-clade vocabulary: Dicot is split into
# Rosid / Asterid, which is what the Figure 6 species panel actually supports).
# --------------------------------------------------------------------------
LINEAGE_MAP = {
    # Monocots
    "Oryza_sativa": "Monocot",
    "Zea_mays": "Monocot",
    "Hordeum_vulgare": "Monocot",
    "Brachypodium_distachyon": "Monocot",
    "Sorghum_bicolor": "Monocot",
    "Setaria_italica": "Monocot",
    "Setaria_viridis": "Monocot",
    "Panicum_virgatum": "Monocot",
    "Cenchrus_americanus": "Monocot",
    "Triticum_aestivum": "Monocot",
    "Saccharum_spontaneum": "Monocot",
    "Lolium_perenne": "Monocot",
    "Elaeis_guineensis": "Monocot",
    "Paspalum_vaginatum": "Monocot",
    # Rosids
    "Arabidopsis_thaliana": "Rosid",
    "Brassica_napus": "Rosid",
    "Brassica_oleracea": "Rosid",
    "Brassica_rapa": "Rosid",
    "Camelina_sativa": "Rosid",
    "Glycine_max": "Rosid",
    "Medicago_truncatula": "Rosid",
    "Medicago_sativa": "Rosid",
    "Phaseolus_vulgaris": "Rosid",
    "Arachis_hypogaea": "Rosid",
    "Cicer_arietinum": "Rosid",
    "Vigna_radiata": "Rosid",
    "Cucumis_sativus": "Rosid",
    "Gossypium_hirsutum": "Rosid",
    "Prunus_persica": "Rosid",
    "Malus_domestica": "Rosid",
    "Rosa_chinensis": "Rosid",
    "Populus_trichocarpa": "Rosid",
    "Manihot_esculenta": "Rosid",
    "Jatropha_curcas": "Rosid",
    "Citrus_sinensis": "Rosid",
    "Vitis_vinifera": "Rosid",
    "Ziziphus_jujuba": "Rosid",
    # Asterids
    "Solanum_lycopersicum": "Asterid",
    "Solanum_tuberosum": "Asterid",
    "Nicotiana_tabacum": "Asterid",
    "Nicotiana_benthamiana": "Asterid",
    "Capsicum_annuum": "Asterid",
    "Lactuca_sativa": "Asterid",
    "Helianthus_annuus": "Asterid",
    "Sesamum_indicum": "Asterid",
    "Camellia_sinensis": "Asterid",
    "Vaccinium_corymbosum": "Asterid",
    # Other seed plants / early diverging (kept for completeness)
    "Beta_vulgaris": "Caryophyllid",
    "Spinacia_oleracea": "Caryophyllid",
    "Chenopodium_quinoa": "Caryophyllid",
    "Picea_abies": "Gymnosperm",
    "Selaginella_moellendorffii": "Lycophyta",
    "Physcomitrella_patens": "Bryophyta",
    "Physcomitrium_patens": "Bryophyta",
    "Marchantia_polymorpha": "Bryophyta",
    "Klebsormidium_nitens": "Charophyta",
    "Chlamydomonas_reinhardtii": "Chlorophyta",
}

# --------------------------------------------------------------------------
# 0.3  Organ harmonisation.  Deliberately conservative: only unambiguous
# synonyms and sub-regions are merged into a core organ.  Multi-tissue strings
# ("Leaf and Stem", "Root, leaf"), developmental stages and junk values are sent
# to Other and excluded from organ-stratified analysis.
# --------------------------------------------------------------------------
ORGAN_MAP = {
    "Leaf": "Leaf",
    "Rosette": "Leaf",
    "Leaf Sheath": "Leaf",
    "Root": "Root",
    "Root tips": "Root",
    "Root Cortex": "Root",
    "Root Stele": "Root",
    "Root Rhizodermis": "Root",
    "Root Endodermis": "Root",
    "Seedling": "Seedling",
    "seedling": "Seedling",
    "Shoot": "Shoot",
    "Shoot tip": "Shoot",
}
CORE_ORGANS = ["Leaf", "Root", "Seedling", "Shoot"]


def strip_transcript_suffix(gene_id):
    """Strip _T1 / .1 transcript suffixes to get the locus ID (as in 02_prepare_labels)."""
    gene_id = re.sub(r"_T\d+$", "", gene_id)
    gene_id = re.sub(r"\.\d+$", "", gene_id)
    return gene_id


# --------------------------------------------------------------------------
def load_gene_species(feature_table):
    """gene_id -> species, and species -> set(gene_id). The gene universe."""
    g2s, s2g = {}, defaultdict(set)
    with gzip.open(feature_table, "rt") as fh:
        header = fh.readline().rstrip("\n").split(",")
        assert header[0] == "gene_id" and header[1] == "species", header[:3]
        for line in fh:
            p = line.split(",", 2)
            g2s[p[0]] = p[1]
            s2g[p[1]].add(p[0])
    return g2s, s2g


def load_id_mappings(id_mapping_dir):
    """species -> {deg_id: gtf_id}. Skips the duplicated 'X 2.tsv' files."""
    mappings = {}
    for fn in sorted(os.listdir(id_mapping_dir)):
        if not fn.endswith(".tsv") or " 2.tsv" in fn:
            continue
        species = fn[:-4]
        df = pd.read_csv(os.path.join(id_mapping_dir, fn), sep="\t")
        if {"deg_id", "gtf_id"}.issubset(df.columns):
            mappings[species] = dict(zip(df["deg_id"], df["gtf_id"]))
    return mappings


def load_split_membership(dataset_dir):
    """(stress, direction) -> {gene_id: split}, from the existing pickles."""
    membership = {}
    for stress in STRESSES:
        for direction in DIRECTIONS:
            p = os.path.join(dataset_dir, f"dataset_{stress}_{direction}.pkl")
            if not os.path.exists(p):
                continue
            with open(p, "rb") as fh:
                splits = pickle.load(fh)
            m = {}
            orig = {}
            for split_name, (gids, labs) in splits.items():
                for g, l in zip(gids, labs):
                    m[g] = split_name
                    orig[g] = int(l)
            membership[(stress, direction)] = (m, orig)
    return membership


def load_deg(deg_table, id_mappings):
    """Load the DEG table and attach harmonised organ + mapped gene_locus."""
    cols = ["species", "stress", "organ", "experiment", "direction", "gene"]
    print(f"  reading {deg_table} ...")
    df = pd.read_csv(deg_table, usecols=cols)
    print(f"  {len(df):,} rows")

    df["stress"] = df["stress"].str.replace("High Light", "High light", regex=False)
    df["organ_harmonised"] = df["organ"].map(ORGAN_MAP).fillna("Other")
    df["gene_locus"] = df["gene"].map(strip_transcript_suffix)

    # Apply per-species ID mapping.
    for species, mapping in id_mappings.items():
        mask = df["species"] == species
        if mask.any():
            df.loc[mask, "gene_locus"] = df.loc[mask, "gene_locus"].map(
                lambda x: mapping.get(x, x))
    return df


# --------------------------------------------------------------------------
def compute_freq(sub_species_stress, all_genes, direction):
    """DEG frequency for one species+stress+direction. Mirrors the original recipe."""
    n_experiments = sub_species_stress["experiment"].nunique()
    if n_experiments < 2:
        return None, 0
    sub = sub_species_stress[sub_species_stress["direction"] == direction]
    if len(sub) == 0:
        return None, n_experiments
    counts = sub.groupby("gene_locus")["experiment"].nunique()
    freq = pd.Series(0.0, index=sorted(all_genes), dtype=float)
    common = counts.index.intersection(freq.index)
    freq.loc[common] = (counts.loc[common] / n_experiments).astype(float)
    return freq, n_experiments


def label_quartile(freq):
    """Original scheme: <=p25 -> 0, >=p75 -> 1, middle dropped."""
    p25, p75 = np.percentile(freq.values, 25), np.percentile(freq.values, 75)
    if p25 == p75:
        return None
    lab = pd.Series(np.nan, index=freq.index)
    lab[freq <= p25] = 0.0
    lab[freq >= p75] = 1.0
    return lab.dropna()


def label_balanced(freq, frac=0.25, seed=42):
    """Rank-based: top frac -> 1, bottom frac -> 0, ties broken deterministically.

    Forces the same positive rate in every species regardless of assay depth,
    which is what removes the depth artefact quantified in Phase 0.1.
    """
    rng = np.random.RandomState(seed)
    jitter = rng.rand(len(freq)) * 1e-9          # deterministic tie-break
    order = np.argsort(freq.values + jitter)     # ascending
    k = int(len(freq) * frac)
    if k < 10:
        return None
    idx = freq.index.values
    lab = pd.Series(np.nan, index=freq.index)
    lab[idx[order[:k]]] = 0.0
    lab[idx[order[-k:]]] = 1.0
    return lab.dropna()


def build_scheme(deg, s2g, membership, scheme, organ=None, mode="quartile"):
    """Recompute labels for every stress-direction under one scheme.

    Returns {(stress, direction): {gene_id: label}} plus a per-species diagnostic table.
    """
    out, diag = {}, []
    work = deg if organ is None else deg[deg["organ_harmonised"] == organ]

    for stress in STRESSES:
        for direction in DIRECTIONS:
            labels = {}
            sub_stress = work[work["stress"] == stress]
            for species, genes in s2g.items():
                sub = sub_stress[sub_stress["species"] == species]
                if len(sub) == 0:
                    continue
                freq, n_exp = compute_freq(sub, genes, direction)
                if freq is None:
                    continue
                lab = label_quartile(freq) if mode == "quartile" else label_balanced(freq)
                if lab is None or lab.empty:
                    continue
                labels.update(lab.astype(int).to_dict())
                diag.append({
                    "scheme": scheme, "stress": stress, "direction": direction,
                    "species": species, "n_experiments": n_exp,
                    "n_labelled": len(lab), "n_pos": int(lab.sum()),
                    "pos_rate": float(lab.mean()),
                })
            out[(stress, direction)] = labels
            print(f"    {scheme:20s} {stress:9s} {direction:4s}: "
                  f"{len(labels):7d} genes labelled")
    return out, pd.DataFrame(diag)


def emit_pickles(labels_by_set, membership, out_dir, scheme):
    """Write dataset pickles that keep the ORIGINAL split membership but new labels."""
    d = os.path.join(out_dir, scheme)
    os.makedirs(d, exist_ok=True)
    rows = []
    for (stress, direction), labels in labels_by_set.items():
        if (stress, direction) not in membership:
            continue
        split_of, _ = membership[(stress, direction)]
        buckets = {"train": ([], []), "val": ([], []), "test": ([], [])}
        for g, l in labels.items():
            sp = split_of.get(g)
            if sp is None:
                continue                    # gene not in the original split
            buckets[sp][0].append(g)
            buckets[sp][1].append(l)
        payload = {k: (v[0], v[1]) for k, v in buckets.items()}
        with open(os.path.join(d, f"dataset_{stress}_{direction}.pkl"), "wb") as fh:
            pickle.dump(payload, fh)
        rows.append({
            "scheme": scheme, "stress": stress, "direction": direction,
            **{f"n_{k}": len(v[0]) for k, v in buckets.items()},
            **{f"pos_rate_{k}": (float(np.mean(v[1])) if v[1] else float("nan"))
               for k, v in buckets.items()},
        })
    return pd.DataFrame(rows)


def gate_against_original(labels_by_set, membership):
    """Agreement between the recomputed all-organ labels and the published ones."""
    rows = []
    for (stress, direction), labels in labels_by_set.items():
        if (stress, direction) not in membership:
            continue
        _, orig = membership[(stress, direction)]
        shared = set(labels) & set(orig)
        if not shared:
            rows.append({"stress": stress, "direction": direction,
                         "n_shared": 0, "agreement": float("nan")})
            continue
        agree = sum(1 for g in shared if labels[g] == orig[g])
        rows.append({
            "stress": stress, "direction": direction,
            "n_original": len(orig), "n_recomputed": len(labels),
            "n_shared": len(shared), "agreement": agree / len(shared),
        })
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="Stratified label construction")
    ap.add_argument("--deg_table", required=True)
    ap.add_argument("--feature_table", required=True)
    ap.add_argument("--id_mapping_dir", required=True)
    ap.add_argument("--dataset_dir", required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--schemes", default="all_organ_rebuild,balanced,Leaf,Root",
                    help="comma-separated: all_organ_rebuild, balanced, and/or organ names")
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    schemes = [s.strip() for s in args.schemes.split(",") if s.strip()]

    print("[1/6] Gene universe from the feature table ...")
    g2s, s2g = load_gene_species(args.feature_table)
    print(f"      {len(g2s):,} genes across {len(s2g)} species")

    print("[2/6] Persisting the lineage map (0.2) ...")
    present = sorted(s2g)
    lin = pd.DataFrame([
        {"species": s, "lineage": LINEAGE_MAP.get(s, "UNASSIGNED"),
         "n_genes_with_features": len(s2g[s])}
        for s in present
    ])
    lin.to_csv(os.path.join(args.output_dir, "species_lineages.csv"), index=False)
    missing = lin[lin["lineage"] == "UNASSIGNED"]["species"].tolist()
    print(f"      {len(lin)} species; unassigned: {missing if missing else 'none'}")
    print(lin["lineage"].value_counts().to_string())

    print("[3/6] Split membership from the existing pickles ...")
    membership = load_split_membership(args.dataset_dir)
    print(f"      {len(membership)} stress-direction sets")

    # The organ harmonisation table is a static artefact; persist it regardless.
    pd.DataFrame([{"organ_raw": k, "organ_harmonised": v}
                  for k, v in sorted(ORGAN_MAP.items())]
                 ).to_csv(os.path.join(args.output_dir, "organ_harmonisation.csv"),
                          index=False)

    if not schemes:
        print("[4/6] no --schemes requested; maps written, skipping the DEG table")
        print(f"\nWritten to {args.output_dir}")
        return

    print("[4/6] ID mappings + DEG table ...")
    id_mappings = load_id_mappings(args.id_mapping_dir)
    print(f"      {len(id_mappings)} species ID maps")
    deg = load_deg(args.deg_table, id_mappings)

    oh = deg["organ_harmonised"].value_counts()
    print("      harmonised organ counts:")
    for k, v in oh.items():
        print(f"        {k:10s} {v:,}")

    print("[5/6] Building label schemes ...")
    all_diag, all_manifest = [], []
    for scheme in schemes:
        if scheme == "all_organ_rebuild":
            labels, diag = build_scheme(deg, s2g, membership, scheme, None, "quartile")
        elif scheme == "balanced":
            labels, diag = build_scheme(deg, s2g, membership, scheme, None, "balanced")
        elif scheme in CORE_ORGANS:
            labels, diag = build_scheme(deg, s2g, membership, scheme, scheme, "quartile")
        else:
            print(f"  [skip] unknown scheme {scheme}")
            continue
        all_diag.append(diag)
        all_manifest.append(emit_pickles(labels, membership, args.output_dir, scheme))

        if scheme == "all_organ_rebuild":
            gate = gate_against_original(labels, membership)
            gate.to_csv(os.path.join(args.output_dir, "gate_vs_original_labels.csv"),
                        index=False)
            print("\n  [GATE] recomputed all-organ labels vs published labels:")
            print(gate.to_string(index=False))
            mean_agree = gate["agreement"].mean()
            print(f"  mean agreement = {mean_agree:.3f}")
            if mean_agree < 0.90:
                print("  WARNING: below 0.90 — investigate before trusting organ rebuilds.")

    print("\n[6/6] Writing diagnostics ...")
    if all_diag:
        pd.concat(all_diag).to_csv(
            os.path.join(args.output_dir, "scheme_species_diagnostics.csv"), index=False)
    if all_manifest:
        man = pd.concat(all_manifest)
        man.to_csv(os.path.join(args.output_dir, "scheme_manifest.csv"), index=False)
        print(man.to_string(index=False))

    # Positive-rate spread per scheme: the number A3 is designed to flatten.
    if all_diag:
        diag = pd.concat(all_diag)
        print("\nPOSITIVE-RATE SPREAD ACROSS SPECIES (lower is better for A3)")
        for scheme, grp in diag.groupby("scheme"):
            g = grp[grp["n_labelled"] >= 500]
            if g.empty:
                continue
            print(f"  {scheme:20s} min={g['pos_rate'].min():.3f} "
                  f"max={g['pos_rate'].max():.3f} "
                  f"spread={g['pos_rate'].max()/max(g['pos_rate'].min(),1e-9):.1f}x")

    print(f"\nWritten to {args.output_dir}")


if __name__ == "__main__":
    main()
