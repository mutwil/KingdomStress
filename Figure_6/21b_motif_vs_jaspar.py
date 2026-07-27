#!/usr/bin/env python3
"""
Phase C1 for the Figure 6 revision: do the model-discovered motifs correspond to
real plant TF binding motifs?

Compares every discovered PWM against JASPAR2024 CORE plants (805 PFMs) and reports
matches with an empirical significance estimate.

Why not Tomtom
  The MEME Suite is not installed. The only Homebrew formula is meme@4 from an
  untrusted third-party tap, which would mean trusting that tap and using a
  decade-old release. Instead the comparison is done directly with a standard
  column-similarity statistic and a permutation null, which is auditable and needs
  no external dependency.

Similarity
  Pearson correlation between aligned PWM columns, averaged over the overlap,
  maximised over all offsets (with a minimum overlap) and both strands. This is the
  same family of statistic Tomtom offers as its PCC option.

Null
  For each query motif, the observed score is its BEST match across the 805 real
  JASPAR motifs. The null is the best match across a size-matched set of 805
  column-shuffled JASPAR motifs, repeated R times. Column shuffling preserves each
  motif's length and per-column information content while destroying its grammar,
  so the null answers "how good a best-match would this query get by chance against
  a database of this size?". Empirical p, then BH across queries.

Inputs are the *.meme files written by the interpretability scripts. The RNA ones
use ALPHABET= ACGU and a single strand; they are converted to DNA here.

Usage
  python 21b_motif_vs_jaspar.py \
      --motif_dirs cnn=".../cnn_results/motifs",cad2=".../plantcad2_interpretability/motifs",rna=".../rna_interpretability/motifs" \
      --jaspar ".../jaspar2024_plants.txt" \
      --output_dir ".../revision/motif_match" \
      --top_per_set 10 --n_null 20

Kingdom Stress Atlas, Mutwil lab.
"""

import argparse
import os
import re
import sys
from collections import defaultdict

import numpy as np
import pandas as pd

STRESSES = ["Heat", "Cold", "Drought", "Salt", "Pathogen"]
DIRECTIONS = ["UP", "DOWN"]
MIN_OVERLAP = 5
PSEUDO = 1e-6
SEED = 42


# ---------------------------------------------------------------------------
def parse_meme(path):
    """Return [(name, consensus, PWM (w x 4) as probabilities)] from a MEME file.

    Handles the RNA-alphabet files by treating U as T; column order is A,C,G,T/U
    in all files written by the interpretability scripts.
    """
    motifs = []
    with open(path) as fh:
        lines = fh.readlines()
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        m = re.match(r"^MOTIF\s+(\S+)(?:\s+(\S+))?", line)
        if not m:
            i += 1
            continue
        name, alt = m.group(1), (m.group(2) or "")
        # find the letter-probability matrix header
        j = i + 1
        while j < len(lines) and not lines[j].strip().startswith("letter-probability"):
            j += 1
        if j >= len(lines):
            break
        wm = re.search(r"w=\s*(\d+)", lines[j])
        w = int(wm.group(1)) if wm else 0
        rows = []
        k = j + 1
        while k < len(lines) and len(rows) < w:
            vals = lines[k].split()
            if len(vals) >= 4:
                try:
                    rows.append([float(v) for v in vals[:4]])
                except ValueError:
                    pass
            k += 1
        if rows:
            pwm = np.asarray(rows, dtype=float)
            pwm = np.clip(pwm, PSEUDO, None)
            pwm = pwm / pwm.sum(axis=1, keepdims=True)
            motifs.append((name, alt, pwm))
        i = k
    return motifs


def parse_jaspar(path):
    """JASPAR raw PFM format -> [(id, name, PWM)] with rows A,C,G,T."""
    motifs = []
    with open(path) as fh:
        lines = [l.rstrip("\n") for l in fh]
    i = 0
    while i < len(lines):
        if not lines[i].startswith(">"):
            i += 1
            continue
        head = lines[i][1:].split()
        mid = head[0]
        mname = head[1] if len(head) > 1 else mid
        counts = []
        for k in range(1, 5):
            if i + k >= len(lines):
                break
            nums = re.findall(r"[-\d.]+", lines[i + k].split("[")[-1])
            counts.append([float(x) for x in nums])
        if len(counts) == 4 and all(len(c) == len(counts[0]) for c in counts):
            pfm = np.asarray(counts, dtype=float).T          # w x 4
            pfm = np.clip(pfm, 0, None) + PSEUDO
            pwm = pfm / pfm.sum(axis=1, keepdims=True)
            motifs.append((mid, mname, pwm))
        i += 5
    return motifs


def revcomp_pwm(p):
    """Reverse-complement a PWM whose columns are ordered A,C,G,T."""
    return p[::-1, ::-1]


def best_similarity(q, targets_fwd, targets_rc):
    """Best mean-column Pearson correlation of q against a list of target PWMs."""
    best = -1.0
    best_j = -1
    for j, (t_f, t_r) in enumerate(zip(targets_fwd, targets_rc)):
        for t in (t_f, t_r):
            s = _align_score(q, t)
            if s > best:
                best, best_j = s, j
    return best, best_j


def _align_score(a, b):
    """Max over offsets of the Sandelin-Wasserman similarity.

    SW = 1 - mean_over_columns( euclidean_distance^2 / 2 ), which is 1 for identical
    columns and 0 for maximally different ones. Per-column Pearson correlation was
    tried first and saturates near 1.0 for low-information A/T-rich columns, making
    real and shuffled motifs indistinguishable.
    """
    la, lb = len(a), len(b)
    best = -1.0
    for off in range(-(lb - MIN_OVERLAP), la - MIN_OVERLAP + 1):
        ia0, ib0 = max(0, off), max(0, -off)
        n = min(la - ia0, lb - ib0)
        if n < MIN_OVERLAP:
            continue
        x = a[ia0:ia0 + n]
        y = b[ib0:ib0 + n]
        d2 = ((x - y) ** 2).sum(axis=1)          # in [0, 2]
        s = float(1.0 - (d2 / 2.0).mean())
        if s > best:
            best = s
    return best


def shuffle_columns(p, rng):
    """Permute the four base probabilities WITHIN each column.

    Permuting whole columns is not a valid null here: many discovered motifs are
    homopolymeric, so a column-permuted copy is identical to the original. Shuffling
    within columns destroys base identity while preserving column-wise information
    content and motif length.
    """
    out = p.copy()
    for i in range(len(out)):
        out[i] = out[i][rng.permutation(4)]
    return out


def bh(pvals):
    p = np.asarray(pvals, dtype=float)
    n = len(p)
    order = np.argsort(p)
    q = np.empty(n)
    prev = 1.0
    for rank, i in enumerate(order[::-1]):
        k = n - rank
        prev = min(prev, p[i] * n / k)
        q[i] = prev
    return q


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="Discovered motifs vs JASPAR plants")
    ap.add_argument("--motif_dirs", required=True,
                    help="comma list of label=path")
    ap.add_argument("--jaspar", required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--top_per_set", type=int, default=10)
    ap.add_argument("--n_null", type=int, default=20)
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    rng = np.random.RandomState(SEED)

    print("[1/4] JASPAR ...")
    jaspar = parse_jaspar(args.jaspar)
    print(f"      {len(jaspar)} plant PFMs")
    if not jaspar:
        sys.exit("failed to parse JASPAR")
    j_fwd = [p for _, _, p in jaspar]
    j_rc = [revcomp_pwm(p) for p in j_fwd]

    print("[2/4] Discovered motifs ...")
    queries = []
    for spec in args.motif_dirs.split(","):
        label, path = spec.split("=", 1)
        path = path.strip().strip('"')
        if not os.path.isdir(path):
            print(f"      [skip] {label}: {path} not found")
            continue
        for stress in STRESSES:
            for d in DIRECTIONS:
                f = os.path.join(path, f"{stress}_{d}_motifs.meme")
                if not os.path.exists(f):
                    continue
                ms = parse_meme(f)[:args.top_per_set]
                for name, alt, pwm in ms:
                    queries.append(dict(model=label, stress=stress, direction=d,
                                        motif=name, consensus=alt, pwm=pwm))
        print(f"      {label}: {sum(1 for q in queries if q['model']==label)} motifs")
    if not queries:
        sys.exit("no discovered motifs parsed")

    print(f"[3/4] Matching {len(queries)} queries vs {len(jaspar)} targets "
          f"({args.n_null} null rounds) ...")
    # Pre-build null databases once; every query is scored against the same ones.
    null_dbs = []
    for r in range(args.n_null):
        sh = [shuffle_columns(p, rng) for p in j_fwd]
        null_dbs.append((sh, [revcomp_pwm(p) for p in sh]))

    rows = []
    for qi, q in enumerate(queries):
        obs, j_idx = best_similarity(q["pwm"], j_fwd, j_rc)
        nulls = [best_similarity(q["pwm"], nf, nr)[0] for nf, nr in null_dbs]
        nulls = np.asarray(nulls)
        p_emp = (1 + int((nulls >= obs).sum())) / (len(nulls) + 1)
        mid, mname, _ = jaspar[j_idx]
        rows.append(dict(model=q["model"], stress=q["stress"], direction=q["direction"],
                         motif=q["motif"], consensus=q["consensus"],
                         best_match_id=mid, best_match_tf=mname,
                         similarity=round(obs, 4),
                         null_mean=round(float(nulls.mean()), 4),
                         null_max=round(float(nulls.max()), 4),
                         p_empirical=p_emp))
        if (qi + 1) % 25 == 0:
            print(f"      {qi+1}/{len(queries)}")

    df = pd.DataFrame(rows)
    df["q_value"] = bh(df["p_empirical"].values)
    df["z_vs_null"] = (df["similarity"] - df["null_mean"]) / (
        df.groupby("model")["null_mean"].transform("std").replace(0, np.nan) + 1e-9)
    df.sort_values(["q_value", "similarity"], ascending=[True, False]).to_csv(
        os.path.join(args.output_dir, "discovered_vs_jaspar.csv"), index=False)

    print("\n[4/4] Summary")
    print("=" * 78)
    sig = df[df.q_value < 0.05]
    print(f"significant matches (q<0.05): {len(sig)}/{len(df)}")
    for model, g in df.groupby("model"):
        gs = g[g.q_value < 0.05]
        print(f"  {model:6s} {len(gs):3d}/{len(g):3d} significant; "
              f"median similarity {g['similarity'].median():.3f} "
              f"(null {g['null_mean'].median():.3f})")

    print("\nTop matches:")
    for _, r in df.sort_values("similarity", ascending=False).head(20).iterrows():
        print(f"  {r.model:5s} {r.stress:9s} {r.direction:4s} {str(r.consensus)[:12]:12s} "
              f"-> {r.best_match_tf:14s} sim={r.similarity:.3f} "
              f"null={r.null_mean:.3f} q={r.q_value:.3g}")

    top_tfs = sig["best_match_tf"].value_counts().head(15)
    if len(top_tfs):
        print("\nMost frequently matched TFs among significant hits:")
        for tf, n in top_tfs.items():
            print(f"  {n:3d}  {tf}")

    print(f"\nWritten to {args.output_dir}")


if __name__ == "__main__":
    main()
