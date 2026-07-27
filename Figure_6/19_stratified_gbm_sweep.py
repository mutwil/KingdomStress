#!/usr/bin/env python3
"""
Phase A1 + A3 for the Figure 6 revision: does restricting training to a lineage,
or removing the assay-depth prevalence artefact, raise auROC?

Every comparison is scored on a FIXED test set (the stratum's genes in the original
test split), so the arms differ only in what the model was trained on.

A1 - lineage restriction, three arms per lineage L:
     pan          train on all species                       (published setting)
     stratum      train on L only
     pan_matched  train on all species, subsampled to n(L)   (several draws)
     stratum vs pan_matched isolates lineage specificity from training-set size;
     pan vs pan_matched isolates the size effect on its own.

A3 - prevalence balancing:
     balanced     train on all species, but within each species the majority class
                  is downsampled so every species contributes pos_rate 0.5.
     Phase 0.1 showed positive rate tracks assay depth (Spearman rho = 0.771,
     spread up to 20.9x). If the pooled auROC leans on that artefact, this arm
     should lower pooled auROC while leaving within-species auROC intact.

Learner: HistGradientBoostingClassifier. The published Combined model used
GradientBoostingClassifier on ~134 per-stress features; running that learner over
the full 658-column union for ~100 fits is not tractable. The `baseline` arm
reproduces the published setting closely enough to license the comparison, and its
value is reported so the substitution is auditable. All arms share the learner, so
the contrasts are internally valid.

Usage
  python 19_stratified_gbm_sweep.py \
      --feature_table ".../feature_table.csv.gz" \
      --dataset_dir   ".../plantrna_fm_results" \
      --lineage_csv   ".../stratified/species_lineages.csv" \
      --output_dir    ".../revision/sweep" \
      --lineages Monocot,Rosid,Asterid

Kingdom Stress Atlas, Mutwil lab.
"""

import argparse
import gzip
import os
import pickle
import sys
from collections import defaultdict

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score

STRESSES = ["Heat", "Cold", "Drought", "Salt", "Pathogen"]
DIRECTIONS = ["UP", "DOWN"]

SEED = 42
MAX_TRAIN = 50_000        # matches the cap used by the published ablation
N_MATCHED_DRAWS = 3       # repeats for the size-matched control
MIN_TRAIN = 2_000         # below this a stratum arm is not attempted
MIN_TEST = 200
MIN_CLASS = 20

HGB_PARAMS = dict(
    max_iter=300, max_depth=6, learning_rate=0.1,
    min_samples_leaf=20, l2_regularization=1.0,
    early_stopping=False, random_state=SEED,
)


# ---------------------------------------------------------------------------
def load_pickles(dataset_dir):
    """(stress, direction) -> {'train'|'val'|'test': (gene_ids, labels)}"""
    out = {}
    for s in STRESSES:
        for d in DIRECTIONS:
            p = os.path.join(dataset_dir, f"dataset_{s}_{d}.pkl")
            if os.path.exists(p):
                with open(p, "rb") as fh:
                    out[(s, d)] = pickle.load(fh)
    return out


def gene_union(pickles):
    u = set()
    for splits in pickles.values():
        for gids, _ in splits.values():
            u.update(gids)
    return u


def load_features(feature_table, keep_genes):
    """Stream the table once, keeping only the genes we need.

    Each row is converted to float32 immediately. Accumulating parsed strings
    instead would spike to tens of GB for ~1e6 genes x 658 columns.
    """
    gene_ids, species, rows = [], [], []
    with gzip.open(feature_table, "rt") as fh:
        header = fh.readline().rstrip("\n").split(",")
        assert header[0] == "gene_id" and header[1] == "species", header[:3]
        feat_names = header[2:]
        n_feat = len(feat_names)
        for line in fh:
            i = line.index(",")
            g = line[:i]
            if g not in keep_genes:
                continue
            j = line.index(",", i + 1)
            vec = np.fromstring(line[j + 1:], dtype=np.float32, sep=",")
            if vec.shape[0] != n_feat:      # ragged row; skip rather than corrupt
                continue
            gene_ids.append(g)
            species.append(line[i + 1:j])
            rows.append(vec)
    X = np.vstack(rows) if rows else np.zeros((0, len(feat_names)), dtype=np.float32)
    idx = {g: i for i, g in enumerate(gene_ids)}
    return X, np.asarray(species), idx, feat_names


def subset(idx, X, species_arr, gids, labs):
    """Rows of X for the given genes, dropping any without features."""
    ii, yy = [], []
    for g, l in zip(gids, labs):
        j = idx.get(g)
        if j is not None:
            ii.append(j)
            yy.append(int(l))
    ii = np.asarray(ii, dtype=np.int64)
    return ii, np.asarray(yy, dtype=np.int8), species_arr[ii]


def fit_score(Xtr, ytr, Xte, yte, sp_te):
    """Train and return pooled + within-species auROC."""
    if len(np.unique(ytr)) < 2 or len(np.unique(yte)) < 2:
        return None
    clf = HistGradientBoostingClassifier(**HGB_PARAMS)
    clf.fit(Xtr, ytr)
    p = clf.predict_proba(Xte)[:, 1]
    pooled = roc_auc_score(yte, p)

    aur, wts = [], []
    for s in np.unique(sp_te):
        m = sp_te == s
        n, npos = int(m.sum()), int(yte[m].sum())
        if n >= MIN_TEST and npos >= MIN_CLASS and (n - npos) >= MIN_CLASS:
            aur.append(roc_auc_score(yte[m], p[m]))
            wts.append(n)
    within = float(np.average(aur, weights=wts)) if aur else float("nan")
    return pooled, within, len(aur)


def cap(rng, ii, yy, n_max):
    if len(ii) <= n_max:
        return ii, yy
    sel = rng.choice(len(ii), n_max, replace=False)
    return ii[sel], yy[sel]


def match_size(rng, ii, yy, n_target, pos_rate_target):
    """Subsample to n_target while matching the target positive rate."""
    pos = np.where(yy == 1)[0]
    neg = np.where(yy == 0)[0]
    n_pos = min(len(pos), int(round(n_target * pos_rate_target)))
    n_neg = min(len(neg), n_target - n_pos)
    if n_pos < MIN_CLASS or n_neg < MIN_CLASS:
        return None, None
    sel = np.concatenate([rng.choice(pos, n_pos, replace=False),
                          rng.choice(neg, n_neg, replace=False)])
    rng.shuffle(sel)
    return ii[sel], yy[sel]


def balance_by_species(rng, ii, yy, sp):
    """Downsample the majority class within each species to pos_rate 0.5."""
    keep = []
    for s in np.unique(sp):
        m = np.where(sp == s)[0]
        pos = m[yy[m] == 1]
        neg = m[yy[m] == 0]
        k = min(len(pos), len(neg))
        if k < MIN_CLASS:
            continue
        keep.append(rng.choice(pos, k, replace=False))
        keep.append(rng.choice(neg, k, replace=False))
    if not keep:
        return None, None
    sel = np.concatenate(keep)
    rng.shuffle(sel)
    return ii[sel], yy[sel]


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="Lineage / prevalence stratification sweep")
    ap.add_argument("--feature_table", required=True)
    ap.add_argument("--dataset_dir", required=True)
    ap.add_argument("--lineage_csv", required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--lineages", default="Monocot,Rosid,Asterid")
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    res_path = os.path.join(args.output_dir, "sweep_results.csv")
    done = set()
    if args.resume and os.path.exists(res_path):
        prev = pd.read_csv(res_path)
        done = set(zip(prev.stress, prev.direction, prev.arm, prev.stratum))
        print(f"resuming: {len(done)} arms already done")

    lineages = [x.strip() for x in args.lineages.split(",") if x.strip()]
    lin_df = pd.read_csv(args.lineage_csv)
    sp2lin = dict(zip(lin_df["species"], lin_df["lineage"]))

    print("[1/3] Loading dataset pickles ...")
    pickles = load_pickles(args.dataset_dir)
    keep = gene_union(pickles)
    print(f"      {len(pickles)} sets, {len(keep):,} distinct genes")

    print("[2/3] Loading features (single pass) ...")
    X, species_arr, idx, feat_names = load_features(args.feature_table, keep)
    print(f"      X = {X.shape[0]:,} x {X.shape[1]} float32 "
          f"({X.nbytes/1e9:.2f} GB), {len(feat_names)} features")
    lin_arr = np.asarray([sp2lin.get(s, "UNASSIGNED") for s in species_arr])

    print("[3/3] Running arms ...")
    results = []

    # Fixed column order. Arms populate different keys, so appended rows must be
    # reindexed onto a canonical schema or the CSV silently misaligns.
    COLUMNS = ["stress", "direction", "arm", "stratum", "n_train", "n_test",
               "train_pos_rate", "test_pos_rate", "pooled_auroc", "pooled_auroc_sd",
               "within_species_auroc", "n_species_scored", "n_draws"]

    def record(row):
        results.append(row)
        (pd.DataFrame([row])
           .reindex(columns=COLUMNS)
           .to_csv(res_path, mode="a",
                   header=not os.path.exists(res_path), index=False))

    for (stress, direction), splits in pickles.items():
        tr_i, tr_y, tr_sp = subset(idx, X, species_arr, *splits["train"])
        te_i, te_y, te_sp = subset(idx, X, species_arr, *splits["test"])
        tr_lin = lin_arr[tr_i]
        te_lin = lin_arr[te_i]
        rng = np.random.RandomState(SEED)

        # ---- baseline: published setting, full test set -------------------
        key = (stress, direction, "baseline", "all")
        if key not in done:
            ii, yy = cap(rng, tr_i, tr_y, MAX_TRAIN)
            out = fit_score(X[ii], yy, X[te_i], te_y, te_sp)
            if out:
                pooled, within, nsp = out
                print(f"  {stress:9s} {direction:4s} baseline        "
                      f"pooled={pooled:.4f} within={within:.4f}")
                record(dict(stress=stress, direction=direction, arm="baseline",
                            stratum="all", n_train=len(ii), n_test=len(te_y),
                            train_pos_rate=float(yy.mean()),
                            test_pos_rate=float(te_y.mean()),
                            pooled_auroc=pooled, within_species_auroc=within,
                            n_species_scored=nsp))

        # ---- A3: prevalence-balanced training, same fixed test set --------
        key = (stress, direction, "balanced", "all")
        if key not in done:
            bi, by = balance_by_species(rng, tr_i, tr_y, tr_sp)
            if bi is not None:
                bi, by = cap(rng, bi, by, MAX_TRAIN)
                out = fit_score(X[bi], by, X[te_i], te_y, te_sp)
                if out:
                    pooled, within, nsp = out
                    print(f"  {stress:9s} {direction:4s} balanced        "
                          f"pooled={pooled:.4f} within={within:.4f}")
                    record(dict(stress=stress, direction=direction, arm="balanced",
                                stratum="all", n_train=len(bi), n_test=len(te_y),
                                train_pos_rate=float(by.mean()),
                                test_pos_rate=float(te_y.mean()),
                                pooled_auroc=pooled, within_species_auroc=within,
                                n_species_scored=nsp))

        # ---- A1: per-lineage, three arms on a fixed test set --------------
        for L in lineages:
            te_mask = te_lin == L
            if te_mask.sum() < MIN_TEST:
                continue
            teL_i, teL_y, teL_sp = te_i[te_mask], te_y[te_mask], te_sp[te_mask]
            if teL_y.sum() < MIN_CLASS or (len(teL_y) - teL_y.sum()) < MIN_CLASS:
                continue

            trL_mask = tr_lin == L
            n_L = int(trL_mask.sum())
            if n_L < MIN_TRAIN:
                continue
            trL_i, trL_y = tr_i[trL_mask], tr_y[trL_mask]

            common = dict(stress=stress, direction=direction, stratum=L,
                          n_test=len(teL_y), test_pos_rate=float(teL_y.mean()))

            # (i) pan
            if (stress, direction, "pan", L) not in done:
                ii, yy = cap(rng, tr_i, tr_y, MAX_TRAIN)
                out = fit_score(X[ii], yy, X[teL_i], teL_y, teL_sp)
                if out:
                    pooled, within, nsp = out
                    print(f"  {stress:9s} {direction:4s} pan/{L:8s}  "
                          f"pooled={pooled:.4f} within={within:.4f}")
                    record(dict(arm="pan", n_train=len(ii),
                                train_pos_rate=float(yy.mean()),
                                pooled_auroc=pooled, within_species_auroc=within,
                                n_species_scored=nsp, **common))

            # (ii) stratum only
            if (stress, direction, "stratum", L) not in done:
                ii, yy = cap(rng, trL_i, trL_y, MAX_TRAIN)
                out = fit_score(X[ii], yy, X[teL_i], teL_y, teL_sp)
                if out:
                    pooled, within, nsp = out
                    print(f"  {stress:9s} {direction:4s} stratum/{L:8s} "
                          f"pooled={pooled:.4f} within={within:.4f}")
                    record(dict(arm="stratum", n_train=len(ii),
                                train_pos_rate=float(yy.mean()),
                                pooled_auroc=pooled, within_species_auroc=within,
                                n_species_scored=nsp, **common))

            # (iii) size-matched pan control
            if (stress, direction, "pan_matched", L) not in done:
                n_target = min(n_L, MAX_TRAIN)
                pr = float(trL_y.mean())
                vals = []
                for draw in range(N_MATCHED_DRAWS):
                    r2 = np.random.RandomState(SEED + draw)
                    mi, my = match_size(r2, tr_i, tr_y, n_target, pr)
                    if mi is None:
                        continue
                    out = fit_score(X[mi], my, X[teL_i], teL_y, teL_sp)
                    if out:
                        vals.append(out)
                if vals:
                    pooled = float(np.mean([v[0] for v in vals]))
                    within = float(np.nanmean([v[1] for v in vals]))
                    sd = float(np.std([v[0] for v in vals]))
                    print(f"  {stress:9s} {direction:4s} matched/{L:8s} "
                          f"pooled={pooled:.4f}+-{sd:.4f} within={within:.4f}")
                    record(dict(arm="pan_matched", n_train=n_target,
                                train_pos_rate=pr, pooled_auroc=pooled,
                                pooled_auroc_sd=sd, within_species_auroc=within,
                                n_species_scored=vals[0][2], n_draws=len(vals),
                                **common))

    # ---- summary ---------------------------------------------------------
    if not results:
        print("no new results")
        return
    df = pd.DataFrame(results)
    print("\n" + "=" * 78)
    print("MEAN OVER STRESS-DIRECTIONS")
    print("=" * 78)
    for (arm, stratum), g in df.groupby(["arm", "stratum"]):
        print(f"  {arm:12s} {stratum:9s} pooled={g['pooled_auroc'].mean():.4f}  "
              f"within={g['within_species_auroc'].mean():.4f}  n={len(g)}")

    print("\nA1 CONTRAST (stratum minus size-matched pan, positive = lineage helps)")
    piv = df.pivot_table(index=["stress", "direction", "stratum"],
                         columns="arm", values="pooled_auroc")
    for c in ["stratum", "pan_matched", "pan"]:
        if c not in piv.columns:
            piv[c] = np.nan
    piv["stratum_minus_matched"] = piv["stratum"] - piv["pan_matched"]
    piv["pan_minus_matched"] = piv["pan"] - piv["pan_matched"]
    print(piv.round(4).to_string())
    piv.to_csv(os.path.join(args.output_dir, "a1_contrasts.csv"))
    for L, g in piv.groupby(level="stratum"):
        print(f"  {L:9s} mean(stratum - matched) = "
              f"{g['stratum_minus_matched'].mean():+.4f}")

    print(f"\nWritten to {args.output_dir}")


if __name__ == "__main__":
    main()
