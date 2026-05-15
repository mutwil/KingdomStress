#!/usr/bin/env python3
"""
PlantCAD2 fine-tuning for stress-responsive gene prediction.

Fine-tunes PlantCaduceus (Mamba-based plant DNA language model) with a
classification head to predict heat-responsive genes from promoter + gene body
DNA sequence.

Requires CUDA GPU (mamba_ssm). Run on Colab or server with NVIDIA GPU.

Usage (Colab):
    !pip install mamba_ssm caduceus transformers peft accelerate pyranges pyfaidx
    %run PlantCAD2_finetune.py
"""

import os, re, json
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForMaskedLM
from peft import LoraConfig, get_peft_model, TaskType
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score
from sklearn.utils.class_weight import compute_class_weight
import pyranges as pr
from pyfaidx import Fasta

# ===========================================================================
# Configuration
# ===========================================================================

# Paths -- adjust for Colab
DRIVE_DIR = Path("/content/drive/MyDrive/Projects/2026_KingdomStress/CC-Fig 5 data_CNN")
WORK_DIR = Path("/content/plantcad2_work")
WORK_DIR.mkdir(exist_ok=True)

DEG_FILE = DRIVE_DIR / "kingdom_stress_dict v3_.csv"
ORTHOGROUPS_FILE = DRIVE_DIR / "Orthogroups.txt"
ID_MAPPING_DIR = DRIVE_DIR / "id_mappings"

# Model
MODEL_NAME = "kuleshov-group/PlantCaduceus_l24"  # 40M params, 512bp context
# For PlantCAD2 8kb context: "kuleshov-group/PlantCAD2" (676M params, needs more VRAM)
MAX_SEQ_LEN = 512  # match model context; for PlantCAD2 use 8192

# Species
SPECIES = {
    "Arabidopsis_thaliana": {"ensembl": "arabidopsis_thaliana", "assembly": "TAIR10"},
    "Hordeum_vulgare": {"ensembl": "hordeum_vulgare", "assembly": "MorexV3_pseudomolecules_assembly"},
    "Brachypodium_distachyon": {"ensembl": "brachypodium_distachyon", "assembly": "Brachypodium_distachyon_v3.0"},
}

STRESS = "Heat"
ORGAN = "Leaf"
DIRECTIONS = ["UP", "DOWN"]
ENSEMBL_RELEASE = 59

# Training
BATCH_SIZE = 16
EPOCHS = 10
LR = 2e-4
LORA_R = 8
LORA_ALPHA = 16
LORA_DROPOUT = 0.1

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {DEVICE}")

# ===========================================================================
# Data helpers (reused from CNN pipeline)
# ===========================================================================

def strip_transcript_suffix(gene_id):
    gene_id = re.sub(r"_T\d+$", "", gene_id)
    gene_id = re.sub(r"\.\d+$", "", gene_id)
    return gene_id

def load_deg_table(deg_file, species_list=None, organ=None):
    cols = ["species", "stress", "organ", "experiment", "direction", "gene"]
    df = pd.read_csv(deg_file, usecols=cols)
    if species_list: df = df[df["species"].isin(species_list)]
    if organ: df = df[df["organ"].str.lower().str.contains(organ.lower())]
    df["stress"] = df["stress"].str.replace("High Light", "High light")
    df["gene_locus"] = df["gene"].map(strip_transcript_suffix)
    print(f"DEG table: {len(df)} entries, {df['species'].nunique()} species")
    return df

def load_id_mapping(species):
    f = ID_MAPPING_DIR / f"{species}.tsv"
    if f.exists():
        df = pd.read_csv(f, sep="\t")
        return dict(zip(df["deg_id"], df["gtf_id"]))
    return None

def compute_deg_frequency(deg_df, species, stress, all_genes, direction):
    subset = deg_df[(deg_df["species"] == species) & (deg_df["stress"] == stress)]
    if len(subset) == 0: return None
    n_exp = subset["experiment"].nunique()
    if n_exp < 2: return None
    dir_subset = subset[subset["direction"] == direction]
    if len(dir_subset) == 0: return None
    counts = dir_subset.groupby("gene_locus")["experiment"].nunique()
    freq = pd.Series(0.0, index=sorted(all_genes))
    for g in set(counts.index) & all_genes:
        freq[g] = counts[g] / n_exp
    return freq, n_exp

def label_by_percentile(freq):
    n_nonzero = (freq > 0).sum()
    if n_nonzero < 50: return None
    p25, p75 = np.percentile(freq, 25), np.percentile(freq, 75)
    if p25 != p75:
        labels = pd.Series(2, index=freq.index)
        labels[freq <= p25] = 0
        labels[freq >= p75] = 1
        return labels
    # Binary fallback
    labels = pd.Series(0, index=freq.index)
    labels[freq > 0] = 1
    return labels

def load_orthogroups(og_file):
    gene_to_og = {}
    with open(og_file) as f:
        for line in f:
            line = line.strip()
            if not line or ":" not in line: continue
            og_id, genes_str = line.split(":", 1)
            for g in genes_str.strip().split():
                gene_to_og[strip_transcript_suffix(g)] = og_id.strip()
    print(f"Loaded {len(gene_to_og)} orthogroup mappings")
    return gene_to_og

def split_by_orthogroups(gene_ids, gene_to_og, train_frac=0.7, val_frac=0.15, seed=42):
    rng = np.random.default_rng(seed)
    og_to_genes = defaultdict(list)
    no_og = []
    for gid in gene_ids:
        og = gene_to_og.get(gid)
        if og: og_to_genes[og].append(gid)
        else: no_og.append(gid)

    ogs = list(og_to_genes.keys())
    rng.shuffle(ogs)
    n_t = int(len(ogs) * train_frac)
    n_v = int(len(ogs) * val_frac)

    assignment = {}
    for og in ogs[:n_t]:
        for g in og_to_genes[og]: assignment[g] = "train"
    for og in ogs[n_t:n_t+n_v]:
        for g in og_to_genes[og]: assignment[g] = "val"
    for og in ogs[n_t+n_v:]:
        for g in og_to_genes[og]: assignment[g] = "test"

    rng.shuffle(no_og)
    nt2 = int(len(no_og) * train_frac)
    nv2 = int(len(no_og) * val_frac)
    for g in no_og[:nt2]: assignment[g] = "train"
    for g in no_og[nt2:nt2+nv2]: assignment[g] = "val"
    for g in no_og[nt2+nv2:]: assignment[g] = "test"

    counts = defaultdict(int)
    for v in assignment.values(): counts[v] += 1
    print(f"  Split: {dict(counts)}")
    return assignment

# ===========================================================================
# Sequence extraction (raw DNA strings for tokenizer)
# ===========================================================================

def load_gene_models(gtf_path):
    gm = pr.read_gtf(gtf_path, as_df=True)
    gm = gm[(gm["Feature"] == "gene") & (gm["gene_biotype"] == "protein_coding")]
    return gm[["Chromosome", "Start", "End", "Strand", "gene_id"]].copy()

def get_chromosomes(gtf_df):
    chroms = gtf_df["Chromosome"].unique()
    numbered = sorted([c for c in chroms if re.match(r"^\d+$", str(c))], key=lambda x: int(x))
    if numbered: return numbered
    lettered = sorted([c for c in chroms if re.match(r"^\d+[A-Z]$", str(c))])
    if lettered: return lettered
    return sorted(chroms)[:20]

def extract_dna_strings(fasta_path, gene_models, seq_len):
    """Extract raw DNA strings centered on TSS. Returns {gene_id: (dna_str, chrom)}."""
    fasta = Fasta(str(fasta_path), as_raw=True, sequence_always_upper=True, read_ahead=10000)
    upstream = seq_len // 2
    downstream = seq_len - upstream
    results = {}
    for _, row in gene_models.iterrows():
        chrom = str(row["Chromosome"])
        start, end, strand, gid = row["Start"], row["End"], row["Strand"], row["gene_id"]
        if chrom not in fasta: continue
        chrom_len = len(fasta[chrom])
        if strand == "+":
            s = start - upstream
            e = start + downstream
        else:
            s = end - downstream
            e = end + upstream
        if s < 0 or e > chrom_len: continue
        try:
            seq = fasta[chrom][s:e]
        except (KeyError, ValueError): continue
        if strand == "-":
            comp = {"A": "T", "T": "A", "C": "G", "G": "C", "N": "N"}
            seq = "".join(comp.get(c, "N") for c in reversed(seq))
        if len(seq) == seq_len:
            results[gid] = (seq, chrom)
    return results

# ===========================================================================
# PyTorch Dataset
# ===========================================================================

class GeneSeqDataset(Dataset):
    def __init__(self, gene_ids, sequences, labels, tokenizer, max_len):
        self.gene_ids = gene_ids
        self.sequences = sequences
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.gene_ids)

    def __getitem__(self, idx):
        seq = self.sequences[idx]
        enc = self.tokenizer(seq, max_length=self.max_len, truncation=True,
                             padding="max_length", return_tensors="pt")
        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "labels": torch.tensor(self.labels[idx], dtype=torch.float32),
        }

# ===========================================================================
# Model with classification head
# ===========================================================================

class PlantCADClassifier(torch.nn.Module):
    def __init__(self, base_model, hidden_size, dropout=0.1):
        super().__init__()
        self.base = base_model
        self.dropout = torch.nn.Dropout(dropout)
        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(hidden_size, 128),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(128, 1),
        )

    def forward(self, input_ids, attention_mask=None):
        outputs = self.base(input_ids=input_ids, output_hidden_states=True)
        # Use mean pooling of last hidden state
        hidden = outputs.hidden_states[-1]  # (batch, seq_len, hidden)
        if attention_mask is not None:
            mask = attention_mask.unsqueeze(-1).float()
            pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        else:
            pooled = hidden.mean(dim=1)
        logits = self.classifier(self.dropout(pooled))
        return logits.squeeze(-1)

# ===========================================================================
# Training loop
# ===========================================================================

def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss, n = 0, 0
    for batch in loader:
        ids = batch["input_ids"].to(device)
        mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        optimizer.zero_grad()
        logits = model(ids, mask)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * len(labels)
        n += len(labels)
    return total_loss / n

def evaluate(model, loader, device):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in loader:
            ids = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            labels = batch["labels"]

            logits = model(ids, mask)
            probs = torch.sigmoid(logits).cpu().numpy()
            all_preds.extend(probs)
            all_labels.extend(labels.numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    if len(np.unique(all_labels)) < 2:
        return {"auROC": 0.5, "F1": 0.0, "acc": 0.5, "preds": all_preds, "labels": all_labels}

    auroc = roc_auc_score(all_labels, all_preds)
    f1 = f1_score(all_labels, (all_preds > 0.5).astype(int))
    acc = accuracy_score(all_labels, (all_preds > 0.5).astype(int))
    return {"auROC": auroc, "F1": f1, "acc": acc, "preds": all_preds, "labels": all_labels}

# ===========================================================================
# Main
# ===========================================================================

def main():
    import subprocess

    # Download genomes
    genome_dir = WORK_DIR / "genomes"
    gtf_dir = WORK_DIR / "gene_models"
    genome_dir.mkdir(exist_ok=True)
    gtf_dir.mkdir(exist_ok=True)

    FTP_BASE = f"http://ftp.ensemblgenomes.org/pub/plants/release-{ENSEMBL_RELEASE}"
    for sp_name, sp_info in SPECIES.items():
        ens = sp_info["ensembl"]
        asm = sp_info["assembly"]
        cap = ens[0].upper() + ens[1:]
        for ftype, template in [
            ("fa", f"{cap}.{asm}.dna.toplevel.fa.gz"),
            ("gtf", f"{cap}.{asm}.{ENSEMBL_RELEASE}.gtf.gz"),
        ]:
            out_dir = genome_dir if ftype == "fa" else gtf_dir
            out_path = out_dir / template.replace(".gz", "")
            if out_path.exists(): continue
            url = f"{FTP_BASE}/{'fasta' if ftype == 'fa' else 'gtf'}/{ens}/{'dna/' if ftype == 'fa' else ''}{template}"
            print(f"Downloading {template}...")
            subprocess.run(["wget", "-q", "--no-check-certificate", "-O",
                            str(out_dir / template), url])
            subprocess.run(["gunzip", "-f", str(out_dir / template)])

    # Load DEG table
    deg_df = load_deg_table(DEG_FILE, list(SPECIES.keys()), organ=ORGAN)
    for sp in SPECIES:
        id_map = load_id_mapping(sp)
        if id_map:
            mask = deg_df["species"] == sp
            deg_df.loc[mask, "gene_locus"] = deg_df.loc[mask, "gene_locus"].map(lambda x: id_map.get(x, x))

    # Load genomes + extract sequences
    species_data = {}
    for sp_name, sp_info in SPECIES.items():
        ens = sp_info["ensembl"]
        asm = sp_info["assembly"]
        cap = ens[0].upper() + ens[1:]
        fa = genome_dir / f"{cap}.{asm}.dna.toplevel.fa"
        gtf = gtf_dir / f"{cap}.{asm}.{ENSEMBL_RELEASE}.gtf"
        if not fa.exists() or not gtf.exists():
            print(f"  {sp_name}: SKIP (missing files)")
            continue
        gm = load_gene_models(str(gtf))
        chroms = get_chromosomes(gm)
        gm = gm[gm["Chromosome"].isin(chroms)]
        seq_dict = extract_dna_strings(fa, gm, MAX_SEQ_LEN)
        print(f"  {sp_name}: {len(seq_dict)} sequences ({MAX_SEQ_LEN}bp)")
        species_data[sp_name] = {"seq_dict": seq_dict, "all_genes": set(gm["gene_id"].values)}

    # Prepare labels
    labels_store = {}
    for sp_name in species_data:
        labels_store[sp_name] = {}
        all_genes = species_data[sp_name]["all_genes"]
        seq_genes = set(species_data[sp_name]["seq_dict"].keys())
        for direction in DIRECTIONS:
            result = compute_deg_frequency(deg_df, sp_name, STRESS, all_genes, direction)
            if result is None: continue
            freq, n_exp = result
            lab = label_by_percentile(freq)
            if lab is None: continue
            valid = {g: int(lab[g]) for g in seq_genes & set(lab.index) if lab[g] in [0, 1]}
            n1 = sum(1 for v in valid.values() if v == 1)
            n0 = len(valid) - n1
            print(f"  {sp_name}/{direction}: {n1} responsive, {n0} non-responsive")
            labels_store[sp_name][direction] = valid

    # Load orthogroups
    gene_to_og = load_orthogroups(str(ORTHOGROUPS_FILE))

    # Load tokenizer
    print(f"\nLoading {MODEL_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)

    # Train per direction
    all_results = []

    for direction in DIRECTIONS:
        print(f"\n{'='*60}")
        print(f"PlantCAD2 fine-tuning: {STRESS} / {direction}")
        print(f"{'='*60}")

        # Pool genes
        all_gids, all_seqs, all_labels_list, all_sp = [], [], [], []
        for sp_name in species_data:
            if direction not in labels_store.get(sp_name, {}): continue
            gene_labels = labels_store[sp_name][direction]
            seq_dict = species_data[sp_name]["seq_dict"]
            for gid, lab in gene_labels.items():
                if gid in seq_dict:
                    dna, _ = seq_dict[gid]
                    all_gids.append(gid)
                    all_seqs.append(dna)
                    all_labels_list.append(lab)
                    all_sp.append(sp_name)

        n1 = sum(all_labels_list)
        print(f"  Pooled: {len(all_gids)} genes, {n1} responsive, {len(all_gids)-n1} non-responsive")

        # Orthogroup split
        assignment = split_by_orthogroups(all_gids, gene_to_og)

        train_ids, train_seqs, train_labels = [], [], []
        val_ids, val_seqs, val_labels = [], [], []
        test_ids, test_seqs, test_labels, test_sp_tags = [], [], [], []

        for gid, seq, lab, sp in zip(all_gids, all_seqs, all_labels_list, all_sp):
            split = assignment.get(gid, "train")
            if split == "train":
                train_ids.append(gid); train_seqs.append(seq); train_labels.append(lab)
            elif split == "val":
                val_ids.append(gid); val_seqs.append(seq); val_labels.append(lab)
            else:
                test_ids.append(gid); test_seqs.append(seq); test_labels.append(lab)
                test_sp_tags.append(sp)

        print(f"  Train: {len(train_ids)}, Val: {len(val_ids)}, Test: {len(test_ids)}")

        # Balance training
        train_labels_arr = np.array(train_labels)
        pos_idx = np.where(train_labels_arr == 1)[0]
        neg_idx = np.where(train_labels_arr == 0)[0]
        n_min = min(len(pos_idx), len(neg_idx))
        sel = np.concatenate([np.random.choice(pos_idx, n_min, replace=False),
                              np.random.choice(neg_idx, n_min, replace=False)])
        np.random.shuffle(sel)
        train_ids = [train_ids[i] for i in sel]
        train_seqs = [train_seqs[i] for i in sel]
        train_labels = [train_labels[i] for i in sel]
        print(f"  Balanced train: {len(train_ids)} ({n_min}/class)")

        # Datasets
        train_ds = GeneSeqDataset(train_ids, train_seqs, train_labels, tokenizer, MAX_SEQ_LEN)
        val_ds = GeneSeqDataset(val_ids, val_seqs, val_labels, tokenizer, MAX_SEQ_LEN)
        test_ds = GeneSeqDataset(test_ids, test_seqs, test_labels, tokenizer, MAX_SEQ_LEN)

        train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE)
        test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE)

        # Load model + LoRA
        base_model = AutoModelForMaskedLM.from_pretrained(MODEL_NAME, trust_remote_code=True)
        hidden_size = base_model.config.d_model

        # Apply LoRA
        lora_config = LoraConfig(
            r=LORA_R, lora_alpha=LORA_ALPHA, lora_dropout=LORA_DROPOUT,
            target_modules=["in_proj", "out_proj"],  # Mamba projection layers
            bias="none",
        )
        base_model = get_peft_model(base_model, lora_config)
        base_model.print_trainable_parameters()

        model = PlantCADClassifier(base_model, hidden_size).to(DEVICE)

        # Class weights for loss
        weights = compute_class_weight("balanced", classes=np.array([0, 1]),
                                       y=np.array(train_labels))
        pos_weight = torch.tensor(weights[1] / weights[0], dtype=torch.float32).to(DEVICE)
        criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)

        optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)

        # Training
        best_val_auroc = 0
        best_model_path = WORK_DIR / f"best_{direction}.pt"
        patience_counter = 0

        for epoch in range(EPOCHS):
            loss = train_epoch(model, train_loader, optimizer, criterion, DEVICE)
            val_res = evaluate(model, val_loader, DEVICE)

            print(f"  Epoch {epoch+1}: loss={loss:.4f}, val auROC={val_res['auROC']:.4f}")

            if val_res["auROC"] > best_val_auroc:
                best_val_auroc = val_res["auROC"]
                torch.save(model.state_dict(), best_model_path)
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= 3:
                    print(f"  Early stopping at epoch {epoch+1}")
                    break

        # Evaluate on test
        model.load_state_dict(torch.load(best_model_path, weights_only=True))
        test_res = evaluate(model, test_loader, DEVICE)
        print(f"\n  TEST: auROC={test_res['auROC']:.4f}, F1={test_res['F1']:.4f}, acc={test_res['acc']:.4f}")

        # Per-species test
        test_sp_arr = np.array(test_sp_tags)
        for sp_name in sorted(set(test_sp_tags)):
            m = test_sp_arr == sp_name
            if m.sum() < 10 and len(np.unique(test_res["labels"][m])) < 2: continue
            if len(np.unique(test_res["labels"][m])) < 2: continue
            sp_auroc = roc_auc_score(test_res["labels"][m], test_res["preds"][m])
            print(f"    {sp_name}: auROC={sp_auroc:.4f} (n={m.sum()})")

        all_results.append({"direction": direction, "test_auROC": test_res["auROC"],
                            "test_F1": test_res["F1"], "val_auROC": best_val_auroc})

    # Save results
    pd.DataFrame(all_results).to_csv(WORK_DIR / "plantcad2_results.csv", index=False)
    print(f"\nResults saved to {WORK_DIR / 'plantcad2_results.csv'}")


if __name__ == "__main__":
    main()
