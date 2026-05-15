# Figure 6 — Cis-regulatory sequence features predict stress-responsive genes across the plant kingdom

[Figure 6.pdf](Figure%206.pdf)

This folder contains all scripts used to generate Figure 6 of the Kingdom Stress Atlas paper, plus the final assembled figure.

## Panel-to-script mapping

| # | Script | Panel(s) | Description |
|---|--------|----------|-------------|
| 01 | `01_build_id_mapping.py` | data prep | Map DEG gene IDs to genome gene IDs via BLAST. |
| 02 | `02_prepare_labels.py` | data prep | Build orthogroup-based train/val/test splits with quartile DEG labels. |
| 03 | `03_cnn_baseline.py` | B (CNN bars) | Train CNN baseline (3× Conv1D on 600 bp one-hot DNA). |
| 04 | `04_plantcad2_finetune.py` | B (PlantCAD2 bars) | Fine-tune PlantCAD2 (Mamba/Caduceus, 88 M) with LoRA r=8. |
| 05a | `05a_tokenize_rna.py` | B (PlantRNA-FM bars) | Tokenize mRNA windows for PlantRNA-FM. |
| 05b | `05b_preprocess_data.py` | B | Build `ready/*.pt` tensors from token cache + dataset pickles. |
| 05c | `05c_train_rna.py` | B | Fine-tune PlantRNA-FM (ESM, 33 M) with LoRA r=16. |
| 05d | `05d_train_all.slurm` | B | SLURM array for the 10 RNA models. |
| 06 | `06_late_fusion.py` | B, C, D | Align DNA and RNA predictions; compute fusion auROC and gain. |
| 07a | `07a_motif_model_combined.py` | B, G, H | GBM on motifs + dinucleotides + RNA structure (combined). |
| 07b | `07b_motif_model_ablation.py` | B (DinucFreq/Motif/Structure) | Train GBM on each feature group separately. |
| 07c | `07c_interpretable_model_replication.py` | B (replication) | Replicate panel B's right side from the published feature table. |
| 08a | `08a_tokenize_windows.py` | E | Tokenize transcript windows for retraining. |
| 08b | `08b_train_windows.slurm` | E | SLURM job for the window scan. |
| 09a | `09a_cnn_interpretability.py` | F (CNN motifs) | CNN integrated gradients + top motif extraction. |
| 09b | `09b_rna_interpretability.py` | F (RNA motifs) | PlantRNA-FM gradient × input attribution + motif extraction. |
| 10 | `10_figure6.py` | All | Assemble the final multi-panel figure. |

## Run order

```bash
# Stage 1 — data prep (builds shared caches used downstream)
python 01_build_id_mapping.py
python 02_prepare_labels.py

# Stage 2 — train sequence models (independent, can run in parallel)
python 03_cnn_baseline.py
python 04_plantcad2_finetune.py            # Colab A100
python 05a_tokenize_rna.py                  # LUMI: build rna_tokens cache
python 05b_preprocess_data.py               # LUMI: build ready/*.pt
sbatch 05d_train_all.slurm                  # LUMI: 10 RNA models

# Stage 3 — window scan
python 08a_tokenize_windows.py
sbatch 08b_train_windows.slurm              # 40 window models

# Stage 4 — fusion and interpretability
python 06_late_fusion.py
python 09a_cnn_interpretability.py
python 09b_rna_interpretability.py
# (PlantCAD2 ISM was performed inline in PlantCAD2_expanded.ipynb on Colab.)

# Stage 5 — interpretable feature models
python 07a_motif_model_combined.py          # also builds structure_cache.pkl
python 07b_motif_model_ablation.py          # reuses structure_cache.pkl
# OR for replication only:
python 07c_interpretable_model_replication.py \
    --feature_table feature_table_all_genes.csv.gz \
    --dataset_dir plantrna_fm_results/ \
    --output_dir replication/

# Stage 6 — figure
python 10_figure6.py
```

## Input data paths

All scripts have absolute paths hard-coded near the top; edit once per environment. The main inputs are:

- `gene_seqs/*.fa` — per-gene 10 kb sequences (5 kb up + 5 kb down of TSS), 25 species.
- `dataset_{stress}_{direction}.pkl` — orthogroup-split gene IDs + labels.
- `Orthogroups.txt` — OrthoFinder output (~2.5 M gene-to-orthogroup mappings).
- `kingdom_stress_dict v3_.csv` — master DEG table (6.2 M rows).

Pre-computed:
- `feature_table_all_genes.csv.gz` — 1,687,829 genes × 658 features (figshare Supplementary Dataset 6).
- `structure_cache.pkl` — ViennaRNA features for 970 K genes.

## Methods summary

- **Data**: 25 plant species, 5 stresses (Heat, Cold, Drought, Salt, Pathogen), 2 directions (UP/DOWN). DEG labels by quartile of DEG frequency across experiments.
- **Splits**: orthogroup-based 70/15/15 train/val/test, to prevent homology leakage.
- **CNN baseline**: 3× Conv1D + 2 dense layers on 600 bp one-hot DNA.
- **PlantCAD2** (DNA LLM): Mamba/Caduceus 88 M, fine-tuned with LoRA r=8, 2.4 M trainable parameters, on Colab A100.
- **PlantRNA-FM** (RNA LLM): ESM transformer 33 M, fine-tuned with LoRA r=16, 369 K trainable parameters, on LUMI (AMD MI250X, ROCm 6.2).
- **Late fusion**: simple averaging of DNA and RNA prediction probabilities per gene.
- **Window scan**: PlantRNA-FM retrained independently on four 1 kb transcript windows.
- **Interpretable features**: 16 dinucleotide frequencies × 2 regions + GC content + 16 known cis-element counts × 2 regions + ~292 discovered 8-mers × 2 regions + 8 RNA structure descriptors from ViennaRNA. Models: sklearn `GradientBoostingClassifier`.
- **Attribution**: integrated gradients (CNN), in silico mutagenesis (PlantCAD2 on Colab), gradient × input (PlantRNA-FM on M3 Max).
- **Statistical comparison** (panel G): Mann-Whitney U with Bonferroni correction across 400 tests.

## Dependencies

- Python 3.9+
- PyTorch 2.5.1
- transformers, peft==0.11.1 (pinned — newer versions break on ROCm)
- scikit-learn, xgboost
- ViennaRNA (RNA secondary structure)
- TensorFlow/Keras (CNN)
- matplotlib, pandas, numpy, scipy

## Hardware used

- **LUMI**: AMD MI250X GPUs, ROCm 6.2, Singularity container built with cotainr.
- **Colab**: NVIDIA A100 40 GB, CUDA, bfloat16.
- **Local**: Apple M3 Max (Metal GPU acceleration for interpretability).

## Related data

- **Supplementary Dataset 5** (figshare) — trained model weights for CNN, PlantCAD2 (LoRA adapters), and PlantRNA-FM.
- **Supplementary Dataset 6** (figshare) — `feature_table_all_genes.csv.gz`, 1,687,829 genes × 658 features.

## Citation

If you use these scripts, please cite the Kingdom Stress Atlas paper (in preparation).
