# KingdomStress

Scripts used to generate data and figures for the Kingdom Stress Atlas paper.

The Kingdom Stress Atlas is a multi-species stress transcriptomics resource covering RNA-seq stress experiments across 36 plant species spanning monocots, dicots, gymnosperms, lycophytes, bryophytes, charophytes, and chlorophytes. Stresses include heat, cold, drought, salt, high light, pathogen, flooding, heavy metal, and herbivory.

## Pipeline

- Quantification: Kallisto (LSTRAP-Cloud for public data)
- Differential expression: DESeq2 (|log2FC| > 1, adjusted p < 0.05)
- Orthogroups: OrthoFinder across 36 species (275,222 orthogroups)
- Functional annotation: TAIR GO biological process terms

## Figures

- [Figure 2](Figure_2/) -- Gene family analysis: conservation, functional composition, phylostratigraphic analysis, and stress overlap

## Input data

The scripts expect three input files (paths are hard-coded near the top of each script and may need to be updated):

| File | Description |
|------|-------------|
| `kingdom_stress_dict v3.csv` | DEG table (13.6M rows, 1.9 GB) |
| `Orthogroups.txt` | OrthoFinder output (275K orthogroups) |
| `ATH_GO_GOSLIM.txt` | TAIR Arabidopsis GO annotations |

## Citation

If you use these scripts, please cite the Kingdom Stress Atlas paper (in preparation).
