# Figure 2 -- Gene family analysis

Scripts that generate the 10 panels of Figure 2, characterizing stress-responsive orthogroups across 36 plant species and 9 stress types.

## Scripts

| Script | Panels | Description |
|--------|--------|-------------|
| `fig2_gene_families.py` | B, C, D | Organ x stress experiment coverage; conservation score histogram; conservation profile per stress |
| `fig2b_v1_core_functions.py` | E | Functional profile of core stress responders (>=7/9 stresses), with stress composition coloring and supplementary fold-enrichment heatmap |
| `fig2e_phylostrata.py` | F, G, H | Phylostratigraphic analysis: OG count, stress breadth, and response strength per evolutionary age category |
| `fig2_hierarchy_overlap.py` | A, I | Data hierarchy pyramid; overlap coefficient schematic (Heat vs Cold example) |

## Run order

1. **Run `fig2_gene_families.py` first.** It builds shared cache files (`_cache_*.csv`) used by the other scripts.
2. The remaining three scripts can run in any order once the caches exist.

## Input data

Paths are hard-coded near the top of each script. Update them to point to your local copies:

```python
DEG_FILE = "/path/to/kingdom_stress_dict v3.csv"
OG_FILE  = "/path/to/Orthogroups.txt"
GO_FILE  = "/path/to/ATH_GO_GOSLIM.txt"
```

## Methods summary

- **Gene ID reconciliation**: transcript suffixes (`.1`, `.2`) stripped from both orthogroup and DEG gene IDs for locus-level matching.
- **Conservation score**: fraction of 36 species in which an orthogroup has at least one DEG. Classified as universal (>50%), moderate (10-50%), or lineage-specific (<10%).
- **Core stress responder**: orthogroup with DEGs in >=7 of 9 stress types.
- **Phylostratum assignment**: most basal clade represented among an orthogroup's member species, ordered Viridiplantae > Streptophyta > Embryophyta > Tracheophyta > Spermatophyta > Angiospermae > Class-specific (monocot or eudicot only).
- **Overlap coefficient**: OC(A, B) = |A intersect B| / min(|A|, |B|), used to quantify pairwise stress similarity at the orthogroup level.
- **Functional categories**: 22 manually curated stress-relevant categories grouping 117 specific GO biological process terms, mapped to orthogroups via Arabidopsis member genes.

## Dependencies

- Python 3.9+
- pandas
- numpy
- scipy
- matplotlib
- seaborn

## Notes

- Nitrogen stress is excluded (only 2 species had data).
- Plot styling uses fixed stress colors across panels for consistency.
- The 1.9 GB DEG file is read in chunks; caches are generated on first run.
