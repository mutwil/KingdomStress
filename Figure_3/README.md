# Figure 3 -- Overlap-coefficient analysis of stress-responsive orthogroups

Scripts that generate panels B-E of Figure 3 and Figure S4, quantifying conservation of stress-responsive one-to-one orthologues across organs, clades, phylogenetic distance, and stress pairs.

The assembled figure is available as [`Figure 3.pdf`](Figure%203.pdf).

## Scripts

| Script | Panel | Description |
|--------|-------|-------------|
| `fig3_phylo_oc_decay.py` | C scatter / S4 | Cross-species OC cache builder; faceted scatter + LOWESS of mean OC vs TimeTree divergence (Figure S4) |
| `fig3b_cross_clade_heatmap.py` | B | Cross-clade mean OC heatmap (UP / DOWN) with Monte Carlo permutation significance (1,000 iter) |
| `fig3c_binned_permutation.py` | C (null) | Permutation null distribution per divergence-time bin (10,000 iter); writes `fig3c_permutation_results.csv` |
| `fig3c_binned_with_null.py` | C | Final binned bar chart of mean OC vs divergence time with null 95% CI overlay |
| `fig3c_rho_heatmap.py` | D | Spearman rho heatmap of divergence time vs OC, per stress and direction |
| `fig3_stress_pair_counts.py` | E | Horizontal bar chart of species counts per significant stress pair (visualization of upstream permutation result) |

Panel A (organ-restricted vs cross-organ OC) is assembled separately and is not produced by a script in this folder.

## Run order

Panels share intermediate CSVs, so the scripts must run in this order. Panels B and E are independent and can run at any time.

1. **`fig3_phylo_oc_decay.py`** -- builds `_cache_cross_species_oc.csv` (large, gitignored) and `fig3c_phylo_oc_data.csv`. Required by steps 2-4.
2. **`fig3c_binned_permutation.py`** -- writes `fig3c_permutation_results.csv`. Required by step 3.
3. **`fig3c_binned_with_null.py`** -- final Panel C.
4. **`fig3c_rho_heatmap.py`** -- Panel D. Can run any time after step 1.
5. **`fig3b_cross_clade_heatmap.py`** -- Panel B. Builds its own `_cache_cross_clade_all.csv`. Independent.
6. **`fig3_stress_pair_counts.py`** -- Panel E. Independent.

## Input data

Paths are hard-coded near the top of each script. Update them to point to your local copies:

```python
INPUT_FOLDER = "/tmp/kingdom_stress_oc"
CLADES_FILE  = "/tmp/kingdom_stress_oc/kingdom_stress_species_clades.csv"
```

`INPUT_FOLDER` must contain per-reference-species OC tables matching `unrestricted_one_to_one_*.csv`, produced upstream by the OrthoFinder + Kallisto + DESeq2 pipeline. `CLADES_FILE` maps each of the 36 species to one of seven clades (Chlorophyte, Charophyte, Bryophyte, Lycophyte, Gymnosperm, Monocot, Dicot).

## Methods summary

- **Overlap coefficient**: OC(A, B) = |A ∩ B| / min(|A|, |B|), computed pairwise between stress-responsive orthogroup sets at the one-to-one orthologue level. Sensitive to set similarity, robust to large size asymmetry.
- **Cross-species filter**: same stress, same DEG direction (UP or DOWN), different species. Each species pair contributes a mean OC per stress and direction.
- **Divergence times**: pairwise MRCA ages (Mya) computed by walking a hard-coded TimeTree-derived taxonomy. 36 species, 630 pairs.
- **Cross-clade significance (Panel B)**: 1,000 Monte Carlo permutations per reference species, shuffling OC values across stresses within each direction.
- **Binned distance significance (Panel C)**: 10,000 permutations of OC values across species pairs within each direction; per-bin p-value = fraction of null means >= observed.
- **Decay correlation (Panel D)**: Spearman rho of divergence time vs mean OC, per stress and direction (n >= 5 pairs required).

Direction-specific stress-responsive sets are defined upstream (|log2FC| > 1, adjusted p < 0.05 from DESeq2).

## Dependencies

- Python 3.10+
- numpy, pandas, scipy, statsmodels (LOWESS), matplotlib, joblib

## Notes

- The `_cache_*.csv` files speed up reruns (the first pass over `unrestricted_*.csv` is slow). Delete them to force a full recomputation.
- Panel E counts are hard-coded in `fig3_stress_pair_counts.py` from the upstream stress-pair permutation analysis; the script is visualization only.
- Six stresses are analysed (Cold, Drought, Heat, Heavy metal, Pathogen, Salt); flooding, herbivory, high light, and nitrogen are excluded for insufficient cross-species sampling.
