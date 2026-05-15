# Figure 4 -- MapMan functional bin co-occurrence networks

Scripts that generate the data-driven panels of Figure 4 and Supplementary Figures S5-S7. The figure characterizes the functional architecture of plant stress responses using MapMan bin co-occurrence networks across 36 species and 6 core stresses (Heat, Cold, Drought, Salt, Pathogen, Heavy metal).

The assembled figure is available as [`Figure 4.pdf`](Figure%204.pdf).

## Scripts by panel

### Figure 4

| Script | Panel | Description |
|--------|-------|-------------|
| `bin_direction_analysis.py` | A (data) | Computes per-bin UP/DOWN/bias scores from MapMan pathway enrichment analysis (PEA) summary |
| `combine_L1_panels.py` | A (figure) | Two-part panel: diverging bar chart of overall direction bias + per-stress heatmap for top/bottom 25 L1 bins |
| `panel_B_network_L1.py` | B | L1 co-occurrence network with nodes colored by direction bias and edges colored by UP/DOWN regulation; top 3 edges per node |
| `panel_B_network.py` | B (L0 alternative) | L0 version of the panel B network |
| `compute_species_cooccurrence.py` | C, E (data) | Computes per-edge species co-occurrence counts (required for edge filtering) |
| `stress_clustermap_v5.py` | C + E | Stress-unique edge bar chart (C) and Jaccard clustermaps for UP and DOWN edges (E) |
| `stress_specific_L0.py` | D | Per-stress L0 co-occurrence networks plus stress-specificity deviation |

### Supplementary Figure S5
| Script | Description |
|--------|-------------|
| `supplementary_figure.py` | Combined supplementary: species cutoff curve, Jaccard clustermaps (UP + DOWN), phylogenetic conservation key bins, clade dendrogram |

### Supplementary Figure S6
| Script | Description |
|--------|-------------|
| `phylo_conservation.py` | L1 direction bias across 7 plant clades (Chlorophyte to Monocot) with summed mean bias sidebar |

### Supplementary Figure S7
| Script | Description |
|--------|-------------|
| `hormone_per_stress_networks.py` | 49 individual hormone x stress XGMML networks for Cytoscape |
| `hormone_L0_network.py` | L1 hormone bin to L0 pathway bin edges with activity filtering |
| `hormone_combined_xgmml.py` | Combined grid XGMML: hormones as columns, stresses as rows |
| `hormone_grid_v2.py` | Grid v2: filtered to active hormone-stress combinations |
| `interpret_hormone_networks.py` | KG-based text interpretation of hormone-pathway edges |
| `kg_validation_figure.py` | KG support heatmap + direction comparison + summary bar chart |
| `kg_hormone_stress_heatmap.py` | Literature bias (KG entries) vs transcriptomic responsiveness heatmap |

### Supporting analyses (referenced by the paper)
| Script | Purpose |
|--------|---------|
| `fig4_mercator_analyses.py` | Master analysis script: 9 analyses including consensus networks, edge asymmetry, clade similarity, organ comparison, hubs, communities, drilldown |
| `analysis1_level1.py` | Consensus stress network at Level 1 |
| `conserved_specific_organ.py` | Conserved vs stress-specific bin associations, leaf vs root |
| `organ_rewiring.py` | Leaf vs root co-occurrence rewiring at L1 |
| `organ_flips.py` | Direction-flipping edges (UP in one organ, DOWN in the other) |
| `stress_clustermap.py`, `stress_clustermap_v3.py`, `stress_clustermap_v4.py` | Earlier iterations of the stress clustermap (kept for reproducibility) |
| `stress_signature_network.py` | L1 network with edges colored by most-specific stress |
| `stress_specific_edges_figure.py`, `stress_specific_edges_L1.py` | Stress-specific edges at L0 and L1 |
| `stress_specific_figure.py` | Per-stress hormone-L0 heatmaps (6 panels) |
| `edge_summary_figure.py` | Universal vs stress-specific vs organ-shared stacked bar |
| `enzyme_classification_breakdown.py`, `enzyme_genes_functions.py` | Deep breakdown of the Enzyme classification MapMan bin |
| `hub_gene_annotation.py` | Map top hub MapMan bins to Arabidopsis genes using Mercator4 |
| `tf_breakdown.py`, `tf_target_bins.py` | TF family classification and TF-to-target bin mapping |
| `export_cytoscape.py`, `export_cytoscape_xgmml.py` | Cytoscape network exports (CSV and XGMML) |

## Run order

Scripts share intermediate caches in `/tmp/mercator_outputs/`. Run in this order:

1. **`bin_direction_analysis.py`** -- builds `bin_summary_{L0,L1}.csv` and `bin_per_stress_{L0,L1}.csv` (consumed by panels A, D and many supporting analyses).
2. **`compute_species_cooccurrence.py`** -- builds `species_cooccurrence_L1.pkl` and `species_cutoff_curve.csv` (required by `stress_clustermap_v5.py` and `supplementary_figure.py`).
3. **`fig4_mercator_analyses.py`** -- master script that produces `analysis{1-9}_*.csv` and consensus network plots.
4. **Panel-specific scripts** (`combine_L1_panels.py`, `panel_B_network_L1.py`, `stress_clustermap_v5.py`, `stress_specific_L0.py`) and supplementary scripts (`supplementary_figure.py`, `phylo_conservation.py`, hormone/KG scripts) can be run in any order after the caches are built.

## Input data

Paths are hard-coded near the top of each script. Update them to point to your local copies:

```python
BASE = "/tmp/mercator_data"          # Mercator network matrices (All_stress/, Stresses/, Clades/)
OUT  = "/tmp/mercator_outputs"        # Intermediate and final outputs

# Specific files used by bin_direction_analysis.py
PEA_LEVEL0 = "/tmp/Mercator_pathway_analysis_summary_level0.csv"
PEA_LEVEL1 = "/tmp/Mercator_pathway_analysis_summary_level1.csv"
BIN_NAMES  = "/tmp/mercator_process_list.csv"

# Arabidopsis annotation used by hub/enzyme/TF scripts
ATH_MERCATOR = "/tmp/Arabidopsis_thaliana_Mercator.txt"

# Knowledge graph entries used by hormone/KG scripts
KG_ENTRIES = "/tmp/kg/hormone_kg_entries.csv"
```

| File | Used by | Description |
|------|---------|-------------|
| `mercator_data/All_stress/*.csv` | network scripts | Cross-stress normalized Mercator networks (UP/DOWN/MERGE, L0/L1/L2, All/Leaf/Root) |
| `mercator_data/Stresses/{stress}/*.csv` | per-stress scripts | Per-stress networks (Heat, Cold, Drought, Salt, Pathogen, Heavy metal) |
| `mercator_data/Clades/{clade}/*.csv` | phylo scripts | Per-clade normalized networks (7 clades, basal to derived) |
| `Mercator_pathway_analysis_summary_level{0,1}.csv` | `bin_direction_analysis.py` | Pathway enrichment summary with US/DS/UDS/NS scores per bin per experiment |
| `mercator_process_list.csv` | many | Bincode-to-bin-name lookup |
| `Arabidopsis_thaliana_Mercator.txt` | hub/enzyme/TF scripts | Arabidopsis gene-to-bin assignments |
| `kg/hormone_kg_entries.csv` | hormone/KG scripts | PlantConnectome knowledge graph entries for hormones |

## Methods summary

- **Direction bias**: For each MapMan bin, the fraction of experiments with UP enrichment (US + UDS) minus the fraction with DOWN enrichment (DS + UDS), averaged across all species/stress/organ combinations. Positive = predominantly upregulated.
- **Co-occurrence network**: Edges between MapMan bins weighted by how often the bins are jointly enriched among DEGs of the same direction across experiments. Networks available at L0, L1, L2; per stress; per organ (All, Leaf, Root); per clade; and cross-stress normalized.
- **Edge filtering (dual criterion)**: Edges retained only if (i) |UP weight - DOWN weight| > 2/N experiments (where N is the per-stress experiment count: Heat 96, Cold 70, Drought 92, Salt 68, Pathogen 61, Heavy metal 46) AND (ii) the two bins co-occur as enriched in >= 5 species. The dual filter prevents single dominant species from driving the network.
- **Excluded L0 bins**: Enzyme classification, not assigned, Protein modification, Protein biosynthesis (too generic or too large for interpretable networks).
- **Stress similarity (Panel E)**: Pairwise Jaccard index between filtered edge sets, separately for UP and DOWN edges, after removing universal edges (those present in all 6 stresses). Average of UP and DOWN Jaccard matrices used for UPGMA hierarchical clustering.
- **Phylogenetic conservation (S5D, S6)**: Per-clade direction bias averaged across stresses, displayed in phylogenetic order (Chlorophyte to Monocot).
- **KG validation (S7)**: Hormone-pathway edges cross-referenced against PlantConnectome KG; agreement between KG direction (normal conditions) and stress network direction reported.

## Dependencies

- Python 3.9+
- pandas, numpy
- matplotlib, seaborn
- scipy (hierarchical clustering, squareform/linkage)
- networkx (network construction and spring layout)

## Notes

- Six core stresses are used in this figure (Heat, Cold, Drought, Salt, Pathogen, Heavy metal). High light, Flooding, Nitrogen, and Herbivory are excluded due to limited species coverage.
- Cytoscape XGMML files for the per-stress hormone networks (Supplementary Figure S7) require manual layout refinement in Cytoscape after generation.
- Multiple `stress_clustermap_v*.py` scripts are retained for reproducibility; the final figure uses `stress_clustermap_v5.py`.
