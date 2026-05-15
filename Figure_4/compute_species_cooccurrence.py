"""
Compute species-level co-occurrence for L1 bin edges.
For each stress: count how many species have both bins enriched (UP or DOWN).
Output edge tables with species count per stress.
"""

import os
import pandas as pd
import numpy as np
from collections import defaultdict
import pickle

OUT = "/tmp/mercator_outputs"

STRESSES = ['Heat', 'Cold', 'Drought', 'Salt', 'Pathogen', 'Heavy metal']
EXCLUDE_L0 = {'Enzyme classification', 'not assigned', 'Protein modification',
              'Protein biosynthesis'}

# ── Load PEA data ─────────────────────────────────────────────────────────
print("Loading PEA data...")
pea = pd.read_csv('/tmp/Mercator_pathway_analysis_summary_level1.csv')
pea['PARENT_BINCODE'] = pea['PARENT_BINCODE'].astype(str)
pea['UP'] = pea['US'] + pea['UDS']
pea['DOWN'] = pea['DS'] + pea['UDS']

proc = pd.read_csv('/tmp/mercator_process_list.csv', index_col=0)
proc['Bincode'] = proc['Bincode'].astype(str)
bc2name = dict(zip(proc['Bincode'], proc['Bincode name']))
name2bc = dict(zip(proc['Bincode name'], proc['Bincode']))

# ── For each species x stress: which L1 bins are UP / DOWN? ──────────────
print("Building per-species bin sets...")

# Aggregate across organs: a bin is UP for a species if ANY organ shows UP
species_up = {}   # (species, stress) -> set of L1 bincode names
species_down = {}

for (species, stress), group in pea.groupby(['species', 'stress']):
    up_bcs = set(group[group['UP'] > 0]['PARENT_BINCODE'])
    down_bcs = set(group[group['DOWN'] > 0]['PARENT_BINCODE'])

    # Convert to bin names and filter
    up_names = set()
    down_names = set()
    for bc in up_bcs:
        name = bc2name.get(bc, '')
        if name and name.split('.')[0] not in EXCLUDE_L0:
            up_names.add(name)
    for bc in down_bcs:
        name = bc2name.get(bc, '')
        if name and name.split('.')[0] not in EXCLUDE_L0:
            down_names.add(name)

    species_up[(species, stress)] = up_names
    species_down[(species, stress)] = down_names

# ── Count species co-occurrence per edge per stress ──────────────────────
print("Counting species co-occurrence per edge...")

# For each stress, get unique bins and compute pairwise species counts
stress_edge_species = {}

for stress in STRESSES:
    combos = [(sp, st) for (sp, st) in species_up if st == stress]
    n_species = len(combos)
    print(f"\n  {stress}: {n_species} species")

    # Collect all bins active in this stress
    all_up_bins = set()
    all_down_bins = set()
    for key in combos:
        all_up_bins |= species_up[key]
        all_down_bins |= species_down[key]

    # For each bin, which species have it UP or DOWN?
    bin_species_up = defaultdict(set)
    bin_species_down = defaultdict(set)
    for (sp, st) in combos:
        for b in species_up[(sp, st)]:
            bin_species_up[b].add(sp)
        for b in species_down[(sp, st)]:
            bin_species_down[b].add(sp)

    # Compute pairwise species co-occurrence
    # For UP edges: species where BOTH bins are UP
    # For DOWN edges: species where BOTH bins are DOWN
    all_bins = sorted(all_up_bins | all_down_bins)
    edge_data = {}

    for i, bin_a in enumerate(all_bins):
        for j, bin_b in enumerate(all_bins):
            if j <= i:
                continue
            key = tuple(sorted([bin_a, bin_b]))

            sp_both_up = bin_species_up[bin_a] & bin_species_up[bin_b]
            sp_both_down = bin_species_down[bin_a] & bin_species_down[bin_b]

            if len(sp_both_up) > 0 or len(sp_both_down) > 0:
                edge_data[key] = {
                    'n_species_up': len(sp_both_up),
                    'n_species_down': len(sp_both_down),
                    'n_species_total': n_species,
                }

    stress_edge_species[stress] = edge_data
    print(f"    Edges with >=1 species: {len(edge_data)}")
    print(f"    Edges with >=2 species UP: {sum(1 for v in edge_data.values() if v['n_species_up'] >= 2)}")
    print(f"    Edges with >=2 species DOWN: {sum(1 for v in edge_data.values() if v['n_species_down'] >= 2)}")

# ── Save ──────────────────────────────────────────────────────────────────
# Save as pickle for fast loading
with open(os.path.join(OUT, "species_cooccurrence_L1.pkl"), 'wb') as f:
    pickle.dump(stress_edge_species, f)

# Also save summary CSV
rows = []
for stress in STRESSES:
    for key, val in stress_edge_species[stress].items():
        rows.append({
            'stress': stress,
            'source': key[0], 'target': key[1],
            'n_species_up': val['n_species_up'],
            'n_species_down': val['n_species_down'],
            'n_species_total': val['n_species_total'],
        })

pd.DataFrame(rows).to_csv(os.path.join(OUT, "species_cooccurrence_L1.csv"), index=False)

print(f"\nSaved: species_cooccurrence_L1.pkl, species_cooccurrence_L1.csv ({len(rows)} rows)")
