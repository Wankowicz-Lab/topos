# Biogenesis Grouped Structural Analysis

## Overview

There are two primary use cases:

| Use Case | Question | Key Scripts |
|----------|----------|-------------|
| **Multi-structure** | What structural features are conserved or variable across conformations? Why do some positions tolerate mutations? | `plot_all.py`, `identify_variable_residues.py`, `export_dms_annotations.py` |

| **Comparison** | How does a mutant or ligand-bound structure differ from WT/apo? Why does a specific mutation disrupt function? | `run_comparison_metrics.py`, `export_dms_annotations.py` |

Both modes produce **graphical outputs** and **annotation CSVs** that can be merged directly with DMS data.

---

## Overview
These scripts are meant to be used AFTER you have generated CSVs on metrics from individual PDBs.

```bash
# 1. Edit the template config for your protein
# Edit my_protein_config.toml: add your PDB IDs, set reference_pdb, define pairs

# 2. Run the full pipeline (renumber → variability → plots → DMS annotations → PyMOL)
python run_pipeline.py --config my_protein_config.toml

# Preview all commands without running them
python run_pipeline.py --config my_protein_config.toml --dry-run

# Skip plot generation (annotation CSVs still produced — faster)
python run_pipeline.py --config my_protein_config.toml --skip-plots
```

## Config TOML Reference

See `template_config.toml` for the full annotated template. Summary of all fields:

```toml
# ── Global settings (all optional; shown with defaults) ─────────────────────
output_dir          = "my_output/"   # all outputs written here
reference_pdb       = "4AKE"        # pdb_id of the reference for renumbering
chain               = "A"           # chain to analyse throughout
max_mismatches      = 5             # max allowed seq mismatches for renumbering
proximity_angstroms = 8.0           # Å radius for local comparison zone
top_n_variable      = 20            # top N variable residues to highlight

[analysis]
run_multi      = true   # multi-structure variability analysis
run_comparison = true   # pairwise comparison (requires [[pairs]])

# ── Structure block (one per PDB) ────────────────────────────────────────────
[[structures]]
label    = "WT_apo"               # unique name used in [[pairs]] and file names
pdb_id   = "4AKE"                 # RCSB ID (auto-downloaded)
pdb_path = "/path/to/local.pdb"  # optional: local file overrides download
group    = "open"                 # optional: group label for multi-structure
state    = "apo"                  # "apo" or "bound"
genotype = "wt"                   # "wt" or "mutant"
chain    = "A"

# Optional: ligand for proximity detection
ligand = {name = "AP5", chain = "A"}

# Optional: point mutations (one block per mutation)
[[structures.mutations]]
resi   = 64     # residue number (reference numbering)
wt_aa  = "S"    # single-letter WT amino acid
mut_aa = "D"    # single-letter mutant amino acid

# ── Comparison pairs ─────────────────────────────────────────────────────────
[[pairs]]
reference   = "WT_apo"       # label of reference structure
comparison  = "R167A"        # label of structure to compare
description = "WT vs R167A"  # used in output file names
```

---


---

## Use Case 1 — Multi-Structure Analysis

### Purpose

Analyze multiple structures of the same protein (different conformations, crystal forms, or homologs) to identify:
- **Conserved positions**: structurally invariant across conformations
- **Variable positions**: change between conformations 


### Step-by-Step (Manual)

All scripts auto-discover PDB IDs from their input directory when `--pdbs` is omitted. Every script also accepts explicit `--pdbs` for manual use.

```bash

# 1. Align all structures to a common reference numbering
python renumber_to_reference.py --output-dir my_output/ --ref 4AKE
# → my_output/renumbered/{PDBID}_features.csv

# 2. Identify residues with the largest metric changes across structures
python identify_variable_residues.py --renumbered-dir my_output/renumbered/ \
    --chain A --top 20 --out my_output/variability/
# → my_output/variability/residue_variability_ranking.csv
# → my_output/variability/variability_heatmap.png
# → my_output/variability/overall_variability_score.png

# 3. Generate comprehensive per-residue plots
python plot_all.py --renumbered-dir my_output/renumbered/ \
    --chain A --out my_output/residue_profiles/
# → my_output/residue_profiles/{metric}.png        (line plots)
# → my_output/residue_profiles/{metric}_boxplot.png
# → my_output/residue_profiles/{metric}_heatmap.png

# 4. Export perturbation annotation CSV
python export_structural_annotations.py --mode multi --chain A \
    --renumbered-dir my_output/renumbered/ \
    --variability-dir my_output/variability/ \
    --out my_output/dms_annotations/
# → my_output/annotations/structural_annotations_multi.csv

# 5. Generate PyMOL visualization scripts
python map_to_pymol.py \
    --csv my_output/annotations/structural_annotations_multi.csv \
    --metric variability_score \
    --pdb 4AKE --output my_output/pymol/4AKE_variability
# → my_output/pymol/4AKE_variability_spectrum.pml
# → my_output/pymol/4AKE_variability_bfactor.pdb
```

### Graphical Outputs

| File | Description |
|------|-------------|
| `variability_heatmap.png` | All residues × all metrics coloured by variability (rank-normalised SD) |
| `overall_variability_score.png` | Bar chart: one bar per residue, top-20 highlighted in red |
| `top10_variable_metrics.png` | SD per residue for the 10 most variable metrics |
| `{metric}.png` (line plots) | Mean ± SD across structures + individual structure traces |
| `{metric}_boxplot.png` | Per-residue boxes with individual structure points |
| `{metric}_heatmap.png` | Structures × residues colour matrix |

### Annotation CSV: `dms_structural_annotations_multi.csv`

One row per residue position (reference numbering). Key columns:

| Column | Description | DMS Use |
|--------|-------------|---------|
| `resi` | Residue number (reference) | Merge key |
| `resn` | Residue name (most common) | Identity check |
| `variability_score` | 0–1; higher = more variable across structures | 
| `sasa_class` | buried / partially_buried / exposed |
| `mean_sasa` | Mean solvent-accessible surface area (Å²) |
| `mean_total_hbond_count` | Mean H-bond count | 
| `mean_packing_contact_density` | Mean packing density |
| `mean_graph_all_graph_betweenness_centrality` |
| `ss_group_consensus` | Most common secondary structure element | 

---

## Use Case 2 — Pairwise Comparison

### Purpose

Compare two structures to find which residues change structurally:
- **WT vs. point mutant**: which residues are allosterically affected by the mutation?
- **Apo vs. ligand-bound**: which residues reorganize upon ligand binding?

This explains *why* a particular mutation might disrupt function.

### Input

A TOML config file defining structures and comparison pairs. Copy and edit `template_config.toml`:

```toml
output_dir          = "my_output/"
reference_pdb       = "4AKE"
proximity_angstroms = 8.0

[[structures]]
label    = "WT_apo"
pdb_id   = "4AKE"
state    = "apo"
genotype = "wt"
chain    = "A"

[[structures]]
label    = "R167A_mutant"
pdb_id   = "1XYZ"        # PDB ID of your mutant structure
state    = "apo"
genotype = "mutant"
chain    = "A"

[[structures.mutations]]
resi   = 167
wt_aa  = "R"
mut_aa = "A"

[[pairs]]
reference  = "WT_apo"
comparison = "R167A_mutant"
description = "WT vs R167A"
```

### Running the Full Pipeline

```bash
# One command — reads everything (structures, pairs, output_dir) from config
python run_pipeline.py --config my_protein_config.toml

# Manual steps

python run_comparison_metrics.py \
    --config my_protein_config.toml \
    --metrics-dir my_output/ \
    --output-dir my_output/comparisons/

python export_dms_annotations.py \
    --mode comparison \
    --local-dir my_output/comparisons/local/ \
    --out my_output/dms_annotations/

python map_to_pymol.py \
    --csv my_output/dms_annotations/dms_comparison_annotations_WT_vs_R167A.csv \
    --metric composite_change_score \
    --pdb 4AKE --output my_output/pymol/WT_vs_R167A_changes
```

### Graphical Outputs

| File | Description |
|------|-------------|
| `histograms/{metric}.jpg` | Distribution overlays for all structures |
| `local/{pair}_local_diffs.csv` | Per-residue, per-metric Δ values sorted by magnitude |
| `global_stats.csv` | Mann-Whitney U test + descriptive stats per metric |
| `{pair}_spectrum.pml` | PyMOL script: residues coloured by composite change score |
| `{pair}_groups.pml` | PyMOL script: residues coloured by change class |

### DMS Annotation CSV: `dms_comparison_annotations_{pair}.csv`

One row per residue position. Key columns:

| Column | Description | DMS Use |
|--------|-------------|---------|
| `resi` | Residue number | Merge key |
| `composite_change_score` | Normalized mean Δ across key metrics | 
| `in_proximity` | Within N Å of mutation/ligand site | 
| `sasa_delta` | Change in solvent accessibility |
| `total_hbond_count_delta` | Change in H-bond count |
| `packing_contact_density_delta` | Change in packing |
| `max_abs_delta` | Largest single-metric change | 
| `top_changed_metric` | Which metric changed most | 


## PyMOL Visualization

```bash
# Open any .pml script in PyMOL
pymol viz/4AKE_variability_spectrum.pml

# Or load the B-factor PDB and color in PyMOL
pymol viz/4AKE_variability_score_bfactor.pdb
# In PyMOL: spectrum b, blue_white_red, all

# In ChimeraX:
# File → Open → viz/4AKE_variability_score_bfactor.pdb
# Color → By Attribute → bfactor
```

