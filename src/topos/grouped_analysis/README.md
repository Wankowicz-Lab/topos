# Topos — Grouped Structural Analysis

Compare multiple pre-processed protein structures to identify conserved features,
variable positions, and pairwise structural differences.

Run this pipeline **after** the main topos pipeline has produced
`{PDB_ID}_features.csv` files for each structure.

---

## Quick Start

### 1. Generate per-structure features

Run the main topos pipeline on each structure first:

```python
from topos.pipeline.runner import Runner

for pdb_id in ["4AKE", "1AKE", "1ANK", "3HPQ", "6F7U"]:
    runner = Runner(pdb_id=pdb_id)
    runner.run()
    runner.save_results(output_dir="my_output/")
```

### 2. Write a config file

Copy `examples/grouped_analysis_example/grouped_analysis_config.toml` and edit it
for your protein system.

### 3. Run the grouped analysis pipeline

```python
from topos.grouped_analysis.run_grouped_pipeline import GroupedPipelineRunner

runner = GroupedPipelineRunner(config_path="my_protein_config.toml")
runner.run()
```

Skip plot generation (annotation CSVs still produced — faster):

```python
runner = GroupedPipelineRunner(config_path="my_protein_config.toml", skip_plots=True)
runner.run()
```
---

## Config Reference

All settings live in a single TOML file. See the fully annotated template at
`examples/grouped_analysis_example/grouped_analysis_config.toml`.

### Global settings

```toml
output_dir          = "my_output/"   # all outputs written here
reference_pdb       = "4AKE"        # pdb_id of the reference for residue renumbering
chain               = "A"           # chain to analyse throughout
max_mismatches      = 5             # max sequence mismatches before a structure is excluded
proximity_angstroms = 8.0           # Å radius for local comparison zone
top_n_variable      = 20            # top-N variable residues to highlight

# Analysis modes (can be at top level or inside [analysis] section)
run_multi      = true   # multi-structure variability analysis
run_comparison = true   # pairwise comparison (requires [[pairs]])
```

### Structure blocks

One `[[structures]]` block per PDB structure:

```toml
[[structures]]
label    = "4AKE"              # unique name used in [[pairs]] and file names
pdb_id   = "4AKE"             # RCSB PDB ID (must match features CSV prefix)
group    = "open"              # optional: group label (e.g. "open", "closed")
state    = "apo"               # optional: "apo" or "bound"
genotype = "wt"                # optional: "wt" or "mutant"
chain    = "A"                 # optional: overrides global chain

# Optional: ligand for proximity detection
ligand = {name = "AP5", chain = "A"}

# Optional: point mutations (one block per mutation; residue numbers in reference numbering)
[[structures.mutations]]
resi   = 56
wt_aa  = "G"
mut_aa = "A"
```

### Comparison pairs

Define which structures to compare:

```toml
[[pairs]]
reference   = "4AKE"                      # label of reference structure
comparison  = "1AKE"                      # label of structure to compare
description = "Open apo vs Closed bound"  # used in output file names
```

---

## Pipeline Steps

`GroupedPipelineRunner.run()` executes these steps in order:

| Step | Function | Output |
|------|----------|--------|
| 1 | Renumber all structures to reference numbering | `output_dir/renumbered/` |
| 2 | Compute per-residue variability scores | `output_dir/variability/` |
| 3 | Generate line plots, boxplots, heatmaps | `output_dir/residue_profiles/` |
| 4 | Export multi-structure annotation CSV | `output_dir/dms_annotations/` |
| 5 | Generate PyMOL scripts (multi) | `output_dir/pymol/` |
| 5b | Compute pairwise CA-RMSD | `output_dir/rmsd/` |
| 6 | Compute pairwise metric differences | `output_dir/comparisons/` |
| 7 | Export per-pair comparison annotation CSVs | `output_dir/dms_annotations/` |

Steps 2–5 only run when `run_multi = true`.
Steps 6–8 only run when `run_comparison = true` and at least one `[[pairs]]` block is defined.

---

## Use Case 1 — Multi-Structure Analysis

### Purpose

Identify which residues are **structurally conserved** or **variable** across multiple
conformations, crystal forms, or homologs of the same protein.

### What the pipeline does

1. Aligns all structure residue numberings to the reference PDB.
2. Computes per-residue standard deviation and range across structures for every metric.
3. Rank-normalizes and averages SDs into an overall **variability score** (0–1).
4. Produces an annotation CSV for merging with DMS data.

### Annotation CSV: `dms_structural_annotations_multi.csv`

One row per residue position (reference numbering):

| Column | Description |
|--------|-------------|
| `resi` | Residue number (reference) |
| `resn` | Residue name (most common across structures) |
| `variability_score` | 0–1; higher = more variable across structures |
| `variability_class` | `low` / `medium` / `high` |
| `sasa_class` | `buried` / `partially_buried` / `exposed` |
| `mean_sasa` | Mean solvent-accessible surface area (Å²) |
| `mean_total_hbond_count` | Mean hydrogen bond count |
| `mean_packing_contact_density` | Mean packing density |
| `mean_graph_all_graph_betweenness_centrality` | Mean betweenness centrality |
| `ss_group_consensus` | Most common secondary structure element |

---

## Use Case 2 — Pairwise Comparison

### Purpose

Compare two structures (WT vs mutant, apo vs bound) to identify which residues
show the largest structural changes, and which are near a mutation site or ligand.

### What the pipeline does

1. Computes per-residue, per-metric differences (Δ) between the two structures.
2. Flags residues within `proximity_angstroms` of a mutation site or ligand.
3. Summarizes differences into a **composite change score**.
4. Produces a per-pair annotation CSV for merging with DMS data.

### Annotation CSV: `dms_comparison_annotations_{description}.csv`

One row per residue position:

| Column | Description |
|--------|-------------|
| `resi` | Residue number |
| `composite_change_score` | Normalized mean Δ across key metrics |
| `change_class` | `unchanged` / `moderate` / `large` |
| `in_proximity` | `True` if within N Å of mutation/ligand site |
| `sasa_delta` | Change in solvent accessibility |
| `total_hbond_count_delta` | Change in H-bond count |
| `packing_contact_density_delta` | Change in packing |
| `max_abs_delta` | Largest single-metric change |
| `top_changed_metric` | Which metric changed most |

---

## Outputs

```
output_dir/
├── renumbered/                        # reference-renumbered feature CSVs
│   └── {PDB_ID}_features.csv
├── variability/                       # multi-structure variability
│   ├── per_residue_sd.csv
│   ├── per_residue_range.csv
│   ├── residue_variability_ranking.csv
│   ├── variability_heatmap.png
│   └── overall_variability_score.png
├── residue_profiles/                  # per-metric distribution plots
│   ├── {metric}.png
│   ├── {metric}_boxplot.png
│   └── {metric}_heatmap.png
├── rmsd/                              # pairwise CA-RMSD matrix
│   └── pairwise_rmsd.csv
├── comparisons/                       # pairwise metric differences
│   ├── global_stats.csv
│   ├── histograms/
│   └── local/{description}_local_diffs.csv
├── dms_annotations/                   # annotation CSVs for DMS merging
│   ├── dms_structural_annotations_multi.csv
│   └── dms_comparison_annotations_{description}.csv
```

