This folder contains scripts for analyzing groups of PDBs. We expect there to be a PDB alongside the corresponding metrics CSV for each PDB. The analysis is built for PDBs with highly similar sequences. Up to three missense mutations are allowed. 

To group comparison PDBs (Unbound v. Bound or WT v. Mutant), fill out the config file with comparisons. Results will include global distributions or differences in counts across all metrics and areas where differences are greatest. 


## Scripts

| Script | Description |
|---|---|
| `run_analysis.py` | **Main analysis script.** Runs global stats, histograms, and local difference tables for all pairs defined in a config. |
| `pairwise_rmsd.py` | Pairwise Cα RMSD for all structures in the config. |
| `figures/map_metric_to_b_factor.py` | Map any metric column to B-factors in a PDB for PyMOL visualization. |


## Quick Start

1. Copy `example_config.toml` and edit it for your structures.
2. Run the analysis:

```bash
python run_analysis.py --config my_config.toml
```

## Config File (See example: example_config.toml)

**Settings**
metrics_dir        = "/path/to/pipeline/output"  # folder with per-PDB sub-dirs
output_dir         = "results/"
output_prefix      = "myproject_"
proximity_angstroms = 8.0   # Å radius around mutation/ligand for local analysis

**Structures**
label    = "WT_apo"
pdb_id   = "1KE4"
state    = "apo"      # "apo" | "bound"
genotype = "wt"       # "wt"  | "mutant"
chain    = "A"

**Sructures**
label    = "S64D_mutant"
pdb_id   = "1L0D"
state    = "apo"
genotype = "mutant"
chain    = "A"

**structures.mutations**
resi   = 64   # residue number in structure
wt_aa  = "S"  # single-letter WT amino acid
mut_aa = "D"  # single-letter mutant amino acid

**Structures**
label   = "WT_bound_BZB"
pdb_id  = "1C3B"
state   = "bound"
genotype = "wt"
chain   = "A"
ligand  = {name = "BZB", chain = "A"}   # HET residue name in PDB

**Pairs**
reference   = "WT_apo"
comparison  = "S64D_mutant"
description = "WT apo vs S64D mutant"

If you omit pairs, every structure is automatically paired against the first `wt`/`apo` entry.

