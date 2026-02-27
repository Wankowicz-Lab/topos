# biogenesis

A toolkit for computing structural and sequence metrics on protein structures.

Given a PDB file and/or mutation data, biogenesis produces per-residue feature tables
useful for downstream analysis of mutational effects, structural variation, and protein
function.

---

## Table of Contents

- [Installation](#installation)
- [Quick Start](#quick-start)
- [Inputs](#inputs)
  - [Structure file](#1-structure-file-required)
  - [Config file](#2-config-file-required)
  - [Mutation / DMS data](#3-mutation--dms-data-optional)
- [Outputs](#outputs)
  - [Features CSV](#features-csv-prefix_featurescsv)
  - [Metadata CSV](#metadata-csv-prefix_metadatacsv)
  - [Run log](#run-log-prefix_run_logtxt)
- [Config reference](#config-reference)
- [Output column reference](#output-column-reference)
- [Examples](#examples)
- [Logging](#logging)
- [Development and Testing](#development-and-testing)

---

## Installation

Requires **Python ≥ 3.11**.

```bash
git clone https://github.com/Wankowicz-Lab/biogenesis.git
cd biogenesis
pip install -e .
```

The required conda environment (Python 3.11 + all dependencies) is used for
development and testing:

```bash
conda create -n biogenesis-py311 python=3.11
conda activate biogenesis-py311
pip install -e .
```

---

## Quick Start

### Structure only (no mutation data)

```python
from src.pipeline.runner import Runner

runner = Runner(
    pdb_id='1HCK',           # PDB ID (downloaded from RCSB) — OR use pdb_path for a local file
    config_path='examples/1HCK_structure_only_example/1HCK_config.toml',
)

runner.run()                  # compute all structural metrics
runner.save_results('output') # writes 3 files: _features.csv, _metadata.csv, _run_log.txt
```

### With deep mutational scanning (DMS) data

```python
from src.pipeline.runner import Runner

runner = Runner(config_path='examples/B2AR_DMS_example/B2AR_config.toml')

runner.run()
runner.save_results('output')
```

Run the ready-made example scripts from the repository root:

```bash
conda activate biogenesis-py311

# Structure-only (1HCK kinase, local PDB file)
python examples/1HCK_structure_only_example/run_example.py

# DMS data (B2AR membrane receptor, downloads 4LDE from RCSB)
python examples/B2AR_DMS_example/run_example.py
```

---

## Inputs

### 1. Structure file (required)

Provide either a **PDB ID** (structure downloaded from RCSB) or a **local PDB / mmCIF file**:

| Method | Config key | Example |
|--------|-----------|---------|
| Download from RCSB | `pdb_id` | `pdb_id = "4LDE"` |
| Local PDB file | `pdb_path` | `pdb_path = "/abs/path/protein.pdb"` |
| Local mmCIF file | `pdb_path` | `pdb_path = "/abs/path/protein.cif"` |

If both are specified, `pdb_path` takes precedence.

**Hydrogens**: The pipeline logs whether hydrogens were present in the loaded
structure and whether they were removed (controlled by `remove_hydrogens`).
This is captured in the run log.

**Alternate locations**: Structures with multiple conformers are handled via
`altloc_policy`. With `"highest"` (default), only the highest-occupancy conformer
is kept; with `"all"`, all conformers are retained (one row per conformer).

### 2. Config file (required)

A [TOML](https://toml.io) file that controls every aspect of the pipeline.
See the [Config reference](#config-reference) section below and the
example configs in `examples/`.

### 3. Mutation / DMS data (optional)

A CSV file with one row per mutation. Required columns (names configurable):

| Column | Default name | Description |
|--------|-------------|-------------|
| Residue position | `position` | Integer residue number |
| Wildtype amino acid | `wildtype` | 1-letter or 3-letter code |
| Mutant amino acid | `mutation` | 1-letter or 3-letter code |
| Mutation type | `type` | `"missense"`, `"synonymous"`, `"stop_codon"` |
| Effect score | `effect` | Numerical fitness / effect score |

The pipeline aligns the mutation data sequence to the PDB chain you specify via
`mutation_data_chain`.  Alignment warnings (mismatches, gaps) are reported in the
run log and as Python warnings — these are expected when the experimental construct
differs from the deposited structure.

---

## Outputs

Three files are written to `output_dir` for each run, prefixed with the PDB ID
(or the `name` you provide):

### Features CSV (`{prefix}_features.csv`)

One row per residue (structure-only mode) or per mutation (DMS mode).
Contains all computed metrics. See [Output column reference](#output-column-reference).

### Metadata CSV (`{prefix}_metadata.csv`)

Residue-level structural annotation table:

| Column | Description |
|--------|-------------|
| `chain` | Chain ID |
| `resi_struct` | Residue number in the PDB structure |
| `resn_struct` | Residue name (3-letter code) from structure |
| `resi_mut` | Residue number from mutation data (NaN if no alignment) |
| `resn_mut` | Residue name from mutation data |
| `struct_info` | `True` if this residue has structural data |
| `mut_info` | `True` if this residue is covered by mutation data |
| `ss_domains` | Secondary structure domain label (e.g. `helix_1`, `coil_3`) |
| `ss_group` | Secondary structure class (`helix`, `sheet`, `coil`) |
| `resm` | Mutant residue (only present when DMS data provided) |

### Run log (`{prefix}_run_log.txt`)

A human-readable summary of the run, written automatically by `save_results()`.
It records:

- **Run date and time**
- **Configuration file** path
- **Structure information**: PDB ID or file path, source (RCSB vs local), chains present, chains used for structural features, number of residues
- **Hydrogen handling**: were hydrogens present in the loaded file? Was `remove_hydrogens = true`? What action was taken?
- **Alternate locations**: were altlocs present? Which `altloc_policy` was applied?
- **Membrane protein settings**: membrane_protein flag, membrane thickness, PDBTM annotation
- **Mutation/DMS data**: file path, chain, alignment cutoff, number of mutations loaded, number of positions covered, whether sequence metrics were enabled
- **Metrics computed**: full list of metrics that ran
- **Output file paths** and row/column counts

Example run log snippet:

```
biogenesis Run Log
==================
Run Date: 2026-02-25 12:00:00
Configuration File: examples/B2AR_DMS_example/B2AR_config.toml

Structure Information
---------------------
  PDB ID:  4LDE
  Source: RCSB (downloaded)
  Chains in structure: A, B
  Chains used for structural features: all (A, B)
  Unique residue positions: 574

Hydrogen Handling
-----------------
  Hydrogens present in loaded structure: No
  remove_hydrogens setting: True
  Action: No hydrogens present; no removal needed

Membrane Protein Settings
-------------------------
  membrane_protein: True
  membrane_thickness: 15.0 Å (half-thickness)
  PDBTM annotation: fetched to orient structure in membrane reference frame
```

---

## Config reference

### Structure parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `pdb_id` | `str` | — | PDB identifier; structure downloaded from RCSB |
| `pdb_path` | `str` | — | Path to local PDB or mmCIF file. Takes precedence over `pdb_id` |
| `membrane_protein` | `bool` | `false` | Set `true` for membrane proteins. Fetches PDBTM annotation to orient the structure in the membrane reference frame and enables membrane-specific metrics (`distance_from_membrane_edge`, membrane-aware secondary structure) |
| `membrane_thickness` | `float` | `15` | Half-thickness of the membrane in Ångströms, used to compute distances from the membrane centre |
| `remove_hydrogens` | `bool` | `true` | Remove hydrogen atoms after loading. The run log records whether hydrogens were present in the file |
| `altloc_policy` | `"highest"` / `"all"` | `"highest"` | How to handle alternate conformers. `"highest"` keeps the highest-occupancy conformer; `"all"` retains all conformers |
| `structural_feature_chains` | `list[str]` | `[]` (all) | Restrict structural metric calculation to specific chains. If empty or omitted, all chains are used |

### Mutagenesis data parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `mutation_data_path` | `str` | — | Path to CSV with mutation scores |
| `mutation_data_chain` | `str` | — | Chain ID to align mutation data against (required if `mutation_data_path` is set) |
| `alignment_cutoff` | `float` | `0.95` | Minimum sequence identity between structure and mutation data before a warning is raised |
| `mutation_residue_col_name` | `str` | `"wildtype"` | Column name for wildtype residues |
| `mutation_residue_idx_name` | `str` | `"position"` | Column name for residue positions |
| `mutation_col_name` | `str` | `"mutation"` | Column name for mutant residues |
| `mutation_type_col_name` | `str` | `"type"` | Column name for mutation type |
| `mutation_score_col_name` | `str` | `"effect"` | Column name for mutation effect scores |

### Sequence feature parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `aaindex_path` | `str` | `"data/aaindex_parsed_small.csv"` | Path to amino acid index database |
| `kidera_path` | `str` | `"data/kidera_factors.csv"` | Path to Kidera factors data |

### Pipeline parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `output_dir` | `str` | — | Directory for output files. Created if it does not exist |
| `output_prefix` | `str` | `""` | Optional prefix prepended to output file names |

### Minimal config templates

**Structure only:**
```toml
pdb_id = "1HCK"
membrane_protein = false
remove_hydrogens = true
altloc_policy = "highest"
aaindex_path = "data/aaindex_parsed_small.csv"
kidera_path  = "data/kidera_factors.csv"
output_dir   = "output"
```

**With DMS data:**
```toml
pdb_id = "4LDE"
membrane_protein = true
membrane_thickness = 15

mutation_data_path  = "path/to/mutations.csv"
mutation_data_chain = "A"
alignment_cutoff    = 0.95

aaindex_path = "data/aaindex_parsed_small.csv"
kidera_path  = "data/kidera_factors.csv"
output_dir   = "output"
```

---

## Output column reference

### Identity columns (always present)

| Column | Description |
|--------|-------------|
| `chain` | Chain ID |
| `resi_struct` | Residue number from the PDB structure |
| `resn_struct` | Residue name (3-letter) from the structure |
| `resi_mut` | Residue number from mutation data (same as `resi_struct` in structure-only mode) |
| `resn_mut` | Residue name from mutation data |
| `resm` | Mutant residue 1-letter code (DMS mode only) |
| `name` | Run name (derived from PDB ID or the `name` parameter) |
| `ss_domains` | Secondary structure domain label (e.g. `helix_1`, `sheet_2`, `coil_3`) |

### Structural metrics

| Column | Description |
|--------|-------------|
| `sasa` | Total solvent accessible surface area (Å²) |
| `sasa_backbone` | Backbone SASA (Å²) |
| `sasa_sidechain` | Sidechain SASA (Å²) |
| `sasa_polar` | Polar atom SASA (Å²) |
| `sasa_nonpolar` | Non-polar atom SASA (Å²) |
| `distance_to_nearest_surface_residue` | Distance to the nearest surface-exposed residue (Å) |
| `kyte_doolittle` | Kyte–Doolittle hydropathy score |
| `distance_from_membrane_edge` | Distance from the membrane boundary (Å; membrane proteins only) |
| `packing_n_atoms` | Number of heavy atoms within 5 Å |
| `packing_n_neighbor_residues` | Number of residues within 5 Å |
| `packing_contact_density` | Ratio of neighbors to atoms (contact density) |
| `distance_to_center_of_mass` | Distance from residue Cα to protein centre of mass (Å) |

### Bond / interaction metrics

| Column | Description |
|--------|-------------|
| `bb_hbond_count` | Number of backbone hydrogen bonds |
| `sc_hbond_count` | Number of sidechain hydrogen bonds |
| `total_hbond_count` | Total hydrogen bonds |
| `salt_bridge_count` | Salt bridge interactions |
| `ionic_bond_count` | Ionic bond interactions |
| `disulfide_bond_count` | Disulfide bridges |
| `pi_stacking_count` | Aromatic π–π stacking interactions |
| `cation_pi_count` | Cation–π interactions |
| `vdw_contact_count` | Van der Waals contacts |

### Sequence / DMS metrics (present only when mutation data is provided)

| Column | Description |
|--------|-------------|
| `effect` | Raw DMS effect score for this mutation |
| `pos_effect` | Mean effect score across all mutations at this position |
| `effect_quartile` | Quartile of `pos_effect` (1 = lowest, 4 = highest) |
| `effect_variance` | Variance of effect scores at this position |
| `effect_variance_rank` | Rank of effect variance among all positions |
| `effect_ranking` | Rank of this specific mutation's effect score |
| `blosum90` | BLOSUM90 log-odds score for this substitution |
| `phat_score` | PHAT substitution matrix score |
| `wildtype_aa_group` | Amino acid physicochemical group of the wildtype residue |
| `mut_aa_group` | Amino acid physicochemical group of the mutant residue |
| `wildtype_mut_aa_group` | Combined wildtype→mutant group label |
| `AAIndex_{accession}_wt` | AA index property value for wildtype residue |
| `AAIndex_{accession}_mut` | AA index property value for mutant residue |
| `AAIndex_{accession}_diff` | Difference (mut − wt) for this AA index |
| `kidera_f{1-10}_wt` | Kidera factor for wildtype residue |
| `kidera_f{1-10}_mut` | Kidera factor for mutant residue |
| `kidera_f{1-10}_diff` | Difference (mut − wt) for this Kidera factor |

### Secondary structure domain metrics (averaged per domain)

Each of the structural and DMS metrics listed above has a corresponding
`ss_domain_{metric}` column containing the mean value for that secondary
structure domain.  Additional domain-level columns:

| Column | Description |
|--------|-------------|
| `ss_domain_length` | Number of residues in this secondary structure domain |
| `ss_domain_log2_aa_group_ratio_{group}` | log2 ratio of amino acid group frequency in this domain vs the whole protein |

Amino acid groups: `Nonpolar_Aliphatic`, `Aromatic`, `Polar_Uncharged`, `Positively_Charged`, `Negatively_Charged`, `Special`.

### Neighborhood metrics

| Column | Description |
|--------|-------------|
| `n_ala_neighbors` | Number of alanine residues within the 5 Å neighbor shell |

### Ligand interaction metrics

One column per detected ligand:

| Column | Values | Description |
|--------|--------|-------------|
| `ligand_{chain}_{res_id}_{resn}_interactions` | `"contact"`, `"binding site"`, `"second shell"`, NaN | Interaction category with respect to each ligand |

- **contact**: residue is in direct atomic contact with the ligand (within 4.5 Å) and is listed in the bonds table
- **binding site**: residue is within 4.5 Å of any ligand atom
- **second shell**: residue is within 5 Å of any binding-site residue

### Graph / network metrics

Computed on three bond-type graphs: `all` (all bonds), `vdw_contact`, `hbond`.

| Column pattern | Description |
|----------------|-------------|
| `graph_{type}_graph_betweenness_centrality` | Betweenness centrality |
| `graph_{type}_graph_closeness_centrality` | Closeness centrality |
| `graph_{type}_graph_eigenvector_centrality` | Eigenvector centrality |
| `graph_{type}_graph_core_number` | k-core number |
| `graph_{type}_graph_community_id` | Community membership ID |
| `graph_{type}_graph_in_lcc` | `True` if residue is in the largest connected component |

---

## Examples

Two ready-to-run examples are provided:

### Example 1 — Structure only (`examples/1HCK_structure_only_example/`)

1HCK is a CDK2/cyclin kinase complex. This example loads a local PDB file, computes
all structural metrics, and writes the output to the example directory.

```bash
python examples/1HCK_structure_only_example/run_example.py
```

Outputs (in `examples/1HCK_structure_only_example/output/`):
- `1HCK_features.csv`  — 294 residues × 68 columns
- `1HCK_metadata.csv`
- `1HCK_run_log.txt`

Config: `examples/1HCK_structure_only_example/1HCK_config.toml`

### Example 2 — DMS data + membrane protein (`examples/B2AR_DMS_example/`)

4LDE is the beta-2 adrenergic receptor (B2AR), a GPCR membrane protein.
This example downloads the structure from RCSB, fetches PDBTM orientation,
and combines it with deep mutational scanning data.

```bash
python examples/B2AR_DMS_example/run_example.py
```

Outputs (in `examples/B2AR_DMS_example/output/`):
- `4LDE_features.csv`  — 9,328 mutation rows × 124 columns
- `4LDE_metadata.csv`
- `4LDE_run_log.txt`

Config: `examples/B2AR_DMS_example/B2AR_config.toml`

---

## Logging

The pipeline uses Python's standard `logging` module.  By default, only WARNING-level
messages are shown.  To see informational messages about each step:

```python
import logging
import sys

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s - %(message)s",
    stream=sys.stdout,
)

from src.pipeline.runner import Runner
# ... rest of your code
```

### Alignment warnings

When mutation data is provided, the pipeline aligns the DMS sequence to the PDB chain
and emits Python warnings if:

- **Sequence identity below `alignment_cutoff`** (default 0.95): the sequences differ
  significantly; check that `mutation_data_chain` is correct
- **Mismatches**: individual residue differences between DMS and PDB sequences (common
  at crystal construct boundaries or engineered mutations)
- **Indels**: insertions or deletions between the DMS and PDB sequences
- **Terminal gaps**: residues at the ends of the sequence that are present in one
  source but not the other (common for expression tags, truncations, or ICL3 loops)

These warnings are expected for real-world datasets where the DMS experiment uses a
slightly different construct than the PDB entry.  They are captured in the run log.

