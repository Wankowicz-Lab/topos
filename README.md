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
- [Sequence alignment](#sequence-alignment)
- [Outputs](#outputs)
  - [Features CSV](#features-csv-prefix_featurescsv)
  - [Metadata CSV](#metadata-csv-prefix_metadatacsv)
  - [Run log](#run-log-prefix_run_logtxt)
- [Config reference](#config-reference)
- [Output column reference](#output-column-reference)
- [Developers](#developers)

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

### DSSP requirement for secondary structure

The pipeline uses `mkdssp` when available for secondary-structure annotation, and falls back to `pydssp` if `mkdssp` is not on `PATH`.

Install `mkdssp`:

- macOS (Homebrew):
```bash
brew install brewsci/bio/dssp
```
- Ubuntu/Debian:
```bash
sudo apt-get update
sudo apt-get install -y dssp
```
- Conda:
```bash
conda install -c conda-forge dssp
```

Verify installation:
```bash
which mkdssp
mkdssp --version
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
```

### With deep mutational scanning (DMS) data

```python
from src.pipeline.runner import Runner

runner = Runner(config_path='examples/B2AR_DMS_example/B2AR_config.toml')

runner.run()
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

### 2. Config file (optional)

A [TOML](https://toml.io) file that controls every aspect of the pipeline.
See the [Config reference](#config-reference) section below and the
example configs in `examples/`.

If no config file is provided, the pipeline will use default settings.

### 3. Mutation / DMS data (optional)

A CSV file with one row per mutation. Default required columns (names configurable within config file):

| Column | Default name | Description |
|--------|-------------|-------------|
| Residue position | `position` | Integer residue number |
| Wildtype amino acid | `wildtype` | 1-letter or 3-letter code |
| Mutant amino acid | `mutation` | 1-letter or 3-letter code |
| Mutation type | `type` | `"missense"`, `"synonymous"`, `"stop"`, `"deletion"`, or `"insertion"` |
| Effect score | `effect` | Numerical fitness / effect score |

#### Mutation input requirements

- `wildtype` must use a single code system across the file: either all 1-letter amino acid codes or all 3-letter amino acid codes.
- `mutation` may use standard 1-letter amino acid codes, standard 3-letter amino acid codes, `*`, or the shorthand indel tokens `DEL`, `DEL1`, `DEL2`, `DEL3`, `INS1`, `INS2`, and `INS3`.
- Mixed `mutation` formats are allowed. Standard 1-letter mutant amino acid codes are normalized to 3-letter codes during loading; the explicit indel shorthand tokens remain unchanged.
- `type` must use one of the canonical values `missense`, `synonymous`, `stop`, `deletion`, or `insertion`.

Loader validation errors reference this section as `README.md#mutation-input-requirements`.

The pipeline aligns the mutation data sequence to the PDB chain you specify via
`mutation_data_chain`.  Alignment warnings (mismatches, gaps) are reported in the
run log and as Python warnings — these are expected when the experimental construct
differs from the deposited structure.

---

## Sequence alignment

When `mutation_data_path` is set, the pipeline performs a pairwise alignment of the **mutation wildtype sequence** to the **PDB chain** given by `mutation_data_chain`. Each row of the resulting table is one **alignment column**: residues on the same row are paired; `NaN` on one side means a gap (insertion or deletion relative to the other sequence).

### Accessing the alignment table

After you construct `Runner` with mutation data, the merged alignment is stored on the context:

```python
runner = Runner(config_path="...")
alignment_df = runner.context.extras["sequence_alignment_merged"]
```

The table includes `align_pos`, `chain`, `resi_mut`, `resn_mut`, `resi_struct`, and `resn_struct` (see also the [metadata CSV](#metadata-csv-prefix_metadatacsv) and [Output column reference](#output-column-reference)).

### Worked example

**Row index** below is **0-based** (pandas `iloc`). **Residue numbers** in `resi_mut` / `resi_struct` are the numbering from each source. The table is a toy illustration, not a real protein.

| Row index | align_pos | resi_mut | resn_mut | resi_struct | resn_struct | Notes |
|-----------|-----------|----------|----------|-------------|-------------|--------|
| 0 | 0 | 1 | ALA | 10 | ALA | Match |
| 1 | 1 | 2 | ARG | 11 | LYS | **Mismatch** — same alignment row, different wildtype letters |
| 2 | 2 | 3 | GLY | — | — | **Internal indel** — paired gap; positions depend on mapping |
| 3 | 3 | — | — | 12 | ASN | **Internal indel** — residue only on structural side |

- **Mismatch:** At **row index 1**, `resn_mut` is ARG and `resn_struct` is LYS while both `resi_mut` and `resi_struct` are present. A mismatch warning lists **residue positions** (e.g. mutation sequence `2`, structural sequence `11`) as compact ranges, not per-row indices.
- **Internal indel:** Rows where either `resn_mut` or `resn_struct` is missing (NaN) and the row is **not** classified as a terminal gap, e.g. **row index 3** with structure only.
- **Terminal gap:** Rows at the **beginning** or **end** of the alignment where one sequence has no residue for the partner; these are called out in a separate warning and **excluded** from the alignment-quality error-rate.
- **Alignment quality below cutoff:** Compares the count of mismatch + internal indel rows (excluding terminal gaps) to `alignment_cutoff`; see the warning text and cross-check against filtered rows in `alignment_df`.

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



## Output column reference

### Identity columns (always present)

| Column | Description |
|--------|-------------|
| `chain` | Chain ID |
| `resi_struct` | Residue number from the PDB structure |
| `resn_struct` | Residue name (3-letter) from the structure |
| `resi_mut` | Residue number from mutation data (same as `resi_struct` in structure-only mode) |
| `resn_mut` | Residue name from mutation data |
| `resm` | Mutant residue token after loading, typically 3-letter for substitutions and unchanged for explicit indel shorthand tokens |
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
| `{accession}_{category}_wt` | AA index property value for wildtype residue |
| `{accession}_{category}_mut` | AA index property value for mutant residue |
| `{accession}_{category}_diff` | Difference (mut − wt) for this AA index |
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

## Developers

Use the environment setup in [Installation](#installation). For development tooling, install test extras once:

```bash
pip install -e ".[test]"
```

The code below allows you to run formatting checks locally, this will flag errors prior automatic CI/CD

Run Ruff (configured in `pyproject.toml`):

```bash
ruff check src tests
```

Apply safe Ruff autofixes:

```bash
ruff check src tests --fix
```

Run mypy (current high-value scope):

```bash
mypy
```

