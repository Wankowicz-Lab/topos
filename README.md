# topos

A toolkit for computing structural and sequence metrics on protein structures.

Given a PDB file and/or mutation data, topos produces per-residue feature tables
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

Install from GitHub:

```bash
pip install "git+https://github.com/Wankowicz-Lab/topos.git"
```

Or clone for development:

```bash
git clone https://github.com/Wankowicz-Lab/topos.git
cd topos
pip install -e ".[test]"
```

The required conda environment (Python 3.11 + all dependencies) is used for
development and testing:

```bash
conda create -n topos-py311 python=3.11
conda activate topos-py311
pip install -e ".[test]"
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

### Optional: TMbed for membrane-parameter estimation

When `membrane_protein = true` but PDBTM has no usable entry (or the structure has no PDB ID, e.g. AlphaFold), topos can estimate TM helix spans with [TMbed](https://github.com/BernhoferM/TMbed) and a membrane frame from helix geometry. This path is controlled by `estimate_membrane_protein_parameters` (default `true`).

TMbed is **optional** and is **not** installed with core topos (it pulls in PyTorch and large ProtT5 weights). Install it only if you need the estimate fallback:

```bash
# CPU PyTorch example; see pytorch.org for GPU builds
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install "git+https://github.com/BernhoferM/TMbed.git"
# First-time ProtT5 / CNN model download (~GB):
tmbed download
```

Verify the CLI is on `PATH`:

```bash
which tmbed
tmbed --help
```

If TMbed is missing when estimation is requested (`estimate_membrane_protein_parameters = true`), the pipeline **errors** and tells you to either install TMbed or set `estimate_membrane_protein_parameters = false` to continue without membrane features.

When PDBTM **is** available, geometric `side1`/`side2` regions are annotated as usual. Absolute Inside/Outside labels (`membrane_side`) come from [TOPDB](https://topdb.unitmp.org) SideDefinition. Provide `uniprot_id` in config when possible; otherwise topos resolves it via PDBe SIFTS from `pdb_id`. If that lookup fails, a warning explains why and how to set `uniprot_id` explicitly—geometric PDBTM annotation still proceeds.

---

## Quick Start

### Structure only (no mutation data)

```python
from topos.pipeline.runner import Runner

runner = Runner(
    pdb_id='1HCK',           # PDB ID (downloaded from RCSB) — OR use pdb_path for a local file
    config_path='examples/1HCK_structure_only_example/1HCK_config.toml',
)

runner.run()                  # compute all structural metrics
```

### With deep mutational scanning (DMS) data

```python
from topos.pipeline.runner import Runner

runner = Runner(config_path='examples/B2AR_DMS_example/B2AR_config.toml')

runner.run()
```

Run the ready-made example scripts from the repository root:

```bash
conda activate topos-py311

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

The pipeline aligns the mutation wildtype sequence to the **PDB polymer construct**
for `mutation_data_chain` when mmCIF polymer scheme data are available (including
structures fetched by `pdb_id`). Alignment warnings (mismatches, gaps, unmodeled
residues) are reported as Python warnings and summarized in the run log — these are
expected when the experimental DMS construct differs from the deposited structure.

---

## Sequence alignment

When `mutation_data_path` is set, the pipeline performs a pairwise alignment of the **mutation wildtype sequence** to the **construct sequence** for `mutation_data_chain`.

- **Deposited mmCIF** (local `.cif` / `.mmcif`, or `pdb_id` RCSB fetch): the construct comes from `_pdbx_poly_seq_scheme`. Residues in the polymer without coordinates are retained and labeled `unmodeled`.
- **Classic PDB / AlphaFold PDB / CIF without a usable scheme:** the construct falls back to coordinate residues (`construct_source="coordinates"`). A warning is issued; `unmodeled` cannot appear because construct≡modeled chain.

Each row of the alignment table is one **alignment column**: residues on the same row are paired; `NaN` on one side means a gap. `coverage_status` classifies mutation-side positions as `modeled`, `unmodeled`, `missing_from_construct`, or `construct_mismatch`. **Construct coverage** is the fraction of DMS wildtype positions that match the construct AA (`modeled` + `unmodeled`). **Coordinate coverage** is the fraction of DMS wildtype positions that align to a construct residue with coordinates (includes `construct_mismatch` when that residue is modeled). `struct_info` remains “has coordinates” (the `modeled` flag), so structure metrics stay gated correctly for unresolved loops.

`runner.context.extras` also stores `construct_residue_table`, `construct_source`, `construct_coverage`, and `coordinate_coverage`.

### Accessing the alignment table

After you construct `Runner` with mutation data, the merged alignment is stored on the context:

```python
runner = Runner(config_path="...")
alignment_df = runner.context.extras["sequence_alignment_merged"]
```

The table includes `align_pos`, `chain`, `resi_mut`, `resn_mut`, `resi_struct`, `resn_struct`, `modeled`, and `coverage_status` (see also the [metadata CSV](#metadata-csv-prefix_metadatacsv) and [Output column reference](#output-column-reference)).

### Worked example

**Row index** below is **0-based** (pandas `iloc`). **Residue numbers** in `resi_mut` / `resi_struct` are the numbering from each source. The table is a toy illustration, not a real protein.

| Row index | align_pos | resi_mut | resn_mut | resi_struct | resn_struct | coverage_status | Notes |
|-----------|-----------|----------|----------|-------------|-------------|-----------------|--------|
| 0 | 0 | 1 | ALA | 10 | ALA | modeled | Match with coordinates |
| 1 | 1 | 2 | ARG | 11 | LYS | construct_mismatch | Same alignment row, different wildtype letters |
| 2 | 2 | 3 | GLY | 12 | GLY | unmodeled | In construct polymer, no coordinates |
| 3 | 3 | 4 | SER | — | — | missing_from_construct | DMS residue absent from construct |
| 4 | 4 | — | — | 13 | ASN | — | Construct-only column (no DMS) |

- **Construct mismatch:** Both sides present with different amino acids (engineered mutation, ortholog, isoform, etc.).
- **Unmodeled:** AA matches the construct but lacks coordinates; excluded from the alignment-quality error rate (like terminal gaps).
- **Missing from construct:** Gap on the construct side for a DMS position (truncation/deletion relative to the deposited polymer, or vs coordinates when `construct_source="coordinates"`).
- **Terminal gap:** Contiguous gaps at the beginning or end of the alignment; excluded from the alignment-quality error rate.
- **Alignment quality below cutoff:** Compares mismatch + internal indel rows (excluding terminal gaps and unmodeled matches) to `alignment_cutoff`.

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
| `struct_info` | `True` if this residue has coordinates (`modeled`) |
| `mut_info` | `True` if this residue is covered by mutation data |
| `modeled` | `True` if coordinates are present for this construct residue |
| `coverage_status` | DMS coverage class: `modeled`, `unmodeled`, `missing_from_construct`, `construct_mismatch` (NaN if no mutation residue on the row) |
| `ss_domains` | Secondary structure domain label (e.g. `helix_1`, `coil_3`); `unmodeled` for construct residues without coordinates (excluded from SS-domain aggregations) |
| `ss_category` | Domain category with unique index stripped (e.g. `TMD`, `side1`, `alpha-helix`); `unmodeled` when coordinates are absent |
| `ss_group` | Secondary structure class (`helix`, `sheet`, `coil`); `unmodeled` when coordinates are absent |
| `resm` | Mutant residue (only present when DMS data provided) |

### Run snapshot JSON (`{prefix}_run_log.json`)
It records:

- **Run date and time**
- **Configuration file** path
- **Structure information**: PDB ID or file path, source (RCSB vs local), chains present, chains used for structural features, number of residues
- **Hydrogen handling**: were hydrogens present in the loaded file? Was `remove_hydrogens = true`? What action was taken?
- **Alternate locations**: were altlocs present? Which `altloc_policy` was applied?
- **Membrane protein settings**: membrane_protein flag, membrane thickness, PDBTM / estimate fallback
- **Mutation/DMS data**: file path, chain, alignment cutoff, number of mutations loaded, number of positions covered, whether sequence metrics were enabled
- **Metrics computed**: full list of metrics that ran
- **Output file paths** and row/column counts

## Config reference

### Structure parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `pdb_id` | `str` | — | PDB identifier; structure downloaded from RCSB |
| `pdb_path` | `str` | — | Path to local PDB or mmCIF file. Takes precedence over `pdb_id` |
| `uniprot_id` | `str` | — | UniProt accession. Used to download AlphaFold models when no PDB is given, and to look up TOPDB SideDefinition for absolute Inside/Outside labels when `membrane_protein = true`. If omitted with only a `pdb_id`, topos tries PDBe SIFTS; on failure it warns and skips absolute side labels |
| `membrane_protein` | `bool` | `false` | Set `true` for membrane proteins. Fetches PDBTM annotation to orient the structure in the membrane reference frame and enables membrane-specific metrics (`distance_from_membrane_edge`, membrane-aware secondary structure) |
| `membrane_thickness` | `float` | `15` | Half-thickness of the membrane in Ångströms, used to compute distances from the membrane centre |
| `estimate_membrane_protein_parameters` | `bool` | `true` | When PDBTM is unavailable (missing entry or no PDB ID), estimate TM helices with TMbed and orient the membrane frame from helix axes. Requires TMbed when `true`; set `false` to skip membrane features instead. See [Optional: TMbed](#optional-tmbed-for-membrane-parameter-estimation). |
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
| `aaindex_path` | `str` | packaged CSV | Path to amino acid index database (defaults to data bundled with the install) |
| `kidera_path` | `str` | packaged CSV | Path to Kidera factors data (defaults to data bundled with the install) |

### Pipeline parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `output_dir` | `str` | — | Directory for output files (CSV, run snapshot JSON). Created if it does not exist |
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
| `ss_domains` | Secondary structure domain label (e.g. `helix_1`, `sheet_2`, `coil_3`); `unmodeled` when coordinates are absent (excluded from SS-domain aggregations) |
| `ss_category` | Domain category with unique index stripped (e.g. `TMD`, `side1`, `alpha-helix`); `unmodeled` when coordinates are absent |
| `ss_group` | Secondary structure class; `unmodeled` when coordinates are absent |

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
| `avg_effect` | Mean `effect` across non-synonymous mutations at this position  |
| `avg_effect_quartile` | Quartile label (`Q1`–`Q4`) from the distribution of `avg_effect` across positions |
| `effect_variance` | Variance of effect scores at this position |
| `effect_variance_rank` | Rank of effect variance among all positions |
| `effect_ranking` | Rank of this specific mutation's effect score |
| `mutation_category` | `LOF`, `neutral`, or `GOF` based on synonymous or stop mutations |
| `total_lof` | Count of mutations at this position classified as `LOF` |
| `total_gof` | Count of mutations at this position classified as `GOF` |
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

