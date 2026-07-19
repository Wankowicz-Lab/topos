# Topos Examples

This directory contains ready-to-run examples demonstrating the two main biogenesis
use cases.  Run all scripts from the **repository root** directory.

```bash
conda activate biogenesis-py311
```

---

## Example 1 — DMS data + membrane protein: B2AR / 4LDE

**Directory**: `examples/B2AR_DMS_example/`

**Protein**: Beta-2 adrenergic receptor, a GPCR (PDB: 4LDE, downloaded from RCSB)

**Inputs**:
- Structure: downloaded from RCSB via `pdb_id = "4LDE"`
- DMS scores: `B2AR_processed_scores.csv` (9,064 mutations across 412 positions)

**What it shows**:
- Downloading a structure by PDB ID
- Membrane protein handling: PDBTM annotation, membrane orientation, `distance_from_membrane_edge`
- Aligning DMS data to a specific chain (`mutation_data_chain = "A"`)
- Full metric set including sequence-level metrics (blosum, aaindex, kidera, effect scores)
- Expected alignment warnings when the DMS construct differs from the deposited structure

```bash
python examples/B2AR_DMS_example/run_example.py
```

**Outputs** (written to `examples/B2AR_DMS_example/output/`):

| File | Description |
|------|-------------|
| `4LDE_features.csv` | 9,328 mutation rows × 124 columns (structural + sequence metrics) |
| `4LDE_metadata.csv` | Residue-level structural annotation with mutation coverage |
| `4LDE_run_log.txt` | Human-readable summary including alignment statistics |

**Config**: `B2AR_config.toml`

```toml
pdb_id = "4LDE"
membrane_protein = true
membrane_thickness = 15

mutation_data_path  = "examples/B2AR_DMS_example/B2AR_processed_scores.csv"
mutation_data_chain = "A"
alignment_cutoff    = 0.95

aaindex_path = "data/aaindex_parsed_small.csv"
kidera_path  = "data/kidera_factors.csv"
output_dir   = "examples/B2AR_DMS_example/output"
```

---

## Example 2 — Structure only: 1HCK

**Directory**: `examples/1HCK_structure_only_example/`

**Protein**: CDK2/cyclin kinase complex (PDB: 1HCK)

**Input**: Local PDB file (`examples/1HCK.pdb`)

**What it shows**:
- Loading a local PDB file using `pdb_path`
- Running all structural metrics (SASA, packing, hydrogen bonds, etc.)
- No mutation data — sequence-level metrics are skipped automatically
- The run log records that no hydrogens were present in the file

```bash
python examples/1HCK_structure_only_example/run_example.py
```

**Outputs** (written to `examples/1HCK_structure_only_example/output/`):

| File | Description |
|------|-------------|
| `1HCK_features.csv` | 294 residues × 68 structural metrics |
| `1HCK_metadata.csv` | Residue-level structural annotation |
| `1HCK_run_log.txt` | Human-readable summary of the run |

**Config**: `1HCK_config.toml`

```toml
# Structure data (pdb_path and name passed in run_example.py)
membrane_protein = false
remove_hydrogens = true
altloc_policy = "highest"
aaindex_path = "data/aaindex_parsed_small.csv"
kidera_path  = "data/kidera_factors.csv"
output_dir   = ""
output_prefix = ""
```
---

---

## Example 3 — Grouped Analysis

**Directory**: `examples/grouped_analysis/`

**Protein**: E. coli adenylate kinase (PDBs: 1AKE, 1ANK, 3HPQ, 4AKE, 4JLB, 8RJ8)

**Input**: Topos CSVs and local PDBs

For each PDB, we expect to see the following (only showing 1AKE): 
| File | Description |
|------|-------------|
| `1AKE_features.csv` | 294 residues × 68 structural metrics |
| `1AKE_metadata.csv` | Residue-level structural annotation |
| `1AKE_run_log.txt` | Human-readable summary of the run |
| `1AKE.cif` | mmCIF/PDBx file |


**What it shows**:
- Loading Topos metrics files
- Renumber Topos metric files based on sequence alignment so all outputs have the residue/chain for comparison.
- Running figures and descriptive statistics on distribution of structures
- Outputs figures and CSVs to compare and/or use multiple structures to analyze outputs

```bash
python examples/grouped_analysis/run_grouped_example.py
```

**Outputs** (written to `examples/grouped_analysis/output/`):

| Folder | Description |
|------|-------------|
| `renumbered` | features, bonds, metadata renumbered |
| `rmsd` | CSV of alpha carbon RMSD between PDBs |
| `residue profiles` | Joint CSV with mean/median of metrics across all PDBs and corresponding figures |
| `variability` | CSV and corresponding figures with largest variability of metrics between PDBs |
| `comparisons` | CSV and corresponding figures on differences between PDBs if toml file indicates comparisons |



**Config**: `ADK_grouped_config.toml`

```toml
# Required fields:
#   label   — unique name used in [[pairs]] and output file names
#   pdb_id  — RCSB PDB ID (downloaded automatically) OR use pdb_path
#
# Optional fields:
#   pdb_path — path to a local .pdb or .cif file (overrides pdb_id download)
#   group    — group label for multi-structure grouping (e.g. "open", "closed")
#   state    — "apo" or "bound" (default: "apo")
#   genotype — "wt" or "mutant" (default: "wt")
#   chain    — chain ID(s) to analyze (default: inherits global chain above)
#   ligand   — HET residue for proximity detection: {name = "AP5", chain = "A"}
#   mutations — list of point mutations (used for proximity detection):
#               [[structures.mutations]]
#               resi   = 64     # residue number in reference numbering
#               wt_aa  = "S"    # single-letter WT amino acid
#               mut_aa = "D"    # single-letter mutant amino acid
```
---


## Troubleshooting

### `ModuleNotFoundError: No module named 'src'`

Run the scripts from the **repository root** directory, or use the
`sys.path` manipulation already in the example scripts (they add the repo
root automatically).

### `ValueError: Alignment quality below cutoff`

The DMS sequence identity with the PDB chain is below `alignment_cutoff` (default 0.95).
Check that:
1. `mutation_data_chain` refers to the correct chain
2. The DMS data covers the same protein as the structure
3. Lower the cutoff if large loop regions are missing from the crystal structure

### `RuntimeError: Failed to fetch PDBTM annotation`

The PDBTM API was unreachable.  Either retry, or set `membrane_protein = false` and
rerun with standard (non-membrane) secondary structure assignment.

### Alignment warnings in DMS mode

Warnings are produced when there is a mismatch between sequence in the mutation data and protein sequence in the structural data. See the [Sequence alignment](../README.md#sequence-alignment) section in the main README for more information.
