# biogenesis Examples

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

## DMS data format

The mutation CSV must contain these columns (column names are configurable in the
TOML config):

| Column | Default name | Required | Description |
|--------|-------------|----------|-------------|
| Residue position | `position` | Yes | Integer residue number matching the DMS construct numbering |
| Wildtype amino acid | `wildtype` | Yes | 1-letter amino acid code |
| Mutant amino acid | `mutation` | Yes | 1-letter amino acid code |
| Mutation type | `type` | Yes | `"missense"`, `"synonymous"`, `"stop_codon"` |
| Effect score | `effect` | Yes | Numerical fitness / effect score (lower = more deleterious is conventional) |

Example rows from `B2AR_processed_scores.csv`:

```
position,wildtype,mutation,type,effect
2,M,A,missense,0.242
2,M,C,missense,0.319
2,M,D,missense,-0.104
```

---

## Structure-only use cases

When **no mutation data** is provided:
- Sequence-level metrics are skipped automatically (blosum, aaindex, kidera, effect scores)
- The features CSV has one row per residue instead of one row per mutation
- The run log notes "Sequence metrics: disabled (structure-only mode)"

---

## Using a local PDB file

To use a local structure file instead of downloading from RCSB, set `pdb_path` in
your config or pass it directly to `Runner`:

**In config** (use an absolute path or a path relative to where you run the script):
```toml
pdb_path = "/absolute/path/to/my_protein.pdb"
```

**Or directly in Python**:
```python
from pathlib import Path
from src.pipeline.runner import Runner

runner = Runner(
    pdb_path=Path("my_protein.pdb").resolve(),
    config_path="my_config.toml",
    name="my_protein",
)
```

Both `.pdb` and `.cif` (mmCIF) formats are supported.

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

```
UserWarning: Found N residues with indels...
UserWarning: Found gaps at the termini...
```

These are expected and do not prevent the run.  They occur when the DMS construct
differs from the deposited PDB sequence (e.g. engineered ICL3 loops in GPCRs,
expression tags, terminal truncations).  Residues that cannot be aligned receive
`NaN` for structural metrics.
