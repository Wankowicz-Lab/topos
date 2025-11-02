# biogenesis

A toolkit for computing and analyzing sequence/structure metrics in proteins.

## Quick Start

### Basic Usage

```python
from structure.structure_context import Context, load_structure_with_id
from structure.run_metrics import compute_all
import structure.metrics  # Register all metrics

# Load a PDB file
arr, pdb_id = load_structure_with_id("path/to/file.pdb")

# Create Context (automatically creates base DF, SASA, and secondary structure)
ctx = Context(arr, pdb_id=pdb_id)

# Access basic metrics
print(f"Residues: {ctx.n_residues}, Chains: {ctx.n_chains}")
print(f"Residue types: {ctx.residue_type_distribution}")

# View baseline DataFrame with SASA and secondary structure
print(ctx.baseline_df.head())

# Run all metrics
results = compute_all("path/to/file.pdb")
print(results.head())
```

### Running Examples

Run the example script:

```bash
# With a local PDB file
python examples/structure_example.py /path/to/file.pdb

# Without arguments (will download example PDB)
python examples/structure_example.py
```
