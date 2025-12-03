# biogenesis

A toolkit for computing and analyzing sequence and structure metrics.

## Installation

Install the package from source:

```bash
git clone https://github.com/Wankowicz-Lab/biogenesis.git
cd biogenesis
pip install -e .
```

### Development and Testing

To install with development/testing dependencies:

```bash
pip install -e ".[test]"
```

Run the test suite:

```bash
pytest tests/ -v
```

Run tests with code coverage:

```bash
pytest tests/ -v --cov=src --cov-report=term --cov-report=xml
```

Coverage reports are automatically generated and uploaded to Coveralls when running through GitHub Actions.

## Quick Start
Given a PDBs and/or FASTAs, users can use the scripts within here to calculate sequence and/or structural metrics for downstream use. All functions produce an output CSV file containing PDB, chain, residue number, and residue name, along with calculated metrics. If PDBs include alternative conformers, metrics can be averaged across the multiple conformers, or each metric can be provided individually for each altloc.  

All metrics will also output individual metrics for each PDB and/or FASTA.

Structure metrics included: secondary structure, solvent exposure, hydrogen bonding patterns, residue packing


We also provide grouped analysis scripts that look at and compare multiple structures. All scripts depend on input metrics. 


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



