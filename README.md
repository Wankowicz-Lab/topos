# biogenesis

A toolkit for computing and analyzing sequence and structure metrics.

## Installation

Install the package from source:

```bash
git clone https://github.com/Wankowicz-Lab/biogenesis.git
cd biogenesis
pip install -e .
```

## Quick Start
Given PDBs and/or mutation information for a given protein, users can use the scripts within here to calculate sequence and/or structural metrics for downstream use. This pipeline produces an output CSV file for each protein containing the chain, residue number, and residue name, along with calculated metrics. If mutation information is provided, these metrics are calculated for each mutation. If PDBs include alternative conformers, metrics can be averaged across the multiple conformers, or each metric can be provided individually for each altloc.  

Structure metrics included: secondary structure, solvent exposure, hydrogen bonding patterns, residue packing

We also provide grouped analysis scripts that look at and compare multiple structures. All scripts depend on input metrics. 

The `examples` directory contains example data for each of these use cases. 


### Basic Usage

```python
from src.pipeline import runner

# Set up pipeline using B2AR example data
pdb_id = '4LDE'
config_path = 'examples/B2AR_DMS_example/B2AR_config.toml'
b2ar_runner = runner.Runner(pdb_id=pdb_id, config_path=config_path)

# Provide a list of specific metrics to calculate
metrics = ['define_secondary_structure', 'sasa', 'kyte_doolittle', 'calculate_blosum_score'] 
b2ar_runner.run(metrics=metrics)

# Or calculate using all available metrics
b2ar_runner.run()

# Access the metrics directly
metrics = b2ar_runner.features

# Save metrics and associated metadata to specified directory
output_dir = 'examples/B2AR_DMS_example/'
b2ar_runner.save_results(output_dir)
```

### Config file
The easiest way to control the behavior of the runner is by modifying the config file that is provided to the `runner.Runner(config_path=config_path)` initialization. 

#### Structure Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `pdb_id` | The PDB ID of the structure | - |
| `pdb_path` | The path to the PDB file. Only one of `pdb_id` or `pdb_path` needs to be provided | - |
| `membrane_protein` | Whether or not the protein is a membrane protein. If it is, calculates additional features | `false` |
| `membrane_thickness` | The thickness of the membrane in Angstroms, used for calculating distances from the center of the membrane | `15` |

#### Mutagenesis Data Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `mutation_data_path` | Path to CSV file containing mutation data | - |
| `mutation_data_chain` | The chain of the PDB file that corresponds to the sequence being mutated in the mutation data | - |
| `mutation_residue_col_name` | Column name for wildtype residues in mutation data CSV | `"wildtype"` |
| `mutation_residue_idx_name` | Column name for residue positions in mutation data CSV | `"position"` |
| `mutation_col_name` | Column name for mutant residues in mutation data CSV | `"mutation"` |
| `mutation_type_col_name` | Column name for mutation types in mutation data CSV | `"type"` |
| `mutation_score_col_name` | Column name for mutation effect scores in mutation data CSV | `"effect"` |

#### Sequence Feature Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `aaindex_path` | Path to the data file containing amino acid indices | `"data/aaindex_parsed_small.csv"` | 


#### Pipeline Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `output_dir` | Path to the directory where output files will be saved | `"examples/B2AR_DMS_example/output"` | 
| `output_prefix` | Prefix to append to generated output files | - | 


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



