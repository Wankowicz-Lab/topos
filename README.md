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
b2ar_runner = runner.Runner(pbd_id=pdb_id, config_path=config_path)

# Provide a list of specific metrics to calculate
metrics = ['define_secondary_structure', 'sasa', 'kyte_doolittle', 'calculate_blosum_score'] 
computed_metrics = b2ar_runner.run(metrics=metrics)

# Or calculate using all available metrics
all_computed_metrics = b2ar_runner.run()
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



