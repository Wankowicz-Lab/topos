"""
Deep Mutational Scanning (DMS) example: B2AR (Beta-2 Adrenergic Receptor)
==========================================================================
This example demonstrates the full topos workflow combining a membrane
protein structure (PDB: 4LDE) with deep mutational scanning data.

Inputs:
    - Structure:       PDB 4LDE downloaded from RCSB
    - DMS scores:      examples/B2AR_DMS_example/B2AR_processed_scores.csv
    - Config:          examples/B2AR_DMS_example/B2AR_config.toml

Outputs (written to examples/B2AR_DMS_example/output/):
    - 4LDE_features.csv   — per-residue/per-mutation metric table
    - 4LDE_metadata.csv   — residue-level structural metadata
    - 4LDE_run_log.txt    — human-readable run summary

Run from anywhere:
    conda activate topos-py311
    python examples/B2AR_DMS_example/run_example.py
"""

import logging
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Path setup — allows the script to be run from any directory
# ---------------------------------------------------------------------------
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# Configure logging before importing topos so all pipeline messages are visible
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s - %(message)s",
    stream=sys.stdout,
)

from src.pipeline.runner import Runner  # noqa: E402

# ---------------------------------------------------------------------------
# File paths
# ---------------------------------------------------------------------------
CONFIG_PATH = _REPO_ROOT / "examples" / "B2AR_DMS_example" / "B2AR_config.toml"
OUTPUT_DIR  = _REPO_ROOT / "examples" / "B2AR_DMS_example" / "output"

# ---------------------------------------------------------------------------
# Run pipeline
# ---------------------------------------------------------------------------
print("\n=== topos: DMS Example (B2AR / 4LDE) ===\n")

# Initialise Runner from the config file.
# The config specifies:
#   pdb_id = "4LDE"                 — structure downloaded from RCSB
#   membrane_protein = true          — enables PDBTM orientation + membrane metrics
#   mutation_data_path = "..."       — CSV with DMS effect scores
#   mutation_data_chain = "A"        — chain to align mutation data against
#
# NOTE: Sequence alignment warnings about gaps/mismatches are expected when the
# DMS experiment covers a slightly different sequence construct than the PDB entry.
runner = Runner(config_path=CONFIG_PATH)

print("\nStructure loaded.")
print(f"  PDB ID:            {runner.context.config.pdb_id}")
print(f"  Chains:            {sorted(runner.context.residue_table['chain'].unique().tolist())}")
print(f"  Residues:          {runner.context.residue_table['resi_struct'].nunique()}")
print(f"  Hydrogens in file: {runner._had_hydrogens}")
print(f"  remove_hydrogens:  {runner.context.config.remove_hydrogens}")
print(f"  altloc_policy:     {runner.context.config.altloc_policy}")
print(f"  Membrane protein:  {runner.context.config.membrane_protein}")

mut_data = runner.context.extras.get("mutation_data")
if mut_data is not None:
    print(f"\nMutation data loaded.")
    print(f"  Rows:      {len(mut_data)}")
    print(f"  Positions: {mut_data['resi'].nunique()}")
    print(f"  Chain:     {runner.context.config.mutation_data_chain}")
print()

# Run all metrics.
# Because mutation_data_path is set in the config, sequence-level metrics
# (blosum, aaindex, kidera, effect scores) are included automatically.
runner.run()

print(f"\nFeatures computed: {len(runner.features)} rows x {len(runner.features.columns)} columns")

# Save results
runner.save_results(output_dir=OUTPUT_DIR)

print(f"\nOutput written to: {OUTPUT_DIR}")
for f in sorted(OUTPUT_DIR.iterdir()):
    print(f"  {f.name}")

print("\nDone.")
