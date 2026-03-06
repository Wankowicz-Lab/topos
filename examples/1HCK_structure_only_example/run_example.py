"""
Structure-only example: 1HCK (CDK2/cyclin kinase inhibitor complex)
====================================================================
This example demonstrates how to use biogenesis with a local PDB file and
no mutation/DMS data.  All structural metrics are computed and saved to CSV.

Run from anywhere — the script resolves paths relative to the repository root:

    conda activate biogenesis-py311
    python examples/1HCK_structure_only_example/run_example.py
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

# Configure logging before importing biogenesis so all pipeline messages are visible
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s - %(message)s",
    stream=sys.stdout,
)

from src.pipeline.runner import Runner  # noqa: E402 (import after logging/path setup)

# ---------------------------------------------------------------------------
# File paths — all derived from the repository root so the script can be run
# from any working directory
# ---------------------------------------------------------------------------
PDB_PATH    = _REPO_ROOT / "examples" / "1HCK.pdb"
CONFIG_PATH = _REPO_ROOT / "examples" / "1HCK_structure_only_example" / "1HCK_config.toml"
OUTPUT_DIR  = _REPO_ROOT / "examples" / "1HCK_structure_only_example" / "output"

# ---------------------------------------------------------------------------
# Run pipeline
# ---------------------------------------------------------------------------
print("\n=== biogenesis: Structure-Only Example (1HCK) ===\n")

# Initialise Runner from the config file.
# The config specifies:
#   pdb_id = "1HCK"                 — structure downloaded from RCSB
#   membrane_protein = false         — will not run PDBTM orientation + membrane metrics
#
# Because no mutation_data_path is set in the config, only structural metrics
# will be computed — sequence-level metrics are skipped automatically.
runner = Runner(
    pdb_path=PDB_PATH,
    config_path=CONFIG_PATH,
    name="1HCK",
)

print("\nStructure loaded.")
print(f"  PDB file:          {PDB_PATH}")
print(f"  Chains:            {sorted(runner.context.residue_table['chain'].unique().tolist())}")
print(f"  Residues:          {runner.context.residue_table['resi_struct'].nunique()}")
print(f"  Hydrogens in file: {runner._had_hydrogens}")
print(f"  remove_hydrogens:  {runner.context.config.remove_hydrogens}")
print(f"  altloc_policy:     {runner.context.config.altloc_policy}")
print(f"  Membrane protein:  {runner.context.config.membrane_protein}")
print()

# Run all metrics.  Sequence-level metrics are skipped automatically when
# no mutation data is provided.
runner.run()

print(f"\nFeatures computed: {len(runner.features)} rows x {len(runner.features.columns)} columns")

# Save features CSV, metadata CSV, and run log to OUTPUT_DIR
runner.save_results(output_dir=OUTPUT_DIR)

print(f"\nOutput written to: {OUTPUT_DIR}")
for f in sorted(OUTPUT_DIR.iterdir()):
    print(f"  {f.name}")

print("\nDone.")
