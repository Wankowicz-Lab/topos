"""
Grouped analysis example: adenylate kinase open vs. closed
===========================================================
This example demonstrates how to use the grouped analysis pipeline to compare
multiple structures after they have been processed by the topos pipeline.

Structures compared
-------------------
  4AKE  — open, apo          (reference for residue numbering)
  1AKE  — closed, AP5-bound
  1ANK  — open, apo
  3HPQ  — closed, bound
  6F7U  — intermediate, apo
  G56A  — mutant (placeholder; replace XXXX in the config with a real PDB ID)

Prerequisites
-------------
Run the topos pipeline on each structure first so that
``{PDB_ID}_features.csv`` files exist in output_dir (set in the config):

    from src.pipeline.runner import Runner

    for pdb_id in ["4AKE", "1AKE", "1ANK", "3HPQ", "6F7U"]:
        runner = Runner(pdb_id=pdb_id)
        runner.run()
        runner.save_results(output_dir="examples/grouped_analysis_example/output")

The grouped analysis pipeline then:
    Step 1  — renumber all structures to the 4AKE reference numbering
    Step 2  — compute per-residue variability scores across all structures
    Step 3  — generate lineplots, boxplots, and heatmaps for every metric
    Step 4  — export a consolidated per-residue annotation CSV
    Step 5  — generate PyMOL colouring scripts (variability, SASA, H-bonds)
    Step 5b — compute pairwise CA-RMSD for all structure pairs
    Step 6  — compute pairwise metric differences for each defined pair
    Step 7  — export per-pair comparison annotation CSVs
    Step 8  — generate PyMOL scripts for comparison metrics

Run from anywhere — the script resolves paths relative to the repository root:

    conda activate topos-py311
    python examples/grouped_analysis_example/run_example.py
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

# Configure logging before importing so all pipeline messages are visible
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s - %(message)s",
    stream=sys.stdout,
)

from src.grouped_analysis.run_grouped_pipeline import GroupedPipelineRunner  # noqa: E402

# ---------------------------------------------------------------------------
# File paths — all derived from the repository root so the script can be run
# from any working directory
# ---------------------------------------------------------------------------
CONFIG_PATH = _REPO_ROOT / "examples" / "grouped_analysis_example" / "grouped_analysis_config.toml"

# ---------------------------------------------------------------------------
# Run grouped analysis pipeline
# ---------------------------------------------------------------------------
print("\n=== topos: grouped analysis (adenylate kinase) ===\n")

# Initialise the runner from the config file.
# GroupedPipelineRunner reads all settings from the TOML and resolves output paths.
runner = GroupedPipelineRunner(config_path=CONFIG_PATH)

print("Resolved settings:")
print(f"  Config:        {runner.config_path}")
print(f"  PDB IDs:       {runner.pdb_ids}")
print(f"  Reference PDB: {runner.reference_pdb}")
print(f"  Chain:         {runner.chain}")
print(f"  Output dir:    {runner.output_dir}")
print(f"  Run multi:     {runner.run_multi_flag}")
print(f"  Run compare:   {runner.run_comparison_flag}")
if runner.pairs:
    print(f"  Pairs:         {[p.get('description', '') for p in runner.pairs]}")
print()

# Run all enabled pipeline stages in order.
runner.run()

print(f"\nAll outputs written to: {runner.output_dir}")
