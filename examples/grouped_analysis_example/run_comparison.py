#!/usr/bin/env python3
"""
Grouped Analysis Example
========================
Compares per-residue metrics across multiple protein structures using the Comparator.

Prerequisites
-------------
Run the 1HCK structure-only example first to generate the required features CSV:
    cd examples/1HCK_structure_only_example
    python run_example.py

Usage
-----
From the repo root:
    conda run -n biogenesis-py311 python examples/grouped_analysis_example/run_comparison.py
"""
import sys
from pathlib import Path

# Add repo root to path so 'src' is importable
repo_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(repo_root))

from src.grouped_analysis import Comparator

config_path = Path(__file__).parent / "comparison_config.toml"

# Check that the required features CSV exists
features_csv = Path(__file__).parent.parent / "1HCK_structure_only_example" / "output" / "1HCK_features.csv"
if not features_csv.exists():
    print(
        f"ERROR: Required file not found: {features_csv}\n"
        "Please run examples/1HCK_structure_only_example/run_example.py first."
    )
    sys.exit(1)

print(f"Running comparison from: {config_path}")
c = Comparator(config_path)
c.run()

print("\nDone! Output files written to:", (Path(__file__).parent / "output").resolve())
