#!/usr/bin/env python3
"""
Analyze and compare packing among a group of structures.

Requirements
------------
The clusters CSV **must** contain a column named `PDB` (listing PDB filenames or IDs)
and a column named `Cluster` defining differences between PDBs you would like to examine.
If you just want descriptive statistics, assign every PDB the same cluster.

Examples
--------
python analyze_compare_packing.py \
  --csv-glob '/path/to/PDBs/*_packing.csv' \
  --clusters-csv '/path/to/HBDScan_PDB_clusters.csv' \
  --output-dir results

# If your PDB name is not simply the start of the filename, you can control extraction
# and also declare a different default suffix than '.pdb' (e.g., '.ent'):
python analyze_compare_packing.py \
  --csv-glob 'PDBs/*_packing.csv' \
  --clusters-csv '/path/to/HBDScan_PDB_clusters.csv' \
  --pdb-suffix '.ent'
"""
from __future__ import annotations

import argparse
import glob
import os
import re
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Packing Analysis among PDBs")
    # Inputs
    p.add_argument("--csv-glob", required=True,
                   help="Glob for per-structure packing CSVs (e.g., 'PDBs/PDBs_hold/*_packing.csv').")
    p.add_argument("--clusters-csv", required=True,
                   help="Path to HDBSCAN clusters CSV. MUST include 'PDB' and 'Cluster' columns.")

    p.add_argument("--pdb-suffix", default=".pdb",
                   help="Default PDB suffix to strip from cluster names when --clusters-strip-suffix is ''. "
                        "[default: .pdb]")

    # Column names in packing CSVs
    p.add_argument("--resi-col", default="resi",
                   help="Residue index column in packing CSVs. [default: resi]")
    p.add_argument("--packing-col", default="contact_density",
                   help="Packing score column in packing CSVs. [default: contact_density]")

    # Residue clusters
    p.add_argument("--resi-clusters-json", default=None,
                   help="Optional path to JSON mapping {cluster_id: [resi, ...]}. "
                        "If not provided, a built-in mapping is used.")

    # Outputs
    p.add_argument("--output-dir", default=".",
                   help="Directory to write figures/CSVs. [default: current dir]")
    p.add_argument("--output-prefix", default="",
                   help="Optional prefix for all output filenames, e.g., 'mac1_'. [default: '']")
    p.add_argument("--make-figures", action=argparse.BooleanOptionalAction, default=True,
                   help="Whether to generate and save figures. [default: True]")

    # Figure options
    p.add_argument("--fig-dpi", type=int, default=300, help="DPI for saved figures. [default: 300]")
    p.add_argument("--bins", type=int, default=30, help="Histogram bins. [default: 30]")

    return p.parse_args()


def ensure_dir(path: str | Path):
    Path(path).mkdir(parents=True, exist_ok=True)


def savefig(path: Path, dpi: int, make_figures: bool):
    """Save figure only if make_figures=True."""
    if not make_figures:
        plt.close()
        return
    plt.tight_layout()
    plt.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close()


# --------------------------- Main workflow ---------------------------

def main():
    args = parse_args()
    ensure_dir(args.output_dir)
    out = lambda name: Path(args.output_dir) / f"{args.output_prefix}{name}"

    # Load packing CSVs
    csv_files = sorted(glob.glob(args.csv_glob))
    if not csv_files:
        raise FileNotFoundError(f"No CSVs found for pattern: {args.csv_glob}")

    all_dfs = []
    for f in csv_files:
        df = pd.read_csv(f)
        if args.resi_col not in df.columns or args.packing_col not in df.columns:
            raise KeyError(f"Expected columns '{args.packing_col}' and '{args.resi_col}' in {f}.")
        pdb_name = os.path.basename(f).split("_")[0].replace(args.pdb_suffix, "")
        df["PDB_Name"] = pdb_name
        all_dfs.append(df)
    combined_df = pd.concat(all_dfs, ignore_index=True)

    # Average packing per PDB
    avg_by_pdb = (
        combined_df.groupby("PDB_Name", as_index=False)[args.packing_col]
        .mean()
        .rename(columns={args.packing_col: "avg_packing"})
    )
    avg_by_pdb.to_csv(out("average_packing_by_PDB.csv"), index=False)

    if args.make_figures:
        plt.figure(figsize=(12, 8))
        plt.hist(avg_by_pdb["avg_packing"], bins=args.bins, alpha=0.5, edgecolor="black")
        plt.title("Distribution of Average Packing Scores by Residue", fontsize=16)
        plt.xlabel("Average Packing Score", fontsize=14)
        plt.ylabel("Frequency", fontsize=14)
        savefig(out("average_packing_scores_distribution.png"), dpi=args.fig_dpi, make_figures=args.make_figures)

    # Load clusters
    clusters_df = pd.read_csv(args.clusters_csv)
    if "PDB" not in clusters_df.columns or "Cluster" not in clusters_df.columns:
        raise KeyError(f"Cluster CSV must have 'PDB' and 'Cluster' columns. Found: {clusters_df.columns.tolist()}")

    clusters_df["PDB_Name"] = clusters_df["PDB"].astype(str).str.replace(args.pdb_suffix, "", regex=False)

    merged = pd.merge(avg_by_pdb, clusters_df, on="PDB_Name", how="inner")
    if merged.empty:
        print("No overlap between packing and cluster data. Exiting after writing average packing CSV.")
        return

    if args.make_figures:
        plt.figure(figsize=(14, 10))
        sns.boxplot(data=merged, x="Cluster", y="avg_packing",
                    order=merged.groupby("Cluster")["avg_packing"].median().sort_values().index)
        plt.xlabel("Cluster", fontsize=22)
        plt.ylabel("Average Packing Score", fontsize=22)
        plt.xticks(rotation=45, fontsize=20)
        plt.yticks(fontsize=20)
        savefig(out("average_packing_scores_by_cluster.png"), dpi=args.fig_dpi, make_figures=args.make_figures)

    print(f"\nAnalysis complete. Outputs written to {args.output_dir}")
    print(f"Figures generated: {args.make_figures}")


if __name__ == "__main__":
    main()
