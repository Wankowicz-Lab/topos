#!/usr/bin/env python3
"""
Analyze and compare packing among a group of structures.

Requirements
------------
The clusters CSV **must** contain a column named `PDB` (listing PDB filenames or IDs)
and a column named `Cluster` defining differences between PDBs you would like to examine. If you just want descriptive statistics, assign every PDB the same cluster. 
If you are providing multiple clusters, the script will remove any clusters with fewer than 2 PDBs. 

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
                   help="Optional prefix for output filenames, e.g., 'mac1_'. [default: '']")

    # Figure options
    p.add_argument("--fig-dpi", type=int, default=300, help="DPI for saved figures. [default: 300]")
    p.add_argument("--bins", type=int, default=30, help="Histogram bins. [default: 30]")

    return p.parse_args()


# --------------------------- Utils ---------------------------

def ensure_dir(path: str | Path):
    Path(path).mkdir(parents=True, exist_ok=True)


def extract_pdb_name(basename: str, method: str, slice_spec: str, regex: str) -> str:
    """Return a PDB identifier parsed from a filename basename (no extension)."""
    stem = os.path.splitext(basename)[0]  # filename w/o extension
    if method == "slice":
        # slice_spec like "0:5" or "0:4"; support optional step "start:stop:step"
        parts = [int(x) if x else None for x in (slice_spec.split(":") + ["", ""])[:3]]
        sl = slice(*parts)
        return stem[sl]
    else:
        m = re.match(regex, stem)
        if not m or "pdb" not in m.groupdict():
            raise ValueError(
                f"Regex did not match or missing named group 'pdb' for '{basename}'. "
                f"Regex: {regex}"
            )
        return m.group("pdb")


def savefig(path: Path, dpi: int):
    plt.tight_layout()
    plt.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close()


# --------------------------- I/O ---------------------------

def load_packing_tables(csv_glob: str,
                        pdb_name_from: str,
                        pdb_slice: str,
                        pdb_regex: str,
                        packing_col: str,
                        resi_col: str) -> pd.DataFrame:
    """Load all packing CSVs and tag each row with PDB_Name parsed from filename."""
    files = sorted(glob.glob(csv_glob))
    if not files:
        raise FileNotFoundError(f"No CSVs found for glob: {csv_glob}")

    dfs = []
    for f in files:
        df = pd.read_csv(f)
        if packing_col not in df.columns or resi_col not in df.columns:
            raise KeyError(f"Expected columns '{packing_col}' and '{resi_col}' in {f}. "
                           f"Found: {df.columns.tolist()}")
        pdb_name = extract_pdb_name(os.path.basename(f), pdb_name_from, pdb_slice, pdb_regex)
        df = df.copy()
        df["PDB_Name"] = pdb_name
        dfs.append(df)

    return pd.concat(dfs, ignore_index=True)


def load_clusters(clusters_csv: str,
                  strip_suffix: str,
                  pdb_suffix: str) -> pd.DataFrame:
    """Load clusters CSV, requiring columns 'PDB' and 'Cluster', and create PDB_Name."""
    df = pd.read_csv(clusters_csv)
    required = {"PDB", "Cluster"}
    missing = required - set(df.columns)
    if missing:
        raise KeyError(f"Missing required column(s) {sorted(missing)} in {clusters_csv}. "
                       f"Columns present: {df.columns.tolist()}")

    df = df.copy()
    if strip_suffix:
        # Strip e.g., "_qFit_H.pdb"
        df["PDB_Name"] = df["PDB"].astype(str).str.replace(strip_suffix, "", regex=False)
    else:
        # Fall back to stripping generic suffix like ".pdb" (or user-provided)
        if pdb_suffix:
            df["PDB_Name"] = df["PDB"].astype(str).str.replace(pdb_suffix, "", regex=False)
        else:
            # If user explicitly passes empty suffix, leave as-is then strip a trailing .pdb if present
            df["PDB_Name"] = df["PDB"].astype(str).str.replace(r"\.pdb$", "", regex=True)

    return df[["PDB_Name", "Cluster"]]


# --------------------------- Defaults ---------------------------

def default_resi_clusters() -> Dict[int, List[int]]:
    # Built-in mapping from your snippet
    return {
        1: [104, 105, 103],
        2: [157, 162, 165],
        3: [29, 30, 31],
        4: [36, 78, 79, 80, 81, 82, 83, 93, 109, 110, 114, 115, 116, 117, 118, 119, 142, 143],
        5: [37, 39, 45, 59, 60, 62, 94, 95],
        6: [13, 15, 16, 114, 146, 147, 148, 149, 151, 152],
    }


def resi_cluster_df_from_mapping(mapping: Dict[int, List[int]]) -> pd.DataFrame:
    rows = []
    for cid, residues in mapping.items():
        for r in residues:
            rows.append({"resi": r, "resi_cluster": cid})
    return pd.DataFrame(rows)


# --------------------------- Main workflow ---------------------------

def main():
    args = parse_args()
    ensure_dir(args.output_dir)
    out = lambda name: Path(args.output_dir) / f"{args.output_prefix}{name}"

    # Load packing CSVs
    combined_df = load_packing_tables(
        csv_glob=args.csv_glob,
        pdb_name_from=args.pdb_name_from,
        pdb_slice=args.pdb_slice,
        pdb_regex=args.pdb_regex,
        packing_col=args.packing_col,
        resi_col=args.resi_col,
    )

    # Average packing per PDB
    avg_by_pdb = (
        combined_df.groupby("PDB_Name", as_index=False)[args.packing_col]
        .mean()
        .rename(columns={args.packing_col: "avg_packing"})
    )
    avg_by_pdb.to_csv(out("average_packing_by_PDB.csv"), index=False)

    # Distribution of average packing scores
    plt.figure(figsize=(12, 8))
    plt.hist(avg_by_pdb["avg_packing"], bins=args.bins, alpha=0.5, edgecolor="black")
    plt.title("Distribution of Average Packing Scores by Residue", fontsize=16)
    plt.xlabel("Average Packing Score", fontsize=14)
    plt.ylabel("Frequency", fontsize=14)
    savefig(out("average_packing_scores_distribution.png"), dpi=args.fig_dpi)

    # Load clusters (requires 'PDB' and 'Cluster')
    hbd = load_clusters(args.clusters_csv, args.clusters_strip_suffix, args.pdb_suffix)

    # Warn if no overlap
    overlap = set(avg_by_pdb["PDB_Name"]) & set(hbd["PDB_Name"])
    if not overlap:
        print("WARNING: No overlap between packing PDBs and cluster PDBs after normalization.\n"
              "Check your --pdb-name-from/--pdb-slice/--pdb-regex and suffix settings.")

    # Merge packing with cluster label
    merged = pd.merge(avg_by_pdb, hbd, on="PDB_Name", how="inner")

    if merged.empty:
        print("No merged rows between packing data and clusters. Exiting after writing initial outputs.")
        print(f"Done. Outputs written to: {Path(args.output_dir).resolve()}")
        return

    # Order clusters by median packing
    cluster_medians = merged.groupby("Cluster")["avg_packing"].median().sort_values()
    cluster_order = cluster_medians.index.tolist()

    # Boxplot of average packing by HDBSCAN cluster
    plt.figure(figsize=(14, 10))
    sns.boxplot(data=merged, x="Cluster", y="avg_packing", order=cluster_order)
    plt.xlabel("Cluster", fontsize=22)
    plt.ylabel("Average Packing Score", fontsize=22)
    plt.xticks(rotation=45, fontsize=20)
    plt.yticks(fontsize=20)
    savefig(out("average_packing_scores_by_cluster.png"), dpi=args.fig_dpi)

    # Residue clusters
    if args.resi_clusters_json:
        resi_mapping = pd.read_json(args.resi_clusters_json, typ="series").to_dict()
        # keys may come as strings—coerce to int
        resi_mapping = {int(k): list(map(int, v)) for k, v in resi_mapping.items()}
    else:
        resi_mapping = default_resi_clusters()
    resi_cluster_df = resi_cluster_df_from_mapping(resi_mapping)

    # Merge residue cluster labels
    combined_with_clusters = pd.merge(
        combined_df.rename(columns={args.resi_col: "resi"}),
        resi_cluster_df,
        on="resi",
        how="inner",
    )

    # Boxplot: packing across all PDBs for each residue cluster
    medians_by_resicluster = (
        combined_with_clusters.groupby("resi_cluster")[args.packing_col]
        .median()
        .sort_values()
    )
    resi_cluster_order = medians_by_resicluster.index.tolist()

    plt.figure(figsize=(14, 8))
    sns.boxplot(data=combined_with_clusters, x="resi_cluster", y=args.packing_col, order=resi_cluster_order)
    plt.xlabel("Residue Cluster", fontsize=16)
    plt.ylabel(f"Packing Score ({args.packing_col})", fontsize=16)
    plt.title("Distribution of Packing Scores by Residue Cluster (All PDBs)", fontsize=18)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    savefig(out("packing_distribution_by_resi_cluster.png"), dpi=args.fig_dpi)

    # Print/Save summary statistics per residue cluster
    stats_rows = []
    for cid in resi_cluster_order:
        vals = combined_with_clusters.loc[
            combined_with_clusters["resi_cluster"] == cid, args.packing_col
        ]
        stats_rows.append(
            {
                "resi_cluster": cid,
                "residues": resi_mapping[cid],
                "n": int(vals.shape[0]),
                "mean": float(vals.mean()),
                "median": float(vals.median()),
                "std": float(vals.std()),
                "min": float(vals.min()),
                "max": float(vals.max()),
            }
        )
    stats_df = pd.DataFrame(stats_rows)
    stats_df.to_csv(out("packing_by_resi_cluster_summary.csv"), index=False)

    # Console summary
    print("\nSummary Statistics by Residue Cluster:")
    print("=" * 70)
    for row in stats_rows:
        print(f"\nResidue Cluster {row['resi_cluster']}:")
        print(f"  Residues: {row['residues']}")
        print(f"  N measurements: {row['n']}")
        print(f"  Mean: {row['mean']:.3f}")
        print(f"  Median: {row['median']:.3f}")
        print(f"  Std: {row['std']:.3f}")
        print(f"  Min: {row['min']:.3f}")
        print(f"  Max: {row['max']:.3f}")

    # Average packing for each (resi_cluster, HDBSCAN cluster)
    cluster_stats = []
    for resi_cluster_id, residues in resi_mapping.items():
        for hbd_cluster in sorted(hbd["Cluster"].unique()):
            cluster_pdbs = hbd.loc[hbd["Cluster"] == hbd_cluster, "PDB_Name"].tolist()
            mask = (combined_df["PDB_Name"].isin(cluster_pdbs)) & (combined_df["resi"].isin(residues))
            subset = combined_df.loc[mask, args.packing_col]
            if subset.empty:
                continue
            cluster_stats.append(
                {
                    "Resi_Cluster": resi_cluster_id,
                    "HBD_Cluster": hbd_cluster,
                    "Mean_Packing": float(subset.mean()),
                    "Median_Packing": float(subset.median()),
                    "Std_Packing": float(subset.std()),
                    "Num_Structures": int(len(set(cluster_pdbs))),
                    "Residues": ",".join(map(str, residues)),
                }
            )
    cluster_stats_df = pd.DataFrame(cluster_stats)
    cluster_stats_df.to_csv(out("residue_cluster_x_hbd_cluster_stats.csv"), index=False)

    # Heatmap: mean packing by (resi_cluster, HDBSCAN cluster)
    if not cluster_stats_df.empty:
        pivot_mean = cluster_stats_df.pivot(index="Resi_Cluster", columns="HBD_Cluster", values="Mean_Packing")
        plt.figure(figsize=(12, 8))
        sns.heatmap(pivot_mean, annot=True, fmt=".3f", cmap="magma",
                    center=pivot_mean.values.mean() if pivot_mean.size else None)
        plt.title("Mean Packing Scores by Residue Cluster and HDBSCAN Cluster", fontsize=16)
        plt.xlabel("HDBSCAN Cluster", fontsize=14)
        plt.ylabel("Residue Cluster", fontsize=14)
        savefig(out("residue_cluster_x_hbd_cluster_heatmap.png"), dpi=args.fig_dpi)

        # Prepare long-form DF for boxplots across (Resi_Cluster, HDBSCAN Cluster)
        long_rows = []
        for _, row in cluster_stats_df.iterrows():
            residues = list(map(int, str(row["Residues"]).split(",")))
            pdbs = hbd.loc[hbd["Cluster"] == row["HBD_Cluster"], "PDB_Name"]
            mask = (combined_df["PDB_Name"].isin(pdbs)) & (combined_df["resi"].isin(residues))
            vals = combined_df.loc[mask, [args.packing_col]].copy()
            if vals.empty:
                continue
            vals["Resi_Cluster"] = row["Resi_Cluster"]
            vals["HDB_Cluster"] = row["HBD_Cluster"]
            long_rows.append(vals)

        if long_rows:
            long_df = pd.concat(long_rows, ignore_index=True)
            plt.figure(figsize=(15, 10))
            long_df["label"] = "R" + long_df["Resi_Cluster"].astype(str) + "\nH" + long_df["HBD_Cluster"].astype(str)
            sns.boxplot(data=long_df, x="label", y=args.packing_col)
            plt.xticks(rotation=45, ha="right")
            plt.title("Distribution of Packing Scores by Residue Cluster and HDBSCAN Cluster", fontsize=16)
            plt.xlabel("Residue Cluster (R) × HDBSCAN Cluster (H)", fontsize=14)
            plt.ylabel("Packing Score", fontsize=14)
            savefig(out("residue_cluster_x_hbd_cluster_distribution.png"), dpi=args.fig_dpi)

        # Clustermap using Median_Packing
        pivot_median = cluster_stats_df.pivot(index="Resi_Cluster", columns="HDB_Cluster", values="Median_Packing")
        g = sns.clustermap(
            pivot_median,
            cmap="magma",
            center=0,
            figsize=(12, 8),
            yticklabels=True,
            xticklabels=True,
            dendrogram_ratio=(.1, .2),
            cbar_pos=(0.02, .2, .03, .4),
        )
        g.ax_heatmap.set_xticklabels(g.ax_heatmap.get_xticklabels(), rotation=45, ha="right")
        g.fig.suptitle("Hierarchically Clustered Packing Statistics", y=1.02)
        g.savefig(out("residue_cluster_x_hbd_cluster_clustermap.png"), dpi=args.fig_dpi)
        plt.close(g.fig)

    print(f"\nDone. Outputs written to: {Path(args.output_dir).resolve()}")


if __name__ == "__main__":
    main()
