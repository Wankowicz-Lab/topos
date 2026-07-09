"""
Plot per-residue metrics across a user-defined group of PDB analyses.

Plot types
----------
lineplot  : mean ± SD across structures, one coloured line per structure
boxplot   : per-residue box + individual points coloured by structure
heatmap   : structures (y) × residues (x), colour = metric value
            use --cluster to add hierarchical clustering dendrograms on
            both axes (residues clustered by similarity across structures,
            structures clustered by similarity across residues)

Usage
-----
python plot_all.py                                  # all csvs in directory (default)
python plot_all.py --pdbs 1ABX 2CDY 3EFG            # only these pdbs
python plot_all.py --dir mydir/                     # look in this input directory
python plot_all.py --chain B --out my_dir/
"""

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd
from scipy.cluster import hierarchy
from scipy.spatial.distance import pdist

# Columns to render as boxplots
BOXPLOT_COLS = {
    "total_hbond_count",
    "total_bond_count",
    "vdw_contact_count",
    "total_between_chain_bonds",
    "ss_domain_total_between_chain_bonds",
    "ss_domain_sc_hbond_count",
    "ss_domain_total_bond_count",
    "ss_domain_total_hbond_count",
    "ss_domain_total_within_chain_bonds",
    "ss_domain_vdw_contact_count",
    "ss_domain_pi_stacking_count",
    "ss_domain_packing_n_atoms",
    "ss_domain_packing_contact_density",
    "ss_domain_log2_aa_group_ratio_Special",
    "ss_domain_log2_aa_group_ratio_Positively_Charged",
    "ss_domain_log2_aa_group_ratio_Polar_Uncharged",
    "ss_domain_log2_aa_group_ratio_Nonpolar_Aliphatic",
    "ss_domain_log2_aa_group_ratio_Negatively_Charged",
    "ss_domain_log2_aa_group_ratio_Aromatic",
    "ss_domain_length",
    "ss_domain_kyte_doolittle",
    "ss_domain_ionic_bond_count",
    "ss_domain_disulfide_bond_count",
    "ss_domain_distance_to_nearest_surface_residue",
    "ss_domain_distance_to_center_of_mass",
    "ss_domain_distance_from_membrane_edge",
    "ss_domain_cation_pi_count",
    "ss_domain_bb_hbond_count",
    "sc_hbond_count",
    "salt_bridge_count",
    "pi_stacking_count",
    "neighborhood_total_between_chain_bonds",
    "neighborhood_pi_stacking_count",
    "neighborhood_ionic_bond_count",
    "neighborhood_disulfide_bond_count",
    "neighborhood_cation_pi_count",
    "n_different_chain_neighbors",
    "graph_hbond_graph_eigenvector_centrality",
    "graph_hbond_graph_core_number",
    "disulfide_bond_count",
    "graph_hbond_graph_betweenness_centrality",
    "graph_all_graph_betweenness_centrality",
    "distance_to_nearest_surface_residue",
    "cation_pi_count",
    "graph_vdw_contact_graph_core_number",
    "ionic_bond_count",
    "ss_domain_packing_n_neighbor_residues",
    "bb_hbond_count",
}

# Columns to skip entirely
SKIP_COLS = {
    "resi_struct",
    "resi_mut",
    "graph_all_graph_community_id",
    "graph_all_graph_core_number",
    "graph_hbond_graph_community_id",
    "graph_hbond_graph_closeness_centrality",
    "graph_hbond_graph_core_number",
    "graph_hbond_graph_eigenvector_centrality",
    "graph_vdw_contact_graph_community_id",
    "kyte_doolittle",
    "mean_neighbor_sequence_distance",
}

DEFAULT_COLORS = [
    "#4C72B0", "#DD8452", "#55A868", "#C44E52",
    "#8172B3", "#937860", "#DA8BC3", "#8C8C8C", "#FFD700", "#27AE60", "#E67E22", "#E74C3C", "#B2BABB", "#A569BD"
]

def get_pdb_ids_and_paths(input_dir: Path, allowed_pdbs=None):
    """
    Scan directory for *_features.csv files, extract PDB ids and return dict {pdbid: path}.
    If allowed_pdbs is set, only keep those.
    """
    paths = list(input_dir.glob("*_features.csv"))
    pdb_id_to_path = {}
    for p in sorted(paths):
        name = p.name
        if not name.endswith("_features.csv"):
            continue
        pdbid = name.split("_features.csv")[0]
        if allowed_pdbs and pdbid not in allowed_pdbs:
            continue
        pdb_id_to_path[pdbid] = p
    return pdb_id_to_path

def assign_colors_to_pdbs(pdb_ids):
    ncolors = len(DEFAULT_COLORS)
    return {pdb: DEFAULT_COLORS[i % ncolors] for i, pdb in enumerate(sorted(pdb_ids))}

# ── Data loading ──────────────────────────────────────────────────────────────

def load_data(chain: str, pdb_id_to_path: dict):
    frames = []
    available_pdb_ids = []
    for pdb_id, path in pdb_id_to_path.items():
        if not path.exists():
            print(f"WARNING: {path.name} not found, skipping.", file=sys.stderr)
            continue
        df = pd.read_csv(path)
        df = df.assign(pdb_id=pdb_id)
        if chain not in df["chain"].values:
            print(f"WARNING: chain {chain} not in {pdb_id}, skipping.", file=sys.stderr)
            continue
        frames.append(df[df["chain"] == chain])
        available_pdb_ids.append(pdb_id)
    if not frames:
        sys.exit(f"No data loaded for chain {chain}.")
    return pd.concat(frames, ignore_index=True), available_pdb_ids

# ── Line plot ─────────────────────────────────────────────────────────────────
def per_residue_stats(df: pd.DataFrame, col: str) -> pd.DataFrame:
    pivot = (
        df.pivot_table(index="resi_struct", columns="pdb_id", values=col)
        .sort_index()
    )
    stats = pd.DataFrame(index=pivot.index)
    stats["mean"] = pivot.mean(axis=1)
    stats["std"]  = pivot.std(axis=1)
    stats["cv"]   = (stats["std"] / stats["mean"].abs()).replace([np.inf, -np.inf], np.nan)
    stats["min"]  = pivot.min(axis=1)
    stats["max"]  = pivot.max(axis=1)
    stats["range"] = stats["max"] - stats["min"]
    stats["n_structures"] = pivot.notna().sum(axis=1)
    for pdb in pivot.columns:
        stats[pdb] = pivot[pdb]
    return stats

def plot_lineplot(col: str, df: pd.DataFrame, out_dir: Path, pdb_ids, pdb_color) -> pd.DataFrame:
    st = per_residue_stats(df, col)
    resi = st.index.values
    present_pdbs = [p for p in pdb_ids if p in st.columns]

    fig, ax = plt.subplots(figsize=(13, 5))
    fig.suptitle(col, fontsize=13, fontweight="bold")

    ax.fill_between(resi, st["mean"] - st["std"], st["mean"] + st["std"],
                    alpha=0.18, color="steelblue", label="mean ± 1 SD")
    ax.plot(resi, st["mean"], color="steelblue", lw=2, label="mean", zorder=4)

    for i, pdb_id in enumerate(present_pdbs):
        ax.plot(resi, st[pdb_id], color=pdb_color[pdb_id],
                lw=0.9, alpha=0.75, label=pdb_id)

    ax.set_xlabel("Residue number", fontsize=11)
    ax.set_ylabel(col, fontsize=11)
    ax.tick_params(labelsize=9)
    ax.xaxis.set_minor_locator(ticker.AutoMinorLocator())
    ax.legend(fontsize=8, framealpha=0.7, ncol=2, loc="upper right")
    plt.tight_layout()

    safe = col.replace("/", "_").replace(" ", "_")
    fig.savefig(out_dir / f"{safe}_lineplot.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    return st[["mean", "std", "cv", "min", "max", "range", "n_structures"]]

# ── Boxplot ───────────────────────────────────────────────────────────────────
def plot_boxplot(col: str, df: pd.DataFrame, out_dir: Path, pdb_ids, pdb_color) -> None:
    residues = sorted(df["resi_struct"].dropna().unique().astype(int))

    data_per_resi = [
        df.loc[df["resi_struct"] == r, col].dropna().values
        for r in residues
    ]

    fig_width = max(20, len(residues) * 0.18)
    fig, ax = plt.subplots(figsize=(fig_width, 5))

    ax.boxplot(
        data_per_resi,
        positions=residues,
        widths=0.65,
        patch_artist=True,
        showfliers=False,
        medianprops=dict(color="black", lw=1.5),
        boxprops=dict(facecolor="#D0E4F5", edgecolor="#4C72B0", lw=0.8),
        whiskerprops=dict(color="#4C72B0", lw=0.8),
        capprops=dict(color="#4C72B0", lw=0.8),
    )

    rng = np.random.default_rng(42)
    for pdb_id in pdb_ids:
        sub = df[df["pdb_id"] == pdb_id][["resi_struct", col]].dropna()
        if sub.empty:
            continue
        jitter = rng.uniform(-0.25, 0.25, size=len(sub))
        ax.scatter(sub["resi_struct"].values + jitter, sub[col].values,
                   color=pdb_color[pdb_id], s=14, alpha=0.75, zorder=3,
                   label=pdb_id, linewidths=0)

    ax.set_xlabel("Residue number", fontsize=11)
    ax.set_ylabel(col, fontsize=11)
    ax.set_title(col, fontsize=13, fontweight="bold")
    ax.set_xlim(residues[0] - 1.5, residues[-1] + 1.5)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(10))
    ax.xaxis.set_minor_locator(ticker.MultipleLocator(1))
    ax.tick_params(axis="x", labelsize=7, rotation=45)
    ax.tick_params(axis="y", labelsize=9)
    ax.grid(axis="y", lw=0.4, alpha=0.5)

    handles, labels = ax.get_legend_handles_labels()
    seen = {}
    for h, l in zip(handles, labels):
        if l not in seen:
            seen[l] = h
    ax.legend(seen.values(), seen.keys(), fontsize=8, framealpha=0.7,
              loc="upper right", ncol=2)

    plt.tight_layout()
    safe = col.replace("/", "_").replace(" ", "_")
    fig.savefig(out_dir / f"{safe}_boxplot.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

# ── Heatmap ───────────────────────────────────────────────────────────────────
def plot_heatmap(col: str, df: pd.DataFrame, out_dir: Path, pdb_ids, cluster: bool = False) -> None:
    # Pivot: rows = structures (pdb_id), columns = residues
    pivot = (
        df.pivot_table(index="pdb_id", columns="resi_struct", values=col)
        .reindex(index=[p for p in pdb_ids if p in df["pdb_id"].values])
        .sort_index(axis=1)
    )

    # Fill NaN with column mean for distance calculations; replace any remaining
    # non-finite values (e.g. -inf from log2(0) columns) with 0
    pivot_filled = pivot.fillna(pivot.mean()).replace([np.inf, -np.inf], np.nan).fillna(0)

    if cluster:
        # ── Hierarchical clustering ───────────────────────────────────────────
        # Cluster columns (residues) — group by similarity across structures
        col_dist = pdist(pivot_filled.T.values, metric="euclidean")
        col_link = hierarchy.linkage(col_dist, method="ward")
        col_order = hierarchy.leaves_list(col_link)

        # Cluster rows (structures) — group by similarity across residues
        row_dist = pdist(pivot_filled.values, metric="euclidean")
        row_link = hierarchy.linkage(row_dist, method="ward")
        row_order = hierarchy.leaves_list(row_link)

        # Reorder pivot
        pivot = pivot.iloc[row_order, col_order]

    residues   = pivot.columns.astype(int)
    structures = pivot.index.tolist()

    if cluster:
        # Layout: dendrogram above (col) + dendrogram left (row) + heatmap
        fig_width  = max(16, len(residues) * 0.09)
        fig_height = max(4, len(structures) * 0.55) + 1.5  # extra for dendrograms

        fig = plt.figure(figsize=(fig_width, fig_height), layout="constrained")
        gs  = fig.add_gridspec(
            2, 2,
            width_ratios=[0.12, 1],
            height_ratios=[0.18, 1],
            hspace=0.01, wspace=0.01,
        )
        ax_col_dend = fig.add_subplot(gs[0, 1])   # top: column dendrogram
        ax_row_dend = fig.add_subplot(gs[1, 0])   # left: row dendrogram
        ax_heat     = fig.add_subplot(gs[1, 1])   # main: heatmap

        # Column dendrogram
        hierarchy.dendrogram(
            col_link, ax=ax_col_dend,
            orientation="top", no_labels=True,
            link_color_func=lambda k: "#555555",
        )
        ax_col_dend.set_axis_off()

        # Row dendrogram
        hierarchy.dendrogram(
            row_link, ax=ax_row_dend,
            orientation="left", no_labels=True,
            link_color_func=lambda k: "#555555",
        )
        ax_row_dend.set_axis_off()

    else:
        fig_width  = max(16, len(residues) * 0.09)
        fig_height = max(3, len(structures) * 0.55)
        fig, ax_heat = plt.subplots(figsize=(fig_width, fig_height))

    # ── Heatmap image ─────────────────────────────────────────────────────────
    im = ax_heat.imshow(
        pivot.values,
        aspect="auto",
        cmap="viridis",
        interpolation="nearest",
    )

    # X-axis: residue numbers, tick every 10
    step = 10
    xtick_pos = [i for i, r in enumerate(residues) if r % step == 0]
    xtick_lab = [residues[i] for i in xtick_pos]
    ax_heat.set_xticks(xtick_pos)
    ax_heat.set_xticklabels(xtick_lab, fontsize=7, rotation=90)

    # Y-axis: structure names
    ax_heat.set_yticks(range(len(structures)))
    ax_heat.set_yticklabels(structures, fontsize=9)

    ax_heat.set_xlabel("Residue number" + (" (clustered)" if cluster else ""), fontsize=11)
    ax_heat.set_ylabel("Structure" + (" (clustered)" if cluster else ""), fontsize=11)

    suffix = " — clustered" if cluster else ""
    fig.suptitle(f"{col}{suffix}", fontsize=13, fontweight="bold", y=1.01 if cluster else 0.98)

    plt.colorbar(im, ax=ax_heat, fraction=0.015, pad=0.02, label=col)
    if not cluster:
        plt.tight_layout()

    safe    = col.replace("/", "_").replace(" ", "_")
    outsuffix = "_heatmap_clustered.png" if cluster else "_heatmap.png"
    fig.savefig(out_dir / f"{safe}{outsuffix}", dpi=150, bbox_inches="tight")
    plt.close(fig)

# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Per-residue metric plots for provided group of structures."
    )
    parser.add_argument(
        "--chain", default="A",
        help="Chain to use (default: A)",
    )
    parser.add_argument(
        '--dir',
        type=Path,
        default=Path("adk_output/renumbered"),  # Default matches the old RENUMBERED_DIR
        help="Directory containing *_features.csv files (default: adk_output/renumbered)"
    )
    parser.add_argument(
        '--pdbs',
        nargs='+', default=None,
        help="Optional: List of PDB IDs to restrict analysis to"
    )
    parser.add_argument(
        "--out", type=Path,
        default=Path("adk_output/residue_profiles"),
        help="Output directory (default: adk_output/residue_profiles)",
    )
    parser.add_argument(
        "--only",
        choices=["lineplots", "boxplots", "heatmaps"],
        default=None,
        help="Run only one plot type. Omit to run all three.",
    )
    parser.add_argument(
        "--cluster",
        action="store_true",
        help="Apply hierarchical clustering to heatmaps (both residues and structures axes).",
    )
    args = parser.parse_args()

    # Find all PDB features files
    pdb_id_to_path = get_pdb_ids_and_paths(args.dir, allowed_pdbs=args.pdbs)
    pdb_ids = sorted(pdb_id_to_path.keys())
    if not pdb_ids:
        sys.exit(f"No feature csvs found in {args.dir} for given --pdbs.")

    pdb_color = assign_colors_to_pdbs(pdb_ids)

    run_line = args.only in (None, "lineplots")
    run_box  = args.only in (None, "boxplots")
    run_heat = args.only in (None, "heatmaps")

    df, available_pdb_ids = load_data(args.chain, pdb_id_to_path)
    args.out.mkdir(parents=True, exist_ok=True)

    all_num_cols = [
        c for c in df.select_dtypes(include="number").columns
        if c not in SKIP_COLS
    ]
    line_cols = [c for c in all_num_cols if c not in BOXPLOT_COLS]
    box_cols  = [c for c in all_num_cols if c in BOXPLOT_COLS]

    # ── Line plots ────────────────────────────────────────────────────────────
    if run_line:
        all_stats = {}
        for i, col in enumerate(line_cols, 1):
            st = plot_lineplot(col, df, args.out, available_pdb_ids, pdb_color)
            all_stats[col] = st

        if all_stats:
            pd.DataFrame({c: s["mean"] for c, s in all_stats.items()}) \
              .rename_axis("resi_struct") \
              .to_csv(args.out / "all_columns_mean_per_residue.csv")
            pd.DataFrame({c: s["std"] for c, s in all_stats.items()}) \
              .rename_axis("resi_struct") \
              .to_csv(args.out / "all_columns_std_per_residue.csv")

    # ── Boxplots ──────────────────────────────────────────────────────────────
    if run_box:
        for i, col in enumerate(box_cols, 1):
            plot_boxplot(col, df, args.out, available_pdb_ids, pdb_color)

    # ── Heatmaps ──────────────────────────────────────────────────────────────
    if run_heat:
        cluster_label = " (clustered)" if args.cluster else ""
        for i, col in enumerate(all_num_cols, 1):
            plot_heatmap(col, df, args.out, available_pdb_ids, cluster=args.cluster)

    total = (len(line_cols) if run_line else 0) + \
            (len(box_cols)  if run_box  else 0) + \
            (len(all_num_cols) if run_heat else 0)

if __name__ == "__main__":
    main()
