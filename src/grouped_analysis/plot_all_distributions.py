"""
Plot types
----------
lineplot  : mean ± SD across structures, one coloured line per structure
boxplot   : per-residue box + individual points coloured by structure
heatmap   : structures (y) × residues (x), colour = metric value
            use --cluster to add hierarchical clustering dendrograms on
            both axes (residues clustered by similarity across structures,
            structures clustered by similarity across residues)

"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd
from scipy.cluster import hierarchy
from scipy.spatial.distance import pdist
import itertools


COLORS = [
    "#4C72B0", "#DD8452", "#55A868", "#C44E52",
    "#8172B3", "#937860", "#DA8BC3", "#8C8C8C",
    "#2ECC71", "#E74C3C", "#9B59B6", "#F39C12",
]

def color_cycle(n, palette=None):
    """
    Returns an iterator of colors for n items, cycling if needed.
    """
    palette = palette or COLORS
    if n <= len(palette):
        return palette[:n]
    return list(itertools.islice(itertools.cycle(palette), n))


BOXPLOT_COLS = {
    # Columns to render as boxplots (set defines which metrics get boxplot representation)
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

# Columns to totally exclude from output plots and statistics
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

# ── Data loading ──────────────────────────────────────────────────────────────

def load_data(chain: str, pdb_ids: list[str], renumbered_dir: Path) -> pd.DataFrame:
    """
    Load and concatenate renumbered features CSVs for all PDB IDs, filtered to the specified chain.
    Returns a single DataFrame holding all matching structures.
    """
    frames = []
    for pdb_id in pdb_ids:
        path = renumbered_dir / f"{pdb_id}_features.csv"
        df = pd.read_csv(path)
        df = df.assign(pdb_id=pdb_id)
        # Keep only rows matching the requested chain
        frames.append(df[df["chain"] == chain])
    if not frames:
        sys.exit(f"No data loaded for chain {chain}.")
    # Concatenate all per-structure DataFrames into one large frame
    return pd.concat(frames, ignore_index=True)

# ── Line plot ─────────────────────────────────────────────────────────────────

def per_residue_stats(df: pd.DataFrame, col: str) -> pd.DataFrame:
    """
    Compute per-residue descriptive statistics
    (mean, SD, CV, min, max, range, n_structures) across all structures for a metric.
    Output: DataFrame indexed by residue positions, columns as statistics and all PDBs individually.
    """
    pivot = (
        df.pivot_table(index="resi_struct", columns="pdb_id", values=col)
        .sort_index()
    )
    stats = pd.DataFrame(index=pivot.index)
    # Compute statistics per residue across all PDBs
    stats["mean"] = pivot.mean(axis=1)
    stats["std"]  = pivot.std(axis=1)
    stats["cv"]   = (stats["std"] / stats["mean"].abs()).replace([np.inf, -np.inf], np.nan)
    stats["min"]  = pivot.min(axis=1)
    stats["max"]  = pivot.max(axis=1)
    stats["range"] = stats["max"] - stats["min"]
    stats["n_structures"] = pivot.notna().sum(axis=1)
    # Add individual structure values for each residue
    for pdb in pivot.columns:
        stats[pdb] = pivot[pdb]
    return stats

def plot_lineplot(col: str, df: pd.DataFrame, out_dir: Path) -> pd.DataFrame:
    """
    For a given metric (col), plot mean ± SD trace across residues,
    with one colored line per structure, and shade for mean ± SD.
    Saves plot as PNG.
    Returns DataFrame with per-residue statistics for this metric.
    """
    st = per_residue_stats(df, col)
    resi = st.index.values
    present_pdbs = [p for p in df["pdb_id"].unique() if p in st.columns]

    fig, ax = plt.subplots(figsize=(13, 5))
    fig.suptitle(col, fontsize=13, fontweight="bold")

    # Plot mean ± std shading
    ax.fill_between(resi, st["mean"] - st["std"], st["mean"] + st["std"],
                    alpha=0.18, color="steelblue", label="mean ± 1 SD")
    ax.plot(resi, st["mean"], color="steelblue", lw=2, label="mean", zorder=4)

    # Plot each structure's metric as an individual colored (thin) line
    for i, pdb_id in enumerate(present_pdbs):
        ax.plot(resi, st[pdb_id], color=COLORS[i % len(COLORS)],
                lw=0.9, alpha=0.75, label=pdb_id)

    ax.set_xlabel("Residue number", fontsize=11)
    ax.set_ylabel(col, fontsize=11)
    ax.tick_params(labelsize=9)
    ax.xaxis.set_minor_locator(ticker.AutoMinorLocator())
    ax.legend(fontsize=8, framealpha=0.7, ncol=2, loc="upper right")
    plt.tight_layout()

    safe = col.replace("/", "_").replace(" ", "_")
    fig.savefig(out_dir / f"{safe}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # Return stats DataFrame (used for CSV ouput summary)
    return st[["mean", "std", "cv", "min", "max", "range", "n_structures"]]

# ── Boxplot ───────────────────────────────────────────────────────────────────

def plot_boxplot(col: str, df: pd.DataFrame, out_dir: Path) -> None:
    """
    Boxplot representation:
    - Each residue gets a box, showing the distribution of the metric value across all structures.
    - Overplotted points for individual structures, colored by PDB.
    - Jitter used for scatter points to avoid overlap.
    Saves plot as PNG.
    """
    residues = sorted(df["resi_struct"].dropna().unique().astype(int))
    all_pdb_ids = sorted(df["pdb_id"].unique())
    pdb_color = {pdb: COLORS[i % len(COLORS)] for i, pdb in enumerate(all_pdb_ids)}

    # Collect values per residue into list for boxplots
    data_per_resi = [
        df.loc[df["resi_struct"] == r, col].dropna().values
        for r in residues
    ]

    fig_width = max(20, len(residues) * 0.18)
    fig, ax = plt.subplots(figsize=(fig_width, 5))

    # Draw the boxplots for all residue positions
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

    # Add colored (jittered) scatter points per structure
    rng = np.random.default_rng(42)
    for pdb_id in all_pdb_ids:
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

    # Deduplicate legend entries (each pdb may appear multiple times in scatter)
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

def plot_heatmap(col: str, df: pd.DataFrame, out_dir: Path, cluster: bool = False) -> None:
    """
    Plot a heatmap of structures (rows) x residue number (columns) for given metric.
    Optionally cluster both dimensions using hierarchical clustering.
    """
    # Pivot: one row per structure, one column per residue number
    pivot = (
        df.pivot_table(index="pdb_id", columns="resi_struct", values=col)
        .reindex(index=[p for p in df["pdb_id"].unique() if p in df["pdb_id"].values])
        .sort_index(axis=1)
    )

    # Fill NaN with column mean for distance calculations; replace any remaining
    # non-finite values (e.g. -inf from log2(0) columns) with 0
    pivot_filled = pivot.fillna(pivot.mean()).replace([np.inf, -np.inf], np.nan).fillna(0)

    if cluster:
        # ── Hierarchical clustering ───────────────────────────────────────────
        # First cluster columns (residues) by values across all PDBs (structures)
        col_dist = pdist(pivot_filled.T.values, metric="euclidean")
        col_link = hierarchy.linkage(col_dist, method="ward")
        col_order = hierarchy.leaves_list(col_link)
        # Then cluster rows (structures) by vector of values over all residues
        row_dist = pdist(pivot_filled.values, metric="euclidean")
        row_link = hierarchy.linkage(row_dist, method="ward")
        row_order = hierarchy.leaves_list(row_link)
        # Reorder DataFrame by clustering order
        pivot = pivot.iloc[row_order, col_order]

    residues   = pivot.columns.astype(int)
    structures = pivot.index.tolist()

    if cluster:
        # Layout for clustered: dendrograms + heatmap
        fig_width  = max(16, len(residues) * 0.09)
        fig_height = max(4, len(structures) * 0.55) + 1.5  # extra for dendrograms

        fig = plt.figure(figsize=(fig_width, fig_height), layout="constrained")
        gs  = fig.add_gridspec(
            2, 2,
            width_ratios=[0.12, 1],
            height_ratios=[0.18, 1],
            hspace=0.01, wspace=0.01,
        )
        ax_col_dend = fig.add_subplot(gs[0, 1])   # top: column dendrogram (residues)
        ax_row_dend = fig.add_subplot(gs[1, 0])   # left: row dendrogram (structures)
        ax_heat     = fig.add_subplot(gs[1, 1])   # main: heatmap

        # Draw column/residue dendrogram
        hierarchy.dendrogram(
            col_link, ax=ax_col_dend,
            orientation="top", no_labels=True,
            link_color_func=lambda k: "#555555",
        )
        ax_col_dend.set_axis_off()
        # Draw structure dendrogram
        hierarchy.dendrogram(
            row_link, ax=ax_row_dend,
            orientation="left", no_labels=True,
            link_color_func=lambda k: "#555555",
        )
        ax_row_dend.set_axis_off()

    else:
        # Non-clustered, just one subplot: structures x residues
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

    # X-axis: residue number, major ticks every 10
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


def run_plots(
    chain: str,
    pdb_ids: list[str],
    renumbered_dir: Path,
    out_dir: Path,
    only: str | None = None,
    cluster: bool = False,
) -> None:
    """
    Programmatic entry point: generate per-residue metric plots across structures.

    Parameters
    ----------
    chain : str
        Chain identifier to analyse (e.g. ``"A"``).
    pdb_ids : list[str]
        PDB IDs whose renumbered features CSVs to load.
    renumbered_dir : Path
        Directory containing ``{PDB_ID}_features.csv`` files.
    out_dir : Path
        Directory where PNG and CSV outputs are written.
    only : str or None
        Restrict to one plot type: ``"lineplots"``, ``"boxplots"``, or
        ``"heatmaps"``.  ``None`` runs all three.
    cluster : bool
        Apply hierarchical clustering to heatmap axes.
    """
    df = load_data(chain, pdb_ids, renumbered_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Determine which plot modes should be run
    run_line = only in (None, "lineplots")
    run_box  = only in (None, "boxplots")
    run_heat = only in (None, "heatmaps")

    # Filter columns: numerical columns minus skip set
    all_num_cols = [
        c for c in df.select_dtypes(include="number").columns
        if c not in SKIP_COLS
    ]
    # Separate columns for lineplots and boxplots
    line_cols = [c for c in all_num_cols if c not in BOXPLOT_COLS]
    box_cols  = [c for c in all_num_cols if c in BOXPLOT_COLS]

    # ── Line plots ────────────────────────────────────────────────────────────
    if run_line:
        all_stats = {}
        for col in line_cols:
            st = plot_lineplot(col, df, out_dir)
            all_stats[col] = st

        if all_stats:
            pd.DataFrame({c: s["mean"] for c, s in all_stats.items()}) \
              .rename_axis("resi_struct") \
              .to_csv(out_dir / "all_columns_mean_per_residue.csv")
            pd.DataFrame({c: s["std"] for c, s in all_stats.items()}) \
              .rename_axis("resi_struct") \
              .to_csv(out_dir / "all_columns_std_per_residue.csv")

    # ── Boxplots ──────────────────────────────────────────────────────────────
    if run_box:
        for col in box_cols:
            plot_boxplot(col, df, out_dir)

    # ── Heatmaps ──────────────────────────────────────────────────────────────
    if run_heat:
        for col in all_num_cols:
            plot_heatmap(col, df, out_dir, cluster=cluster)


