"""
Identify residues with the largest metric changes across structures.

For each numeric metric, compute per-residue standard deviation and range
across all provided structures (from any folder/PDB set). Then:
  1. Heatmap  — residues × metrics, colour = rank-normalised SD
  2. Overall variability score per residue (mean of rank-normalised SDs)
     plotted as a bar chart, with top-N residues highlighted
  3. Per-metric top-N table saved to CSV
  4. Master variability CSV (residue × metric SD values)

"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import pandas as pd
from scipy.stats import rankdata


# Columns that are not meaningful metrics for variability analysis
SKIP_COLS = {
    "resi_struct", "resi_mut",
    "graph_all_graph_community_id",
    "graph_all_graph_core_number",
    "graph_hbond_graph_community_id",
    "graph_hbond_graph_closeness_centrality",
    "graph_hbond_graph_core_number",
    "graph_hbond_graph_eigenvector_centrality",
    "graph_vdw_contact_graph_community_id",
    "kyte_doolittle",
    "mean_neighbor_sequence_distance",
    "disulfide_bond_count",           # zero everywhere
    "ss_domain_disulfide_bond_count",
    "neighborhood_disulfide_bond_count",
    "packing_n_atoms",                # zero variance
    "distance_from_membrane_edge",    # not membrane protein
    "ss_domain_distance_from_membrane_edge",
    "neighborhood_distance_from_membrane_edge",
}

# ── Data loading ──────────────────────────────────────────────────────────────

def load_data(chain: str, pdb_ids: list[str], renumbered_dir: Path) -> pd.DataFrame:
    """Load and concatenate renumbered features CSVs for all PDB IDs, filtered to the specified chain."""
    frames = []
    for pdb_id in pdb_ids:
        path = renumbered_dir / f"{pdb_id}_features.csv"
        if not path.exists():
            # Missing features files are expected for some IDs; warn and move on.
            print(f"WARNING: {path.name} not found in {renumbered_dir}, skipping.", file=sys.stderr)
            continue
        df = pd.read_csv(path)
        df = df.assign(pdb_id=pdb_id)
        if chain not in df["chain"].values:
            # Do not crash on incomplete structures; only keep requested chain.
            print(f"WARNING: Chain {chain} not found in {path.name}, skipping.", file=sys.stderr)
            continue
        frames.append(df[df["chain"] == chain])
    if not frames:
        # Upstream pipeline requires at least one structure for the chain.
        sys.exit(f"No data loaded for chain {chain}.")
    return pd.concat(frames, ignore_index=True)

# ── Variability computation ───────────────────────────────────────────────────

def compute_variability(df: pd.DataFrame, metric_cols: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Compute per-residue variability metrics across structures.

    Inputs
    ------
    df : pd.DataFrame
        Metrics table with `resi_struct` and per-structure rows.
    metric_cols : list[str]
        Numeric metric columns to evaluate.

    Output
    ------
    tuple[pd.DataFrame, pd.DataFrame]
        DataFrames of per-residue standard deviation and range per metric.

    Returns
    -------
    sd_df   : DataFrame (resi × metric) of per-residue SD across structures
    range_df: DataFrame (resi × metric) of per-residue range across structures
    """
  
    sd_rows, rng_rows = {}, {}
    for col in metric_cols:
        # Align values by residue across structures to compute per-residue stats.
        pivot = df.pivot_table(index="resi_struct", columns="pdb_id", values=col)
        sd_rows[col]  = pivot.std(axis=1)
        rng_rows[col] = pivot.max(axis=1) - pivot.min(axis=1)

    sd_df  = pd.DataFrame(sd_rows).sort_index()
    rng_df = pd.DataFrame(rng_rows).sort_index()
    return sd_df, rng_df


def rank_normalise(df: pd.DataFrame) -> pd.DataFrame:
    """
    Rank-normalise each column to [0, 1] for comparable variability scores.

    Inputs
    ------
    df : pd.DataFrame
        Per-residue metric values (e.g., SDs).

    Output
    ------
    pd.DataFrame
        Rank-normalised values with the same index/columns.
    """
    n = len(df)
    # Fill NaNs so missing residues do not get dropped from ranking.
    normed = df.apply(lambda col: rankdata(col.fillna(0)) / n)
    normed.index = df.index
    return normed

# ── Plots ─────────────────────────────────────────────────────────────────────

def plot_heatmap(normed: pd.DataFrame, out_dir: Path) -> None:
    """Heatmap: residues (rows) × metrics (cols), colour = rank-norm SD."""
    # Scale figure size to data dimensions for readability.
    fig_h = max(10, len(normed.columns) * 0.18)
    fig_w = max(18, len(normed) * 0.08)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    im = ax.imshow(
        normed.T.values,
        aspect="auto",
        cmap="YlOrRd",
        vmin=0, vmax=1,
        interpolation="nearest",
    )

    ax.set_xticks(range(len(normed)))
    ax.set_xticklabels(normed.index.astype(int), fontsize=5, rotation=90)
    ax.set_yticks(range(len(normed.columns)))
    ax.set_yticklabels(normed.columns, fontsize=7)
    ax.set_xlabel("Residue number", fontsize=11)
    ax.set_title("Per-residue metric variability across structures\n(rank-normalised SD, higher = more variable)",
                 fontsize=12, fontweight="bold")

    plt.colorbar(im, ax=ax, fraction=0.01, pad=0.01, label="Rank-normalised SD")
    plt.tight_layout()
    fig.savefig(out_dir / "variability_heatmap.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  → variability_heatmap.png")


def plot_overall_score(score: pd.Series, top_n: int, out_dir: Path) -> None:
    """Bar chart of overall variability score per residue."""
    fig, ax = plt.subplots(figsize=(18, 5))

    # Highlight top-N residues in a contrasting color.
    colors = ["#C44E52" if r in score.nlargest(top_n).index else "#AEC6CF"
              for r in score.index]

    ax.bar(score.index, score.values, color=colors, width=0.8, edgecolor="none")

    # Annotate top-N residues
    for resi in score.nlargest(top_n).index:
        ax.text(resi, score[resi] + 0.003, str(int(resi)),
                ha="center", va="bottom", fontsize=6, color="#C44E52", fontweight="bold")

    ax.set_xlabel("Residue number", fontsize=11)
    ax.set_ylabel("Mean rank-normalised SD\n(across all metrics)", fontsize=10)
    ax.set_title(f"Overall residue variability score  —  top {top_n} highlighted in red",
                 fontsize=12, fontweight="bold")
    ax.xaxis.set_major_locator(ticker.MultipleLocator(10))
    ax.xaxis.set_minor_locator(ticker.MultipleLocator(1))
    ax.tick_params(labelsize=8)
    ax.grid(axis="y", lw=0.4, alpha=0.5)

    plt.tight_layout()
    fig.savefig(out_dir / "overall_variability_score.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  → overall_variability_score.png")


def plot_top_metrics(sd_df: pd.DataFrame, top_n: int, out_dir: Path) -> None:
    """For the top-10 most variable metrics, plot SD across residues."""
    # Rank metrics by mean SD across residues
    mean_sd = sd_df.mean()
    # Use a fixed top-10 view for a consistent plot grid.
    top_metrics = mean_sd.nlargest(10).index.tolist()

    fig, axes = plt.subplots(5, 2, figsize=(16, 18), sharex=True)
    axes = axes.flatten()

    for i, col in enumerate(top_metrics):
        ax = axes[i]
        ax.bar(sd_df.index, sd_df[col].fillna(0), width=0.8,
               color="#4C72B0", edgecolor="none", alpha=0.8)
        ax.set_title(col, fontsize=9, fontweight="bold")
        ax.set_ylabel("SD", fontsize=8)
        ax.tick_params(labelsize=7)
        ax.xaxis.set_major_locator(ticker.MultipleLocator(20))
        ax.grid(axis="y", lw=0.3, alpha=0.5)

    for ax in axes[len(top_metrics):]:
        ax.set_visible(False)

    axes[-2].set_xlabel("Residue number", fontsize=10)
    axes[-1].set_xlabel("Residue number", fontsize=10)

    fig.suptitle("Top 10 most variable metrics — SD per residue across structures",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    fig.savefig(out_dir / "top10_variable_metrics.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  → top10_variable_metrics.png")

# ── Main ──────────────────────────────────────────────────────────────────────


def run_variability_analysis(
    chain: str,
    pdb_ids: list[str],
    renumbered_dir: Path,
    out_dir: Path,
    top_n: int = 20,
) -> None:
    """
    Programmatic entry point: compute per-residue variability across structures and write all outputs.

    Parameters
    ----------
    chain : str
        Chain identifier to analyse (e.g. "A").
    pdb_ids : list[str]
        PDB IDs whose renumbered features CSVs to load.
    renumbered_dir : Path
        Directory containing ``{PDB_ID}_features.csv`` files.
    out_dir : Path
        Directory where CSV and PNG outputs are written.
    top_n : int
        Number of top-variable residues to highlight in plots and console summary.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    df = load_data(chain, pdb_ids, renumbered_dir)

    metric_cols = [
        c for c in df.select_dtypes(include="number").columns
        if c not in SKIP_COLS
    ]

    # Compute per-residue SD and range across structures
    sd_df, rng_df = compute_variability(df, metric_cols)

    # Drop metrics with zero variance everywhere (uninformative)
    zero_var = sd_df.columns[sd_df.max() == 0]
    if len(zero_var):
        print(f"  Dropping {len(zero_var)} zero-variance metric(s): {list(zero_var)}")
    sd_df  = sd_df.drop(columns=zero_var)
    rng_df = rng_df.drop(columns=zero_var)

    # Rank-normalise so all metrics contribute equally to the overall score
    normed = rank_normalise(sd_df)

    # Overall score = mean rank-normalised SD across all metrics
    score = normed.mean(axis=1)
    score.index = score.index.astype(int)

    # ── Save CSVs ─────────────────────────────────────────────────────────────
    sd_df.to_csv(out_dir / "per_residue_sd.csv")
    rng_df.to_csv(out_dir / "per_residue_range.csv")
    normed.to_csv(out_dir / "per_residue_normalised_sd.csv")

    score_df = score.rename("variability_score").to_frame()
    score_df["rank"] = score_df["variability_score"].rank(ascending=False).astype(int)
    score_df.index.name = "resi_struct"
    score_df.to_csv(out_dir / "residue_variability_ranking.csv")

    # Per-metric top-N table
    top_per_metric = {}
    for col in sd_df.columns:
        top_per_metric[col] = (
            sd_df[col].nlargest(top_n)
            .rename("sd")
            .to_frame()
            .assign(range=rng_df[col])
        )
    per_metric_df = pd.concat(top_per_metric, names=["metric", "resi_struct"])
    per_metric_df.to_csv(out_dir / f"top{top_n}_residues_per_metric.csv")

    # ── Plots ─────────────────────────────────────────────────────────────────
    plot_heatmap(normed, out_dir)
    plot_overall_score(score, top_n, out_dir)
    plot_top_metrics(sd_df, top_n, out_dir)

    # ── Console summary ───────────────────────────────────────────────────────
    print(f"\nTop {top_n} most variable residues (overall score):")
    print("-" * 45)
    top_residues = score_df.sort_values("rank").head(top_n)
    for resi, row in top_residues.iterrows():
        print(f"  Residue {int(resi):>4}   score={row['variability_score']:.4f}   rank={row['rank']}")

    print("\nTop 10 most variable metrics (mean SD across residues):")
    print("-" * 45)
    for col, val in sd_df.mean().nlargest(10).items():
        print(f"  {col:<55} mean SD={val:.4g}")


