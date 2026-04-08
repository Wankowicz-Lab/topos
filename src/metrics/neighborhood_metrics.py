"""
Neighborhood metrics: per-residue aggregates over neighboring residues.

Neighborhood metric functions take (context, features) and return a DataFrame
with merge columns (chain, resi_struct, resn_struct) plus new columns.
They read the neighbor mapping from context.extras['residue_neighbors'].
"""
from __future__ import annotations

import math
from collections import Counter

import pandas as pd

from src.metrics.averaging_metrics import METRICS_TO_AVERAGE
from src.metrics.secondary_structure import AA_TO_GROUP
from src.pipeline.context import Context
from src.structure.utils import res_key

STRUCT_COLS = ["chain", "resi_struct", "resn_struct"]


def count_ala_neighbors(
    context: Context,
    features: pd.DataFrame,
) -> pd.DataFrame:
    """Number of alanine residues in the neighborhood of each residue.

    Uses context.extras['residue_neighbors'] (residue_key -> [neighbor keys]).
    For each residue, subsets features to neighbor rows and counts rows
    with resn_struct == 'ALA'. Uses one row per (chain, resi_struct) when
    looking up residue type so structure-level neighbors are counted once.

    Parameters
    ----------
    context : Context
        Context with extras['residue_neighbors'] mapping.
    features : pd.DataFrame
        Merged features from Runner (must have chain, resi_struct, resn_struct).
    Returns
    -------
    pd.DataFrame
        Columns: chain, resi_struct, resn_struct, n_ala_neighbors.
    """
    neighbor_map = context.extras["residue_neighbors"]

    # One row per (chain, resi_struct, resn_struct) in features.
    unique = features[STRUCT_COLS].drop_duplicates()
    unique = unique.loc[unique.resi_struct.notna(), :]
    
    key_to_resn = dict(
        zip(
            (res_key(c, r, n) for c, r, n in zip(unique["chain"], unique["resi_struct"], unique["resn_struct"])),
            unique["resn_struct"],
        )
    )

    rows = []
    for _, row in unique.iterrows():
        chain, resi, resn = row["chain"], row["resi_struct"], row["resn_struct"]
        residue_key = res_key(chain, resi, resn)
        neighbor_keys = neighbor_map.get(residue_key, [])
        n_ala = sum(1 for k in neighbor_keys if key_to_resn.get(k) == "ALA")
        rows.append({"chain": chain, "resi_struct": resi, "resn_struct": resn, "n_ala_neighbors": n_ala})

    return pd.DataFrame(rows)


def average_neighbor_metrics(
    context: Context,
    features: pd.DataFrame,
) -> pd.DataFrame:
    """Average selected feature columns across each residue's 3D neighborhood.

    Uses context.extras['residue_neighbors'] (residue_key -> [neighbor keys]).
    Only columns listed in METRICS_TO_AVERAGE and present in features are averaged.
    Output columns are prefixed with ``neighborhood_``.

    Parameters
    ----------
    context : Context
        Context with extras['residue_neighbors'] mapping.
    features : pd.DataFrame
        Merged features from Runner (must have chain, resi_struct, resn_struct).
    Returns
    -------
    pd.DataFrame
        Columns: chain, resi_struct, resn_struct, and neighborhood_<metric> columns.
    """
    neighbor_map = context.extras["residue_neighbors"]
    present_metrics = [c for c in METRICS_TO_AVERAGE if c in features.columns]
    
    # Collapse mutation-level rows to one residue-level row by averaging each metric per residue.
    residue_level = features.loc[
        features["resi_struct"].notna(),
        STRUCT_COLS + present_metrics,
    ].copy()
    residue_level = residue_level.groupby(STRUCT_COLS, as_index=False).agg(
        {metric: "mean" for metric in present_metrics}
    )

    # Build key -> row-index lookup once so each residue can resolve neighbor rows quickly.
    key_series = pd.Series(
        [
            res_key(c, r, n)
            for c, r, n in zip(
                residue_level["chain"],
                residue_level["resi_struct"],
                residue_level["resn_struct"],
            )
        ],
        index=residue_level.index,
        dtype=object,
    )
    key_to_idx = {k: idx for idx, k in key_series.items()}

    rows = []
    for _, row in residue_level.iterrows():
        residue_key = res_key(row["chain"], row["resi_struct"], row["resn_struct"])
        neighbor_keys = neighbor_map.get(residue_key, [])
        # Ignore neighbors outside filtered/output features (e.g., structural_feature_chains).
        neighbor_indices = [key_to_idx[k] for k in neighbor_keys if k in key_to_idx]

        out_row = {
            "chain": row["chain"],
            "resi_struct": row["resi_struct"],
            "resn_struct": row["resn_struct"],
        }
        for metric in present_metrics:
            col_name = f"neighborhood_{metric}"
            if len(neighbor_indices) == 0:
                out_row[col_name] = float("nan")
            else:
                # Pandas mean(skipna=True) matches ss-domain averaging behavior.
                out_row[col_name] = residue_level.loc[neighbor_indices, metric].mean(skipna=True)
        rows.append(out_row)

    return pd.DataFrame(rows)


def _shannon_entropy(labels: list[str]) -> float:
    """Compute Shannon entropy in bits for a list of categorical labels."""
    if not labels:
        return 0.0

    counts = Counter(labels)
    total = sum(counts.values())
    return -sum((count / total) * math.log2(count / total) for count in counts.values())


def _unique_residues(features: pd.DataFrame) -> pd.DataFrame:
    """Collapse a feature table to one row per structural residue."""
    unique = features[STRUCT_COLS].drop_duplicates()
    return unique.loc[unique["resi_struct"].notna(), :].reset_index(drop=True)


def _residue_key_lookup(df: pd.DataFrame, value_col: str) -> dict[str, str]:
    """Build a residue-key lookup for a single residue-level column."""
    return dict(
        zip(
            (res_key(c, r, n) for c, r, n in zip(df["chain"], df["resi_struct"], df["resn_struct"])),
            df[value_col],
        )
    )


def _secondary_structure_coarse_from_label(secondary_structure_granular: object) -> str | None:
    """Map a full ss_domains label to helix/sheet/coil buckets."""
    if pd.isna(secondary_structure_granular):
        return None

    label = str(secondary_structure_granular)
    if label.startswith("alpha-helix_") or label.startswith("TMD_"):
        return "alpha-helix"
    if label.startswith("beta-sheet_"):
        return "beta-sheet"
    if label.startswith("coil_") or "_loop_" in label:
        return "coil"
    return None


def _residue_secondary_structure_granular_lookup(
    context: Context,
    features: pd.DataFrame,
) -> dict[str, object]:
    """Build a residue-key lookup for ss_domains when annotations are available."""
    if "ss_domains" in features.columns:
        ss_lookup = features[STRUCT_COLS + ["ss_domains"]].drop_duplicates(STRUCT_COLS)
    elif hasattr(context, "residue_table") and "ss_domains" in context.residue_table.columns:
        ss_lookup = context.residue_table[STRUCT_COLS + ["ss_domains"]].drop_duplicates(STRUCT_COLS)
    else:
        return {}

    ss_lookup = ss_lookup.loc[ss_lookup["resi_struct"].notna(), :].reset_index(drop=True)
    return _residue_key_lookup(ss_lookup, "ss_domains")


def _parse_residue_key(residue_key: str) -> tuple[str, int, str]:
    """Parse a canonical residue key into chain, residue number, and residue name."""
    chain, resi, resn = residue_key.split(":", 2)
    return chain, int(resi), resn


def neighbor_entropy_metrics(
    context: Context,
    features: pd.DataFrame,
) -> pd.DataFrame:
    """Compute neighborhood entropy over residue identities and residue groups.

    Uses context.extras['residue_neighbors'] (residue_key -> [neighbor keys]).
    For each residue, looks up neighboring residue identities, then computes
    Shannon entropy over both 3-letter amino-acid labels and AA_TO_GROUP labels.

    Parameters
    ----------
    context : Context
        Context with extras['residue_neighbors'] mapping.
    features : pd.DataFrame
        Residue-level or merged features with chain, resi_struct, resn_struct.

    Returns
    -------
    pd.DataFrame
        Columns: chain, resi_struct, resn_struct, n_neighbors,
        neighbor_aa_entropy, neighbor_aa_group_entropy.
    """
    neighbor_map = context.extras["residue_neighbors"]
    unique = _unique_residues(features)
    key_to_resn = _residue_key_lookup(unique, "resn_struct")

    rows = []
    for _, row in unique.iterrows():
        chain, resi, resn = row["chain"], row["resi_struct"], row["resn_struct"]
        residue_key = res_key(chain, resi, resn)
        neighbor_keys = neighbor_map.get(residue_key, [])
        neighbor_residues = [key_to_resn[k] for k in neighbor_keys if k in key_to_resn]
        neighbor_groups = [AA_TO_GROUP[neighbor_resn] for neighbor_resn in neighbor_residues]
        rows.append(
            {
                "chain": chain,
                "resi_struct": resi,
                "resn_struct": resn,
                "n_neighbors": len(neighbor_keys),
                "neighbor_aa_entropy": _shannon_entropy(neighbor_residues),
                "neighbor_aa_group_entropy": _shannon_entropy(neighbor_groups),
            }
        )

    return pd.DataFrame(rows)


def neighbor_sequence_range_metrics(
    context: Context,
    features: pd.DataFrame,
    long_range_threshold: int = 12,
) -> pd.DataFrame:
    """Compute same-chain sequence-range metrics for each residue's neighbors.

    Uses context.extras['residue_neighbors'] (residue_key -> [neighbor keys]).
    Only same-chain neighbors contribute. Long-range neighbors are defined by
    absolute raw resi_struct difference greater than long_range_threshold.

    Parameters
    ----------
    context : Context
        Context with extras['residue_neighbors'] mapping.
    features : pd.DataFrame
        Residue-level or merged features with chain, resi_struct, resn_struct.
    long_range_threshold : int
        Sequence-distance threshold for long-range neighbors. Defaults to 12.

    Returns
    -------
    pd.DataFrame
        Columns: chain, resi_struct, resn_struct, prop_long_range_neighbors,
        mean_neighbor_sequence_distance.
    """
    neighbor_map = context.extras["residue_neighbors"]
    unique = _unique_residues(features)

    rows = []
    for _, row in unique.iterrows():
        residue_key = res_key(row["chain"], row["resi_struct"], row["resn_struct"])
        neighbor_keys = neighbor_map.get(residue_key, [])
        _, target_resi, _ = _parse_residue_key(residue_key)

        same_chain_distances = []
        for neighbor_key in neighbor_keys:
            neighbor_chain, neighbor_resi, _ = _parse_residue_key(neighbor_key)
            if neighbor_chain != row["chain"]:
                continue
            same_chain_distances.append(abs(neighbor_resi - target_resi))

        rows.append(
            {
                "chain": row["chain"],
                "resi_struct": row["resi_struct"],
                "resn_struct": row["resn_struct"],
                "prop_long_range_neighbors": sum(
                    distance > long_range_threshold for distance in same_chain_distances
                ) / len(same_chain_distances),
                "mean_neighbor_sequence_distance": sum(same_chain_distances) / len(same_chain_distances),
            }
        )

    return pd.DataFrame(rows)


def neighbor_secondary_structure_coarse_granular_metrics(
    context: Context,
    features: pd.DataFrame,
) -> pd.DataFrame:
    """Compute neighborhood secondary-structure proportions and entropy metrics.

    Uses context.extras['residue_neighbors'] (residue_key -> [neighbor keys]).
    Neighbor ss_domains labels are mapped into alpha-helix, beta-sheet, and
    coil buckets; membrane TMD labels are treated as helix-like and loop labels
    as coil-like. Unsupported or missing ss_domains are ignored.

    Parameters
    ----------
    context : Context
        Context with residue_table and extras['residue_neighbors'] mapping.
    features : pd.DataFrame
        Residue-level or merged features with chain, resi_struct, resn_struct.

    Returns
    -------
    pd.DataFrame
        Columns: chain, resi_struct, resn_struct, three neighborhood proportion
        columns, and two entropy columns.
    """
    neighbor_map = context.extras["residue_neighbors"]
    unique = _unique_residues(features)

    residue_key_to_secondary_structure_granular = _residue_secondary_structure_granular_lookup(
        context,
        features,
    )

    rows = []
    for _, row in unique.iterrows():
        residue_key = res_key(row["chain"], row["resi_struct"], row["resn_struct"])
        neighbor_keys = neighbor_map.get(residue_key, [])

        secondary_structure_coarse_labels = []
        secondary_structure_granular_labels = []
        for neighbor_key in neighbor_keys:
            if neighbor_key not in residue_key_to_secondary_structure_granular:
                continue
            secondary_structure_granular = residue_key_to_secondary_structure_granular[neighbor_key]
            secondary_structure_coarse = _secondary_structure_coarse_from_label(
                secondary_structure_granular
            )
            if secondary_structure_coarse is None:
                continue
            secondary_structure_coarse_labels.append(secondary_structure_coarse)
            secondary_structure_granular_labels.append(str(secondary_structure_granular))

        secondary_structure_coarse_counts = Counter(secondary_structure_coarse_labels)
        n_secondary_structure_coarse_neighbors = len(secondary_structure_coarse_labels)
        if n_secondary_structure_coarse_neighbors == 0:
            prop_alpha_helix = float("nan")
            prop_beta_sheet = float("nan")
            prop_coil = float("nan")
        else:
            prop_alpha_helix = (
                secondary_structure_coarse_counts["alpha-helix"] / n_secondary_structure_coarse_neighbors
            )
            prop_beta_sheet = (
                secondary_structure_coarse_counts["beta-sheet"] / n_secondary_structure_coarse_neighbors
            )
            prop_coil = secondary_structure_coarse_counts["coil"] / n_secondary_structure_coarse_neighbors

        rows.append(
            {
                "chain": row["chain"],
                "resi_struct": row["resi_struct"],
                "resn_struct": row["resn_struct"],
                "neighbor_prop_alpha_helix": prop_alpha_helix,
                "neighbor_prop_beta_sheet": prop_beta_sheet,
                "neighbor_prop_coil": prop_coil,
                "secondary_structure_coarse_entropy": _shannon_entropy(secondary_structure_coarse_labels),
                "secondary_structure_granular_entropy": _shannon_entropy(secondary_structure_granular_labels),
            }
        )

    return pd.DataFrame(rows)


NEIGHBORHOOD_METRIC_FUNCTIONS = [
    count_ala_neighbors,
    average_neighbor_metrics,
    neighbor_sequence_range_metrics,
    neighbor_secondary_structure_coarse_granular_metrics,
]
