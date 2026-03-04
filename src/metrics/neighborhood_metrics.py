"""
Neighborhood metrics: per-residue aggregates over neighboring residues.

Neighborhood metric functions take (context, features) and return a DataFrame
with merge columns (chain, resi_struct, resn_struct) plus new columns.
They read the neighbor mapping from context.extras['residue_neighbors'].
"""
from __future__ import annotations

import pandas as pd

from src.metrics.averaging_metrics import METRICS_TO_AVERAGE
from src.pipeline.context import Context
from src.structure.utils import res_key

RESIDUE_NEIGHBORS_KEY = "residue_neighbors"


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
    neighbor_map = context.extras[RESIDUE_NEIGHBORS_KEY]

    struct_cols = ["chain", "resi_struct", "resn_struct"]
    missing_struct = [c for c in struct_cols if c not in features.columns]
    if missing_struct:
        raise ValueError(
            f"Missing required structural columns for neighborhood metrics: {missing_struct}"
        )

    # One row per (chain, resi_struct, resn_struct) in features
    unique = features[struct_cols].drop_duplicates()
    unique = unique.loc[unique.resi_struct.notna(), :]
    if unique.empty:
        raise ValueError(
            "No structure-level rows found after filtering; upstream feature merge likely failed."
        )
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
    neighbor_map = context.extras[RESIDUE_NEIGHBORS_KEY]
    struct_cols = ["chain", "resi_struct", "resn_struct"]
    missing_struct = [c for c in struct_cols if c not in features.columns]
    if missing_struct:
        raise ValueError(
            f"Missing required structural columns for neighborhood metrics: {missing_struct}"
        )

    present_metrics = [c for c in METRICS_TO_AVERAGE if c in features.columns]
    if not present_metrics:
        raise ValueError(
            "No configured averaging metrics found in features; run expected upstream metrics first."
        )
    
    # Collapse to one row per residue identity before neighbor traversal.
    # This preserves many_to_one merging on (chain, resi_struct, resn_struct).
    unique = features[struct_cols + present_metrics].drop_duplicates(subset=struct_cols)
    unique = unique.loc[unique.resi_struct.notna(), :].copy()
    if unique.empty:
        raise ValueError(
            "No structure-level rows found after filtering; upstream feature merge likely failed."
        )

    # Build key -> row-index lookup once so each residue can resolve neighbor rows quickly.
    key_series = pd.Series(
        [res_key(c, r, n) for c, r, n in zip(unique["chain"], unique["resi_struct"], unique["resn_struct"])],
        index=unique.index,
        dtype=object,
    )
    key_to_idx = {k: idx for idx, k in key_series.items()}

    rows = []
    for _, row in unique.iterrows():
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
                out_row[col_name] = unique.loc[neighbor_indices, metric].mean(skipna=True)
        rows.append(out_row)

    return pd.DataFrame(rows)


NEIGHBORHOOD_METRIC_FUNCTIONS = [count_ala_neighbors, average_neighbor_metrics]
