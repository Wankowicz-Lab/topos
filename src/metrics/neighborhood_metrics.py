"""
Neighborhood metrics: per-residue aggregates over neighboring residues.

Neighborhood metric functions take (context, features) and return a DataFrame
with merge columns (chain, resi_struct, resn_struct) plus new columns.
They read the neighbor mapping from context.extras['residue_neighbors'].
"""
from __future__ import annotations

import pandas as pd

from src.pipeline.context import Context


def count_ala_neighbors(
    context: Context, features: pd.DataFrame, extras_key: str = "residue_neighbors"
) -> pd.DataFrame:
    """Number of alanine residues in the neighborhood of each residue.

    Uses context.extras[extras_key] (residue_key -> [neighbor keys]).
    For each residue, subsets features to neighbor rows and counts rows
    with resn_struct == 'ALA'. Uses one row per (chain, resi_struct) when
    looking up residue type so structure-level neighbors are counted once.

    Parameters
    ----------
    context : Context
        Context with extras[extras_key] mapping.
    features : pd.DataFrame
        Merged features from Runner (must have chain, resi_struct, resn_struct).
    extras_key : str, optional
        Key in context.extras for the neighbor mapping. Default 'residue_neighbors'.

    Returns
    -------
    pd.DataFrame
        Columns: chain, resi_struct, resn_struct, n_ala_neighbors.
    """
    neighbor_map = context.extras.get(extras_key, {})

    struct_cols = ["chain", "resi_struct", "resn_struct"]
    if not all(c in features.columns for c in struct_cols):
        return pd.DataFrame(columns=struct_cols + ["n_ala_neighbors"])

    # One row per (chain, resi_struct) in features
    unique = features[struct_cols].drop_duplicates()
    key_to_resn = dict(
        zip(
            (f"{c}:{r}" for c, r in zip(features["chain"], features["resi_struct"])),
            features["resn_struct"],
        )
    )

    rows = []
    for _, row in unique.iterrows():
        chain, resi, resn = row["chain"], row["resi_struct"], row["resn_struct"]
        res_key = f"{chain}:{int(resi)}"
        neighbor_keys = neighbor_map.get(res_key, [])
        n_ala = sum(1 for k in neighbor_keys if key_to_resn.get(k) == "ALA")
        rows.append({"chain": chain, "resi_struct": resi, "resn_struct": resn, "n_ala_neighbors": n_ala})

    return pd.DataFrame(rows)


NEIGHBORHOOD_METRIC_FUNCTIONS = [count_ala_neighbors]
