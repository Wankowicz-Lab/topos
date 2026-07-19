from __future__ import annotations

from typing import Iterable

import pandas as pd
from pandas.api.types import is_bool_dtype, is_numeric_dtype

from topos.metrics.averaging_metrics import column_needs_synonym_mask, spatial_pool_metric_columns
from topos.pipeline.context import Context

SEQUENCE_MERGE_COLS = ["chain", "resi_mut", "resn_mut"]


def calculate_sequence_window_features(
    context: Context,
    features: pd.DataFrame,
    seq_metric_columns: Iterable[str],
    window_size: int = 5,
) -> pd.DataFrame:
    """Average numeric sequence-derived features across a centered sequence window.

    Only columns both sequence-tagged and in ``spatial_pool_metric_columns(features)``
    (neighbor / SS-domain pooling eligibility) enter the rolling step.

    Parameters
    ----------
    context : Context
        Pipeline context containing ``residue_table`` with sequence-position ordering
        in ``align_pos``.
    features : pd.DataFrame
        Merged runner feature table. When mutation data is present, this may contain
        multiple rows per residue, one for each mutation.
    seq_metric_columns : Iterable[str]
        Concrete feature columns produced by sequence-tagged metrics during
        ``Runner.run_metrics()``.
    window_size : int
        Size of the centered residue window used for the rolling mean.

    Returns
    -------
    pd.DataFrame
        One row per residue with ``chain``, ``resi_mut``, ``resn_mut``, and
        ``sequence_window_*`` columns for numeric sequence-derived metrics.

    Notes
    -----
    Mutation-level rows are first collapsed to one residue-level value by averaging
    each numeric sequence metric. The rolling mean is then computed in ``align_pos``
    order within each chain, using only residues that already have at least one
    numeric sequence metric value.
    """
    seq_metric_columns = list(dict.fromkeys(seq_metric_columns))
    eligible = set(spatial_pool_metric_columns(features))
    # Numeric sequence outputs that are also spatial-pooling-eligible; skip labels/bool.
    numeric_cols = [
        column
        for column in seq_metric_columns
        if column in features.columns
        and column in eligible
        and is_numeric_dtype(features[column])
        and not is_bool_dtype(features[column])
    ]
    if not numeric_cols:
        return pd.DataFrame(columns=SEQUENCE_MERGE_COLS)

    # Order rows by aligned sequence position within each chain.
    align_pos = (
        context.residue_table[SEQUENCE_MERGE_COLS + ["align_pos"]]
        .drop_duplicates(SEQUENCE_MERGE_COLS)
        .reset_index(drop=True)
    )
    mutation_level_cols = SEQUENCE_MERGE_COLS + numeric_cols + (["type"] if "type" in features.columns else [])
    mutation_level = pd.merge(
        features[mutation_level_cols],
        align_pos,
        on=SEQUENCE_MERGE_COLS,
        how="left",
        validate="many_to_one",
    )
    # Mutation-derived metrics: nan out synonymous rows before averaging across variants.
    if "type" in mutation_level.columns:
        synonymous_mask = mutation_level["type"].eq("synonymous")
        for column in numeric_cols:
            if column_needs_synonym_mask(column):
                mutation_level.loc[synonymous_mask, column] = float("nan")
        mutation_level = mutation_level.drop(columns=["type"])
    # One residue-level value per (chain, resi_mut, resn_mut, align_pos): mean across resm variants.
    residue_level = mutation_level.groupby(
        SEQUENCE_MERGE_COLS + ["align_pos"],
        as_index=False,
    ).agg({column: "mean" for column in numeric_cols})

    # Omit residues that have no numeric sequence metric left after synonym handling.
    residue_level = residue_level.loc[residue_level[numeric_cols].notna().any(axis=1), :]
    if residue_level.empty:
        return pd.DataFrame(columns=SEQUENCE_MERGE_COLS)

    residue_level = residue_level.sort_values(["chain", "align_pos"], kind="mergesort").reset_index(drop=True)
    # Centered window along alignment order within each chain.
    for column in numeric_cols:
        residue_level[f"sequence_window_{column}"] = (
            residue_level.groupby("chain")[column]
            .transform(
                lambda values: values.rolling(
                    window=window_size,
                    center=True,
                    min_periods=1,
                ).mean()
            )
        )

    window_cols = [f"sequence_window_{column}" for column in numeric_cols]
    return residue_level[SEQUENCE_MERGE_COLS + window_cols]
