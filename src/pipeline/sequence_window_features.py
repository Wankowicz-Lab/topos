from __future__ import annotations

from typing import Iterable

import pandas as pd
from pandas.api.types import is_bool_dtype, is_numeric_dtype

from src.pipeline.context import Context

SEQUENCE_MERGE_COLS = ["chain", "resi_mut", "resn_mut"]


def calculate_sequence_window_features(
    context: Context,
    features: pd.DataFrame,
    seq_metric_columns: Iterable[str],
    window_size: int = 5,
) -> pd.DataFrame:
    """Average numeric sequence-derived features across a centered sequence window.

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
    # Limit the rolling step to realized numeric sequence metrics and skip labels.
    numeric_cols = [
        column
        for column in seq_metric_columns
        if column in features.columns
        and is_numeric_dtype(features[column])
        and not is_bool_dtype(features[column])
    ]
    if not numeric_cols:
        return pd.DataFrame(columns=SEQUENCE_MERGE_COLS)

    # Align sequence metrics to a unique residue ordering before collapsing mutation rows.
    align_pos = (
        context.residue_table[SEQUENCE_MERGE_COLS + ["align_pos"]]
        .drop_duplicates(SEQUENCE_MERGE_COLS)
        .reset_index(drop=True)
    )
    residue_level = pd.merge(
        features[SEQUENCE_MERGE_COLS + numeric_cols],
        align_pos,
        on=SEQUENCE_MERGE_COLS,
        how="left",
        validate="many_to_one",
    )
    # Residue-level inputs for the window come from the mean across mutation rows.
    residue_level = residue_level.groupby(
        SEQUENCE_MERGE_COLS + ["align_pos"],
        as_index=False,
    ).agg({column: "mean" for column in numeric_cols})

    # Exclude residues with no numeric sequence metrics from the rolling series.
    residue_level = residue_level.loc[residue_level[numeric_cols].notna().any(axis=1), :]
    if residue_level.empty:
        return pd.DataFrame(columns=SEQUENCE_MERGE_COLS)

    # Apply the centered rolling mean independently within each chain.
    residue_level = residue_level.sort_values(["chain", "align_pos"], kind="mergesort").reset_index(drop=True)
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
