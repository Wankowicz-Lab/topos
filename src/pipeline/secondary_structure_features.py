import pandas as pd

from src.metrics.averaging_metrics import (
    assert_poolable_numeric_columns,
    column_needs_synonym_mask,
    spatial_pool_metric_columns,
)
from src.metrics.secondary_structure import ss_domain_lengths, ss_domain_log2_aa_group_ratios
from src.pipeline.context import Context


def calculate_secondary_structure_features(
    context: Context,
    features: pd.DataFrame,
) -> pd.DataFrame:
    """Aggregate residue-level features onto secondary-structure domains."""
    merge_cols = ["chain", "resi_struct", "resn_struct"]

    rt_subset = context.residue_table[merge_cols + ["ss_domains"]].drop_duplicates(merge_cols)
    merged = pd.merge(features, rt_subset, on=merge_cols, how="left")
    merged = merged.dropna(subset=["ss_domains"])

    cols_to_avg = [column for column in spatial_pool_metric_columns(features) if column in merged.columns]
    assert_poolable_numeric_columns(cols_to_avg, merged)

    if "type" in merged.columns:
        synonymous_mask = merged["type"].eq("synonymous")
        for column in cols_to_avg:
            if column_needs_synonym_mask(column):
                merged.loc[synonymous_mask, column] = float("nan")

    residue_level_cols = merge_cols + ["ss_domains"] + cols_to_avg
    if cols_to_avg:
        residue_level = merged[residue_level_cols].groupby(
            merge_cols + ["ss_domains"],
            as_index=False,
        ).agg({column: "mean" for column in cols_to_avg})
    else:
        residue_level = merged[merge_cols + ["ss_domains"]].drop_duplicates()

    agg_dict = {column: "mean" for column in cols_to_avg}
    if cols_to_avg:
        by_domain = residue_level.groupby(["chain", "ss_domains"], as_index=False).agg(agg_dict)
    else:
        by_domain = residue_level[["chain", "ss_domains"]].drop_duplicates()
    by_domain = by_domain.rename(columns={column: f"ss_domain_{column}" for column in cols_to_avg})

    lengths = ss_domain_lengths(residue_level)
    by_domain = by_domain.merge(lengths, on=["chain", "ss_domains"], how="left")
    log2_df = ss_domain_log2_aa_group_ratios(residue_level)
    by_domain = by_domain.merge(log2_df, on=["chain", "ss_domains"], how="left")

    rt_subset = rt_subset.dropna(subset=["ss_domains"])
    return rt_subset.merge(by_domain, on=["chain", "ss_domains"], how="left")
