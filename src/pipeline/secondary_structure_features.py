from typing import List, Optional

import pandas as pd

from src.metrics.averaging_metrics import METRICS_TO_AVERAGE
from src.metrics.secondary_structure import ss_domain_lengths, ss_domain_log2_aa_group_ratios
from src.pipeline.context import Context

NONSYN_DOMAIN_COLUMNS = {
    "avg_effect",
    "effect_variance",
    "effect_variance_rank",
    "effect",
    "effect_ranking",
}


def calculate_secondary_structure_features(
    context: Context,
    features: pd.DataFrame,
    ss_metrics: Optional[List[str]] = None,
) -> pd.DataFrame:
    """Aggregate residue-level features onto secondary-structure domains."""
    metrics_to_avg = ss_metrics if ss_metrics is not None else METRICS_TO_AVERAGE
    merge_cols = ["chain", "resi_struct", "resn_struct"]

    rt_subset = context.residue_table[merge_cols + ["ss_domains"]].drop_duplicates(merge_cols)
    merged = pd.merge(features, rt_subset, on=merge_cols, how="left")
    merged = merged.dropna(subset=["ss_domains"])

    # Exclude synonymous rows from calculations when mutation type context is present.
    if "type" in merged.columns:
        synonymous_mask = merged["type"].eq("synonymous")
        for column in NONSYN_DOMAIN_COLUMNS:
            if column in merged.columns:
                merged.loc[synonymous_mask, column] = float("nan")

    cols_to_avg = [column for column in metrics_to_avg if column in merged.columns]

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
