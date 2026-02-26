"""
Per-residue statistical comparison between groups.

Computes Cohen's d and Mann-Whitney U for each metric at each residue.
"""
from __future__ import annotations

import warnings
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats

from .config import (
    EXCLUDED_COLUMNS,
    METRIC_CATEGORY_PATTERNS,
    MetricsConfig,
)


def _col_matches_pattern(col: str, patterns: list[tuple[str, str]]) -> bool:
    for match_type, pattern in patterns:
        if match_type == "exact" and col == pattern:
            return True
        if match_type == "prefix" and col.startswith(pattern):
            return True
        if match_type == "suffix" and col.endswith(pattern):
            return True
        if match_type == "contains" and pattern in col:
            return True
    return False


def select_metric_columns(df: pd.DataFrame, metrics: MetricsConfig) -> List[str]:
    """
    Return the list of numeric columns to include in the comparison.

    If ``metrics.custom_columns`` is set, those are returned directly
    (filtered to columns actually present in df).

    Otherwise, columns are selected by ``include_categories``, then
    ``exclude_columns`` and the global ``EXCLUDED_COLUMNS`` are removed.
    """
    if metrics.custom_columns is not None:
        present = set(df.columns)
        cols = [c for c in metrics.custom_columns if c in present]
        return cols

    categories = metrics.include_categories
    use_all = "all" in categories

    selected: list[str] = []
    for col in df.columns:
        if col in EXCLUDED_COLUMNS:
            continue
        if col in metrics.exclude_columns:
            continue
        if not pd.api.types.is_numeric_dtype(df[col]):
            continue

        if use_all:
            selected.append(col)
            continue

        for cat in categories:
            if cat not in METRIC_CATEGORY_PATTERNS:
                continue
            if _col_matches_pattern(col, METRIC_CATEGORY_PATTERNS[cat]):
                selected.append(col)
                break

    return selected


def _cohens_d(a: np.ndarray, b: np.ndarray) -> float:
    """Signed Cohen's d: (mean_b - mean_a) / pooled_std."""
    n_a, n_b = len(a), len(b)
    if n_a < 2 or n_b < 2:
        return np.nan
    var_a = np.var(a, ddof=1)
    var_b = np.var(b, ddof=1)
    pooled_std = np.sqrt(((n_a - 1) * var_a + (n_b - 1) * var_b) / (n_a + n_b - 2))
    if pooled_std == 0:
        return np.nan
    return (np.mean(b) - np.mean(a)) / pooled_std


def _mann_whitney_p(a: np.ndarray, b: np.ndarray) -> float:
    """Mann-Whitney U p-value; NaN if n < 3 per group."""
    if len(a) < 3 or len(b) < 3:
        return np.nan
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = stats.mannwhitneyu(a, b, alternative="two-sided")
        return float(result.pvalue)
    except Exception:
        return np.nan


def compare_two_groups(
    df_long: pd.DataFrame,
    metric_cols: List[str],
    group_a: str,
    group_b: str,
    group_col: str = "_group",
    residue_key_cols: List[str] = None,
) -> pd.DataFrame:
    """
    Compare group_a vs group_b at each residue for each metric column.

    For each residue × metric:
        {metric}_{groupA}_mean, {metric}_{groupA}_std
        {metric}_{groupB}_mean, {metric}_{groupB}_std
        {metric}_diff            (groupB_mean - groupA_mean)
        {metric}_cohens_d        (signed Cohen's d)
        {metric}_pvalue          (Mann-Whitney U two-sided)

    Summary columns per residue:
        mean_abs_cohens_d        mean |Cohen's d| across all metrics
        n_large_effect           count of metrics with |Cohen's d| > 0.8
        top_metric               metric name with largest |Cohen's d|
    """
    if residue_key_cols is None:
        residue_key_cols = ["chain", "resi_struct", "resn_struct"]

    # Use whichever key cols are actually present
    key_cols = [c for c in residue_key_cols if c in df_long.columns]

    df_a = df_long[df_long[group_col] == group_a]
    df_b = df_long[df_long[group_col] == group_b]

    # Get all unique residues across both groups
    all_residues = pd.concat([
        df_a[key_cols].drop_duplicates(),
        df_b[key_cols].drop_duplicates(),
    ]).drop_duplicates().reset_index(drop=True)

    records = []
    for _, res_row in all_residues.iterrows():
        mask_a = pd.Series([True] * len(df_a), index=df_a.index)
        mask_b = pd.Series([True] * len(df_b), index=df_b.index)
        for col in key_cols:
            mask_a = mask_a & (df_a[col] == res_row[col])
            mask_b = mask_b & (df_b[col] == res_row[col])

        rows_a = df_a[mask_a]
        rows_b = df_b[mask_b]

        record: dict = {c: res_row[c] for c in key_cols}
        cohens_d_vals: list[float] = []

        for metric in metric_cols:
            vals_a = rows_a[metric].dropna().values.astype(float) if metric in rows_a.columns else np.array([])
            vals_b = rows_b[metric].dropna().values.astype(float) if metric in rows_b.columns else np.array([])

            mean_a = float(np.mean(vals_a)) if len(vals_a) > 0 else np.nan
            std_a = float(np.std(vals_a, ddof=1)) if len(vals_a) > 1 else np.nan
            mean_b = float(np.mean(vals_b)) if len(vals_b) > 0 else np.nan
            std_b = float(np.std(vals_b, ddof=1)) if len(vals_b) > 1 else np.nan

            diff = (mean_b - mean_a) if not (np.isnan(mean_a) or np.isnan(mean_b)) else np.nan
            cd = _cohens_d(vals_a, vals_b)
            pval = _mann_whitney_p(vals_a, vals_b)

            a_tag = group_a.replace(" ", "_")
            b_tag = group_b.replace(" ", "_")

            record[f"{metric}_{a_tag}_mean"] = mean_a
            record[f"{metric}_{a_tag}_std"] = std_a
            record[f"{metric}_{b_tag}_mean"] = mean_b
            record[f"{metric}_{b_tag}_std"] = std_b
            record[f"{metric}_diff"] = diff
            record[f"{metric}_cohens_d"] = cd
            record[f"{metric}_pvalue"] = pval

            if not np.isnan(cd):
                cohens_d_vals.append(abs(cd))

        # Summary columns
        if cohens_d_vals:
            record["mean_abs_cohens_d"] = float(np.mean(cohens_d_vals))
            record["n_large_effect"] = int(sum(v > 0.8 for v in cohens_d_vals))
            # top_metric: metric with max |Cohen's d|
            abs_cds = {
                m: abs(record.get(f"{m}_cohens_d", np.nan))
                for m in metric_cols
            }
            abs_cds = {k: v for k, v in abs_cds.items() if not np.isnan(v)}
            record["top_metric"] = max(abs_cds, key=abs_cds.get) if abs_cds else None
        else:
            record["mean_abs_cohens_d"] = np.nan
            record["n_large_effect"] = 0
            record["top_metric"] = None

        records.append(record)

    return pd.DataFrame(records)


def compare_all_vs_all(
    df_long: pd.DataFrame,
    metric_cols: List[str],
    label_col: str = "_label",
    residue_key_cols: List[str] = None,
) -> Dict[Tuple[str, str], pd.DataFrame]:
    """
    Pairwise comparison of all label pairs.

    Returns a dict keyed by (label_A, label_B) where label_A < label_B.
    Each value is the output of compare_two_groups using _group = _label.
    """
    if residue_key_cols is None:
        residue_key_cols = ["chain", "resi_struct", "resn_struct"]

    labels = sorted(df_long[label_col].unique())
    results: Dict[Tuple[str, str], pd.DataFrame] = {}

    # Temporarily alias _label to _group for compare_two_groups
    df_tmp = df_long.copy()
    df_tmp["_pairwise_group"] = df_tmp[label_col]

    for i, la in enumerate(labels):
        for lb in labels[i + 1 :]:
            pair_df = df_tmp[df_tmp["_pairwise_group"].isin([la, lb])].copy()
            result = compare_two_groups(
                pair_df,
                metric_cols=metric_cols,
                group_a=la,
                group_b=lb,
                group_col="_pairwise_group",
                residue_key_cols=residue_key_cols,
            )
            results[(la, lb)] = result

    return results
