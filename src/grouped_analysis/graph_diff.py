"""
Community-based graph comparison using co-community matrices.

Avoids the label instability of community IDs by comparing *co-membership*:
M[i,j] = 1 if residues i and j are in the same community in a given structure.
"""
from __future__ import annotations

import logging
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.stats import entropy as scipy_entropy

logger = logging.getLogger(__name__)

GRAPH_TYPES = ["all", "vdw_contact", "hbond"]


def _community_col(graph_type: str) -> str:
    return f"graph_{graph_type}_graph_community_id"


def _is_nan_val(val) -> bool:
    try:
        return np.isnan(float(val))
    except (TypeError, ValueError):
        return val is None or str(val) == "nan"


def _build_matrix_from_ids(community_ids) -> np.ndarray:
    """Internal: build (N×N) boolean matrix from a sequence of community IDs."""
    n = len(community_ids)
    ids = [
        f"__nan_{i}__" if _is_nan_val(c) else str(c)
        for i, c in enumerate(community_ids)
    ]
    mat = np.zeros((n, n), dtype=bool)
    for i in range(n):
        for j in range(i, n):
            same = ids[i] == ids[j]
            mat[i, j] = same
            mat[j, i] = same
    return mat


def build_co_community_matrix(
    df: pd.DataFrame,
    graph_type: str,
    residue_keys: List[str],
) -> Optional[np.ndarray]:
    """
    Build an (N × N) boolean co-community matrix for a single structure's DataFrame.

    Rows of ``df`` must already be sorted in the desired residue order.
    M[i,j] = True iff residues i and j share the same community_id.
    NaN values are treated as unique singletons.

    Returns None if the community column is absent or all-NaN.
    """
    col = _community_col(graph_type)
    if col not in df.columns:
        return None

    community_ids = df[col].values
    if all(_is_nan_val(v) for v in community_ids):
        return None

    return _build_matrix_from_ids(community_ids)


def _group_co_community_mean(
    df_long: pd.DataFrame,
    graph_type: str,
    group: str,
    sorted_residues: pd.DataFrame,
    key_cols: List[str],
    group_col: str = "_group",
    label_col: str = "_label",
) -> Optional[np.ndarray]:
    """
    Build the mean co-community matrix across all structures in a group.

    ``sorted_residues`` is a DataFrame of unique (key_cols) rows in consistent order.
    Returns None if no valid matrices could be built.
    """
    com_col = _community_col(graph_type)
    if com_col not in df_long.columns:
        return None

    group_df = df_long[df_long[group_col] == group]
    if label_col in group_df.columns:
        structure_labels = group_df[label_col].unique()
    else:
        structure_labels = [None]

    n = len(sorted_residues)
    mats = []

    for lbl in structure_labels:
        if lbl is not None:
            struct_df = group_df[group_df[label_col] == lbl]
        else:
            struct_df = group_df

        # Build community_id vector aligned to sorted_residues order
        community_vec = np.full(n, np.nan, dtype=object)
        for idx_r, res_row in sorted_residues.iterrows():
            mask = pd.Series([True] * len(struct_df), index=struct_df.index)
            for c in key_cols:
                mask = mask & (struct_df[c] == res_row[c])
            matched = struct_df[mask]
            if len(matched) > 0 and com_col in matched.columns:
                community_vec[idx_r] = matched[com_col].iloc[0]

        # Only build matrix if not all NaN
        if all(_is_nan_val(v) for v in community_vec):
            continue

        mat = _build_matrix_from_ids(community_vec)
        mats.append(mat)

    if not mats:
        return None

    return np.mean(np.stack(mats, axis=0).astype(float), axis=0)


def community_change_scores(
    df_long: pd.DataFrame,
    groups: Tuple[str, str],
    graph_types: Optional[List[str]] = None,
    residue_key_cols: Optional[List[str]] = None,
    group_col: str = "_group",
    label_col: str = "_label",
) -> pd.DataFrame:
    """
    Compute per-residue community change scores between two groups.

    For each residue r and graph type t:
        community_change_score_{t}   L1 norm of row diff / N (0–1 scale)
        community_entropy_{t}        normalized Shannon entropy of community_id across all structures
        community_switches_{t}       True if majority community differs between groups (n≥2 per group)
    Summary:
        pathway_instability_score    mean community_change_score across graph types

    Returns a DataFrame with one row per residue.
    """
    if graph_types is None:
        graph_types = GRAPH_TYPES
    if residue_key_cols is None:
        residue_key_cols = ["chain", "resi_struct", "resn_struct"]

    group_a, group_b = groups

    # Build aligned, sorted residue list
    key_cols = [c for c in residue_key_cols if c in df_long.columns]
    sorted_residues = (
        df_long[key_cols].drop_duplicates().sort_values(key_cols).reset_index(drop=True)
    )
    n = len(sorted_residues)

    # Pre-compute mean co-community matrices for each group × graph_type
    mean_mats: dict[tuple, Optional[np.ndarray]] = {}
    for gt in graph_types:
        mean_mats[(group_a, gt)] = _group_co_community_mean(
            df_long, gt, group_a, sorted_residues, key_cols, group_col, label_col
        )
        mean_mats[(group_b, gt)] = _group_co_community_mean(
            df_long, gt, group_b, sorted_residues, key_cols, group_col, label_col
        )

    results = []

    for res_idx, res_row in sorted_residues.iterrows():
        record: dict = {c: res_row[c] for c in key_cols}
        change_scores: list[float] = []

        for gt in graph_types:
            com_col = _community_col(gt)

            # --- Community entropy across all structures ---
            if com_col in df_long.columns:
                mask = pd.Series([True] * len(df_long), index=df_long.index)
                for col in key_cols:
                    mask = mask & (df_long[col] == res_row[col])
                all_ids = df_long[mask][com_col].dropna().astype(str).values

                if len(all_ids) > 0:
                    _, counts = np.unique(all_ids, return_counts=True)
                    probs = counts / counts.sum()
                    raw_entropy = float(scipy_entropy(probs))
                    max_entropy = np.log(len(counts)) if len(counts) > 1 else 1.0
                    norm_entropy = raw_entropy / max_entropy if max_entropy > 0 else 0.0
                else:
                    norm_entropy = np.nan

                record[f"community_entropy_{gt}"] = norm_entropy

                # --- Community switches (n≥2 per group) ---
                mask_a = mask & (df_long[group_col] == group_a)
                mask_b = mask & (df_long[group_col] == group_b)
                ids_a = df_long[mask_a][com_col].dropna().astype(str).values
                ids_b = df_long[mask_b][com_col].dropna().astype(str).values

                if len(ids_a) >= 2 and len(ids_b) >= 2:
                    def majority(arr):
                        vals, cnts = np.unique(arr, return_counts=True)
                        return vals[np.argmax(cnts)]
                    record[f"community_switches_{gt}"] = bool(majority(ids_a) != majority(ids_b))
                else:
                    record[f"community_switches_{gt}"] = None
            else:
                record[f"community_entropy_{gt}"] = np.nan
                record[f"community_switches_{gt}"] = None

            # --- Co-community change score ---
            mat_a = mean_mats[(group_a, gt)]
            mat_b = mean_mats[(group_b, gt)]

            if (
                mat_a is not None
                and mat_b is not None
                and mat_a.shape == mat_b.shape
                and res_idx < mat_a.shape[0]
            ):
                row_diff = np.abs(mat_b[res_idx, :] - mat_a[res_idx, :])
                change_score = float(row_diff.sum() / n)
                record[f"community_change_score_{gt}"] = change_score
                change_scores.append(change_score)
            else:
                record[f"community_change_score_{gt}"] = np.nan

        # --- Pathway instability score ---
        record["pathway_instability_score"] = (
            float(np.mean(change_scores)) if change_scores else np.nan
        )
        results.append(record)

    return pd.DataFrame(results)
