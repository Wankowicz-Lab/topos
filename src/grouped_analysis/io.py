"""
Load, validate, and align multiple _features.csv files for group comparison.
"""
from __future__ import annotations

import logging
import warnings
from typing import Dict, List

import pandas as pd

from .config import ComparisonConfig, StructureEntry

logger = logging.getLogger(__name__)

# Structural columns to use when deduplicating DMS-style CSVs
_STRUCTURAL_PREFIXES = (
    "sasa", "distance_", "kyte_", "packing_",
    "distance_to_center_of_mass", "distance_from_membrane_edge",
    "distance_to_nearest_surface_residue",
    "salt_bridge_count", "ionic_bond_count", "disulfide_bond_count",
    "pi_stacking_count", "cation_pi_count", "vdw_contact_count",
    "bb_hbond_count", "sc_hbond_count", "total_hbond_count",
    "n_ala_neighbors",
    "graph_all_", "graph_vdw_", "graph_hbond_",
    "ligand_",
    "ss_domain_",
)


def _is_structural_col(col: str) -> bool:
    """Return True if this column holds a per-residue structural metric."""
    return any(col.startswith(p) for p in _STRUCTURAL_PREFIXES)


def load_features(entry: StructureEntry) -> pd.DataFrame:
    """
    Load a *_features.csv produced by Runner.

    If the file is DMS-style (multiple rows per residue), deduplicate to one row
    per (chain, resi_struct) by taking the mean of structural metric columns.

    Adds ``_label`` and ``_group`` columns from the entry.
    """
    df = pd.read_csv(entry.path, low_memory=False)

    required = {"chain", "resi_struct"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"Features file {entry.path} missing required columns: {missing}"
        )

    key_cols = ["chain", "resi_struct"]

    # Detect DMS-style: multiple rows per (chain, resi_struct)
    if df.duplicated(subset=key_cols).any():
        logger.debug(
            "%s: DMS-style CSV detected — deduplicating to one row per residue",
            entry.label,
        )
        structural_cols = [c for c in df.columns if _is_structural_col(c)]
        meta_cols = [c for c in df.columns if c not in structural_cols and c not in key_cols]

        # Mean of numeric structural columns; first value of metadata columns
        agg_dict: dict = {c: "mean" for c in structural_cols if pd.api.types.is_numeric_dtype(df[c])}
        for c in meta_cols:
            agg_dict[c] = "first"

        df = df.groupby(key_cols, as_index=False).agg(agg_dict)

    df["_label"] = entry.label
    df["_group"] = entry.group
    return df


def validate_compatibility(dfs: Dict[str, pd.DataFrame]) -> None:
    """
    Warn if residue sets differ; raise if no residues are shared across all structures.
    """
    if not dfs:
        return

    key_cols = ["chain", "resi_struct"]
    residue_sets: dict[str, set] = {}
    for label, df in dfs.items():
        residue_sets[label] = set(
            zip(df["chain"].astype(str), df["resi_struct"].astype(str))
        )

    labels = list(residue_sets.keys())
    shared = residue_sets[labels[0]].copy()
    for lbl in labels[1:]:
        shared &= residue_sets[lbl]

    if not shared:
        raise ValueError(
            "No residues are shared across all structures. "
            "Check that all structures have the same sequence."
        )

    for lbl, rset in residue_sets.items():
        only_here = rset - shared
        if only_here:
            warnings.warn(
                f"Structure '{lbl}' has {len(only_here)} residue(s) not present "
                f"in all other structures: {sorted(only_here)[:5]}{'...' if len(only_here) > 5 else ''}",
                UserWarning,
                stacklevel=2,
            )


def align_structures(
    entries: List[StructureEntry],
    comparison: ComparisonConfig,
) -> pd.DataFrame:
    """
    Load all structures, optionally filter to a single chain, and outer-join on
    (chain, resi_struct) to produce a long-format DataFrame.

    Returns one row per (chain, resi_struct, _label).
    """
    dfs: Dict[str, pd.DataFrame] = {}
    for entry in entries:
        df = load_features(entry)
        if comparison.chain is not None:
            df = df[df["chain"] == comparison.chain].copy()
        dfs[entry.label] = df

    validate_compatibility(dfs)

    long_df = pd.concat(list(dfs.values()), ignore_index=True)
    return long_df
