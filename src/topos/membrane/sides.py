"""Absolute membrane_side labels from region types or TOPDB SideDefinition."""

from __future__ import annotations

import pandas as pd

from topos.membrane.topdb import map_geometric_side

# TMbed region types already carry absolute topology.
_TMBED_REGION_TO_SIDE = {
    "inside": "Inside",
    "outside": "Outside",
    "transmembrane_helix": "transmembrane",
    "transmembrane_beta_strand": "transmembrane",
    "transmembrane_coil": "transmembrane",
    "unknown": "unknown",
}


def membrane_side_from_tmbed_regions(residue_table: pd.DataFrame) -> pd.Series:
    """Map estimate-path ``pdbtm_region`` values to absolute ``membrane_side``."""
    return residue_table["pdbtm_region"].map(
        lambda r: _TMBED_REGION_TO_SIDE.get(r, "unknown") if pd.notna(r) else pd.NA
    )


def membrane_side_from_side_definition(
    residue_table: pd.DataFrame,
    side1_definition: str,
) -> pd.Series:
    """Map geometric PDBTM ``pdbtm_region`` values using TOPDB Side1."""
    def _map(region: object) -> object:
        if pd.isna(region):
            return pd.NA
        return map_geometric_side(str(region), side1_definition)

    return residue_table["pdbtm_region"].map(_map)
