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


def membrane_side_from_tmbed_spans(
    residue_table: pd.DataFrame,
    regions: pd.DataFrame,
    base: pd.Series,
) -> pd.Series:
    """Overlay TMbed spans onto ``membrane_side`` by chain and author residue number.

    Uses the same region-type map as ``membrane_side_from_tmbed_regions``.
    Spans are ``pdb_beg`` / ``pdb_end`` inclusive. Residues outside every span
    keep ``base``. This does not read or write ``pdbtm_region``.
    """
    sides = base.copy()
    if regions.empty:
        return sides

    for _, region in regions.iterrows():
        label = _TMBED_REGION_TO_SIDE.get(str(region["type"]), "unknown")
        mask = (
            (residue_table["chain"] == region["chain"])
            & (residue_table["resi"] >= int(region["pdb_beg"]))
            & (residue_table["resi"] <= int(region["pdb_end"]))
        )
        sides.loc[mask] = label
    return sides


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
