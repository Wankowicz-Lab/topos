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

# On the PDBTM path, TMbed only names sides that PDBTM already defined.
_TMBED_INSIDE_OUTSIDE = {
    "inside": "Inside",
    "outside": "Outside",
}
_GEOMETRIC_SIDES = frozenset({"side1", "side2"})


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
    """Assign Inside/Outside from TMbed onto residues that already have a side.

    Only ``inside`` and ``outside`` spans are applied, and only where
    ``pdbtm_region`` is ``side1`` or ``side2``. Spans are ``pdb_beg`` /
    ``pdb_end`` inclusive. A later span wins when two of those ranges overlap.
    Transmembrane and unknown spans are ignored. Every other residue keeps
    ``base``, including residues already marked transmembrane. This does not
    write ``pdbtm_region``.
    """
    sides = base.copy()
    if regions.empty:
        return sides

    defined_side = residue_table["pdbtm_region"].isin(_GEOMETRIC_SIDES)
    for _, region in regions.iterrows():
        label = _TMBED_INSIDE_OUTSIDE.get(str(region["type"]))
        if label is None:
            continue
        mask = (
            defined_side
            & (residue_table["chain"] == region["chain"])
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
