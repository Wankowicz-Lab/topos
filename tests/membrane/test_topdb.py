"""Tests for TOPDB SideDefinition parsing and geometric→absolute mapping."""

import pandas as pd
import pytest

from topos.membrane.sides import (
    membrane_side_from_side_definition,
    membrane_side_from_tmbed_regions,
    membrane_side_from_tmbed_spans,
)
from topos.membrane.topdb import (
    SideDefinition,
    TopdbSideDefinitionNotFound,
    clear_topdb_cache,
    map_geometric_side,
    parse_side_definition,
)


SAMPLE_TOPDB_XML = b"""<?xml version="1.0" encoding="UTF-8"?>
<TOPDB xmlns="https://topdb.unitmp.org" ID="OPSD_BOVIN" type="Alpha_Polytopic">
  <CrossRef>
    <PDB ID="1u19">
      <SideDefinition Side1="Inside">
        <Note>by similarity</Note>
      </SideDefinition>
    </PDB>
    <PDB ID="2ped">
      <SideDefinition Side1="Outside">
        <Note>Figure 1</Note>
      </SideDefinition>
    </PDB>
  </CrossRef>
</TOPDB>
"""


@pytest.fixture(autouse=True)
def _clear_cache():
    clear_topdb_cache()
    yield
    clear_topdb_cache()


@pytest.mark.parametrize(
    "region,side1,expected",
    [
        ("side1", "Inside", "Inside"),
        ("side2", "Inside", "Outside"),
        ("side1", "Outside", "Outside"),
        ("side2", "Outside", "Inside"),
        ("transmembrane_helix", "Inside", "transmembrane"),
        ("unknown", "Inside", "unknown"),
    ],
)
def test_map_geometric_side_table(region, side1, expected):
    assert map_geometric_side(region, side1) == expected


def test_parse_side_definition_for_pdb():
    side = parse_side_definition(SAMPLE_TOPDB_XML, "1U19")
    assert side == SideDefinition(side1="Inside", note="by similarity")
    side2 = parse_side_definition(SAMPLE_TOPDB_XML, "2ped")
    assert side2.side1 == "Outside"


def test_parse_side_definition_missing_pdb_raises():
    with pytest.raises(TopdbSideDefinitionNotFound, match="no PDB crossref"):
        parse_side_definition(SAMPLE_TOPDB_XML, "9ZZZ")


def test_membrane_side_from_tmbed_regions():
    rt = pd.DataFrame(
        {
            "pdbtm_region": [
                "inside",
                "transmembrane_helix",
                "outside",
                "unknown",
            ]
        }
    )
    assert list(membrane_side_from_tmbed_regions(rt)) == [
        "Inside",
        "transmembrane",
        "Outside",
        "unknown",
    ]


def test_membrane_side_from_tmbed_spans_overlays_author_residue_ranges():
    """Spans label membrane_side and leave residues outside the spans on the base value."""
    rt = pd.DataFrame(
        {
            "chain": ["A", "A", "A", "A", "B"],
            "resi": [1, 2, 3, 4, 1],
            "pdbtm_region": ["side2", "side2", "transmembrane_helix", "side1", "side1"],
        }
    )
    regions = pd.DataFrame(
        {
            "chain": ["A", "A"],
            "type": ["outside", "inside"],
            "pdb_beg": [1, 4],
            "pdb_end": [2, 4],
        }
    )
    base = pd.Series([pd.NA, pd.NA, "transmembrane", pd.NA, pd.NA])
    sides = membrane_side_from_tmbed_spans(rt, regions, base)
    assert list(sides.iloc[:4]) == ["Outside", "Outside", "transmembrane", "Inside"]
    assert pd.isna(sides.iloc[4])
    assert list(rt["pdbtm_region"]) == [
        "side2",
        "side2",
        "transmembrane_helix",
        "side1",
        "side1",
    ]


def test_rhodopsin_style_side2_is_outside_when_side1_inside():
    """Classic GPCR: Side1=Inside → extracellular (side2) loops are Outside."""
    rt = pd.DataFrame(
        {
            "pdbtm_region": [
                "side2",  # N-term / extracellular
                "transmembrane_helix",
                "side1",
                "transmembrane_helix",
                "side2",
            ]
        }
    )
    sides = membrane_side_from_side_definition(rt, "Inside")
    assert list(sides) == [
        "Outside",
        "transmembrane",
        "Inside",
        "transmembrane",
        "Outside",
    ]
