"""TOPDB SideDefinition: map geometric PDBTM sides to absolute Inside/Outside."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, Optional

import requests
from lxml import etree

logger = logging.getLogger(__name__)

TOPDB_API = "https://topdb.unitmp.org/api/v1/entry/{accession}.xml"

_OPPOSITE = {"Inside": "Outside", "Outside": "Inside"}

# Process-local XML cache keyed by UniProt accession.
_XML_CACHE: Dict[str, bytes] = {}


class TopdbSideDefinitionNotFound(LookupError):
    """Raised when TOPDB has no usable SideDefinition for a UniProt/PDB pair."""


@dataclass(frozen=True)
class SideDefinition:
    """Absolute meaning of geometric Side1 for one PDB cross-reference."""

    side1: str
    note: Optional[str] = None


def map_geometric_side(region_type: str, side1_definition: str) -> str:
    """
    Map a geometric PDBTM region type to absolute Inside/Outside/transmembrane.

    ``side1`` takes ``side1_definition``; ``side2`` takes the opposite.
    Membrane-embedded types map to ``transmembrane``; anything else to ``unknown``.
    """
    side1 = side1_definition.strip().capitalize()
    if side1 not in _OPPOSITE:
        raise ValueError(f"Invalid Side1 definition: {side1_definition!r}")

    if region_type == "side1":
        return side1
    if region_type == "side2":
        return _OPPOSITE[side1]
    if region_type in {"transmembrane_helix", "transmembrane_beta_strand", "transmembrane_coil"}:
        return "transmembrane"
    if region_type in {"inside", "outside"}:
        return region_type.capitalize()
    return "unknown"


def fetch_topdb_xml(accession: str, *, use_cache: bool = True) -> bytes:
    """Download TOPDB entry XML for a UniProt accession (cached per process)."""
    key = accession.strip().upper()
    if use_cache and key in _XML_CACHE:
        return _XML_CACHE[key]

    url = TOPDB_API.format(accession=key)
    try:
        response = requests.get(url, timeout=60)
        response.raise_for_status()
    except requests.RequestException as e:
        raise TopdbSideDefinitionNotFound(
            f"TOPDB request failed for UniProt {key}: {e}"
        ) from e

    xml_bytes = response.content
    if use_cache:
        _XML_CACHE[key] = xml_bytes
    return xml_bytes


def clear_topdb_cache() -> None:
    """Clear the in-process TOPDB XML cache (tests)."""
    _XML_CACHE.clear()


def parse_side_definition(xml_bytes: bytes, pdb_id: str) -> SideDefinition:
    """Extract SideDefinition for ``pdb_id`` from a TOPDB entry XML document."""
    parser = etree.XMLParser(ns_clean=True, recover=True)
    root = etree.fromstring(xml_bytes, parser=parser)
    pdb_upper = pdb_id.strip().upper()

    for pdb_elem in root.xpath("//*[local-name() = 'PDB']"):
        elem_id = (pdb_elem.get("ID") or "").strip().upper()
        if elem_id != pdb_upper:
            continue
        side_elems = pdb_elem.xpath(".//*[local-name() = 'SideDefinition']")
        if not side_elems:
            raise TopdbSideDefinitionNotFound(
                f"TOPDB PDB crossref {pdb_id} has no SideDefinition"
            )
        side_elem = side_elems[0]
        side1 = (side_elem.get("Side1") or "").strip()
        if side1.capitalize() not in _OPPOSITE:
            raise TopdbSideDefinitionNotFound(
                f"TOPDB SideDefinition for {pdb_id} has invalid Side1={side1!r}"
            )
        notes = side_elem.xpath(".//*[local-name() = 'Note']/text()")
        note = " ".join(n.strip() for n in notes if n and n.strip()) or None
        return SideDefinition(side1=side1.capitalize(), note=note)

    raise TopdbSideDefinitionNotFound(
        f"TOPDB entry has no PDB crossref for {pdb_id}"
    )


def fetch_side_definition(accession: str, pdb_id: str) -> SideDefinition:
    """Fetch TOPDB XML and return SideDefinition for the given PDB id."""
    return parse_side_definition(fetch_topdb_xml(accession), pdb_id)
