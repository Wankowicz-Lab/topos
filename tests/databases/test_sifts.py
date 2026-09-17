"""Tests for SIFTS PDB→UniProt resolution."""

import pytest

from topos.databases.sifts import (
    UNIPROT_CONFIG_HINT,
    UniprotLookupError,
    pick_uniprot_for_chains,
    resolve_uniprot_accession,
)


def _sifts_payload(pdb_id: str = "1u19") -> dict:
    return {
        pdb_id.lower(): {
            "UniProt": {
                "P02699": {
                    "mappings": [
                        {
                            "chain_id": "A",
                            "unp_start": 1,
                            "unp_end": 348,
                            "start": {"residue_number": 1},
                            "end": {"residue_number": 348},
                        }
                    ]
                },
                "P99999": {
                    "mappings": [
                        {
                            "chain_id": "B",
                            "unp_start": 1,
                            "unp_end": 10,
                            "start": {"residue_number": 1},
                            "end": {"residue_number": 10},
                        }
                    ]
                },
            }
        }
    }


def test_pick_uniprot_prefers_feature_chain_coverage():
    acc, segs = pick_uniprot_for_chains(_sifts_payload(), "1U19", chains=["A"])
    assert acc == "P02699"
    assert segs[0]["chain_id"] == "A"


def test_resolve_uniprot_uses_config_without_sifts(monkeypatch):
    monkeypatch.setattr(
        "topos.databases.sifts.fetch_sifts_uniprot_mappings",
        lambda _pdb: (_ for _ in ()).throw(AssertionError("SIFTS should not be called")),
    )
    acc, source = resolve_uniprot_accession(
        pdb_id="1U19",
        config_uniprot_id="P02699",
    )
    assert acc == "P02699"
    assert source == "config"


def test_resolve_uniprot_via_sifts(monkeypatch):
    monkeypatch.setattr(
        "topos.databases.sifts.fetch_sifts_uniprot_mappings",
        lambda _pdb: _sifts_payload(),
    )
    acc, source = resolve_uniprot_accession(pdb_id="1U19", chains=["A"])
    assert acc == "P02699"
    assert source == "sifts"


def test_resolve_uniprot_no_mapping_raises(monkeypatch):
    monkeypatch.setattr(
        "topos.databases.sifts.fetch_sifts_uniprot_mappings",
        lambda _pdb: {"1u19": {"UniProt": {}}},
    )
    with pytest.raises(UniprotLookupError, match="no UniProt mapping"):
        resolve_uniprot_accession(pdb_id="1U19")


def test_resolve_uniprot_http_error_raises(monkeypatch):
    def boom(_pdb):
        raise UniprotLookupError("SIFTS request failed for PDB 1U19: boom")

    monkeypatch.setattr("topos.databases.sifts.fetch_sifts_uniprot_mappings", boom)
    with pytest.raises(UniprotLookupError, match="SIFTS request failed"):
        resolve_uniprot_accession(pdb_id="1U19")
    assert "uniprot_id" in UNIPROT_CONFIG_HINT


def test_resolve_uniprot_missing_pdb_and_config_raises():
    with pytest.raises(UniprotLookupError, match="No uniprot_id in config"):
        resolve_uniprot_accession(pdb_id=None, config_uniprot_id=None)
