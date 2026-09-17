"""Tests for TMbed label parsing and region collapse."""

import pandas as pd
import pytest

from topos.membrane.tmbed import (
    TmbedNotAvailable,
    extract_chain_sequences,
    labels_to_regions,
    parse_tmbed_format1,
    predict_tmbed_labels,
)


def test_parse_tmbed_format1():
    text = (
        ">A\n"
        "ACDEFGHIK\n"
        "iiiHHHHHo\n"
        ">B\n"
        "AAAA\n"
        "oooo\n"
    )
    out = parse_tmbed_format1(text)
    assert out == {"A": "iiiHHHHHo", "B": "oooo"}


def test_labels_to_regions_emits_full_alphabet_runs():
    labels = "iiiHHHHHHoooHHiiii"
    resis = list(range(10, 10 + len(labels)))
    regions = labels_to_regions(labels, "A", resis)
    assert list(regions["type"]) == [
        "inside",
        "transmembrane_helix",
        "outside",
        "transmembrane_helix",
        "inside",
    ]
    assert regions.iloc[0]["pdb_beg"] == 10
    assert regions.iloc[0]["pdb_end"] == 12
    assert regions.iloc[1]["pdb_beg"] == 13
    assert regions.iloc[1]["pdb_end"] == 18
    assert regions.iloc[2]["type"] == "outside"
    assert regions.iloc[3]["pdb_beg"] == 22
    assert regions.iloc[3]["pdb_end"] == 23


def test_labels_to_regions_maps_beta_and_signal_to_types():
    labels = "iiBBB..SSoo"
    resis = list(range(1, len(labels) + 1))
    regions = labels_to_regions(labels, "A", resis)
    assert list(regions["type"]) == [
        "inside",
        "transmembrane_beta_strand",
        "unknown",
        "outside",
    ]


def test_labels_to_regions_length_mismatch_raises():
    with pytest.raises(ValueError, match="does not match residue count"):
        labels_to_regions("HHH", "A", [1, 2])


def test_predict_tmbed_labels_raises_when_cli_missing(monkeypatch):
    monkeypatch.setattr("topos.membrane.tmbed.shutil.which", lambda _name: None)
    with pytest.raises(TmbedNotAvailable, match="tmbed"):
        predict_tmbed_labels({"A": "ACDEF"})


def test_extract_chain_sequences_maps_modified_residues_to_single_char():
    """Noncanonical codes like MSE must stay one character for TMbed FASTA."""
    residue_table = pd.DataFrame(
        {
            "chain": ["A", "A", "A"],
            "resi": [10, 11, 12],
            "resn": ["ALA", "MSE", "GLY"],
        }
    )
    seqs = extract_chain_sequences(residue_table)
    seq, resis = seqs["A"]
    assert seq == "AXG"
    assert resis == [10, 11, 12]
    assert len(seq) == len(resis)
