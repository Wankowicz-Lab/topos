"""Tests for TMbed label parsing and region collapse."""

import pytest

from topos.membrane.tmbed import (
    TmbedNotAvailable,
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


def test_labels_to_regions_collapses_h_runs():
    labels = "iiiHHHHHHoooHHiiii"
    resis = list(range(10, 10 + len(labels)))
    regions = labels_to_regions(labels, "A", resis)
    assert list(regions["type"]) == ["transmembrane_helix", "transmembrane_helix"]
    assert regions.iloc[0]["pdb_beg"] == 13
    assert regions.iloc[0]["pdb_end"] == 18
    assert regions.iloc[0]["seq_beg"] == 4
    assert regions.iloc[0]["seq_end"] == 9
    assert regions.iloc[1]["pdb_beg"] == 22
    assert regions.iloc[1]["pdb_end"] == 23


def test_labels_to_regions_length_mismatch_raises():
    with pytest.raises(ValueError, match="does not match residue count"):
        labels_to_regions("HHH", "A", [1, 2])


def test_predict_tmbed_labels_raises_when_cli_missing(monkeypatch):
    monkeypatch.setattr("topos.membrane.tmbed.shutil.which", lambda _name: None)
    with pytest.raises(TmbedNotAvailable, match="tmbed"):
        predict_tmbed_labels({"A": "ACDEF"})
