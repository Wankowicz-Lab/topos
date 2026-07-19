from __future__ import annotations

from typing import Optional

import pandas as pd
import pytest

from src.grouped_analysis.run_comparison_metrics import (
    _classify_columns,
    _find_features_csv,
    analyze_local,
)

def test_flat_uppercase_found(tmp_path):
    f = tmp_path / "AAAA_features.csv"
    f.touch()
    assert _find_features_csv("AAAA", tmp_path) == f

def test_flat_lowercase_found(tmp_path):
    f = tmp_path / "aaaa_features.csv"
    f.touch()
    assert _find_features_csv("AAAA", tmp_path) == f

def test_flat_uppercase_preferred_over_subdir(tmp_path):
    flat = tmp_path / "AAAA_features.csv"
    flat.touch()
    subdir = tmp_path / "aaaa"
    subdir.mkdir()
    (subdir / "run1_features.csv").touch()
    assert _find_features_csv("AAAA", tmp_path) == flat

def test_subdirectory_layout_fallback(tmp_path):
    subdir = tmp_path / "aaaa"
    subdir.mkdir()
    f = subdir / "run_features.csv"
    f.touch()
    assert _find_features_csv("AAAA", tmp_path) == f

def test_returns_none_when_not_found(tmp_path):
    assert _find_features_csv("XXXX", tmp_path) is None

def test_empty_subdir_returns_none(tmp_path):
    (tmp_path / "aaaa").mkdir()
    assert _find_features_csv("AAAA", tmp_path) is None


def _make_df(extra_cols: Optional[dict] = None):
    """Return a minimal df with continuous + count columns."""
    base = {
        "chain": ["A"],
        "resi_struct": [1],
        "resn_struct": ["ALA"],
        "sasa": [25.0],
        "total_hbond_count": [2],
        "vdw_contact_count": [8],
        "packing_contact_density": [0.45],
    }
    if extra_cols:
        base.update(extra_cols)
    return pd.DataFrame(base)

def test_continuous_excludes_identifiers():
    df = _make_df()
    cont, counts = _classify_columns(df)
    assert "chain" not in cont
    assert "resi_struct" not in cont
    assert "resn_struct" not in cont

def test_sasa_is_continuous():
    df = _make_df()
    cont, _ = _classify_columns(df)
    assert "sasa" in cont

def test_packing_density_is_continuous():
    df = _make_df()
    cont, _ = _classify_columns(df)
    assert "packing_contact_density" in cont

def test_hbond_count_classified_as_count():
    df = _make_df()
    _, counts = _classify_columns(df)
    assert "total_hbond_count" in counts

def test_vdw_count_classified():
    df = _make_df()
    _, counts = _classify_columns(df)
    assert "vdw_contact_count" in counts

def test_community_id_excluded():
    df = _make_df({"graph_all_graph_community_id": [1]})
    cont, counts = _classify_columns(df)
    assert "graph_all_graph_community_id" not in cont
    assert "graph_all_graph_community_id" not in counts

def test_returns_two_lists():
    df = _make_df()
    result = _classify_columns(df)
    assert len(result) == 2


def _make_pair():
    """Return (ref_df, cmp_df) with identical residue numbering."""
    ref = pd.DataFrame({
        "chain": ["A", "A", "A"],
        "resi_struct": [1, 2, 3],
        "sasa": [10.0, 20.0, 30.0],
        "total_hbond_count": [2, 3, 1],
        "dssp_phi": [180.0, 90.0, 45.0],
    })
    cmp = pd.DataFrame({
        "chain": ["A", "A", "A"],
        "resi_struct": [1, 2, 3],
        "sasa": [15.0, 20.0, 25.0],
        "total_hbond_count": [3, 3, 2],
        "dssp_phi": [170.0, 95.0, 40.0],
    })
    return ref, cmp

def test_output_columns_present():
    ref, cmp = _make_pair()
    result = analyze_local(ref, cmp, "WT", "MUT", set())
    assert "resi" in result.columns
    assert "metric" in result.columns
    assert "delta" in result.columns
    assert "abs_delta" in result.columns

def test_delta_correct():
    ref, cmp = _make_pair()
    result = analyze_local(ref, cmp, "WT", "MUT", set())
    sasa_rows = result[result["metric"] == "sasa"]
    resi1 = sasa_rows[sasa_rows["resi"] == 1]
    assert resi1["delta"].iloc[0] == pytest.approx(5.0)

def test_abs_delta_is_nonnegative():
    ref, cmp = _make_pair()
    result = analyze_local(ref, cmp, "WT", "MUT", set())
    assert (result["abs_delta"] >= 0).all()

def test_sorted_by_abs_delta_descending():
    ref, cmp = _make_pair()
    result = analyze_local(ref, cmp, "WT", "MUT", set())
    abs_deltas = result["abs_delta"].values
    assert list(abs_deltas) == sorted(abs_deltas, reverse=True)

def test_in_proximity_flag():
    ref, cmp = _make_pair()
    result = analyze_local(ref, cmp, "WT", "MUT", proximity_resi={2})
    resi2_rows = result[result["resi"] == 2]
    assert resi2_rows["in_proximity"].all()
    resi1_rows = result[result["resi"] == 1]
    assert not resi1_rows["in_proximity"].all()

def test_empty_proximity_all_false():
    ref, cmp = _make_pair()
    result = analyze_local(ref, cmp, "WT", "MUT", proximity_resi=set())
    assert not result["in_proximity"].any()

def test_circular_correction_for_dssp_phi():
    """Angles crossing ±180° boundary must use circular subtraction."""
    ref = pd.DataFrame({
        "chain": ["A"],
        "resi_struct": [1],
        "dssp_phi": [170.0],
    })
    cmp = pd.DataFrame({
        "chain": ["A"],
        "resi_struct": [1],
        "dssp_phi": [-170.0],  # 20° difference across boundary, not 340°
    })
    result = analyze_local(ref, cmp, "WT", "MUT", set())
    phi_row = result[result["metric"] == "dssp_phi"]
    assert not phi_row.empty
    assert abs(phi_row["delta"].iloc[0]) <= 180.0

def test_ref_and_cmp_labels_in_columns():
    ref, cmp = _make_pair()
    result = analyze_local(ref, cmp, "RefLabel", "CmpLabel", set())
    assert "RefLabel_value" in result.columns

def test_zero_delta_rows_included():
    """Residues with identical values in ref/cmp should still appear."""
    ref, cmp = _make_pair()
    # resi 2: sasa is 20.0 in both → delta = 0
    result = analyze_local(ref, cmp, "WT", "MUT", set())
    resi2_sasa = result[(result["resi"] == 2) & (result["metric"] == "sasa")]
    assert not resi2_sasa.empty
    assert resi2_sasa["delta"].iloc[0] == pytest.approx(0.0)

def test_only_inner_join_residues():
    """Residues present in only one structure should be excluded."""
    ref = pd.DataFrame({
        "chain": ["A", "A"],
        "resi_struct": [1, 99],  # 99 only in ref
        "sasa": [10.0, 5.0],
    })
    cmp = pd.DataFrame({
        "chain": ["A", "A"],
        "resi_struct": [1, 88],  # 88 only in cmp
        "sasa": [15.0, 8.0],
    })
    result = analyze_local(ref, cmp, "WT", "MUT", set())
    resis = set(result["resi"].unique())
    assert 99 not in resis
    assert 88 not in resis
    assert 1 in resis
