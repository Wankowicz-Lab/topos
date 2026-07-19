"""
Tests for identify_variable_metrics.py
"""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.grouped_analysis.identify_variable_metrics import (
    SKIP_COLS,
    compute_variability,
    load_data,
)


def _make_variability_df():
    """3 residues × 2 structures, one metric."""
    return pd.DataFrame({
        "resi_struct": [1, 1, 2, 2, 3, 3],
        "pdb_id":      ["A", "B", "A", "B", "A", "B"],
        "metric1":     [1.0, 1.0, 2.0, 4.0, 5.0, 5.0],
        "metric2":     [10.0, 20.0, 10.0, 10.0, 10.0, 10.0],
    })

def test_compute_variability_sd_zero_for_identical():
    df = _make_variability_df()
    sd_df, _ = compute_variability(df, ["metric1"])
    assert sd_df.loc[1, "metric1"] == pytest.approx(0.0)
    assert sd_df.loc[3, "metric1"] == pytest.approx(0.0)

def test_compute_variability_sd_nonzero_for_variable():
    df = _make_variability_df()
    sd_df, _ = compute_variability(df, ["metric1"])
    # resi 2 has values 2.0 and 4.0 → SD > 0
    assert sd_df.loc[2, "metric1"] > 0

def test_compute_variability_range_values():
    df = _make_variability_df()
    _, rng_df = compute_variability(df, ["metric2"])
    # metric2: resi 1 has 10, 20 → range = 10
    assert rng_df.loc[1, "metric2"] == pytest.approx(10.0)
    assert rng_df.loc[2, "metric2"] == pytest.approx(0.0)

def test_compute_variability_multiple_metrics():
    df = _make_variability_df()
    sd_df, rng_df = compute_variability(df, ["metric1", "metric2"])
    assert "metric1" in sd_df.columns
    assert "metric2" in sd_df.columns
    assert "metric1" in rng_df.columns
    assert "metric2" in rng_df.columns

def test_compute_variability_returns_sorted_index():
    df = _make_variability_df()
    sd_df, _ = compute_variability(df, ["metric1"])
    assert list(sd_df.index) == sorted(sd_df.index)


def _write_test_features(directory: Path, pdb_id: str):
    df = pd.DataFrame({
        "chain": ["A", "A", "B"],
        "resi_struct": [1, 2, 1],
        "metric1": [1.0, 2.0, 3.0],
    })
    df.to_csv(directory / f"{pdb_id}_features.csv", index=False)

def test_load_data_loads_correct_chain(tmp_path):
    _write_test_features(tmp_path, "AAAA")
    df = load_data("A", ["AAAA"], tmp_path)
    assert all(df["chain"] == "A")

def test_load_data_assigns_pdb_id_column(tmp_path):
    _write_test_features(tmp_path, "AAAA")
    df = load_data("A", ["AAAA"], tmp_path)
    assert "pdb_id" in df.columns
    assert (df["pdb_id"] == "AAAA").all()

def test_load_data_multiple_pdbs_concatenated(tmp_path):
    _write_test_features(tmp_path, "AAAA")
    _write_test_features(tmp_path, "BBBB")
    df = load_data("A", ["AAAA", "BBBB"], tmp_path)
    assert set(df["pdb_id"].unique()) == {"AAAA", "BBBB"}

def test_load_data_missing_file_skipped(tmp_path, capsys):
    _write_test_features(tmp_path, "AAAA")
    # BBBB not on disk
    df = load_data("A", ["AAAA", "BBBB"], tmp_path)
    assert "BBBB" not in df["pdb_id"].values
    err = capsys.readouterr().err
    assert "WARNING" in err

def test_load_data_chain_not_in_pdb_raises(tmp_path):
    pd.DataFrame({"chain": ["B"], "resi_struct": [1], "metric1": [0.0]}).to_csv(
        tmp_path / "CCCC_features.csv", index=False
    )
    # chain A not present → sys.exit called when no data loaded
    with pytest.raises(SystemExit):
        load_data("A", ["CCCC"], tmp_path)


def _write_pipeline_features(directory: Path, pdb_id: str, seed: int = 0):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({
        "chain": ["A"] * 5,
        "resi_struct": list(range(1, 6)),
        "metric1": rng.uniform(0, 10, 5).tolist(),
        "metric2": rng.uniform(0, 5, 5).tolist(),
        "disulfide_bond_count": [0] * 5,  # in SKIP_COLS
    })
    df.to_csv(directory / f"{pdb_id}_features.csv", index=False)

def _run_variability_pipeline(rdir: Path, pdb_ids=("AAAA", "BBBB")):
    df = load_data("A", list(pdb_ids), rdir)
    metric_cols = [
        c for c in df.select_dtypes(include="number").columns
        if c not in SKIP_COLS
    ]
    sd_df, rng_df = compute_variability(df, metric_cols)
    zero_var = sd_df.columns[sd_df.max() == 0]
    sd_df = sd_df.drop(columns=zero_var)
    rng_df = rng_df.drop(columns=zero_var)
    normed = rank_normalise(sd_df)
    score = normed.mean(axis=1)
    score_df = score.rename("variability_score").to_frame()
    score_df["rank"] = score_df["variability_score"].rank(ascending=False).astype(int)
    return score_df, sd_df, rng_df

def test_variability_pipeline_variability_ranking_has_correct_columns(tmp_path):
    rdir = tmp_path / "renumbered"
    rdir.mkdir()
    _write_pipeline_features(rdir, "AAAA", seed=3)
    _write_pipeline_features(rdir, "BBBB", seed=4)
    score_df, _, _ = _run_variability_pipeline(rdir)
    assert "variability_score" in score_df.columns
    assert "rank" in score_df.columns

def test_variability_pipeline_skip_cols_excluded_from_variability(tmp_path):
    rdir = tmp_path / "renumbered"
    rdir.mkdir()
    _write_pipeline_features(rdir, "AAAA", seed=5)
    _write_pipeline_features(rdir, "BBBB", seed=6)
    _, sd_df, _ = _run_variability_pipeline(rdir)
    assert "disulfide_bond_count" not in sd_df.columns

def test_variability_pipeline_ranking_covers_all_residues(tmp_path):
    rdir = tmp_path / "renumbered"
    rdir.mkdir()
    _write_pipeline_features(rdir, "AAAA", seed=7)
    _write_pipeline_features(rdir, "BBBB", seed=8)
    score_df, _, _ = _run_variability_pipeline(rdir)
    # 5 residues in the test data
    assert len(score_df) == 5
    assert set(score_df["rank"]) == set(range(1, 6))
