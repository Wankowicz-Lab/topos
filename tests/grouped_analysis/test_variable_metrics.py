"""
Tests for identify_variable_residues.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from identify_variable_residues import (
    compute_variability,
    load_data,
    rank_normalise,
)


# ── rank_normalise ────────────────────────────────────────────────────────────

class TestRankNormalise:
    def test_output_in_zero_one(self):
        df = pd.DataFrame({"a": [3.0, 1.0, 2.0], "b": [10.0, 30.0, 20.0]})
        normed = rank_normalise(df)
        assert normed.min().min() >= 0.0
        assert normed.max().max() <= 1.0

    def test_shape_preserved(self):
        df = pd.DataFrame({"x": range(5), "y": range(5, 10)})
        assert rank_normalise(df).shape == df.shape

    def test_ascending_rank_order(self):
        df = pd.DataFrame({"v": [1.0, 2.0, 3.0, 4.0, 5.0]})
        normed = rank_normalise(df)
        # Ranks should be monotonically increasing
        assert list(normed["v"]) == sorted(normed["v"])

    def test_nan_treated_as_zero(self):
        df = pd.DataFrame({"v": [np.nan, 1.0, 2.0]})
        normed = rank_normalise(df)
        # NaN filled with 0 before ranking — no NaN in output
        assert not normed.isnull().any().any()

    def test_single_column(self):
        df = pd.DataFrame({"x": [5.0, 5.0, 5.0]})
        normed = rank_normalise(df)
        assert normed.shape == (3, 1)

    def test_index_preserved(self):
        df = pd.DataFrame({"m": [1.0, 2.0, 3.0]}, index=[10, 20, 30])
        normed = rank_normalise(df)
        assert list(normed.index) == [10, 20, 30]


# ── compute_variability ───────────────────────────────────────────────────────

class TestComputeVariability:
    def _make_df(self):
        """3 residues × 2 structures, one metric."""
        return pd.DataFrame({
            "resi_struct": [1, 1, 2, 2, 3, 3],
            "pdb_id":      ["A", "B", "A", "B", "A", "B"],
            "metric1":     [1.0, 1.0, 2.0, 4.0, 5.0, 5.0],
            "metric2":     [10.0, 20.0, 10.0, 10.0, 10.0, 10.0],
        })

    def test_sd_zero_for_identical(self):
        df = self._make_df()
        sd_df, _ = compute_variability(df, ["metric1"])
        # resi 1 and 3 have identical values across structures
        assert sd_df.loc[1, "metric1"] == pytest.approx(0.0)
        assert sd_df.loc[3, "metric1"] == pytest.approx(0.0)

    def test_sd_nonzero_for_variable(self):
        df = self._make_df()
        sd_df, _ = compute_variability(df, ["metric1"])
        # resi 2 has values 2.0 and 4.0 → SD = 1.414...
        assert sd_df.loc[2, "metric1"] > 0

    def test_range_values(self):
        df = self._make_df()
        _, rng_df = compute_variability(df, ["metric2"])
        # metric2: resi 1 has 10, 20 → range = 10
        assert rng_df.loc[1, "metric2"] == pytest.approx(10.0)
        # resi 2 and 3 have same value → range = 0
        assert rng_df.loc[2, "metric2"] == pytest.approx(0.0)

    def test_multiple_metrics(self):
        df = self._make_df()
        sd_df, rng_df = compute_variability(df, ["metric1", "metric2"])
        assert "metric1" in sd_df.columns
        assert "metric2" in sd_df.columns
        assert "metric1" in rng_df.columns
        assert "metric2" in rng_df.columns

    def test_returns_sorted_index(self):
        df = self._make_df()
        sd_df, _ = compute_variability(df, ["metric1"])
        assert list(sd_df.index) == sorted(sd_df.index)


# ── load_data ─────────────────────────────────────────────────────────────────

class TestLoadData:
    def _write_features(self, directory: Path, pdb_id: str):
        df = pd.DataFrame({
            "chain": ["A", "A", "B"],
            "resi_struct": [1, 2, 1],
            "metric1": [1.0, 2.0, 3.0],
        })
        df.to_csv(directory / f"{pdb_id}_features.csv", index=False)

    def test_loads_correct_chain(self, tmp_path):
        self._write_features(tmp_path, "AAAA")
        df = load_data("A", ["AAAA"], tmp_path)
        assert all(df["chain"] == "A")

    def test_assigns_pdb_id_column(self, tmp_path):
        self._write_features(tmp_path, "AAAA")
        df = load_data("A", ["AAAA"], tmp_path)
        assert "pdb_id" in df.columns
        assert (df["pdb_id"] == "AAAA").all()

    def test_multiple_pdbs_concatenated(self, tmp_path):
        self._write_features(tmp_path, "AAAA")
        self._write_features(tmp_path, "BBBB")
        df = load_data("A", ["AAAA", "BBBB"], tmp_path)
        assert set(df["pdb_id"].unique()) == {"AAAA", "BBBB"}

    def test_missing_file_skipped(self, tmp_path, capsys):
        self._write_features(tmp_path, "AAAA")
        # BBBB not on disk
        df = load_data("A", ["AAAA", "BBBB"], tmp_path)
        assert "BBBB" not in df["pdb_id"].values
        err = capsys.readouterr().err
        assert "WARNING" in err

    def test_chain_not_in_pdb_skipped(self, tmp_path):
        pd.DataFrame({"chain": ["B"], "resi_struct": [1], "metric1": [0.0]}).to_csv(
            tmp_path / "CCCC_features.csv", index=False
        )
        # chain A not present in CCCC → should be skipped
        df = load_data("A", ["CCCC"], tmp_path)
        assert df.empty or len(df) == 0


# ── Integration: main via CLI ─────────────────────────────────────────────────

class TestIdentifyVariableMain:
    def _write_features(self, directory: Path, pdb_id: str, seed: int = 0):
        rng = np.random.default_rng(seed)
        df = pd.DataFrame({
            "chain": ["A"] * 5,
            "resi_struct": list(range(1, 6)),
            "metric1": rng.uniform(0, 10, 5).tolist(),
            "metric2": rng.uniform(0, 5, 5).tolist(),
            "disulfide_bond_count": [0] * 5,  # in SKIP_COLS
        })
        df.to_csv(directory / f"{pdb_id}_features.csv", index=False)

    def test_main_creates_output_files(self, tmp_path):
        rdir = tmp_path / "renumbered"
        rdir.mkdir()
        out_dir = tmp_path / "variability"

        self._write_features(rdir, "AAAA", seed=1)
        self._write_features(rdir, "BBBB", seed=2)

        import identify_variable_residues as ivr
        old_argv = sys.argv
        sys.argv = [
            "identify_variable_residues.py",
            "--pdbs", "AAAA,BBBB",
            "--renumbered-dir", str(rdir),
            "--chain", "A",
            "--top", "3",
            "--out", str(out_dir),
        ]
        try:
            ivr.main()
        finally:
            sys.argv = old_argv

        assert (out_dir / "residue_variability_ranking.csv").exists()
        assert (out_dir / "per_residue_sd.csv").exists()
        assert (out_dir / "per_residue_range.csv").exists()
        assert (out_dir / "per_residue_normalised_sd.csv").exists()

    def test_variability_ranking_has_correct_columns(self, tmp_path):
        rdir = tmp_path / "renumbered"
        rdir.mkdir()
        out_dir = tmp_path / "variability"

        self._write_features(rdir, "AAAA", seed=3)
        self._write_features(rdir, "BBBB", seed=4)

        import identify_variable_residues as ivr
        old_argv = sys.argv
        sys.argv = [
            "identify_variable_residues.py",
            "--pdbs", "AAAA,BBBB",
            "--renumbered-dir", str(rdir),
            "--chain", "A",
            "--out", str(out_dir),
        ]
        try:
            ivr.main()
        finally:
            sys.argv = old_argv

        df = pd.read_csv(out_dir / "residue_variability_ranking.csv")
        assert "variability_score" in df.columns
        assert "rank" in df.columns

    def test_skip_cols_excluded_from_variability(self, tmp_path):
        """disulfide_bond_count (all-zero) should be dropped as zero-variance."""
        rdir = tmp_path / "renumbered"
        rdir.mkdir()
        out_dir = tmp_path / "variability"

        self._write_features(rdir, "AAAA", seed=5)
        self._write_features(rdir, "BBBB", seed=6)

        import identify_variable_residues as ivr
        old_argv = sys.argv
        sys.argv = [
            "identify_variable_residues.py",
            "--pdbs", "AAAA,BBBB",
            "--renumbered-dir", str(rdir),
            "--chain", "A",
            "--out", str(out_dir),
        ]
        try:
            ivr.main()
        finally:
            sys.argv = old_argv

        sd_df = pd.read_csv(out_dir / "per_residue_sd.csv", index_col=0)
        assert "disulfide_bond_count" not in sd_df.columns
