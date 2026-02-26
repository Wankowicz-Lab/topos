"""Tests for src/grouped_analysis/io.py"""
import warnings
import pytest
import pandas as pd
from pathlib import Path

from src.grouped_analysis.config import StructureEntry, ComparisonConfig


def make_entry(tmp_path, df, label, group, filename=None) -> StructureEntry:
    filename = filename or f"{label}.csv"
    p = tmp_path / filename
    df.to_csv(p, index=False)
    return StructureEntry(path=p, label=label, group=group)


def minimal_df(residues=None, chain="A"):
    if residues is None:
        residues = [1, 2, 3]
    return pd.DataFrame({
        "chain": chain,
        "resi_struct": residues,
        "resn_struct": ["ALA"] * len(residues),
        "sasa": [50.0 + i for i in range(len(residues))],
        "packing_contact_density": [1.0 + i * 0.1 for i in range(len(residues))],
    })


class TestLoadFeatures:
    def test_basic_load(self, tmp_path):
        from src.grouped_analysis.io import load_features

        df = minimal_df()
        entry = make_entry(tmp_path, df, "apo", "apo")
        result = load_features(entry)

        assert "_label" in result.columns
        assert "_group" in result.columns
        assert result["_label"].iloc[0] == "apo"
        assert result["_group"].iloc[0] == "apo"
        assert len(result) == len(df)

    def test_dms_deduplication(self, tmp_path):
        from src.grouped_analysis.io import load_features

        # DMS-style: multiple rows per residue
        dms_df = pd.DataFrame({
            "chain": "A",
            "resi_struct": [1, 1, 2, 2, 3],
            "resn_struct": ["ALA"] * 5,
            "resi_mut": [1, 1, 2, 2, 3],
            "resn_mut": ["GLY", "VAL", "LEU", "PHE", "ALA"],
            "sasa": [50.0, 50.0, 60.0, 60.0, 70.0],
            "packing_contact_density": [1.0, 1.0, 2.0, 2.0, 3.0],
            "effect": [-0.5, 0.5, 0.1, -0.1, 0.0],
        })
        entry = make_entry(tmp_path, dms_df, "dms_struct", "dms")
        result = load_features(entry)

        # After dedup: 3 unique residues
        assert len(result) == 3
        # Structural columns should be means (identical in this case)
        assert result.loc[result["resi_struct"] == 1, "sasa"].iloc[0] == pytest.approx(50.0)

    def test_missing_required_col_raises(self, tmp_path):
        from src.grouped_analysis.io import load_features

        bad_df = pd.DataFrame({"resi_struct": [1, 2], "sasa": [1.0, 2.0]})
        # missing "chain"
        p = tmp_path / "bad.csv"
        bad_df.to_csv(p, index=False)
        entry = StructureEntry(path=p, label="bad", group="x")

        with pytest.raises(ValueError, match="missing required columns"):
            load_features(entry)


class TestValidateCompatibility:
    def test_shared_residues(self, tmp_path):
        from src.grouped_analysis.io import validate_compatibility

        dfs = {
            "a": pd.DataFrame({"chain": "A", "resi_struct": [1, 2, 3]}),
            "b": pd.DataFrame({"chain": "A", "resi_struct": [1, 2, 3]}),
        }
        validate_compatibility(dfs)  # should not raise

    def test_no_shared_residues_raises(self, tmp_path):
        from src.grouped_analysis.io import validate_compatibility

        dfs = {
            "a": pd.DataFrame({"chain": "A", "resi_struct": [1, 2]}),
            "b": pd.DataFrame({"chain": "A", "resi_struct": [3, 4]}),
        }
        with pytest.raises(ValueError, match="No residues are shared"):
            validate_compatibility(dfs)

    def test_partial_overlap_warns(self, tmp_path):
        from src.grouped_analysis.io import validate_compatibility

        dfs = {
            "a": pd.DataFrame({"chain": "A", "resi_struct": [1, 2, 3]}),
            "b": pd.DataFrame({"chain": "A", "resi_struct": [1, 2, 99]}),
        }
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            validate_compatibility(dfs)
        assert len(caught) > 0
        assert any("not present" in str(w.message) for w in caught)


class TestAlignStructures:
    def test_basic_alignment(self, tmp_path):
        from src.grouped_analysis.io import align_structures

        df1 = minimal_df(residues=[1, 2, 3])
        df2 = minimal_df(residues=[1, 2, 3])
        entry1 = make_entry(tmp_path, df1, "s1", "g1", "s1.csv")
        entry2 = make_entry(tmp_path, df2, "s2", "g2", "s2.csv")

        comparison = ComparisonConfig()
        result = align_structures([entry1, entry2], comparison)

        # 3 residues × 2 structures = 6 rows
        assert len(result) == 6
        assert set(result["_label"].unique()) == {"s1", "s2"}

    def test_chain_filter(self, tmp_path):
        from src.grouped_analysis.io import align_structures

        df_multi = pd.DataFrame({
            "chain": ["A", "A", "B", "B"],
            "resi_struct": [1, 2, 1, 2],
            "resn_struct": ["ALA"] * 4,
            "sasa": [1.0, 2.0, 3.0, 4.0],
        })
        entry = make_entry(tmp_path, df_multi, "s1", "g1")

        comparison = ComparisonConfig(chain="A")
        result = align_structures([entry], comparison)

        assert set(result["chain"].unique()) == {"A"}
        assert len(result) == 2
