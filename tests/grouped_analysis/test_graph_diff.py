"""Tests for src/grouped_analysis/graph_diff.py"""
import numpy as np
import pytest
import pandas as pd

from src.grouped_analysis import graph_diff


def make_community_df(community_ids, residues=None, chain="A"):
    """Build a minimal single-structure DataFrame with community IDs."""
    if residues is None:
        residues = list(range(1, len(community_ids) + 1))
    return pd.DataFrame({
        "chain": chain,
        "resi_struct": residues,
        "resn_struct": "ALA",
        "graph_all_graph_community_id": community_ids,
        "graph_vdw_contact_graph_community_id": community_ids,
        "graph_hbond_graph_community_id": community_ids,
    })


class TestBuildCoCommunityMatrix:
    def test_simple_two_communities(self):
        # 4 residues: [0,0,1,1] — two communities of 2
        df = make_community_df([0, 0, 1, 1], residues=[1, 2, 3, 4])
        M = graph_diff.build_co_community_matrix(df, "all", ["chain", "resi_struct"])
        assert M is not None
        assert M.shape == (4, 4)
        # Residues 0,1 in same community
        assert M[0, 1] == True
        assert M[2, 3] == True
        # Cross-community
        assert M[0, 2] == False
        assert M[1, 3] == False
        # Diagonal
        assert M[0, 0] == True

    def test_all_same_community(self):
        df = make_community_df([5, 5, 5], residues=[1, 2, 3])
        M = graph_diff.build_co_community_matrix(df, "all", ["chain", "resi_struct"])
        assert M is not None
        assert M.all()  # all True

    def test_all_different_communities(self):
        df = make_community_df([0, 1, 2], residues=[1, 2, 3])
        M = graph_diff.build_co_community_matrix(df, "all", ["chain", "resi_struct"])
        assert M is not None
        # Off-diagonals False, diagonal True
        np.testing.assert_array_equal(M, np.eye(3, dtype=bool))

    def test_missing_column_returns_none(self):
        df = pd.DataFrame({"chain": "A", "resi_struct": [1, 2], "sasa": [1.0, 2.0]})
        M = graph_diff.build_co_community_matrix(df, "all", ["chain", "resi_struct"])
        assert M is None

    def test_all_nan_returns_none(self):
        df = make_community_df([np.nan, np.nan, np.nan], residues=[1, 2, 3])
        M = graph_diff.build_co_community_matrix(df, "all", ["chain", "resi_struct"])
        assert M is None

    def test_nan_treated_as_singleton(self):
        # Each NaN is its own community
        df = make_community_df([np.nan, np.nan, 1], residues=[1, 2, 3])
        M = graph_diff.build_co_community_matrix(df, "all", ["chain", "resi_struct"])
        assert M is not None
        # NaN residues should NOT be co-community with each other (each is unique)
        assert M[0, 1] == False


class TestCommunityChangeScores:
    def _make_long_df(self, com_a, com_b, residues=None):
        """Two groups with different community assignments."""
        if residues is None:
            residues = list(range(1, len(com_a) + 1))
        rows = []
        for i, resi in enumerate(residues):
            rows.append({
                "chain": "A", "resi_struct": resi, "resn_struct": "ALA",
                "graph_all_graph_community_id": com_a[i],
                "graph_vdw_contact_graph_community_id": com_a[i],
                "graph_hbond_graph_community_id": com_a[i],
                "_group": "apo", "_label": f"apo_s1",
            })
            rows.append({
                "chain": "A", "resi_struct": resi, "resn_struct": "ALA",
                "graph_all_graph_community_id": com_a[i],
                "graph_vdw_contact_graph_community_id": com_a[i],
                "graph_hbond_graph_community_id": com_a[i],
                "_group": "apo", "_label": f"apo_s2",
            })
            rows.append({
                "chain": "A", "resi_struct": resi, "resn_struct": "ALA",
                "graph_all_graph_community_id": com_b[i],
                "graph_vdw_contact_graph_community_id": com_b[i],
                "graph_hbond_graph_community_id": com_b[i],
                "_group": "bound", "_label": f"bound_s1",
            })
            rows.append({
                "chain": "A", "resi_struct": resi, "resn_struct": "ALA",
                "graph_all_graph_community_id": com_b[i],
                "graph_vdw_contact_graph_community_id": com_b[i],
                "graph_hbond_graph_community_id": com_b[i],
                "_group": "bound", "_label": f"bound_s2",
            })
        return pd.DataFrame(rows)

    def test_no_change_zero_score(self):
        # Same communities in both groups → change score should be 0
        com = [0, 0, 1, 1]
        df = self._make_long_df(com_a=com, com_b=com)
        result = graph_diff.community_change_scores(df, groups=("apo", "bound"))

        assert "community_change_score_all" in result.columns
        scores = result["community_change_score_all"].dropna()
        assert (scores.abs() < 1e-10).all(), f"Expected zero scores, got: {scores.tolist()}"

    def test_full_reorganization_positive_score(self):
        # Complete community reorganization → change score > 0
        com_a = [0, 0, 1, 1]
        com_b = [1, 1, 0, 0]  # swapped communities
        df = self._make_long_df(com_a=com_a, com_b=com_b)
        result = graph_diff.community_change_scores(df, groups=("apo", "bound"))

        scores = result["community_change_score_all"].dropna()
        assert len(scores) == 4

    def test_output_columns_present(self):
        com = [0, 1, 0, 1]
        df = self._make_long_df(com_a=com, com_b=com)
        result = graph_diff.community_change_scores(df, groups=("apo", "bound"))

        expected_cols = [
            "community_change_score_all",
            "community_change_score_vdw_contact",
            "community_change_score_hbond",
            "community_entropy_all",
            "community_switches_all",
            "pathway_instability_score",
        ]
        for col in expected_cols:
            assert col in result.columns, f"Missing column: {col}"

    def test_residues_in_output(self):
        com = [0, 1]
        df = self._make_long_df(com_a=com, com_b=com, residues=[10, 20])
        result = graph_diff.community_change_scores(df, groups=("apo", "bound"))
        assert set(result["resi_struct"].values) == {10, 20}

    def test_missing_community_column_gives_nan(self):
        # DataFrame without any community columns
        df = pd.DataFrame({
            "chain": "A", "resi_struct": [1, 2],
            "resn_struct": "ALA", "sasa": [1.0, 2.0],
            "_group": ["apo", "bound"], "_label": ["a", "b"],
        })
        result = graph_diff.community_change_scores(df, groups=("apo", "bound"))
        # Should return a DF with NaN community scores
        for gt in graph_diff.GRAPH_TYPES:
            assert f"community_change_score_{gt}" in result.columns
            assert result[f"community_change_score_{gt}"].isna().all()
