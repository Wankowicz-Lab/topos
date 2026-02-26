"""Tests for src/grouped_analysis/residue_diff.py"""
import numpy as np
import pytest
import pandas as pd

from src.grouped_analysis.config import MetricsConfig


def make_long_df(n_reps_a=5, n_reps_b=5, seed=42):
    """
    Two groups, two residues, two metrics.
    Group 'apo':   sasa ~ N(50, 2),  packing ~ N(1.0, 0.1)
    Group 'bound': sasa ~ N(60, 2),  packing ~ N(1.5, 0.1)
    Large effect on sasa (d ≈ 5), large effect on packing (d ≈ 5).
    """
    rng = np.random.default_rng(seed)
    rows = []
    for resi in [1, 2]:
        for _ in range(n_reps_a):
            rows.append({
                "chain": "A", "resi_struct": resi, "resn_struct": "ALA",
                "sasa": rng.normal(50, 2),
                "packing_contact_density": rng.normal(1.0, 0.1),
                "_group": "apo", "_label": f"apo_{_}",
            })
        for _ in range(n_reps_b):
            rows.append({
                "chain": "A", "resi_struct": resi, "resn_struct": "ALA",
                "sasa": rng.normal(60, 2),
                "packing_contact_density": rng.normal(1.5, 0.1),
                "_group": "bound", "_label": f"bound_{_}",
            })
    return pd.DataFrame(rows)


class TestSelectMetricColumns:
    def test_structural_category(self):
        from src.grouped_analysis.residue_diff import select_metric_columns

        df = pd.DataFrame({
            "chain": ["A"],
            "resi_struct": [1],
            "sasa": [50.0],
            "packing_contact_density": [1.0],
            "kyte_doolittle": [0.5],
            "graph_all_graph_community_id": [2],  # excluded
            "_label": ["x"],
            "_group": ["g"],
        })
        metrics = MetricsConfig(include_categories=["structural"])
        cols = select_metric_columns(df, metrics)
        assert "sasa" in cols
        assert "packing_contact_density" in cols
        assert "kyte_doolittle" in cols
        assert "graph_all_graph_community_id" not in cols
        assert "_label" not in cols

    def test_custom_columns_override(self):
        from src.grouped_analysis.residue_diff import select_metric_columns

        df = pd.DataFrame({
            "chain": ["A"], "resi_struct": [1],
            "sasa": [1.0], "other_col": [2.0],
        })
        metrics = MetricsConfig(custom_columns=["sasa"])
        cols = select_metric_columns(df, metrics)
        assert cols == ["sasa"]

    def test_custom_columns_filters_missing(self):
        from src.grouped_analysis.residue_diff import select_metric_columns

        df = pd.DataFrame({"chain": ["A"], "resi_struct": [1], "sasa": [1.0]})
        metrics = MetricsConfig(custom_columns=["sasa", "nonexistent"])
        cols = select_metric_columns(df, metrics)
        assert cols == ["sasa"]

    def test_bonds_category(self):
        from src.grouped_analysis.residue_diff import select_metric_columns

        df = pd.DataFrame({
            "chain": ["A"], "resi_struct": [1],
            "hbond_count": [3], "salt_bridge_count": [1],
            "sasa": [50.0],
        })
        metrics = MetricsConfig(include_categories=["bonds"])
        cols = select_metric_columns(df, metrics)
        assert "hbond_count" in cols
        assert "salt_bridge_count" in cols
        assert "sasa" not in cols

    def test_graph_category_excludes_community_id(self):
        from src.grouped_analysis.residue_diff import select_metric_columns

        df = pd.DataFrame({
            "chain": ["A"], "resi_struct": [1],
            "graph_all_graph_betweenness_centrality": [0.1],
            "graph_all_graph_community_id": [2],
            "graph_all_graph_in_lcc": [True],
        })
        metrics = MetricsConfig(include_categories=["graph"])
        cols = select_metric_columns(df, metrics)
        assert "graph_all_graph_betweenness_centrality" in cols
        assert "graph_all_graph_community_id" not in cols


class TestCompareTwoGroups:
    def test_known_cohens_d(self):
        from src.grouped_analysis.residue_diff import compare_two_groups

        df_long = make_long_df(n_reps_a=20, n_reps_b=20)
        result = compare_two_groups(
            df_long,
            metric_cols=["sasa"],
            group_a="apo",
            group_b="bound",
        )
        # Large effect expected (means differ by ~10, std ~2)
        cd_vals = result["sasa_cohens_d"].dropna()
        assert len(cd_vals) == 2  # two residues
        assert all(cd_vals > 2.0), f"Expected large Cohen's d, got {cd_vals.tolist()}"

    def test_summary_columns_present(self):
        from src.grouped_analysis.residue_diff import compare_two_groups

        df_long = make_long_df()
        result = compare_two_groups(
            df_long,
            metric_cols=["sasa", "packing_contact_density"],
            group_a="apo",
            group_b="bound",
        )
        assert "mean_abs_cohens_d" in result.columns
        assert "n_large_effect" in result.columns
        assert "top_metric" in result.columns

    def test_small_n_nan_pvalue(self):
        from src.grouped_analysis.residue_diff import compare_two_groups

        # n < 3 per group → NaN p-value
        df = pd.DataFrame({
            "chain": "A",
            "resi_struct": [1, 1, 1, 1],
            "resn_struct": "ALA",
            "sasa": [50.0, 51.0, 60.0, 61.0],
            "_group": ["apo", "apo", "bound", "bound"],
            "_label": ["a1", "a2", "b1", "b2"],
        })
        result = compare_two_groups(df, metric_cols=["sasa"], group_a="apo", group_b="bound")
        # n=2 per group → p-value should be NaN
        assert pd.isna(result["sasa_pvalue"].iloc[0])

    def test_zero_diff_when_same_data(self):
        from src.grouped_analysis.residue_diff import compare_two_groups

        # Same values in both groups → diff = 0
        df = pd.DataFrame({
            "chain": "A",
            "resi_struct": [1] * 10,
            "resn_struct": "ALA",
            "sasa": [50.0] * 10,
            "_group": ["apo"] * 5 + ["bound"] * 5,
            "_label": [f"s{i}" for i in range(10)],
        })
        result = compare_two_groups(df, metric_cols=["sasa"], group_a="apo", group_b="bound")
        assert result["sasa_diff"].iloc[0] == pytest.approx(0.0)
        assert pd.isna(result["sasa_cohens_d"].iloc[0])  # both std=0 → NaN


class TestCompareAllVsAll:
    def test_returns_correct_pairs(self):
        from src.grouped_analysis.residue_diff import compare_all_vs_all

        df = make_long_df()
        # Add a third label
        extra = df[df["_group"] == "apo"].copy()
        extra["_group"] = "extra"
        extra["_label"] = "extra_label"
        df_3 = pd.concat([df, extra], ignore_index=True)

        results = compare_all_vs_all(df_3, metric_cols=["sasa"], label_col="_label")
        # Labels: apo_0..4, bound_0..4, extra_label → N*(N-1)/2 pairs
        assert isinstance(results, dict)
        assert len(results) > 0
        for (la, lb) in results:
            assert la < lb  # sorted order

    def test_two_labels_one_pair(self):
        from src.grouped_analysis.residue_diff import compare_all_vs_all

        df = make_long_df()
        # Collapse to two labels only
        df2 = df.copy()
        df2["_label"] = df2["_group"]  # "apo" and "bound"
        results = compare_all_vs_all(df2, metric_cols=["sasa"], label_col="_label")
        assert len(results) == 1
        assert ("apo", "bound") in results
