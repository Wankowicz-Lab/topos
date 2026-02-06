"""Tests for graph metrics module."""
import pandas as pd
import pytest

from src.metrics.graph_metrics import calculate_graph_metrics


def test_calculate_graph_metrics_simple():
    """Simple graph: 2-3 residues, 1-2 bonds; verify metrics present and merge correct."""
    bonds_df = pd.DataFrame({
        "chain_1": ["A", "A"],
        "resi_1": [1, 2],
        "chain_2": ["A", "A"],
        "resi_2": [2, 3],
    })
    residue_table = pd.DataFrame({
        "chain": ["A", "A", "A"],
        "resi_struct": [1, 2, 3],
        "resn_struct": ["ALA", "GLY", "SER"],
    })

    result = calculate_graph_metrics(bonds_df, residue_table)

    expected_cols = [
        "graph_betweenness_centrality",
        "graph_closeness_centrality",
        "graph_eigenvector_centrality",
        "graph_pagerank",
        "graph_core_number",
        "graph_community_id",
        "graph_in_lcc",
    ]
    for col in expected_cols:
        assert col in result.columns, f"Missing column: {col}"

    assert len(result) == 3
    assert result["graph_in_lcc"].all()
    assert not result["graph_betweenness_centrality"].isna().any()
    assert not result["graph_closeness_centrality"].isna().any()


def test_calculate_graph_metrics_disconnected_lcc():
    """Disconnected graph: verify LCC subset and excluded residue reporting."""
    # Two components: A:1-A:2 (size 2) and A:3-A:4-A:5 (size 3, LCC)
    bonds_df = pd.DataFrame({
        "chain_1": ["A", "A", "A"],
        "resi_1": [1, 3, 4],
        "chain_2": ["A", "A", "A"],
        "resi_2": [2, 4, 5],
    })
    residue_table = pd.DataFrame({
        "chain": ["A"] * 5,
        "resi_struct": [1, 2, 3, 4, 5],
        "resn_struct": ["ALA", "GLY", "SER", "THR", "VAL"],
    })

    result = calculate_graph_metrics(bonds_df, residue_table)

    # Residues 1, 2 are in smaller component; 3, 4, 5 are in LCC
    in_lcc = result["graph_in_lcc"].values
    assert not in_lcc[0] and not in_lcc[1]  # A:1, A:2 excluded
    assert in_lcc[2] and in_lcc[3] and in_lcc[4]  # A:3, A:4, A:5 in LCC

    # Excluded residues should have NaN for centrality metrics
    assert pd.isna(result.loc[0, "graph_betweenness_centrality"])
    assert pd.isna(result.loc[1, "graph_betweenness_centrality"])
    assert not pd.isna(result.loc[2, "graph_betweenness_centrality"])


def test_calculate_graph_metrics_extra_residues_not_in_bonds():
    """Residue_table with extra residues not in bonds: verify NaN for those."""
    bonds_df = pd.DataFrame({
        "chain_1": ["A"],
        "resi_1": [1],
        "chain_2": ["A"],
        "resi_2": [2],
    })
    # Residue 3 is in residue_table but not in graph (isolated)
    residue_table = pd.DataFrame({
        "chain": ["A", "A", "A"],
        "resi_struct": [1, 2, 3],
        "resn_struct": ["ALA", "GLY", "SER"],
    })

    result = calculate_graph_metrics(bonds_df, residue_table)

    # A:3 is not in the graph at all - should have NaN for all metrics, graph_in_lcc=False
    assert not result.loc[2, "graph_in_lcc"]
    assert pd.isna(result.loc[2, "graph_betweenness_centrality"])
    assert pd.isna(result.loc[2, "graph_closeness_centrality"])


def test_calculate_graph_metrics_centrality_identifies_hub():
    """Star graph: central node should have highest betweenness centrality."""
    # Star: A:2 connected to A:1, A:3, A:4 (central node)
    bonds_df = pd.DataFrame({
        "chain_1": ["A", "A", "A"],
        "resi_1": [2, 2, 2],
        "chain_2": ["A", "A", "A"],
        "resi_2": [1, 3, 4],
    })
    residue_table = pd.DataFrame({
        "chain": ["A", "A", "A", "A"],
        "resi_struct": [1, 2, 3, 4],
        "resn_struct": ["ALA", "GLY", "SER", "THR"],
    })

    result = calculate_graph_metrics(bonds_df, residue_table)

    # Central node A:2 (index 1) should have highest betweenness
    betweenness = result["graph_betweenness_centrality"].values
    central_idx = 1
    assert betweenness[central_idx] == pytest.approx(1.0)
    # Leaf nodes should have 0
    for i in [0, 2, 3]:
        assert betweenness[i] == pytest.approx(0.0)

