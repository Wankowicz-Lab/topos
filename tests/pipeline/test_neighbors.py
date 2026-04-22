import pandas as pd

from src.pipeline.neighbors import calculate_neighborhood_features, compute_residue_neighbors
from src.pipeline.runner import Runner
from src.structure.utils import res_key


def test_compute_residue_neighbors_basic(tmp_path):
    """Test compute_residue_neighbors computes neighbors correctly and stores in extras."""
    myrunner = Runner(
        pdb_id="8smv",
        name="test_neighbors",
        pdb_path=None,
        membrane_protein=False,
        output_dir=tmp_path,
    )
    mapping = compute_residue_neighbors(myrunner.context, cutoff=10.0)

    assert myrunner.context.extras["residue_neighbors"] == mapping
    assert isinstance(mapping, dict)
    assert len(mapping) > 0

    rt = myrunner.context.residue_table
    expected_keys = {
        res_key(row["chain"], row["resi_struct"], row["resn_struct"])
        for _, row in rt.iterrows()
    }
    assert set(mapping.keys()) == expected_keys


def test_compute_residue_neighbors_cutoff_effect(tmp_path):
    """Test that different cutoffs produce different neighbor sets."""
    myrunner = Runner(
        pdb_id="8smv",
        name="test_neighbors",
        pdb_path=None,
        membrane_protein=False,
        output_dir=tmp_path,
    )

    mapping_small = compute_residue_neighbors(myrunner.context, cutoff=5.0)
    mapping_large = compute_residue_neighbors(myrunner.context, cutoff=20.0)

    for residue_key in mapping_small:
        assert residue_key in mapping_large
        assert len(mapping_large[residue_key]) >= len(mapping_small[residue_key])


def test_calculate_neighborhood_features_basic(tmp_path):
    """Test calculate_neighborhood_features loops over functions and aggregates correctly."""
    myrunner = Runner(
        pdb_id="8smv",
        name="test_neighbors",
        pdb_path=None,
        membrane_protein=False,
        output_dir=tmp_path,
    )
    myrunner.features = myrunner.run_metrics(metrics=["sasa"])

    compute_residue_neighbors(myrunner.context, cutoff=10.0)
    result = calculate_neighborhood_features(myrunner.context, myrunner.features)

    merge_cols = ["chain", "resi_struct", "resn_struct"]
    assert all(column in result.columns for column in merge_cols)
    assert "n_ala_neighbors" in result.columns
    assert "neighborhood_sasa" in result.columns

    expected_rows = myrunner.features[merge_cols].drop_duplicates()
    assert len(result) == len(expected_rows)

    merged_check = pd.merge(expected_rows, result[merge_cols], on=merge_cols, how="inner")
    assert len(merged_check) == len(expected_rows)


def test_calculate_neighborhood_features_aggregates_multiple_metrics(tmp_path):
    """Test that calculate_neighborhood_features aggregates multiple metric outputs."""
    myrunner = Runner(
        pdb_id="8smv",
        name="test_neighbors_multi",
        pdb_path=None,
        membrane_protein=False,
        output_dir=tmp_path,
    )
    myrunner.features = myrunner.run_metrics(metrics=["sasa", "kyte_doolittle"])
    compute_residue_neighbors(myrunner.context, cutoff=10.0)

    result = calculate_neighborhood_features(myrunner.context, myrunner.features)

    merge_cols = ["chain", "resi_struct", "resn_struct"]
    assert all(column in result.columns for column in merge_cols)
    assert "n_ala_neighbors" in result.columns
    assert "neighborhood_sasa" in result.columns
    assert "neighborhood_kyte_doolittle" in result.columns

    expected_rows = myrunner.features[merge_cols].drop_duplicates()
    assert len(result) == len(expected_rows)


def test_calculate_neighborhood_features_neighbor_averages_deterministic():
    """Neighborhood averages use only mapped neighbors and ignore NaN values."""
    class DummyContext:
        def __init__(self, neighbor_map):
            self.extras = {"residue_neighbors": neighbor_map}

    features = pd.DataFrame({
        "chain": ["A", "A", "A", "A", "A"],
        "resi_struct": [1, 2, 2, 3, 4],
        "resn_struct": ["ALA", "VAL", "VAL", "GLY", "SER"],
        "sasa": [1.0, 5.0, 7.0, 3.0, 10.0],
        "kyte_doolittle": [2.0, 4.0, 10.0, 6.0, 8.0],
    })
    neighbor_map = {
        "A:1:ALA": ["A:2:VAL", "A:3:GLY", "A:999:UNK"],
        "A:2:VAL": ["A:1:ALA"],
        "A:3:GLY": [],
        "A:4:SER": ["A:2:VAL"],
    }
    context = DummyContext(neighbor_map)

    result = calculate_neighborhood_features(context, features)
    result = result.set_index(["chain", "resi_struct", "resn_struct"])

    assert result.loc[("A", 1, "ALA"), "neighborhood_sasa"] == 4.5
    assert result.loc[("A", 1, "ALA"), "neighborhood_kyte_doolittle"] == 6.5
    assert result.loc[("A", 2, "VAL"), "neighborhood_sasa"] == 1.0
    assert pd.isna(result.loc[("A", 3, "GLY"), "neighborhood_sasa"])
    assert result.loc[("A", 4, "SER"), "neighborhood_sasa"] == 6.0
