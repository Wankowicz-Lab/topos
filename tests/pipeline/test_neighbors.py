import math
import warnings

import pandas as pd
import pytest

from src.metrics.neighborhood_metrics import neighbor_sequence_range_metrics
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
    assert "neighbor_prop_alpha_helix" in result.columns
    assert "secondary_structure_coarse_entropy" in result.columns
    assert "secondary_structure_granular_entropy" in result.columns
    assert "prop_long_range_neighbors" in result.columns
    assert "mean_neighbor_sequence_distance" in result.columns

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
    assert "neighbor_prop_alpha_helix" in result.columns
    assert "secondary_structure_coarse_entropy" in result.columns
    assert "secondary_structure_granular_entropy" in result.columns
    assert "prop_long_range_neighbors" in result.columns
    assert "mean_neighbor_sequence_distance" in result.columns

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
        "A:1:ALA": ["A:2:VAL", "A:3:GLY", "B:999:UNK"],
        "A:2:VAL": ["A:1:ALA"],
        "A:3:GLY": ["A:1:ALA"],
        "A:4:SER": ["A:2:VAL"],
    }
    context = DummyContext(neighbor_map)

    result = calculate_neighborhood_features(context, features)
    result = result.set_index(["chain", "resi_struct", "resn_struct"])

    assert result.loc[("A", 1, "ALA"), "neighborhood_sasa"] == 4.5
    assert result.loc[("A", 1, "ALA"), "neighborhood_kyte_doolittle"] == 6.5
    assert result.loc[("A", 2, "VAL"), "neighborhood_sasa"] == 1.0
    assert result.loc[("A", 3, "GLY"), "neighborhood_sasa"] == 1.0
    assert result.loc[("A", 4, "SER"), "neighborhood_sasa"] == 6.0

    
def test_calculate_neighborhood_features_chain_neighbor_counts_deterministic():
    """Chain-aware neighbor counts separate same-chain from cross-chain neighbors."""
    class DummyContext:
        def __init__(self, neighbor_map):
            self.extras = {"residue_neighbors": neighbor_map}

    features = pd.DataFrame({
        "chain": ["A", "A", "A", "B", "B", "B"],
        "resi_struct": [1, 2, 3, 1, 2, 2],
        "resn_struct": ["ALA", "VAL", "LEU", "GLY", "SER", "SER"],
        "sasa": [1.0, 2.0, 5.0, 3.0, 4.0, 8.0],
    })
    neighbor_map = {
        "A:1:ALA": ["A:2:VAL", "B:1:GLY", "B:999:UNK"],
        "A:2:VAL": ["A:1:ALA", "A:3:LEU"],
        "A:3:LEU": ["A:2:VAL"],
        "B:1:GLY": ["B:2:SER"],
        "B:2:SER": ["A:2:VAL", "B:1:GLY"],
    }
    context = DummyContext(neighbor_map)

    result = calculate_neighborhood_features(context, features)
    result = result.set_index(["chain", "resi_struct", "resn_struct"])

    assert result.loc[("A", 1, "ALA"), "n_same_chain_neighbors"] == 1
    assert result.loc[("A", 1, "ALA"), "n_different_chain_neighbors"] == 1
    assert result.loc[("A", 2, "VAL"), "n_same_chain_neighbors"] == 2
    assert result.loc[("A", 2, "VAL"), "n_different_chain_neighbors"] == 0
    assert result.loc[("B", 1, "GLY"), "n_same_chain_neighbors"] == 1
    assert result.loc[("B", 1, "GLY"), "n_different_chain_neighbors"] == 0
    
    
def test_neighbor_entropy_metrics_filters_missing_and_nonstandard_neighbors():
    """Entropy metrics ignore unresolved and non-standard neighbor residue labels."""

    class DummyContext:
        def __init__(self, neighbor_map):
            self.extras = {"residue_neighbors": neighbor_map}

    features = pd.DataFrame({
        "chain": ["A", "A", "A"],
        "resi_struct": [1, 2, 3],
        "resn_struct": ["ALA", "VAL", "MSE"],
        "sasa": [1.0, 2.0, 3.0],
    })
    neighbor_map = {
        "A:1:ALA": ["A:2:VAL", "A:3:MSE", "A:999:UNK"],
        "A:2:VAL": ["A:1:ALA"],
        "A:3:MSE": ["A:1:ALA"],
    }
    context = DummyContext(neighbor_map)

    result = calculate_neighborhood_features(context, features)
    result = result.set_index(["chain", "resi_struct", "resn_struct"])

    row = result.loc[("A", 1, "ALA")]
    assert row["n_neighbors"] == 2
    assert row["neighbor_aa_entropy"] == pytest.approx(1.0)
    assert row["neighbor_aa_group_entropy"] == pytest.approx(0.0)


def test_calculate_neighborhood_features_secondary_structure_coarse_granular_metrics_deterministic():
    """Secondary-structure neighborhood metrics use coarse and distinct domains."""

    class DummyContext:
        def __init__(self, neighbor_map, residue_table):
            self.extras = {"residue_neighbors": neighbor_map}
            self.residue_table = residue_table

    features = pd.DataFrame({
        "chain": ["A"] * 12,
        "resi_struct": list(range(1, 13)),
        "resn_struct": [
            "ALA", "VAL", "LEU", "ILE", "TYR", "PHE",
            "THR", "SER", "GLY", "PRO", "ASN", "GLN",
        ],
        "sasa": [float(i) for i in range(12)],
    })
    residue_table = features.copy()
    residue_table["ss_domains"] = [
        "alpha-helix_0",
        "alpha-helix_1",
        "alpha-helix_1",
        "alpha-helix_1",
        "beta-sheet_1",
        "beta-sheet_1",
        "beta-sheet_2",
        "beta-sheet_2",
        "coil_1",
        "coil_2",
        "unknown_1",
        pd.NA,
    ]
    neighbor_map = {
        "A:1:ALA": [
            "A:2:VAL",
            "A:3:LEU",
            "A:4:ILE",
            "A:5:TYR",
            "A:6:PHE",
            "A:7:THR",
            "A:8:SER",
            "A:9:GLY",
            "A:10:PRO",
            "A:11:ASN",
            "A:12:GLN",
        ],
        "A:2:VAL": ["A:1:ALA"],
        "A:3:LEU": ["A:1:ALA"],
        "A:4:ILE": ["A:1:ALA"],
        "A:5:TYR": ["A:1:ALA"],
        "A:6:PHE": ["A:1:ALA"],
        "A:7:THR": ["A:1:ALA"],
        "A:8:SER": ["A:1:ALA"],
        "A:9:GLY": ["A:1:ALA"],
        "A:10:PRO": ["A:1:ALA"],
        "A:11:ASN": ["A:1:ALA"],
        "A:12:GLN": ["A:1:ALA"],
    }
    context = DummyContext(neighbor_map, residue_table)

    result = calculate_neighborhood_features(context, features)
    result = result.set_index(["chain", "resi_struct", "resn_struct"])

    expected_ss_entropy = -sum(p * math.log2(p) for p in [3 / 9, 4 / 9, 2 / 9])
    expected_ss_domain_entropy = -sum(
        p * math.log2(p) for p in [3 / 9, 2 / 9, 2 / 9, 1 / 9, 1 / 9]
    )
    row = result.loc[("A", 1, "ALA")]
    assert row["neighbor_prop_alpha_helix"] == pytest.approx(3 / 9)
    assert row["neighbor_prop_beta_sheet"] == pytest.approx(4 / 9)
    assert row["neighbor_prop_coil"] == pytest.approx(2 / 9)
    assert row["secondary_structure_coarse_entropy"] == pytest.approx(expected_ss_entropy)
    assert row["secondary_structure_granular_entropy"] == pytest.approx(expected_ss_domain_entropy)
    assert row["secondary_structure_granular_entropy"] > row["secondary_structure_coarse_entropy"]

    assert result.loc[("A", 2, "VAL"), "neighbor_prop_alpha_helix"] == pytest.approx(1.0)
    assert result.loc[("A", 2, "VAL"), "secondary_structure_coarse_entropy"] == pytest.approx(0.0)
    assert result.loc[("A", 3, "LEU"), "neighbor_prop_alpha_helix"] == pytest.approx(1.0)
    assert result.loc[("A", 3, "LEU"), "secondary_structure_coarse_entropy"] == pytest.approx(0.0)
    assert result.loc[("A", 3, "LEU"), "secondary_structure_granular_entropy"] == pytest.approx(0.0)


def test_calculate_neighborhood_features_secondary_structure_coarse_granular_metrics_membrane_mapping():
    """Membrane TMD and loop labels map to helix/coil while unknowns are ignored."""

    class DummyContext:
        def __init__(self, neighbor_map, residue_table):
            self.extras = {"residue_neighbors": neighbor_map}
            self.residue_table = residue_table

    features = pd.DataFrame({
        "chain": ["A"] * 6,
        "resi_struct": [1, 2, 3, 4, 5, 6],
        "resn_struct": ["ALA", "VAL", "GLY", "SER", "THR", "ASN"],
        "sasa": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
    })
    residue_table = features.copy()
    residue_table["ss_domains"] = [
        "TMD_0",
        "TMD_1",
        "extracellular_loop_1",
        "cytoplasmic_loop_1",
        "unknown_1",
        pd.NA,
    ]
    neighbor_map = {
        "A:1:ALA": ["A:2:VAL", "A:3:GLY", "A:4:SER", "A:5:THR", "A:6:ASN"],
        "A:2:VAL": ["A:5:THR", "A:6:ASN"],
        "A:3:GLY": ["A:2:VAL"],
        "A:4:SER": ["A:2:VAL"],
        "A:5:THR": ["A:2:VAL"],
        "A:6:ASN": ["A:2:VAL"],
    }
    context = DummyContext(neighbor_map, residue_table)

    result = calculate_neighborhood_features(context, features)
    result = result.set_index(["chain", "resi_struct", "resn_struct"])

    expected_ss_entropy = -sum(p * math.log2(p) for p in [1 / 3, 2 / 3])
    row = result.loc[("A", 1, "ALA")]
    assert row["neighbor_prop_alpha_helix"] == pytest.approx(1 / 3)
    assert row["neighbor_prop_beta_sheet"] == pytest.approx(0.0)
    assert row["neighbor_prop_coil"] == pytest.approx(2 / 3)
    assert row["secondary_structure_coarse_entropy"] == pytest.approx(expected_ss_entropy)
    assert row["secondary_structure_granular_entropy"] == pytest.approx(math.log2(3))

    empty_row = result.loc[("A", 2, "VAL")]
    assert pd.isna(empty_row["neighbor_prop_alpha_helix"])
    assert pd.isna(empty_row["neighbor_prop_beta_sheet"])
    assert pd.isna(empty_row["neighbor_prop_coil"])
    assert empty_row["secondary_structure_coarse_entropy"] == pytest.approx(0.0)
    assert empty_row["secondary_structure_granular_entropy"] == pytest.approx(0.0)


def test_neighbor_sequence_range_metrics_same_chain_and_threshold():
    """Sequence-range metrics ignore cross-chain neighbors and honor the threshold."""

    class DummyContext:
        def __init__(self, neighbor_map):
            self.extras = {"residue_neighbors": neighbor_map}

    features = pd.DataFrame({
        "chain": ["A", "A", "A", "A", "A", "B"],
        "resi_struct": [15, 27, 28, 40, 30, 100],
        "resn_struct": ["ALA", "VAL", "GLY", "SER", "LEU", "THR"],
    })
    neighbor_map = {
        "A:15:ALA": ["A:27:VAL", "A:28:GLY", "A:40:SER", "B:100:THR"],
        "A:27:VAL": ["A:15:ALA"],
        "A:28:GLY": ["A:15:ALA"],
        "A:40:SER": ["A:15:ALA"],
        "A:30:LEU": ["A:15:ALA", "B:100:THR"],
        "B:100:THR": ["B:112:ASN", "A:15:ALA"],
    }
    context = DummyContext(neighbor_map)

    result = neighbor_sequence_range_metrics(context, features)
    result = result.set_index(["chain", "resi_struct", "resn_struct"])

    row = result.loc[("A", 15, "ALA")]
    assert row["prop_long_range_neighbors"] == pytest.approx(2 / 3)
    assert row["mean_neighbor_sequence_distance"] == pytest.approx((12 + 13 + 25) / 3)

    assert result.loc[("A", 30, "LEU"), "prop_long_range_neighbors"] == pytest.approx(1.0)
    assert result.loc[("A", 30, "LEU"), "mean_neighbor_sequence_distance"] == pytest.approx(15.0)

    custom = neighbor_sequence_range_metrics(context, features, long_range_threshold=20)
    custom = custom.set_index(["chain", "resi_struct", "resn_struct"])
    assert custom.loc[("A", 15, "ALA"), "prop_long_range_neighbors"] == pytest.approx(1 / 3)


def test_neighbor_sequence_range_metrics_warns_when_same_chain_neighbors_missing():
    """Sequence-range metrics warn and skip residues whose neighbors are only cross-chain."""

    class DummyContext:
        def __init__(self, neighbor_map):
            self.extras = {"residue_neighbors": neighbor_map}

    features = pd.DataFrame({
        "chain": ["A", "B"],
        "resi_struct": [10, 50],
        "resn_struct": ["ALA", "GLY"],
    })
    neighbor_map = {
        "A:10:ALA": ["B:50:GLY"],
        "B:50:GLY": ["A:10:ALA"],
    }
    context = DummyContext(neighbor_map)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = neighbor_sequence_range_metrics(context, features)

    assert len(caught) == 2
    assert "No same-chain neighbors for residue A:10:ALA" in str(caught[0].message)
    result = result.set_index(["chain", "resi_struct", "resn_struct"])
    assert pd.isna(result.loc[("A", 10, "ALA"), "prop_long_range_neighbors"])
    assert pd.isna(result.loc[("A", 10, "ALA"), "mean_neighbor_sequence_distance"])
