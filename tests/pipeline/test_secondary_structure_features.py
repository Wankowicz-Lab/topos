import numpy as np
import pandas as pd

from topos.pipeline.runner import Runner
from topos.pipeline.secondary_structure_features import calculate_secondary_structure_features
from tests.test_utils import _make_config_file


def _make_synthetic_ss_fixture(include_na_ss=False, metric_with_nan=False, aa_groups=None):
    """Build synthetic residue_table and features for SS-domain aggregation tests."""
    residue_table = pd.DataFrame({
        "chain": ["A"] * 6,
        "resi_struct": [1, 2, 3, 4, 5, 6],
        "resn_struct": ["ALA", "ARG", "CYS", "ASP", "GLU", "PHE"],
        "resi_mut": [1, 2, 3, 4, 5, 6],
        "resn_mut": ["ALA", "ARG", "CYS", "ASP", "GLU", "PHE"],
        "ss_domains": ["alpha-helix_1", "alpha-helix_1", "beta-sheet_1", "beta-sheet_1", "coil_1", "coil_1"],
    })
    features = pd.DataFrame({
        "chain": ["A"] * 6,
        "resi_struct": [1, 2, 3, 4, 5, 6],
        "resn_struct": ["ALA", "ARG", "CYS", "ASP", "GLU", "PHE"],
        "resi_mut": [1, 2, 3, 4, 5, 6],
        "resn_mut": ["ALA", "ARG", "CYS", "ASP", "GLU", "PHE"],
        "metric_a": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
    })
    if aa_groups is not None:
        features["wildtype_aa_group"] = aa_groups
    if include_na_ss:
        residue_table = pd.concat([
            residue_table,
            pd.DataFrame({
                "chain": ["A"],
                "resi_struct": [7],
                "resn_struct": ["GLY"],
                "resi_mut": [7],
                "resn_mut": ["GLY"],
                "ss_domains": [pd.NA],
            }),
        ], ignore_index=True)
        features = pd.concat([
            features,
            pd.DataFrame({
                "chain": ["A"],
                "resi_struct": [7],
                "resn_struct": ["GLY"],
                "resi_mut": [7],
                "resn_mut": ["GLY"],
                "metric_a": [7.0],
            }),
        ], ignore_index=True)
    if metric_with_nan:
        features.loc[features["resi_struct"] == 1, "metric_a"] = np.nan
    return residue_table, features


def test_calculate_secondary_structure_features_columns_exist(tmp_path):
    """Each residue in a domain should receive the same SS-domain aggregate values."""
    config_path = tmp_path / "config.toml"
    _make_config_file(config_path, mutation_data_chain="A", mutation_data_path="")
    myrunner = Runner(config_path=config_path)
    residue_table, features = _make_synthetic_ss_fixture()
    myrunner.context.residue_table = residue_table

    out = calculate_secondary_structure_features(myrunner.context, features)

    assert len(out) == len(residue_table)
    assert "ss_domains" in out.columns
    assert "ss_domain_length" in out.columns
    assert "ss_domain_metric_a" in out.columns

    from topos.metrics import secondary_structure as ss_module

    for group_name in ss_module.AA_GROUPS:
        assert f"ss_domain_log2_aa_group_ratio_{group_name}" in out.columns

    feat_with_ss = features.merge(
        residue_table[["chain", "resi_struct", "resn_struct", "ss_domains"]],
        on=["chain", "resi_struct", "resn_struct"],
    )
    merged = feat_with_ss.merge(
        out[["chain", "resi_struct", "resn_struct", "ss_domain_metric_a"]],
        on=["chain", "resi_struct", "resn_struct"],
    )
    merged["expected"] = merged.groupby(["chain", "ss_domains"])["metric_a"].transform("mean")
    np.testing.assert_array_equal(merged["ss_domain_metric_a"].values, merged["expected"].values)


def test_calculate_secondary_structure_features_na_in_metric(tmp_path):
    """NA metric values should not wipe out the SS-domain aggregate output."""
    config_path = tmp_path / "config.toml"
    _make_config_file(config_path, mutation_data_chain="A", mutation_data_path="")
    myrunner = Runner(config_path=config_path)
    residue_table, features = _make_synthetic_ss_fixture(metric_with_nan=True)
    myrunner.context.residue_table = residue_table

    out = calculate_secondary_structure_features(myrunner.context, features)
    assert len(out) == len(residue_table)
    assert "ss_domain_metric_a" in out.columns
    assert "ss_domain_length" in out.columns
    assert not out["ss_domain_metric_a"].isna().all()


def test_calculate_secondary_structure_features_na_ss_domains_excluded(tmp_path):
    """Rows with missing ss_domains should be excluded from the output."""
    config_path = tmp_path / "config.toml"
    _make_config_file(config_path, mutation_data_chain="A", mutation_data_path="")
    myrunner = Runner(config_path=config_path)
    residue_table, features = _make_synthetic_ss_fixture(include_na_ss=True)
    myrunner.context.residue_table = residue_table

    out = calculate_secondary_structure_features(myrunner.context, features)
    assert len(out) == len(residue_table) - 1
    assert "ss_domains" in out.columns
    assert "ss_domain_length" in out.columns
    assert "ss_domain_metric_a" in out.columns


def test_calculate_secondary_structure_features_averages_mutation_rows_per_residue(tmp_path):
    """Mutation-level duplicate rows should collapse before SS-domain averaging."""
    config_path = tmp_path / "config.toml"
    _make_config_file(config_path, mutation_data_chain="A", mutation_data_path="")
    myrunner = Runner(config_path=config_path)

    residue_table, features = _make_synthetic_ss_fixture()
    dup_row = features.loc[features["resi_struct"] == 1].copy()
    dup_row["metric_a"] = 9.0
    features = pd.concat([features, dup_row], ignore_index=True)

    myrunner.context.residue_table = residue_table
    out = calculate_secondary_structure_features(myrunner.context, features)
    alpha = out[out["ss_domains"] == "alpha-helix_1"]

    assert np.isclose(alpha["ss_domain_metric_a"].iloc[0], 3.5)
