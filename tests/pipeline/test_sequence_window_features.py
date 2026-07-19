import numpy as np
import pandas as pd

from topos.pipeline.sequence_window_features import calculate_sequence_window_features


class DummyContext:
    def __init__(self, residue_table: pd.DataFrame):
        self.residue_table = residue_table


def test_calculate_sequence_window_features_collapses_and_rolls():
    residue_table = pd.DataFrame(
        {
            "chain": ["A"] * 6,
            "resi_mut": [1, 2, 3, 4, 5, 6],
            "resn_mut": ["ALA", "ARG", "CYS", "ASP", "GLU", "PHE"],
            "align_pos": [10, 20, 30, 40, 50, 60],
        }
    )
    features = pd.DataFrame(
        {
            "chain": ["A"] * 12,
            "resi_mut": [1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6],
            "resn_mut": ["ALA", "ALA", "ARG", "ARG", "CYS", "CYS", "ASP", "ASP", "GLU", "GLU", "PHE", "PHE"],
            "effect": [2.0, 4.0, 4.0, 6.0, 6.0, 8.0, np.nan, np.nan, 10.0, 12.0, 12.0, 14.0],
            "blosum90": [1.0, 3.0, 2.0, 4.0, 3.0, 5.0, np.nan, np.nan, 5.0, 7.0, 6.0, 8.0],
            "avg_effect_quartile": ["Q1", "Q1", "Q2", "Q2", "Q3", "Q3", None, None, "Q4", "Q4", "Q4", "Q4"],
            "mut_aa_group": ["A", "B", "A", "B", "A", "B", None, None, "A", "B", "A", "B"],
        }
    )

    result = calculate_sequence_window_features(
        DummyContext(residue_table),
        features,
        seq_metric_columns=[
            "effect",
            "blosum90",
            "avg_effect_quartile",
            "mut_aa_group",
        ],
        window_size=5,
    ).set_index(["chain", "resi_mut", "resn_mut"])

    assert len(result) == 5
    assert ("A", 4, "ASP") not in result.index
    assert "sequence_window_avg_effect_quartile" not in result.columns
    assert "sequence_window_mut_aa_group" not in result.columns

    assert np.isclose(result.loc[("A", 1, "ALA"), "sequence_window_effect"], 5.0)
    assert np.isclose(result.loc[("A", 3, "CYS"), "sequence_window_effect"], 7.8)
    assert np.isclose(result.loc[("A", 5, "GLU"), "sequence_window_effect"], 9.0)
    assert np.isclose(result.loc[("A", 3, "CYS"), "sequence_window_blosum90"], 4.4)


def test_calculate_sequence_window_features_returns_empty_without_numeric_metrics():
    residue_table = pd.DataFrame(
        {
            "chain": ["A"],
            "resi_mut": [1],
            "resn_mut": ["ALA"],
            "align_pos": [1],
        }
    )
    features = pd.DataFrame(
        {
            "chain": ["A"],
            "resi_mut": [1],
            "resn_mut": ["ALA"],
            "avg_effect_quartile": ["Q1"],
        }
    )

    result = calculate_sequence_window_features(
        DummyContext(residue_table),
        features,
        seq_metric_columns=["avg_effect_quartile"],
        window_size=5,
    )

    assert result.empty
    assert list(result.columns) == ["chain", "resi_mut", "resn_mut"]
