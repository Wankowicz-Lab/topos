"""Integration tests for src/grouped_analysis/comparator.py"""
import textwrap
import pytest
import numpy as np
import pandas as pd
from pathlib import Path


def make_features_csv(tmp_path: Path, name: str, group: str, n_residues=5, seed=0) -> Path:
    """Write a minimal *_features.csv with realistic column names."""
    rng = np.random.default_rng(seed)
    resi = list(range(1, n_residues + 1))
    df = pd.DataFrame({
        "chain": "A",
        "resi_struct": resi,
        "resn_struct": "ALA",
        "sasa": rng.normal(50 if group == "apo" else 60, 3, n_residues),
        "sasa_backbone": rng.normal(10, 1, n_residues),
        "packing_contact_density": rng.normal(1.0 if group == "apo" else 1.5, 0.1, n_residues),
        "distance_to_center_of_mass": rng.uniform(10, 30, n_residues),
        "total_hbond_count": rng.integers(0, 5, n_residues).astype(float),
        "graph_all_graph_betweenness_centrality": rng.uniform(0, 0.5, n_residues),
        "graph_all_graph_community_id": rng.integers(0, 3, n_residues).astype(float),
        "graph_all_graph_in_lcc": True,
        "graph_vdw_contact_graph_betweenness_centrality": rng.uniform(0, 0.5, n_residues),
        "graph_vdw_contact_graph_community_id": rng.integers(0, 3, n_residues).astype(float),
        "graph_vdw_contact_graph_in_lcc": True,
        "graph_hbond_graph_betweenness_centrality": rng.uniform(0, 0.1, n_residues),
        "graph_hbond_graph_community_id": rng.integers(0, 2, n_residues).astype(float),
        "graph_hbond_graph_in_lcc": False,
        "name": "test_protein",
    })
    p = tmp_path / name
    df.to_csv(p, index=False)
    return p


def write_toml(tmp_path: Path, csv_apo: Path, csv_bound: Path, mode: str = "group") -> Path:
    # Use multiple apo structures to give enough reps for Mann-Whitney (n>=3)
    content = f"""\
        name = "test_comparison"
        output_dir = "output/"

        [[structures]]
        path = "{csv_apo}"
        label = "apo_s1"
        group = "apo"

        [[structures]]
        path = "{csv_apo}"
        label = "apo_s2"
        group = "apo"

        [[structures]]
        path = "{csv_apo}"
        label = "apo_s3"
        group = "apo"

        [[structures]]
        path = "{csv_bound}"
        label = "bound_s1"
        group = "bound"

        [[structures]]
        path = "{csv_bound}"
        label = "bound_s2"
        group = "bound"

        [[structures]]
        path = "{csv_bound}"
        label = "bound_s3"
        group = "bound"

        [comparison]
        mode = "{mode}"
        reference_group = "apo"

        [metrics]
        include_categories = ["structural", "bonds", "graph"]
    """
    toml_file = tmp_path / "config.toml"
    toml_file.write_text(textwrap.dedent(content))
    return toml_file


class TestComparatorIntegration:
    def test_group_mode_writes_files(self, tmp_path):
        from src.grouped_analysis.comparator import Comparator

        csv_apo = make_features_csv(tmp_path, "apo.csv", "apo", seed=1)
        csv_bound = make_features_csv(tmp_path, "bound.csv", "bound", seed=2)
        toml = write_toml(tmp_path, csv_apo, csv_bound, mode="group")

        c = Comparator(toml)
        c.run()

        out_dir = tmp_path / "output"
        assert (out_dir / "test_comparison_residue_comparison.csv").exists()
        assert (out_dir / "test_comparison_graph_comparison.csv").exists()
        assert (out_dir / "test_comparison_summary.txt").exists()

    def test_residue_comparison_has_expected_columns(self, tmp_path):
        from src.grouped_analysis.comparator import Comparator

        csv_apo = make_features_csv(tmp_path, "apo.csv", "apo", seed=3)
        csv_bound = make_features_csv(tmp_path, "bound.csv", "bound", seed=4)
        toml = write_toml(tmp_path, csv_apo, csv_bound)

        c = Comparator(toml)
        c.load()
        c.run_residue_comparison()

        result = c.residue_comparison
        assert result is not None
        assert "mean_abs_cohens_d" in result.columns
        assert "n_large_effect" in result.columns
        assert "top_metric" in result.columns
        assert len(result) == 5  # n_residues

    def test_graph_comparison_has_expected_columns(self, tmp_path):
        from src.grouped_analysis.comparator import Comparator

        csv_apo = make_features_csv(tmp_path, "apo.csv", "apo", seed=5)
        csv_bound = make_features_csv(tmp_path, "bound.csv", "bound", seed=6)
        toml = write_toml(tmp_path, csv_apo, csv_bound)

        c = Comparator(toml)
        c.load()
        c.run_graph_comparison()

        result = c.graph_comparison
        assert result is not None
        assert "pathway_instability_score" in result.columns
        assert "community_change_score_all" in result.columns

    def test_all_vs_all_mode_writes_pair_files(self, tmp_path):
        from src.grouped_analysis.comparator import Comparator

        csv_apo = make_features_csv(tmp_path, "apo.csv", "apo", seed=7)
        csv_bound = make_features_csv(tmp_path, "bound.csv", "bound", seed=8)
        toml = write_toml(tmp_path, csv_apo, csv_bound, mode="all_vs_all")

        c = Comparator(toml)
        c.run()

        out_dir = tmp_path / "output"
        # Should have per-pair CSV files
        pair_files = list(out_dir.glob("*_vs_*_residue_comparison.csv"))
        assert len(pair_files) > 0

    def test_summary_file_content(self, tmp_path):
        from src.grouped_analysis.comparator import Comparator

        csv_apo = make_features_csv(tmp_path, "apo.csv", "apo", seed=9)
        csv_bound = make_features_csv(tmp_path, "bound.csv", "bound", seed=10)
        toml = write_toml(tmp_path, csv_apo, csv_bound)

        c = Comparator(toml)
        c.run()

        summary = (tmp_path / "output" / "test_comparison_summary.txt").read_text()
        assert "test_comparison" in summary
        assert "Mode:" in summary
        assert "Structures:" in summary
