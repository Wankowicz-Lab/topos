"""Tests for src/grouped_analysis/config.py"""
import textwrap
import pytest
from pathlib import Path


def write_toml(tmp_path: Path, content: str, name: str = "config.toml") -> Path:
    p = tmp_path / name
    p.write_text(textwrap.dedent(content))
    return p


def write_features_csv(tmp_path: Path, name: str = "feat.csv") -> Path:
    p = tmp_path / name
    p.write_text("chain,resi_struct,resn_struct,sasa\nA,1,MET,50.0\n")
    return p


class TestGroupedAnalysisConfig:
    def test_from_toml_basic(self, tmp_path):
        from src.grouped_analysis.config import GroupedAnalysisConfig

        csv = write_features_csv(tmp_path, "apo.csv")

        toml_content = f"""\
            name = "test_comparison"
            output_dir = "out/"

            [[structures]]
            path = "apo.csv"
            label = "apo"
            group = "apo"

            [[structures]]
            path = "apo.csv"
            label = "bound"
            group = "bound"

            [comparison]
            mode = "group"
            reference_group = "apo"

            [metrics]
            include_categories = ["structural", "bonds"]
        """
        toml_file = write_toml(tmp_path, toml_content)
        cfg = GroupedAnalysisConfig.from_toml(toml_file)

        assert cfg.name == "test_comparison"
        assert len(cfg.structures) == 2
        assert cfg.structures[0].label == "apo"
        assert cfg.structures[0].group == "apo"
        assert cfg.comparison.mode == "group"
        assert cfg.comparison.reference_group == "apo"
        assert "structural" in cfg.metrics.include_categories

    def test_from_toml_bad_path_raises(self, tmp_path):
        from src.grouped_analysis.config import GroupedAnalysisConfig

        toml_content = """\
            name = "bad_test"

            [[structures]]
            path = "nonexistent_file.csv"
            label = "x"
            group = "x"
        """
        toml_file = write_toml(tmp_path, toml_content)
        with pytest.raises(Exception):  # FileNotFoundError from pydantic validator
            GroupedAnalysisConfig.from_toml(toml_file)

    def test_missing_config_file_raises(self, tmp_path):
        from src.grouped_analysis.config import GroupedAnalysisConfig

        with pytest.raises(FileNotFoundError):
            GroupedAnalysisConfig.from_toml(tmp_path / "does_not_exist.toml")

    def test_invalid_category_raises(self, tmp_path):
        from src.grouped_analysis.config import MetricsConfig
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            MetricsConfig(include_categories=["bogus_category"])

    def test_all_category_accepted(self, tmp_path):
        from src.grouped_analysis.config import MetricsConfig

        cfg = MetricsConfig(include_categories=["all"])
        assert "all" in cfg.include_categories

    def test_custom_columns_override(self, tmp_path):
        from src.grouped_analysis.config import MetricsConfig

        cfg = MetricsConfig(custom_columns=["sasa", "packing_contact_density"])
        assert cfg.custom_columns == ["sasa", "packing_contact_density"]

    def test_relative_output_dir_resolved(self, tmp_path):
        from src.grouped_analysis.config import GroupedAnalysisConfig

        csv = write_features_csv(tmp_path, "s.csv")
        toml_content = """\
            name = "rel_test"
            output_dir = "results/"

            [[structures]]
            path = "s.csv"
            label = "a"
            group = "x"
        """
        toml_file = write_toml(tmp_path, toml_content)
        cfg = GroupedAnalysisConfig.from_toml(toml_file)
        # output_dir should be absolute (resolved relative to TOML)
        assert cfg.output_dir.is_absolute()

    def test_all_vs_all_mode(self, tmp_path):
        from src.grouped_analysis.config import GroupedAnalysisConfig

        csv = write_features_csv(tmp_path, "f.csv")
        toml_content = """\
            name = "ava_test"

            [[structures]]
            path = "f.csv"
            label = "a"
            group = "g1"

            [[structures]]
            path = "f.csv"
            label = "b"
            group = "g2"

            [comparison]
            mode = "all_vs_all"
        """
        toml_file = write_toml(tmp_path, toml_content)
        cfg = GroupedAnalysisConfig.from_toml(toml_file)
        assert cfg.comparison.mode == "all_vs_all"
