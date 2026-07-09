"""
Tests for run_pipeline.py
"""
import sys
import textwrap
from pathlib import Path
from unittest.mock import patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import run_pipeline as rp


# ── load_config ───────────────────────────────────────────────────────────────

class TestLoadConfig:
    def test_loads_valid_toml(self, tmp_path):
        cfg_path = tmp_path / "test.toml"
        cfg_path.write_text(textwrap.dedent("""\
            output_dir = "out/"
            reference_pdb = "4AKE"
            chain = "A"

            [[structures]]
            label = "4AKE"
            pdb_id = "4AKE"
        """))
        cfg = rp.load_config(cfg_path)
        assert cfg["output_dir"] == "out/"
        assert cfg["reference_pdb"] == "4AKE"

    def test_returns_dict(self, tmp_path):
        cfg_path = tmp_path / "t.toml"
        cfg_path.write_text('key = "value"\n')
        assert isinstance(rp.load_config(cfg_path), dict)

    def test_structures_list(self, tmp_path):
        cfg_path = tmp_path / "t.toml"
        cfg_path.write_text(textwrap.dedent("""\
            [[structures]]
            label = "A"
            pdb_id = "AAAA"

            [[structures]]
            label = "B"
            pdb_id = "BBBB"
        """))
        cfg = rp.load_config(cfg_path)
        assert len(cfg["structures"]) == 2

    def test_analysis_section(self, tmp_path):
        cfg_path = tmp_path / "t.toml"
        cfg_path.write_text(textwrap.dedent("""\
            [analysis]
            run_multi = false
            run_comparison = true
        """))
        cfg = rp.load_config(cfg_path)
        assert cfg["analysis"]["run_multi"] is False
        assert cfg["analysis"]["run_comparison"] is True


# ── get_pdb_ids ───────────────────────────────────────────────────────────────

class TestGetPdbIds:
    def test_returns_uppercase_ids(self):
        cfg = {"structures": [{"pdb_id": "4ake"}, {"pdb_id": "1ANK"}]}
        assert rp.get_pdb_ids(cfg) == ["4AKE", "1ANK"]

    def test_empty_structures(self):
        assert rp.get_pdb_ids({}) == []
        assert rp.get_pdb_ids({"structures": []}) == []

    def test_preserves_order(self):
        cfg = {"structures": [
            {"pdb_id": "ZZZZ"},
            {"pdb_id": "AAAA"},
            {"pdb_id": "MMMM"},
        ]}
        assert rp.get_pdb_ids(cfg) == ["ZZZZ", "AAAA", "MMMM"]


# ── get_setting ───────────────────────────────────────────────────────────────

class TestGetSetting:
    def test_returns_value_if_present(self):
        cfg = {"chain": "B"}
        assert rp.get_setting(cfg, "chain", "A") == "B"

    def test_returns_default_if_absent(self):
        assert rp.get_setting({}, "chain", "A") == "A"

    def test_integer_default(self):
        assert rp.get_setting({}, "max_mismatches", 5) == 5

    def test_float_default(self):
        assert rp.get_setting({}, "proximity_angstroms", 8.0) == 8.0


# ── get_analysis ──────────────────────────────────────────────────────────────

class TestGetAnalysis:
    def test_present_in_section(self):
        cfg = {"analysis": {"run_multi": False}}
        assert rp.get_analysis(cfg, "run_multi", True) is False

    def test_returns_default_when_absent(self):
        assert rp.get_analysis({}, "run_multi", True) is True

    def test_returns_default_when_section_missing(self):
        assert rp.get_analysis({"other": {}}, "run_multi", True) is True


# ── _run ─────────────────────────────────────────────────────────────────────

class TestRun:
    def test_dry_run_does_not_execute(self, capsys):
        rp._run(["echo", "hello"], dry_run=True, step_name="test")
        out = capsys.readouterr().out
        assert "dry-run" in out
        assert "STEP" in out

    def test_dry_run_prints_command(self, capsys):
        rp._run(["echo", "hello"], dry_run=True, step_name="TestStep")
        out = capsys.readouterr().out
        assert "echo hello" in out
        assert "TestStep" in out

    def test_real_run_executes(self, tmp_path):
        sentinel = tmp_path / "touched.txt"
        rp._run(["touch", str(sentinel)], dry_run=False, step_name="touch")
        assert sentinel.exists()

    def test_failed_command_exits(self):
        with pytest.raises(SystemExit):
            rp._run(["false"], dry_run=False, step_name="fail")


# ── stage functions (dry-run only) ────────────────────────────────────────────

class TestStageFunctions:
    """Verify stage functions build the correct subprocess command."""

    def _capture_cmd(self, fn, *args, **kwargs):
        """Call a stage function with dry_run=True and capture the printed command."""
        import io
        from contextlib import redirect_stdout
        buf = io.StringIO()
        with redirect_stdout(buf):
            fn(*args, **kwargs)
        return buf.getvalue()

    def test_stage_biogenesis_includes_pdbs(self, tmp_path):
        out = self._capture_cmd(
            rp.stage_biogenesis,
            ["AAAA", "BBBB"], tmp_path, dry_run=True
        )
        assert "AAAA,BBBB" in out
        assert "run_adk_biogenesis" in out

    def test_stage_renumber_includes_ref(self, tmp_path):
        out = self._capture_cmd(
            rp.stage_renumber,
            ["AAAA", "BBBB"], "AAAA", tmp_path, 5, dry_run=True
        )
        assert "--ref" in out
        assert "AAAA" in out
        assert "renumber_to_reference" in out

    def test_stage_variability_includes_top(self, tmp_path):
        out = self._capture_cmd(
            rp.stage_variability,
            ["AAAA"], tmp_path / "renumbered", tmp_path / "variability",
            "A", 20, dry_run=True
        )
        assert "--top" in out
        assert "identify_variable_residues" in out

    def test_stage_comparison_includes_config(self, tmp_path):
        cfg = tmp_path / "test.toml"
        cfg.write_text("")
        out = self._capture_cmd(
            rp.stage_comparison,
            cfg, tmp_path, tmp_path / "comparisons",
            False, 8.0, dry_run=True
        )
        assert "run_comparison_metrics" in out
        assert "--config" in out

    def test_stage_pymol_includes_metric(self, tmp_path):
        out = self._capture_cmd(
            rp.stage_pymol,
            tmp_path / "ann.csv", "variability_score", "spectrum",
            "AAAA", "A", tmp_path / "pymol/AAAA_var",
            dry_run=True
        )
        assert "map_to_pymol" in out
        assert "variability_score" in out


# ── main (integration, dry-run) ───────────────────────────────────────────────

class TestMainDryRun:
    def test_dry_run_with_valid_config(self, minimal_toml, capsys):
        old_argv = sys.argv
        sys.argv = ["run_pipeline.py",
                    "--config", str(minimal_toml),
                    "--dry-run"]
        try:
            rp.main()
        finally:
            sys.argv = old_argv

        out = capsys.readouterr().out
        assert "STEP" in out
        assert "dry-run" in out

    def test_dry_run_shows_all_expected_stages(self, minimal_toml, capsys):
        old_argv = sys.argv
        sys.argv = ["run_pipeline.py",
                    "--config", str(minimal_toml),
                    "--dry-run"]
        try:
            rp.main()
        finally:
            sys.argv = old_argv

        out = capsys.readouterr().out
        # Multi-structure stages should appear
        assert "renumber" in out.lower() or "Renumber" in out
        assert "variability" in out.lower() or "variable" in out.lower()

    def test_missing_config_exits(self, tmp_path):
        old_argv = sys.argv
        sys.argv = ["run_pipeline.py",
                    "--config", str(tmp_path / "nonexistent.toml")]
        try:
            with pytest.raises(SystemExit):
                rp.main()
        finally:
            sys.argv = old_argv

    def test_skip_plots_flag(self, minimal_toml, capsys):
        old_argv = sys.argv
        sys.argv = ["run_pipeline.py",
                    "--config", str(minimal_toml),
                    "--dry-run", "--skip-plots"]
        try:
            rp.main()
        finally:
            sys.argv = old_argv

        out = capsys.readouterr().out
        assert "plot_all" not in out

    def test_output_dir_created_by_main(self, minimal_toml, tmp_path):
        out_dir = tmp_path / "out"
        old_argv = sys.argv
        sys.argv = ["run_pipeline.py",
                    "--config", str(minimal_toml),
                    "--dry-run"]
        try:
            rp.main()
        finally:
            sys.argv = old_argv
        # dry-run still creates the top-level output_dir
        assert out_dir.exists()
