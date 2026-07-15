"""
Tests for run_grouped_pipeline.py
"""
import textwrap
from unittest.mock import patch

import pytest

from src.grouped_analysis import run_grouped_pipeline as rp


def test_loads_valid_toml(tmp_path):
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

def test_load_config_returns_dict(tmp_path):
    cfg_path = tmp_path / "t.toml"
    cfg_path.write_text('key = "value"\n')
    assert isinstance(rp.load_config(cfg_path), dict)

def test_load_config_structures_list(tmp_path):
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

def test_load_config_analysis_section(tmp_path):
    cfg_path = tmp_path / "t.toml"
    cfg_path.write_text(textwrap.dedent("""\
        [analysis]
        run_multi = false
        run_comparison = true
    """))
    cfg = rp.load_config(cfg_path)
    assert cfg["analysis"]["run_multi"] is False
    assert cfg["analysis"]["run_comparison"] is True



def test_get_pdb_ids_returns_uppercase_ids():
    cfg = {"structures": [{"pdb_id": "4ake"}, {"pdb_id": "1ANK"}]}
    assert rp.get_pdb_ids(cfg) == ["4AKE", "1ANK"]

def test_get_pdb_ids_empty_structures():
    assert rp.get_pdb_ids({}) == []
    assert rp.get_pdb_ids({"structures": []}) == []

def test_get_pdb_ids_preserves_order():
    cfg = {"structures": [
        {"pdb_id": "ZZZZ"},
        {"pdb_id": "AAAA"},
        {"pdb_id": "MMMM"},
    ]}
    assert rp.get_pdb_ids(cfg) == ["ZZZZ", "AAAA", "MMMM"]


def test_get_setting_returns_value_if_present():
    cfg = {"chain": "B"}
    assert rp.get_setting(cfg, "chain", "A") == "B"

def test_get_setting_returns_default_if_absent():
    assert rp.get_setting({}, "chain", "A") == "A"

def test_get_setting_integer_default():
    assert rp.get_setting({}, "max_mismatches", 5) == 5

def test_get_setting_float_default():
    assert rp.get_setting({}, "proximity_angstroms", 8.0) == 8.0

def test_get_analysis_present_in_section():
    cfg = {"analysis": {"run_multi": False}}
    assert rp.get_analysis(cfg, "run_multi", True) is False

def test_get_analysis_returns_default_when_absent():
    assert rp.get_analysis({}, "run_multi", True) is True

def test_get_analysis_returns_default_when_section_missing():
    assert rp.get_analysis({"other": {}}, "run_multi", True) is True


def test_run_dry_run_skips_execution(capsys):
    rp._run(["echo", "hello"], dry_run=True, step_name="test")
    out = capsys.readouterr().out
    assert "dry-run" in out

def test_run_dry_run_does_not_run_command(tmp_path):
    sentinel = tmp_path / "should_not_exist.txt"
    rp._run(["touch", str(sentinel)], dry_run=True, step_name="test")
    assert not sentinel.exists()

def test_run_real_run_executes(tmp_path):
    sentinel = tmp_path / "touched.txt"
    rp._run(["touch", str(sentinel)], dry_run=False, step_name="touch")
    assert sentinel.exists()

def test_run_failed_command_exits():
    with pytest.raises(SystemExit):
        rp._run(["false"], dry_run=False, step_name="fail")


def _get_cmd(fn, *args, **kwargs):
    """Call a stage function and return the command list passed to _run."""
    with patch.object(rp, '_run') as mock_run:
        fn(*args, **kwargs)
    return mock_run.call_args[0][0]

def test_stage_renumber_includes_ref(tmp_path):
    cmd = _get_cmd(
        rp.stage_renumber,
        ["AAAA", "BBBB"], "AAAA", tmp_path, 5, dry_run=True
    )
    cmd_str = " ".join(str(c) for c in cmd)
    assert "--ref" in cmd_str
    assert "AAAA" in cmd_str
    assert "renumber_to_reference" in cmd_str

def test_stage_variability_includes_top(tmp_path):
    cmd = _get_cmd(
        rp.stage_variability,
        ["AAAA"], tmp_path / "renumbered", tmp_path / "variability",
        "A", 20, dry_run=True
    )
    cmd_str = " ".join(str(c) for c in cmd)
    assert "--top" in cmd_str
    assert "identify_variable_residues" in cmd_str

def test_stage_comparison_includes_config(tmp_path):
    cfg = tmp_path / "test.toml"
    cfg.write_text("")
    cmd = _get_cmd(
        rp.stage_comparison,
        cfg, tmp_path, tmp_path / "comparisons",
        False, 8.0, dry_run=True
    )
    cmd_str = " ".join(str(c) for c in cmd)
    assert "run_comparison_metrics" in cmd_str
    assert "--config" in cmd_str

def test_stage_pymol_includes_metric(tmp_path):
    cmd = _get_cmd(
        rp.stage_pymol,
        tmp_path / "ann.csv", "variability_score", "spectrum",
        "AAAA", "A", tmp_path / "pymol/AAAA_var",
        dry_run=True
    )
    cmd_str = " ".join(str(c) for c in cmd)
    assert "map_to_pymol" in cmd_str
    assert "variability_score" in cmd_str
