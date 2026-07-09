"""

Usage
-----
python run_pipeline.py --config my_protein_config.toml

# Preview all commands without running them
python run_pipeline.py --config my_protein_config.toml --dry-run

# Skip all plotting (generate annotation CSVs only — faster)
python run_pipeline.py --config my_protein_config.toml --skip-plots

(This pipeline always skips biogenesis; metrics CSVs must already exist.)

Config format
-------------
See template_config.toml for the full annotated template.  Quick example:

    output_dir          = "my_output/"
    reference_pdb       = "4AKE"
    chain               = "A"
    max_mismatches      = 5
    proximity_angstroms = 8.0
    top_n_variable      = 20

    [analysis]
    run_multi      = true
    run_comparison = true

    [[structures]]
    label    = "4AKE"
    pdb_id   = "4AKE"
    group    = "open"
    state    = "apo"
    genotype = "wt"
    chain    = "A"

    [[structures]]
    label    = "1AKE"
    pdb_id   = "1AKE"
    group    = "closed"
    state    = "bound"
    genotype = "wt"
    chain    = "A"

    [[pairs]]
    reference   = "4AKE"
    comparison  = "1AKE"
    description = "Open vs Closed"
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

try:
    import tomllib
except ImportError:
    try:
        import tomli as tomllib  # type: ignore[no-redef]
    except ImportError:
        sys.exit("ERROR: Python 3.11+ has tomllib built-in. "
                 "For older Python, install tomli:  pip install tomli")

_SCRIPT_DIR = Path(__file__).resolve().parent


# ── Config loading ────────────────────────────────────────────────────────────

def load_config(path: Path) -> dict:
    with open(path, "rb") as f:
        return tomllib.load(f)


def get_pdb_ids(cfg: dict) -> list[str]:
    """Return the list of pdb_id values from [[structures]]."""
    return [s["pdb_id"].upper() for s in cfg.get("structures", [])]


def get_setting(cfg: dict, key: str, default):
    return cfg.get(key, default)


def get_analysis(cfg: dict, key: str, default: bool) -> bool:
    return cfg.get("analysis", {}).get(key, default)


# ── Step runner ───────────────────────────────────────────────────────────────

def _run(cmd: list[str], dry_run: bool, step_name: str) -> None:
    cmd_str = " ".join(str(c) for c in cmd)
    if dry_run:
        print("  [dry-run] skipping.")
        return
    result = subprocess.run(cmd)
    if result.returncode != 0:
        sys.exit(f"\nERROR: step '{step_name}' failed (exit code {result.returncode}).")


# ── Pipeline stages ───────────────────────────────────────────────────────────

def stage_renumber(pdb_ids: list[str], reference: str, output_dir: Path,
                   max_mismatches: int, dry_run: bool) -> None:
    _run(
        [
            sys.executable,
            str(_SCRIPT_DIR / "renumber_to_reference.py"),
            "--pdbs", ",".join(pdb_ids),
            "--ref", reference,
            "--output-dir", str(output_dir),
            "--max-mismatches", str(max_mismatches),
        ],
        dry_run=dry_run,
        step_name="Renumber all structures to reference numbering",
    )


def stage_variability(pdb_ids: list[str], renumbered_dir: Path,
                      variability_dir: Path, chain: str, top_n: int,
                      dry_run: bool) -> None:
    _run(
        [
            sys.executable,
            str(_SCRIPT_DIR / "identify_variable_residues.py"),
            "--pdbs", ",".join(pdb_ids),
            "--renumbered-dir", str(renumbered_dir),
            "--chain", chain,
            "--top", str(top_n),
            "--out", str(variability_dir),
        ],
        dry_run=dry_run,
        step_name="Identify structurally variable residues",
    )


def stage_plots(pdb_ids: list[str], renumbered_dir: Path,
                profiles_dir: Path, chain: str, dry_run: bool) -> None:
    _run(
        [
            sys.executable,
            str(_SCRIPT_DIR / "plot_all.py"),
            "--pdbs", ",".join(pdb_ids),
            "--renumbered-dir", str(renumbered_dir),
            "--chain", chain,
            "--out", str(profiles_dir),
        ],
        dry_run=dry_run,
        step_name="Generate per-residue plots (lineplots, boxplots, heatmaps)",
    )


def stage_export_multi(pdb_ids: list[str], renumbered_dir: Path,
                       variability_dir: Path, annotations_dir: Path,
                       chain: str, dry_run: bool) -> None:
    _run(
        [
            sys.executable,
            str(_SCRIPT_DIR / "export_dms_annotations.py"),
            "--mode", "multi",
            "--pdbs", ",".join(pdb_ids),
            "--renumbered-dir", str(renumbered_dir),
            "--variability-dir", str(variability_dir),
            "--chain", chain,
            "--out", str(annotations_dir),
        ],
        dry_run=dry_run,
        step_name="Export DMS-ready annotations (multi-structure)",
    )


def stage_comparison(config_path: Path, metrics_dir: Path,
                     comparison_dir: Path, skip_plots: bool,
                     proximity_angstroms: float, dry_run: bool) -> None:
    cmd = [
        sys.executable,
        str(_SCRIPT_DIR / "run_comparison_metrics.py"),
        "--config", str(config_path),
        "--metrics-dir", str(metrics_dir),
        "--output-dir", str(comparison_dir),
    ]
    if skip_plots:
        cmd.append("--no-histograms")
    _run(cmd, dry_run=dry_run, step_name="Run pairwise comparison analysis")


def stage_export_comparison(local_dir: Path, annotations_dir: Path,
                             dry_run: bool) -> None:
    _run(
        [
            sys.executable,
            str(_SCRIPT_DIR / "export_dms_annotations.py"),
            "--mode", "comparison",
            "--local-dir", str(local_dir),
            "--out", str(annotations_dir),
        ],
        dry_run=dry_run,
        step_name="Export DMS-ready annotations (comparison)",
    )


def stage_pymol(annotation_csv: Path, metric: str, mode: str,
                pdb_id: str, chain: str, out_prefix: Path,
                dry_run: bool) -> None:
    _run(
        [
            sys.executable,
            str(_SCRIPT_DIR / "map_to_pymol.py"),
            "--csv", str(annotation_csv),
            "--metric", metric,
            "--pdb", pdb_id,
            "--chain", chain,
            "--output", str(out_prefix),
            "--mode", mode,
            "--no-bfactor-pdb",
        ],
        dry_run=dry_run,
        step_name=f"Generate PyMOL script: {metric} ({mode})",
    )


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--config", required=True, type=Path,
        help="Path to your pipeline config TOML file (see template_config.toml).",
    )
    parser.add_argument(
        "--skip-plots", action="store_true",
        help="Skip all plot generation (faster; annotation CSVs still produced).",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print all commands without executing them.",
    )
    args = parser.parse_args()

    # Always skip biogenesis step
    skip_biogenesis = True

    if not args.config.exists():
        sys.exit(f"ERROR: config not found: {args.config}\n"
                 "Copy template_config.toml and edit it for your project.")

    cfg = load_config(args.config)

    # ── Read settings ─────────────────────────────────────────────────────────
    pdb_ids           = get_pdb_ids(cfg)
    output_dir        = Path(get_setting(cfg, "output_dir", "output/"))
    reference_pdb     = str(get_setting(cfg, "reference_pdb", pdb_ids[0] if pdb_ids else "")).upper()
    chain             = str(get_setting(cfg, "chain", "A"))
    max_mismatches    = int(get_setting(cfg, "max_mismatches", 5))
    top_n_variable    = int(get_setting(cfg, "top_n_variable", 20))
    proximity_ang     = float(get_setting(cfg, "proximity_angstroms", 8.0))

    run_multi         = get_analysis(cfg, "run_multi", True)
    run_comparison    = get_analysis(cfg, "run_comparison", bool(cfg.get("pairs")))
    pairs             = cfg.get("pairs", [])

    if not pdb_ids:
        sys.exit("ERROR: No [[structures]] entries found in config. "
                 "Add at least one [[structures]] block.")

    # ── Derived paths ─────────────────────────────────────────────────────────
    output_dir.mkdir(parents=True, exist_ok=True)
    renumbered_dir   = output_dir / "renumbered"
    variability_dir  = output_dir / "variability"
    profiles_dir     = output_dir / "residue_profiles"
    comparison_dir   = output_dir / "comparisons"
    annotations_dir  = output_dir / "dms_annotations"
    viz_dir          = output_dir / "pymol"

    # ── Step 1: Biogenesis ────────────────────────────────────────────────────
    stage_biogenesis(pdb_ids, output_dir, args.dry_run)

    # ── Step 2: Renumber ──────────────────────────────────────────────────────
    if run_multi or len(pdb_ids) > 1:
        stage_renumber(pdb_ids, reference_pdb, output_dir,
                       max_mismatches, args.dry_run)

    # ── Multi-structure branch ────────────────────────────────────────────────
    if run_multi:
        stage_variability(pdb_ids, renumbered_dir, variability_dir,
                          chain, top_n_variable, args.dry_run)

        if not args.skip_plots:
            stage_plots(pdb_ids, renumbered_dir, profiles_dir, chain, args.dry_run)

        stage_export_multi(pdb_ids, renumbered_dir, variability_dir,
                           annotations_dir, chain, args.dry_run)

        # PyMOL scripts for multi-structure annotations
        multi_csv = annotations_dir / "dms_structural_annotations_multi.csv"
        for metric, mode_flag in [
            ("variability_score",              "spectrum"),
            ("variability_class",              "groups"),
            ("sasa_class",                     "groups"),
            ("mean_total_hbond_count",         "spectrum"),
            ("mean_graph_all_graph_betweenness_centrality", "spectrum"),
        ]:
            stage_pymol(
                annotation_csv=multi_csv,
                metric=metric,
                mode=mode_flag,
                pdb_id=reference_pdb,
                chain=chain,
                out_prefix=viz_dir / f"{reference_pdb}_{metric}",
                dry_run=args.dry_run,
            )

    # ── Comparison branch ─────────────────────────────────────────────────────
    if run_comparison and pairs:
        # The comparison script reads its own config for pairs/structure info.
        # metrics_dir points to raw (non-renumbered) features CSVs.
        stage_comparison(
            config_path=args.config,
            metrics_dir=output_dir,
            comparison_dir=comparison_dir,
            skip_plots=args.skip_plots,
            proximity_angstroms=proximity_ang,
            dry_run=args.dry_run,
        )

        local_dir = comparison_dir / "local"
        stage_export_comparison(local_dir, annotations_dir, args.dry_run)

        # PyMOL scripts for comparison annotations
        if not args.dry_run:
            ann_files = list(annotations_dir.glob("dms_comparison_annotations_*.csv"))
        else:
            ann_files = [annotations_dir / "dms_comparison_annotations_EXAMPLE.csv"]

        for ann_csv in ann_files:
            pair_name = ann_csv.stem.replace("dms_comparison_annotations_", "")
            for metric, mode_flag in [
                ("composite_change_score", "spectrum"),
                ("change_class",           "groups"),
            ]:
                stage_pymol(
                    annotation_csv=ann_csv,
                    metric=metric,
                    mode=mode_flag,
                    pdb_id=reference_pdb,
                    chain=chain,
                    out_prefix=viz_dir / f"{pair_name}_{metric}",
                    dry_run=args.dry_run,
                )
    elif run_comparison and not pairs:
        print("\n[SKIP] Comparison: run_comparison=true but no [[pairs]] defined in config.")


if __name__ == "__main__":
    main()
