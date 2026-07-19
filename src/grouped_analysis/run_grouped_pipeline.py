"""
Grouped analysis pipeline runner.

Orchestrates all grouped analysis steps by calling the relevant modules
directly as Python functions — no subprocess invocations.

Usage
------------------
from grouped_analysis.run_grouped_pipeline import GroupedPipelineRunner

runner = GroupedPipelineRunner(config_path="my_config.toml")
runner.run()

See template_config.toml for the full annotated template. 
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass, field
from pathlib import Path
import tomllib

from src.grouped_analysis.identify_variable_metrics import run_variability_analysis
from src.grouped_analysis.pairwise_RMSD import compute_pairwise_rmsd
from src.grouped_analysis.plot_all_distributions import run_plots
from src.grouped_analysis.renumber_to_referencePDB import renumber_structures
from src.grouped_analysis.run_comparison_metrics import run_comparison_analysis
from src.grouped_analysis.structural_interpretation import run_comparison, run_multi


@dataclass
class GroupedPipelineRunner:
    """
    Orchestrates the grouped analysis pipeline.

    Loads settings from a TOML config file and exposes a ``run()`` method
    that calls each analysis step directly as Python functions.

    Parameters
    ----------
    config_path : Path or str
        Path to the pipeline config TOML file.
    skip_plots : bool
        If True, skip all plot generation (annotation CSVs are still produced).
    """

    config_path: Path
    skip_plots: bool = False

    # Resolved settings (populated in __post_init__)
    cfg: dict = field(default_factory=dict, init=False, repr=False)
    pdb_ids: list[str] = field(default_factory=list, init=False)
    output_dir: Path = field(default=None, init=False)
    reference_pdb: str = field(default="", init=False)
    chain: str = field(default="A", init=False)
    max_mismatches: int = field(default=5, init=False)
    top_n_variable: int = field(default=20, init=False)
    proximity_ang: float = field(default=8.0, init=False)
    run_multi_flag: bool = field(default=True, init=False)
    run_comparison_flag: bool = field(default=False, init=False)
    pairs: list = field(default_factory=list, init=False)

    # Derived output directories (populated in __post_init__)
    renumbered_dir: Path = field(default=None, init=False)
    variability_dir: Path = field(default=None, init=False)
    profiles_dir: Path = field(default=None, init=False)
    comparison_dir: Path = field(default=None, init=False)
    annotations_dir: Path = field(default=None, init=False)
    viz_dir: Path = field(default=None, init=False)
    rmsd_dir: Path = field(default=None, init=False)

    def __post_init__(self):
        self.config_path = Path(self.config_path).resolve()
        if not self.config_path.exists():
            raise FileNotFoundError(f"Config not found: {self.config_path}")

        with open(self.config_path, "rb") as f:
            self.cfg = tomllib.load(f)

        # PDB IDs from [[structures]]
        self.pdb_ids = [s["pdb_id"].upper() for s in self.cfg.get("structures", [])]
        if not self.pdb_ids:
            raise ValueError(
                "No [[structures]] entries found in config. "
                "Add at least one [[structures]] block."
            )

        # Top-level settings
        self.output_dir      = Path(self.cfg.get("output_dir", "output/")).resolve()
        self.reference_pdb   = str(self.cfg.get("reference_pdb", self.pdb_ids[0])).upper()
        self.chain           = str(self.cfg.get("chain", "A"))
        self.max_mismatches  = int(self.cfg.get("max_mismatches", 5))
        self.top_n_variable  = int(self.cfg.get("top_n_variable", 20))
        self.proximity_ang   = float(self.cfg.get("proximity_angstroms", 8.0))

        # [analysis] section
        analysis = self.cfg.get("analysis", {})
        self.run_multi_flag      = analysis.get("run_multi", True)
        self.pairs               = self.cfg.get("pairs", [])
        self.run_comparison_flag = analysis.get("run_comparison", bool(self.pairs))

        # Derived output subdirectories
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.renumbered_dir  = self.output_dir / "renumbered"
        self.variability_dir = self.output_dir / "variability"
        self.profiles_dir    = self.output_dir / "residue_profiles"
        self.comparison_dir  = self.output_dir / "comparisons"
        self.annotations_dir = self.output_dir / "dms_annotations"
        self.rmsd_dir        = self.output_dir / "rmsd"


    def run(self) -> None:
        """Execute all enabled pipeline stages in order."""

        # Step 1: Renumber all structures to the reference PDB numbering.
        if self.run_multi_flag or len(self.pdb_ids) > 1:
            self._run_renumber()

        # Multi-structure branch
        if self.run_multi_flag:
            self._run_variability()
            if not self.skip_plots:
                self._run_plots()
            self._run_export_multi()

        # Pairwise RMSD (always run when multiple structures present)
        if len(self.pdb_ids) > 1:
            self._run_rmsd()

        # Comparison branch
        if self.run_comparison_flag and self.pairs:
            self._run_comparison()
            self._run_export_comparison()

    def _run_renumber(self) -> None:
        """Align all structures' residue numbering to the reference PDB."""
        renumber_structures(
            ref_pdb=self.reference_pdb,
            max_mismatches=self.max_mismatches,
            input_dir=str(self.output_dir),
            pdb_list=self.pdb_ids,
        )

    def _run_variability(self) -> None:
        """Compute per-residue variability scores across all structures."""
        run_variability_analysis(
            chain=self.chain,
            pdb_ids=self.pdb_ids,
            renumbered_dir=self.renumbered_dir,
            out_dir=self.variability_dir,
            top_n=self.top_n_variable,
        )

    def _run_plots(self) -> None:
        """Generate per-residue lineplots, boxplots, and heatmaps."""
        run_plots(
            chain=self.chain,
            pdb_ids=self.pdb_ids,
            renumbered_dir=self.renumbered_dir,
            out_dir=self.profiles_dir,
        )

    def _run_export_multi(self) -> None:
        """Aggregate per-residue annotations across all structures into a single CSV."""
        run_multi(
            chain=self.chain,
            out_dir=self.annotations_dir,
            pdb_ids=self.pdb_ids,
            renumbered_dir=self.renumbered_dir,
            variability_dir=self.variability_dir,
        )

    def _run_rmsd(self) -> None:
        """Compute pairwise sequence-aligned CA-RMSD for all structures."""
        compute_pairwise_rmsd(config_path=self.config_path, output_dir=self.rmsd_dir)

    def _run_comparison(self) -> None:
        """Compute pairwise metric differences and proximity-flagged residue tables."""
        run_comparison_analysis(
            config_path=self.config_path,
            metrics_dir=self.output_dir,
            out_dir=self.comparison_dir,
            no_histograms=self.skip_plots,
        )

    def _run_export_comparison(self) -> None:
        """Summarise per-pair local-difference CSVs into compact annotation files."""
        local_dir = self.comparison_dir / "local"
        if not local_dir.exists() or not list(local_dir.glob("*_local_diffs.csv")):
            print(f"  WARNING: no *_local_diffs.csv found in {local_dir}; skipping.")
            return
        run_comparison(
            local_dir=local_dir,
            out_dir=self.annotations_dir,
            delta_threshold=0.5,
        )


