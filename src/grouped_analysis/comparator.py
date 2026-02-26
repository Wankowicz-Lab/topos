"""
Main orchestrator for grouped structure comparison.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import pandas as pd

from .config import GroupedAnalysisConfig
from . import io as _io
from . import residue_diff as _rdiff
from . import graph_diff as _gdiff

logger = logging.getLogger(__name__)


@dataclass
class Comparator:
    """
    Orchestrates grouped analysis: load → residue comparison → graph comparison → save.

    Parameters
    ----------
    config_path : str or Path
        Path to the TOML config file.
    """
    config_path: Union[str, Path]
    config: GroupedAnalysisConfig = field(init=False)
    _df_long: Optional[pd.DataFrame] = field(init=False, default=None)
    _metric_cols: Optional[List[str]] = field(init=False, default=None)
    residue_comparison: Optional[pd.DataFrame] = field(init=False, default=None)
    # For all_vs_all mode, keyed by (labelA, labelB)
    residue_comparisons: Optional[Dict[Tuple[str, str], pd.DataFrame]] = field(
        init=False, default=None
    )
    graph_comparison: Optional[pd.DataFrame] = field(init=False, default=None)

    def __post_init__(self):
        self.config = GroupedAnalysisConfig.from_toml(self.config_path)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def load(self) -> None:
        """Load and align all structure feature CSVs into a long DataFrame."""
        logger.info("Loading %d structures...", len(self.config.structures))
        self._df_long = _io.align_structures(
            self.config.structures,
            self.config.comparison,
        )
        logger.info(
            "Loaded %d rows across %d structures",
            len(self._df_long),
            len(self.config.structures),
        )

    def run_residue_comparison(self) -> None:
        """Compute per-residue Cohen's d and Mann-Whitney U between groups."""
        if self._df_long is None:
            raise RuntimeError("Call load() before run_residue_comparison()")

        self._metric_cols = _rdiff.select_metric_columns(
            self._df_long, self.config.metrics
        )
        logger.info("Comparing %d metric columns", len(self._metric_cols))

        mode = self.config.comparison.mode
        groups = list({e.group for e in self.config.structures})

        if mode == "group":
            ref = self.config.comparison.reference_group
            if ref is None:
                # Default: first group vs second group (alphabetical)
                groups_sorted = sorted(groups)
                if len(groups_sorted) < 2:
                    raise ValueError(
                        "At least 2 groups are required for group comparison."
                    )
                group_a, group_b = groups_sorted[0], groups_sorted[1]
            else:
                other_groups = [g for g in groups if g != ref]
                if not other_groups:
                    raise ValueError(
                        f"reference_group '{ref}' is the only group; need at least one other."
                    )
                group_a, group_b = ref, other_groups[0]

            self.residue_comparison = _rdiff.compare_two_groups(
                self._df_long,
                metric_cols=self._metric_cols,
                group_a=group_a,
                group_b=group_b,
                group_col="_group",
            )

        elif mode == "all_vs_all":
            self.residue_comparisons = _rdiff.compare_all_vs_all(
                self._df_long,
                metric_cols=self._metric_cols,
                label_col="_label",
            )
        else:
            raise ValueError(f"Unknown comparison mode: {mode}")

    def run_graph_comparison(self) -> None:
        """Compute community change scores using co-community matrix approach."""
        if self._df_long is None:
            raise RuntimeError("Call load() before run_graph_comparison()")

        groups = sorted({e.group for e in self.config.structures})
        if len(groups) < 2:
            logger.warning("Graph comparison requires at least 2 groups; skipping.")
            return

        group_a = self.config.comparison.reference_group or groups[0]
        group_b = [g for g in groups if g != group_a][0]

        self.graph_comparison = _gdiff.community_change_scores(
            self._df_long,
            groups=(group_a, group_b),
        )

    def save_results(self, output_dir: Optional[Union[str, Path]] = None) -> None:
        """Write output CSV files and summary text."""
        out_dir = Path(output_dir) if output_dir is not None else self.config.output_dir
        out_dir.mkdir(parents=True, exist_ok=True)

        name = self.config.name
        mode = self.config.comparison.mode

        if mode == "group" and self.residue_comparison is not None:
            path = out_dir / f"{name}_residue_comparison.csv"
            self.residue_comparison.to_csv(path, index=False)
            logger.info("Wrote %s", path)

        elif mode == "all_vs_all" and self.residue_comparisons:
            for (la, lb), df in self.residue_comparisons.items():
                fname = f"{name}_{la}_vs_{lb}_residue_comparison.csv"
                path = out_dir / fname
                df.to_csv(path, index=False)
                logger.info("Wrote %s", path)

        if self.graph_comparison is not None:
            path = out_dir / f"{name}_graph_comparison.csv"
            self.graph_comparison.to_csv(path, index=False)
            logger.info("Wrote %s", path)

        self._save_summary(out_dir / f"{name}_summary.txt")

    def _save_summary(self, path: Path) -> None:
        """Write a human-readable summary of the comparison run."""
        lines = [
            f"Grouped Analysis Summary: {self.config.name}",
            "=" * 60,
            f"Mode: {self.config.comparison.mode}",
            f"Structures: {len(self.config.structures)}",
            "",
            "Structures:",
        ]
        for entry in self.config.structures:
            lines.append(f"  [{entry.group}] {entry.label}: {entry.path}")

        if self._metric_cols:
            lines += [
                "",
                f"Metric columns compared: {len(self._metric_cols)}",
                "  " + ", ".join(self._metric_cols[:10])
                + ("..." if len(self._metric_cols) > 10 else ""),
            ]

        if self.residue_comparison is not None:
            n_large = int(
                (self.residue_comparison.get("n_large_effect", pd.Series(dtype=float)) > 0).sum()
            )
            lines += [
                "",
                f"Residues with ≥1 large effect (|d|>0.8): {n_large}",
            ]
            # Top 5 residues by mean_abs_cohens_d
            if "mean_abs_cohens_d" in self.residue_comparison.columns:
                top = self.residue_comparison.nlargest(5, "mean_abs_cohens_d")
                lines.append("Top residues by mean |Cohen's d|:")
                for _, row in top.iterrows():
                    res_id = f"{row.get('chain','?')}{row.get('resi_struct','?')} ({row.get('resn_struct','?')})"
                    lines.append(f"  {res_id}: {row['mean_abs_cohens_d']:.3f} (top metric: {row.get('top_metric','?')})")

        if self.graph_comparison is not None and "pathway_instability_score" in self.graph_comparison.columns:
            top5_path = self.graph_comparison.nlargest(5, "pathway_instability_score")
            lines += [
                "",
                "Top 5 residues by pathway instability score:",
            ]
            key_cols = [c for c in ["chain", "resi_struct", "resn_struct"] if c in top5_path.columns]
            for _, row in top5_path.iterrows():
                res_id = " ".join(str(row[c]) for c in key_cols)
                lines.append(f"  {res_id}: {row['pathway_instability_score']:.4f}")

        lines.append("")
        with open(path, "w") as f:
            f.write("\n".join(lines))
        logger.info("Wrote summary: %s", path)

    def run(self) -> None:
        """Run the full comparison pipeline: load → compare → save."""
        self.load()
        self.run_residue_comparison()
        self.run_graph_comparison()
        self.save_results()
        logger.info("Comparator run complete.")
