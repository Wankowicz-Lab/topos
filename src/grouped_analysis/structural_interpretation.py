"""
Export per-residue structural annotations for interpretation.

Two modes
---------
multi   : Aggregate biogenesis outputs across all structures into a single
          residue-level annotation CSV.  Each row is one residue position
          (reference numbering).  Columns include mean structural properties,
          variability score/class, and an interpretation string.
          → dms_structural_annotations_multi.csv

comparison : Summarise per-pair local-difference tables (from
          run_comparison_metrics.py) into a compact residue × comparison
          annotation CSV.  Each row is one residue in one comparison pair.
          Δ values for key structural metrics, a composite change score, and
          a change_class label are computed.
          → dms_comparison_annotations_{pair}.csv  (one file per pair)

Usage
-----
# Multi-structure mode (reads adk_output/renumbered/ + adk_output/variability/)
python export_dms_annotations.py --mode multi

# Comparison mode (reads a directory containing *_local_diffs.csv files)
python export_dms_annotations.py --mode comparison --local-dir results/local/

# Override output directory
python export_dms_annotations.py --mode multi --out my_annotations/

Merging with your DMS data
--------------------------
The exported CSVs use ``resi`` as the residue key (matches the reference
numbering used throughout the pipeline).  To annotate your DMS table::

    import pandas as pd
    dms = pd.read_csv("my_dms_data.csv")           # must have a column 'position'
    ann = pd.read_csv("dms_structural_annotations_multi.csv")
    merged = dms.merge(ann, left_on="position", right_on="resi", how="left")

You can then ask questions like:
  - Do LOF mutations cluster in conserved, buried positions?
  - Which residues that show large ΔSASA in a mutant also have low DMS fitness?
  - Do high-variability residues tolerate more mutations?
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# ── Default paths (relative to this script) ──────────────────────────────────
_SCRIPT_DIR          = Path(__file__).resolve().parent
_DEFAULT_OUTPUT_DIR  = _SCRIPT_DIR / "adk_output"
_DEFAULT_RENUMBERED  = _DEFAULT_OUTPUT_DIR / "renumbered"
_DEFAULT_VARIABILITY = _DEFAULT_OUTPUT_DIR / "variability"

# Key per-residue structural metrics to average across structures
_KEY_METRICS = [
    "sasa",
    "sasa_sidechain",
    "total_hbond_count",
    "bb_hbond_count",
    "sc_hbond_count",
    "salt_bridge_count",
    "ionic_bond_count",
    "pi_stacking_count",
    "cation_pi_count",
    "vdw_contact_count",
    "packing_contact_density",
    "packing_n_neighbor_residues",
    "distance_to_center_of_mass",
    "distance_to_nearest_surface_residue",
    "graph_all_graph_betweenness_centrality",
    "graph_all_graph_closeness_centrality",
    "graph_all_graph_eigenvector_centrality",
    "n_neighbors",
    "neighbor_aa_entropy",
    "prop_long_range_neighbors",
    "neighbor_prop_alpha_helix",
    "neighbor_prop_beta_sheet",
]

# Key metrics to extract from comparison local-diff files
_CMP_METRICS = [
    "sasa",
    "total_hbond_count",
    "packing_contact_density",
    "vdw_contact_count",
    "salt_bridge_count",
    "distance_to_center_of_mass",
    "graph_all_graph_betweenness_centrality",
    "graph_all_graph_closeness_centrality",
]

# ── SASA classification thresholds (Å²) ──────────────────────────────────────
_SASA_BURIED     = 10.0   # < 10 → buried
_SASA_PARTIAL    = 40.0   # 10–40 → partially buried
# > 40 → exposed


# ── Helpers ───────────────────────────────────────────────────────────────────

def _sasa_class(sasa: float) -> str:
    if sasa < _SASA_BURIED:
        return "buried"
    if sasa < _SASA_PARTIAL:
        return "partially_buried"
    return "exposed"


def _variability_class(score: float, low_thr: float, high_thr: float) -> str:
    """Classify variability into three tiers based on score percentiles."""
    if score <= low_thr:
        return "conserved"
    if score <= high_thr:
        return "moderate"
    return "variable"


def _build_interpretation(row: pd.Series) -> str:
    """Generate a one-sentence structural interpretation for a residue."""
    parts = []

    vc = row.get("variability_class", "")
    if vc == "conserved":
        parts.append("structurally conserved across conformations")
    elif vc == "variable":
        parts.append("structurally variable across conformations")

    sc = row.get("sasa_class", "")
    if sc == "buried":
        parts.append("buried (low solvent exposure)")
    elif sc == "exposed":
        parts.append("solvent-exposed")
    elif sc == "partially_buried":
        parts.append("partially buried")

    hb = row.get("mean_total_hbond_count", float("nan"))
    if not np.isnan(hb):
        if hb >= 3:
            parts.append(f"high H-bond count (mean={hb:.1f})")
        elif hb <= 0.5:
            parts.append("few H-bonds")

    bc = row.get("mean_graph_all_graph_betweenness_centrality", float("nan"))
    if not np.isnan(bc):
        if bc > 0.05:
            parts.append("hub residue (high network centrality)")

    return "; ".join(parts) if parts else "no strong structural annotation"


# ── Mode 1: Multi-structure ───────────────────────────────────────────────────

def run_multi(chain: str, out_dir: Path, pdb_ids: list[str],
              renumbered_dir: Path, variability_dir: Path) -> None:
    """
    Aggregate renumbered features CSVs + variability outputs into a single
    per-residue annotation file for DMS interpretation.
    """
    print(f"[multi] Loading renumbered features from {renumbered_dir}/")

    # ── Load all structure data ───────────────────────────────────────────────
    frames = []
    loaded = []
    for pdb_id in pdb_ids:
        path = renumbered_dir / f"{pdb_id}_features.csv"
        if not path.exists():
            print(f"  WARNING: {path.name} not found — skipping.")
            continue
        df = pd.read_csv(path)
        df = df[df["chain"] == chain].copy()
        if df.empty:
            print(f"  WARNING: no chain {chain} data in {pdb_id} — skipping.")
            continue
        df = df.assign(pdb_id=pdb_id)
        frames.append(df)
        loaded.append(pdb_id)

    if not frames:
        sys.exit(f"ERROR: No usable features CSVs found for chain {chain}.")
    print(f"  Loaded {len(loaded)} structures: {', '.join(loaded)}")

    all_df = pd.concat(frames, ignore_index=True)

    # ── Per-residue averages of structural metrics ────────────────────────────
    avail_metrics = [m for m in _KEY_METRICS if m in all_df.columns]
    missing = set(_KEY_METRICS) - set(avail_metrics)
    if missing:
        print(f"  INFO: {len(missing)} requested metrics not found in CSVs (skipped): "
              f"{sorted(missing)[:5]}{'...' if len(missing) > 5 else ''}")

    grouped = all_df.groupby("resi_struct")

    mean_df = grouped[avail_metrics].mean()
    mean_df.columns = [f"mean_{c}" for c in mean_df.columns]

    sd_df = grouped[avail_metrics].std()
    sd_df.columns = [f"sd_{c}" for c in sd_df.columns]

    n_df = grouped["pdb_id"].count().rename("n_structures")

    # Most common residue name and secondary structure group
    resn_df   = grouped["resn_struct"].agg(lambda s: s.mode().iloc[0] if not s.empty else "UNK")
    ss_df     = grouped["ss_group"].agg(lambda s: s.mode().iloc[0] if not s.empty else "unknown")

    ann = pd.concat([resn_df, ss_df, n_df, mean_df, sd_df], axis=1)
    ann.index.name = "resi"
    ann = ann.rename(columns={"resn_struct": "resn", "ss_group": "ss_group_consensus"})

    # ── Merge variability ranking ─────────────────────────────────────────────
    var_path = variability_dir / "residue_variability_ranking.csv"
    if var_path.exists():
        var_df = pd.read_csv(var_path, index_col="resi_struct")
        ann = ann.join(var_df[["variability_score", "rank"]].rename(
            columns={"rank": "variability_rank"}
        ), how="left")

        # Three-tier variability class based on score percentiles
        low_thr  = ann["variability_score"].quantile(0.33)
        high_thr = ann["variability_score"].quantile(0.67)
        ann["variability_class"] = ann["variability_score"].apply(
            lambda s: _variability_class(s, low_thr, high_thr)
            if not np.isnan(s) else "unknown"
        )
    else:
        print(f"  WARNING: {var_path} not found. Run identify_variable_residues.py first.")
        ann["variability_score"]  = float("nan")
        ann["variability_rank"]   = float("nan")
        ann["variability_class"]  = "unknown"

    # ── SASA classification ───────────────────────────────────────────────────
    if "mean_sasa" in ann.columns:
        ann["sasa_class"] = ann["mean_sasa"].apply(
            lambda v: _sasa_class(v) if not np.isnan(v) else "unknown"
        )
    else:
        ann["sasa_class"] = "unknown"

    # ── Human-readable interpretation ─────────────────────────────────────────
    ann["interpretation"] = ann.apply(_build_interpretation, axis=1)

    # ── Output ────────────────────────────────────────────────────────────────
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "dms_structural_annotations_multi.csv"
    ann.reset_index().to_csv(out_path, index=False)
    print(f"\n[multi] Wrote {len(ann)} residues × {len(ann.columns)} columns → {out_path}")

    # Quick summary
    if "variability_class" in ann.columns:
        vc = ann["variability_class"].value_counts()
        print(f"\nVariability class breakdown:\n{vc.to_string()}")
    if "sasa_class" in ann.columns:
        sc = ann["sasa_class"].value_counts()
        print(f"\nSASA class breakdown:\n{sc.to_string()}")

    print(
        "\nTo merge with your DMS data:\n"
        "  dms = pd.read_csv('your_dms.csv')\n"
        f"  ann = pd.read_csv('{out_path}')\n"
        "  merged = dms.merge(ann, left_on='position', right_on='resi', how='left')"
    )


# ── Mode 2: Comparison ────────────────────────────────────────────────────────

def run_comparison(local_dir: Path, out_dir: Path, delta_threshold: float) -> None:
    """
    Summarise local-difference tables from run_comparison_metrics.py into
    compact per-residue annotations for each comparison pair.
    """
    local_files = sorted(local_dir.glob("*_local_diffs.csv"))
    if not local_files:
        sys.exit(f"ERROR: No *_local_diffs.csv files found in {local_dir}")
    print(f"[comparison] Found {len(local_files)} local-diff file(s) in {local_dir}/")

    out_dir.mkdir(parents=True, exist_ok=True)

    for local_path in local_files:
        pair_name = local_path.stem.replace("_local_diffs", "")
        print(f"\n  Processing: {pair_name}")

        diff_df = pd.read_csv(local_path)
        if diff_df.empty:
            print(f"    WARNING: empty file, skipping.")
            continue

        # ── Pivot: one row per residue, one column per metric delta ──────────
        avail = [m for m in _CMP_METRICS if m in diff_df["metric"].values]
        if not avail:
            print(f"    WARNING: none of the key metrics found in this file.")
            continue

        sub = diff_df[diff_df["metric"].isin(avail)].copy()

        # Pick up value column names (depend on pair labels)
        val_cols = [c for c in sub.columns if c.endswith("_value")]
        if len(val_cols) < 2:
            print(f"    WARNING: cannot identify ref/cmp value columns.")
            continue

        ref_val_col = val_cols[0]
        cmp_val_col = val_cols[1]
        ref_label = ref_val_col.replace("_value", "")
        cmp_label = cmp_val_col.replace("_value", "")

        # Pivot delta values: rows = resi, cols = metric
        pivot_delta = sub.pivot_table(
            index=["resi", "resn"],
            columns="metric",
            values="delta",
            aggfunc="first",
        )
        pivot_delta.columns = [f"{c}_delta" for c in pivot_delta.columns]

        # Proximity flag (any metric → True if any row for that resi is in_proximity)
        prox = (
            sub.groupby("resi")["in_proximity"].any()
            if "in_proximity" in sub.columns
            else pd.Series(False, index=sub["resi"].unique())
        )

        # abs_delta max per resi (across all metrics, not just key ones)
        max_abs = diff_df.groupby("resi")["abs_delta"].max().rename("max_abs_delta")

        # top changed metric name
        top_metric = (
            diff_df.sort_values("abs_delta", ascending=False)
            .drop_duplicates("resi")
            .set_index("resi")["metric"]
            .rename("top_changed_metric")
        )

        ann = (
            pivot_delta.reset_index()
            .set_index("resi")
            .join(prox.rename("in_proximity"))
            .join(max_abs)
            .join(top_metric)
        )

        # ── Composite change score ────────────────────────────────────────────
        # Mean of |delta| for key metrics, normalised by their global SD
        delta_cols = [f"{m}_delta" for m in avail if f"{m}_delta" in ann.columns]
        if delta_cols:
            abs_delta_matrix = ann[delta_cols].abs()
            global_sd = abs_delta_matrix.std().replace(0, 1)  # avoid /0
            normalised = abs_delta_matrix / global_sd
            ann["composite_change_score"] = normalised.mean(axis=1)
        else:
            ann["composite_change_score"] = float("nan")

        # ── Change class ──────────────────────────────────────────────────────
        if "composite_change_score" in ann.columns:
            q33 = ann["composite_change_score"].quantile(0.33)
            q67 = ann["composite_change_score"].quantile(0.67)
            q90 = ann["composite_change_score"].quantile(0.90)

            def _change_class(v):
                if np.isnan(v):
                    return "unknown"
                if v >= q90:
                    return "major"
                if v >= q67:
                    return "moderate"
                if v >= q33:
                    return "minor"
                return "minimal"

            ann["change_class"] = ann["composite_change_score"].apply(_change_class)

        # ── Reference / comparison labels ─────────────────────────────────────
        ann["ref_label"] = ref_label
        ann["cmp_label"] = cmp_label

        # ── Save ──────────────────────────────────────────────────────────────
        ann.index.name = "resi"
        out_path = out_dir / f"dms_comparison_annotations_{pair_name}.csv"
        ann.reset_index().to_csv(out_path, index=False)

        n_changed = int((ann.get("change_class", pd.Series()) == "major").sum())
        n_prox    = int(ann.get("in_proximity", pd.Series(False)).sum())
        print(f"    → {len(ann)} residues  |  {n_changed} major changes  |"
              f"  {n_prox} in proximity  →  {out_path.name}")

    print(
        f"\n[comparison] Done. Files in {out_dir}/\n"
        "\nTo merge with DMS data:\n"
        "  dms = pd.read_csv('your_dms.csv')\n"
        "  ann = pd.read_csv('dms_comparison_annotations_PAIR.csv')\n"
        "  merged = dms.merge(ann, left_on='position', right_on='resi', how='left')"
    )


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--mode", choices=["multi", "comparison"], required=True,
        help=(
            "multi: aggregate multi-structure variability + mean properties. "
            "comparison: summarise per-pair local-diff tables."
        ),
    )
    parser.add_argument("--chain", default="A",
                        help="Chain to analyse (default: A; multi mode only).")
    parser.add_argument(
        "--pdbs", default=None,
        help=(
            "Comma-separated PDB IDs to include (multi mode). "
            "Defaults to all CSVs found in --renumbered-dir."
        ),
    )
    parser.add_argument(
        "--renumbered-dir", type=Path, default=_DEFAULT_RENUMBERED,
        help="Directory containing renumbered features CSVs (default: adk_output/renumbered/).",
    )
    parser.add_argument(
        "--variability-dir", type=Path, default=_DEFAULT_VARIABILITY,
        help="Directory containing variability CSVs (default: adk_output/variability/).",
    )
    parser.add_argument(
        "--local-dir", type=Path, default=None,
        help="Directory containing *_local_diffs.csv files (comparison mode).",
    )
    parser.add_argument(
        "--out", type=Path, default=_DEFAULT_OUTPUT_DIR / "dms_annotations",
        help="Output directory (default: adk_output/dms_annotations/).",
    )
    parser.add_argument(
        "--delta-threshold", type=float, default=0.1,
        help="Minimum normalised delta to flag a change (comparison mode, default: 0.1).",
    )
    args = parser.parse_args()

    if args.mode == "multi":
        renumbered_dir = args.renumbered_dir
        if args.pdbs:
            pdb_ids = [p.strip().upper() for p in args.pdbs.split(",") if p.strip()]
        else:
            pdb_ids = sorted(
                p.name.replace("_features.csv", "").upper()
                for p in renumbered_dir.glob("*_features.csv")
            )
            if not pdb_ids:
                sys.exit(f"No *_features.csv files found in {renumbered_dir}.")
        run_multi(chain=args.chain, out_dir=args.out, pdb_ids=pdb_ids,
                  renumbered_dir=renumbered_dir, variability_dir=args.variability_dir)
    else:
        local_dir = args.local_dir or (_DEFAULT_OUTPUT_DIR / "comparisons" / "local")
        if not local_dir.exists():
            sys.exit(
                f"ERROR: --local-dir not found: {local_dir}\n"
                "Run run_comparison_metrics.py first, then pass --local-dir "
                "pointing to the directory with *_local_diffs.csv files."
            )
        run_comparison(local_dir=local_dir, out_dir=args.out,
                       delta_threshold=args.delta_threshold)


if __name__ == "__main__":
    main()
