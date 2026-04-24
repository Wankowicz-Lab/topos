#!/usr/bin/env python3
"""
For each comparison pair defined in a config TOML, produces:

  Global analysis (per pair)
  --------------------------
  * Descriptive statistics + Mann-Whitney U test for every continuous metric
    → <output_dir>/global_stats.csv
  * Count / total differences for every bond-count metric
    → <output_dir>/count_differences.csv

  Histograms (all structures combined, one JPG per metric)
  ---------------------------------------------------------
    → <output_dir>/histograms/<metric>.jpg

  Local differences (per pair)
  ----------------------------
  Residues within ``proximity_angstroms`` of the mutation site(s) or
  ligand atoms are flagged; the full difference table (all residues, all
  continuous + count metrics) is sorted by the biggest changes.
    → <output_dir>/local/<pair_description>_local_diffs.csv

Usage
-----
python run_analysis.py --config example_config.toml [--output-dir results/]
"""
from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu

import biotite.database.rcsb as rcsb
import biotite.structure as struc
from biotite.structure.io.pdb import PDBFile
from biotite.structure.io.pdbx import CIFFile
from biotite.structure.io.pdbx import get_structure as cif_get_structure


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--config", required=True, help="Path to grouped_structures.toml"
    )
    ap.add_argument(
        "--output-dir",
        default=None,
        help="Override output_dir from config",
    )
    ap.add_argument(
        "--metrics-dir",
        default=None,
        help="Override metrics_dir from config",
    )
    ap.add_argument(
        "--bins", type=int, default=30, help="Histogram bins (default: 30)"
    )
    ap.add_argument(
        "--no-histograms",
        action="store_true",
        help="Skip histogram generation (faster)",
    )
    ap.add_argument(
        "--no-proximity",
        action="store_true",
        help="Skip PDB fetching for proximity detection; report all residues",
    )
    return ap.parse_args()



plt.rcParams.update(
    {
        "font.size": 14,
        "axes.labelsize": 14,
        "axes.labelweight": "bold",
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "legend.fontsize": 11,
    }
)

_PALETTE = [
    "#4C72B0", "#DD8452", "#55A868", "#C44E52",
    "#8172B2", "#937860", "#DA8BC3", "#8C8C8C",
    "#CCB974", "#64B5CD",
]

# Columns that identify a residue (excluded from metric analysis).
_ID_COLS = frozenset(
    {
        "chain", "resi_struct", "resn_struct", "resi_mut", "resn_mut",
        "name", "ss_domains", "ss_group", "struct_info", "mut_info",
    }
)

# DSSP angle columns that are circular (degrees); use circular distance.
_CIRCULAR_ANGLE_COLS = frozenset(
    {"dssp_phi", "dssp_psi", "dssp_alpha", "dssp_kappa", "dssp_tco"}
)

# Sentinel value used by DSSP for undefined angles.
_DSSP_SENTINEL = 360.0

# Suffix patterns that indicate a count / integer metric.
_COUNT_SUFFIXES = ("_count",)
_COUNT_PREFIXES = ("total_",)
_COUNT_EXACT = frozenset(
    {
        "packing_n_atoms", "packing_n_neighbor_residues",
        "ss_domain_packing_n_atoms", "ss_domain_packing_n_neighbor_residues",
        "neighborhood_packing_n_atoms", "neighborhood_packing_n_neighbor_residues",
        "ss_domain_length",
        "graph_all_graph_core_number",
        "graph_vdw_contact_graph_core_number",
        "graph_hbond_graph_core_number",
        "n_ala_neighbors",
    }
)

# Skip these columns entirely (categorical / community IDs / boolean flags).
_SKIP_SUFFIXES = ("_community_id", "_in_lcc")

#_____HELPER FUNCTIONS_________________

def _classify_columns(
    df: pd.DataFrame,
) -> tuple[list[str], list[str]]:
    """Return (continuous_cols, count_cols) from *df*, excluding ID/skip cols."""
    continuous, counts = [], []
    for col in df.columns:
        if col in _ID_COLS:
            continue
        if any(col.endswith(s) for s in _SKIP_SUFFIXES):
            continue
        if not pd.api.types.is_numeric_dtype(df[col]):
            continue
        is_count = (
            any(col.endswith(s) for s in _COUNT_SUFFIXES)
            or any(col.startswith(p) for p in _COUNT_PREFIXES)
            or col in _COUNT_EXACT
            or "bond" in col
        )
        if is_count:
            counts.append(col)
        else:
            continuous.append(col)
    return continuous, counts



def _find_features_csv(pdb_id: str, metrics_dir: Path) -> Optional[Path]:
    """
    Locate the *_features.csv for *pdb_id* inside *metrics_dir*.

    Handles two naming conventions:
      {PDBID}_features.csv          (newer pipeline output)
      {pdbid}_final_qFit_features.csv  (older qFit output)
    """
    pdb_lower = pdb_id.lower()
    pdb_dir = metrics_dir / pdb_lower
    if not pdb_dir.exists():
        return None
    candidates = sorted(pdb_dir.glob("*_features.csv"))
    return candidates[0] if candidates else None


def load_all_metrics(
    entries,
    metrics_dir: Path,
) -> dict[str, pd.DataFrame]:
    """Load features CSVs for all entries; skip any that are missing."""
    out = {}
    for entry in entries:
        csv_path = _find_features_csv(entry.pdb_id, metrics_dir)
        if csv_path is None:
            print(
                f"  WARNING: no features CSV found for {entry.label} "
                f"({entry.pdb_id}) in {metrics_dir} — skipping."
            )
            continue
        df = pd.read_csv(csv_path, low_memory=False)
        df["resi_struct"] = df["resi_struct"].astype(int)
        out[entry.label] = df
    return out



def _load_structure_array(entry):
    """Return a biotite AtomArray for *entry* (fetch from RCSB if needed)."""

    extra = ["b_factor", "occupancy"]
    try:
        if entry.pdb_path is not None:
            path = Path(entry.pdb_path)
            ext = path.suffix.lstrip(".").lower()
            if ext in ("cif", "mmcif"):
                cif = CIFFile.read(str(path))
                return cif_get_structure(cif, model=1, extra_fields=extra, altloc="occupancy")
            pdb = PDBFile.read(str(path))
            return pdb.get_structure(model=1, extra_fields=extra, altloc="occupancy")

        # Fetch from RCSB
        obj = rcsb.fetch(entry.pdb_id, format="cif")
        tmp = NamedTemporaryFile(delete=False, suffix=".cif")
        tmp.write(obj.getvalue().encode("utf-8"))
        tmp.close()
        cif = CIFFile.read(tmp.name)
        return cif_get_structure(cif, model=1, extra_fields=extra, altloc="occupancy")
    except Exception as exc:
        print(f"  WARNING: could not load structure for {entry.label}: {exc}")
        return None


def _residues_near_position(arr, chain: str, resi: int, cutoff: float) -> set[int]:
    """Return residue numbers (res_id) within *cutoff* Å of residue *resi*."""
    try:
        import biotite.structure as struc
    except ImportError:
        return set()

    aa_mask = struc.filter_amino_acids(arr)
    chain_mask = arr.chain_id == chain
    target_mask = aa_mask & chain_mask & (arr.res_id == resi)
    if not target_mask.any():
        return set()

    target_coords = arr.coord[target_mask]  # (N_atoms, 3)
    all_coords = arr.coord[aa_mask & chain_mask]  # (M_atoms, 3)
    all_resi = arr.res_id[aa_mask & chain_mask]

    # Distance: min over target atoms
    dists = np.sqrt(
        ((all_coords[:, None, :] - target_coords[None, :, :]) ** 2).sum(axis=2)
    ).min(axis=1)

    return set(all_resi[dists <= cutoff].tolist())


def _residues_near_ligand(arr, ligand_name: str, chain: str, cutoff: float) -> set[int]:
    """Return protein residue numbers within *cutoff* Å of ligand *ligand_name*."""
    try:
        import biotite.structure as struc
    except ImportError:
        return set()

    lig_mask = arr.hetero & (np.char.strip(arr.res_name) == ligand_name)
    if not lig_mask.any():
        print(f"  WARNING: ligand '{ligand_name}' not found in structure.")
        return set()

    lig_coords = arr.coord[lig_mask]  # (L, 3)

    aa_mask = struc.filter_amino_acids(arr)
    chain_mask = arr.chain_id == chain
    prot_mask = aa_mask & chain_mask
    prot_coords = arr.coord[prot_mask]
    prot_resi = arr.res_id[prot_mask]

    dists = np.sqrt(
        ((prot_coords[:, None, :] - lig_coords[None, :, :]) ** 2).sum(axis=2)
    ).min(axis=1)

    return set(prot_resi[dists <= cutoff].tolist())


def get_proximity_residues(
    ref_entry,
    cmp_entry,
    cutoff: float,
) -> set[int]:
    """
    Return a set of residue numbers that are within *cutoff* Å of:
      - each mutation site listed in cmp_entry (measured on ref_entry's structure)
      - the ligand in either entry (measured on the bound structure)
    """
    proximity: set[int] = set()

    # Mutation proximity — load reference structure once
    if cmp_entry.mutations:
        arr = _load_structure_array(ref_entry)
        if arr is not None:
            chain = ref_entry.chain if isinstance(ref_entry.chain, str) else ref_entry.chain[0]
            for mut in cmp_entry.mutations:
                nbrs = _residues_near_position(arr, chain, mut.resi, cutoff)
                proximity |= nbrs

    # Ligand proximity — prefer the bound entry
    for entry in (cmp_entry, ref_entry):
        if entry.ligand is None:
            continue
        arr = _load_structure_array(entry)
        if arr is not None:
            lig = entry.ligand
            nbrs = _residues_near_ligand(arr, lig.name, lig.chain, cutoff)
            proximity |= nbrs
        break  # only need one bound structure

    return proximity

def _merge_pair(ref_df: pd.DataFrame, cmp_df: pd.DataFrame) -> pd.DataFrame:
    """Inner-join on (chain, resi_struct); disambiguate shared columns."""
    return ref_df.merge(
        cmp_df,
        on=["chain", "resi_struct"],
        suffixes=("_ref", "_cmp"),
        how="inner",
    )


def analyze_global(
    ref_df: pd.DataFrame,
    cmp_df: pd.DataFrame,
    ref_label: str,
    cmp_label: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Compute per-metric global statistics.

    Returns
    -------
    stats_df:
        One row per continuous metric with descriptive stats + Mann-Whitney.
    count_df:
        One row per count metric with per-structure totals and Δ.
    """
    merged = _merge_pair(ref_df, cmp_df)
    continuous_cols, count_cols = _classify_columns(ref_df)

    # -- Continuous metrics --------------------------------------------------
    stat_rows = []
    for col in continuous_cols:
        col_ref = f"{col}_ref" if f"{col}_ref" in merged.columns else col
        col_cmp = f"{col}_cmp" if f"{col}_cmp" in merged.columns else col
        if col_ref not in merged.columns or col_cmp not in merged.columns:
            continue

        ref_vals = pd.to_numeric(merged[col_ref], errors="coerce").dropna()
        cmp_vals = pd.to_numeric(merged[col_cmp], errors="coerce").dropna()
        if len(ref_vals) < 3 or len(cmp_vals) < 3:
            continue

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                stat, pval = mannwhitneyu(ref_vals, cmp_vals, alternative="two-sided")
            except Exception:
                stat, pval = float("nan"), float("nan")

        def _r(v):
            """Round a scalar, replacing inf/-inf/nan with None."""
            try:
                f = float(v)
                return None if not np.isfinite(f) else round(f, 4)
            except (TypeError, ValueError):
                return None

        with np.errstate(invalid="ignore", over="ignore"):
            stat_rows.append({
                "metric": col,
                "ref_label": ref_label,
                "ref_n": len(ref_vals),
                "ref_mean": _r(ref_vals.mean()),
                "ref_median": _r(ref_vals.median()),
                "ref_std": _r(ref_vals.std()),
                "ref_min": _r(ref_vals.min()),
                "ref_max": _r(ref_vals.max()),
                "cmp_label": cmp_label,
                "cmp_n": len(cmp_vals),
                "cmp_mean": _r(cmp_vals.mean()),
                "cmp_median": _r(cmp_vals.median()),
                "cmp_std": _r(cmp_vals.std()),
                "cmp_min": _r(cmp_vals.min()),
                "cmp_max": _r(cmp_vals.max()),
                "mean_delta": _r(cmp_vals.mean() - ref_vals.mean()),
                "median_delta": _r(cmp_vals.median() - ref_vals.median()),
                "mann_whitney_stat": _r(stat),
                "mann_whitney_p": _r(round(pval, 6)) if not np.isnan(pval) else None,
                "significant_p05": bool(pval < 0.05),
            })

    # -- Count / integer metrics ---------------------------------------------
    count_rows = []
    for col in count_cols:
        ref_total = pd.to_numeric(ref_df[col], errors="coerce").sum()
        cmp_total = pd.to_numeric(cmp_df[col], errors="coerce").sum()
        count_rows.append(
            {
                "metric": col,
                "ref_label": ref_label,
                "ref_total": int(ref_total),
                "cmp_label": cmp_label,
                "cmp_total": int(cmp_total),
                "delta": int(cmp_total - ref_total),
            }
        )

    return pd.DataFrame(stat_rows), pd.DataFrame(count_rows)



def generate_histograms(
    metrics_map: dict[str, pd.DataFrame],
    out_dir: Path,
    bins: int = 30,
) -> None:
    """
    For every continuous + count metric, save one overlaid histogram JPG
    showing all structures.
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    # Collect all metric columns across all DataFrames
    all_cols: set[str] = set()
    for df in metrics_map.values():
        cont, cnt = _classify_columns(df)
        all_cols.update(cont)
        all_cols.update(cnt)

    labels = list(metrics_map.keys())
    colours = {lbl: _PALETTE[i % len(_PALETTE)] for i, lbl in enumerate(labels)}

    for col in sorted(all_cols):
        fig, ax = plt.subplots(figsize=(8, 5))
        plotted = False
        for lbl, df in metrics_map.items():
            if col not in df.columns:
                continue
            vals = pd.to_numeric(df[col], errors="coerce").dropna()
            if vals.empty:
                continue
            ax.hist(
                vals,
                bins=bins,
                alpha=0.55,
                color=colours[lbl],
                edgecolor="none",
                label=lbl,
            )
            plotted = True

        if not plotted:
            plt.close(fig)
            continue

        ax.set_xlabel(col.replace("_", " "))
        ax.set_ylabel("Residue count")
        ax.set_title(col.replace("_", " "), fontsize=13, fontweight="bold")
        ax.legend(frameon=True, loc="upper right")
        ax.yaxis.grid(True, linestyle="--", alpha=0.4)
        ax.set_axisbelow(True)
        plt.tight_layout()

        safe_name = col.replace("/", "_").replace(" ", "_")
        fig.savefig(out_dir / f"{safe_name}.jpg", dpi=150)
        plt.close(fig)

def analyze_local(
    ref_df: pd.DataFrame,
    cmp_df: pd.DataFrame,
    ref_label: str,
    cmp_label: str,
    proximity_resi: set[int],
) -> pd.DataFrame:
    """
    Build a flat residue × metric difference table.

    Every residue and every numeric metric is included.  A boolean
    ``in_proximity`` column flags residues inside the proximity zone.
    Rows are sorted by ``abs_delta`` (largest changes first).
    """
    merged = _merge_pair(ref_df, cmp_df)
    if merged.empty:
        return pd.DataFrame()

    continuous_cols, count_cols = _classify_columns(ref_df)
    all_metric_cols = continuous_cols + count_cols

    # Resolve residue name (prefer ref side)
    resn_col = (
        "resn_struct_ref" if "resn_struct_ref" in merged.columns else "resn_struct"
    )

    rows = []
    for _, row in merged.iterrows():
        resi = int(row["resi_struct"])
        resn = str(row.get(resn_col, ""))
        in_prox = resi in proximity_resi

        for col in all_metric_cols:
            col_ref = f"{col}_ref" if f"{col}_ref" in merged.columns else col
            col_cmp = f"{col}_cmp" if f"{col}_cmp" in merged.columns else col
            if col_ref not in row.index or col_cmp not in row.index:
                continue
            try:
                ref_val = float(row[col_ref])
                cmp_val = float(row[col_cmp])
            except (ValueError, TypeError):
                continue
            if not np.isfinite(ref_val) or not np.isfinite(cmp_val):
                continue
            # Skip DSSP sentinel (360 = undefined)
            if col in _CIRCULAR_ANGLE_COLS and (
                ref_val == _DSSP_SENTINEL or cmp_val == _DSSP_SENTINEL
            ):
                continue
            raw_delta = cmp_val - ref_val
            # Circular correction for angle columns
            if col in _CIRCULAR_ANGLE_COLS:
                delta = ((raw_delta + 180) % 360) - 180
            else:
                delta = raw_delta
            rows.append(
                {
                    "resi": resi,
                    "resn": resn,
                    "metric": col,
                    f"{ref_label}_value": round(ref_val, 4),
                    f"{cmp_label}_value": round(cmp_val, 4),
                    "delta": round(delta, 4),
                    "abs_delta": round(abs(delta), 4),
                    "in_proximity": in_prox,
                }
            )

    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values("abs_delta", ascending=False).reset_index(drop=True)
    return df


def main():
    args = parse_args()
    config_path = Path(args.config)
    if not config_path.exists():
        sys.exit(f"ERROR: config not found: {config_path}")

    # Allow running from any directory
    sys.path.insert(0, str(config_path.resolve().parents[2]))
    from src.grouped_analysis.load_grouped_config import load_config

    entries, pairs, settings = load_config(config_path)

    metrics_dir = Path(args.metrics_dir or settings.get("metrics_dir", "."))
    out_dir = Path(args.output_dir or settings.get("output_dir", "results"))
    prefix = settings.get("output_prefix", "")
    cutoff = float(settings.get("proximity_angstroms", 8.0))

    out_dir.mkdir(parents=True, exist_ok=True)
    entry_map = {e.label: e for e in entries}

    # -- Load all metrics ----------------------------------------------------
    metrics_map = load_all_metrics(entries, metrics_dir)
    if not metrics_map:
        sys.exit("ERROR: No metrics CSVs could be loaded.")

    # -- Global histograms (all structures) ----------------------------------
    if not args.no_histograms:
        generate_histograms(
            metrics_map,
            out_dir / "histograms",
            bins=args.bins,
        )

    # -- Per-pair analysis ---------------------------------------------------
    all_stats: list[pd.DataFrame] = []
    all_counts: list[pd.DataFrame] = []

    for pair in pairs:
        ref_lbl = pair.reference
        cmp_lbl = pair.comparison
        desc = pair.description.replace(" ", "_").replace("/", "-")

        if ref_lbl not in metrics_map:
            print(f"  SKIP {pair.description}: reference '{ref_lbl}' metrics missing.")
            continue
        if cmp_lbl not in metrics_map:
            print(f"  SKIP {pair.description}: comparison '{cmp_lbl}' metrics missing.")
            continue

        ref_df = metrics_map[ref_lbl]
        cmp_df = metrics_map[cmp_lbl]
        ref_entry = entry_map[ref_lbl]
        cmp_entry = entry_map[cmp_lbl]

        # Global stats
        stats_df, count_df = analyze_global(ref_df, cmp_df, ref_lbl, cmp_lbl)
        stats_df["pair"] = pair.description
        count_df["pair"] = pair.description
        all_stats.append(stats_df)
        all_counts.append(count_df)

        # Proximity residues
        proximity_resi: set[int] = set()
        if not args.no_proximity:
            has_mutation = bool(cmp_entry.mutations)
            has_ligand = cmp_entry.ligand is not None or ref_entry.ligand is not None
            if has_mutation or has_ligand:
                proximity_resi = get_proximity_residues(ref_entry, cmp_entry, cutoff)
            else:
                print("  No mutations or ligand specified; local zone = all residues.")

        # Local differences
        local_df = analyze_local(
            ref_df, cmp_df, ref_lbl, cmp_lbl, proximity_resi
        )
        if not local_df.empty:
            local_dir = out_dir / "local"
            local_dir.mkdir(exist_ok=True)
            local_path = local_dir / f"{prefix}{desc}_local_diffs.csv"
            local_df.to_csv(local_path, index=False)
            n_prox = int(local_df["in_proximity"].sum())
    # -- Write combined global outputs ---------------------------------------
    if all_stats:
        stats_path = out_dir / f"{prefix}global_stats.csv"
        pd.concat(all_stats, ignore_index=True).to_csv(stats_path, index=False)

    if all_counts:
        count_path = out_dir / f"{prefix}count_differences.csv"
        pd.concat(all_counts, ignore_index=True).to_csv(count_path, index=False)



if __name__ == "__main__":
    main()
