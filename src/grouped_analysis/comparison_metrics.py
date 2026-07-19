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

"""
from __future__ import annotations

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

from grouped_analysis.load_group_config import load_config




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
    """
    Classify numeric metric columns into continuous vs count-like.

    Inputs
    ------
    df : pd.DataFrame
        Metrics table containing residue identifiers and numeric features.

    Output
    ------
    tuple[list[str], list[str]]
        Two lists: continuous metric column names and count metric column names.
    """
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
    Locate the *_features.csv for a PDB entry in supported layouts.

    Inputs
    ------
    pdb_id : str
        PDB identifier to locate.
    metrics_dir : Path
        Root directory containing features CSV files.

    Output
    ------
    Optional[Path]
        Matching CSV path if found; otherwise None.
    """
    # Flat layout (topos default)
    flat = metrics_dir / f"{pdb_id}_features.csv"
    if flat.exists():
        return flat
    flat_lower = metrics_dir / f"{pdb_id.lower()}_features.csv"
    if flat_lower.exists():
        return flat_lower

    # Subdirectory layout (legacy qFit format)
    pdb_dir = metrics_dir / pdb_id.lower()
    if pdb_dir.exists():
        candidates = sorted(pdb_dir.glob("*_features.csv"))
        if candidates:
            return candidates[0]

    return None


def load_all_metrics(
    entries,
    metrics_dir: Path,
) -> dict[str, pd.DataFrame]:
    """
    Load metrics CSVs for all entries listed in the config.

    Inputs
    ------
    entries : iterable
        Config entries with `pdb_id` and `label` fields.
    metrics_dir : Path
        Directory containing features CSVs.

    Output
    ------
    dict[str, pd.DataFrame]
        Mapping from entry label to its metrics DataFrame.
    """
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
    """
    Load a biotite AtomArray for the entry, fetching from RCSB if needed.

    Inputs
    ------
    entry : object
        Config entry with `pdb_id` and optional `pdb_path`.

    Output
    ------
    biotite.structure.AtomArray
        Structure array for model 1 with extra fields populated.
    """

    extra = ["b_factor", "occupancy"]
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


def _residues_near_position(arr, chain: str, resi: int, cutoff: float) -> set[int]:
    """
    Compute protein residues within a cutoff around a target residue.

    Inputs
    ------
    arr : biotite.structure.AtomArray
    chain : str
        Chain identifier containing the target residue.
    resi : int
        Residue number of the target position.
    cutoff : float
        Distance cutoff in Angstroms.

    Output
    ------
    set[int]
        Residue numbers within the cutoff distance.
    """

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
    """
    Compute protein residues within a cutoff around ligand atoms.

    Inputs
    ------
    arr : biotite.structure.AtomArray
        Structure array to search.
    ligand_name : str
        Ligand residue name.
    chain : str
        Chain identifier containing the ligand.
    cutoff : float
        Distance cutoff in Angstroms.

    Output
    ------
    set[int]
        Residue numbers within the cutoff distance.
    """
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
    Compute per-metric global statistics for one comparison pair.

    Inputs
    ------
    ref_df : pd.DataFrame
        Reference metrics table.
    cmp_df : pd.DataFrame
        Comparison metrics table.
    ref_label : str
        Label for the reference structure.
    cmp_label : str
        Label for the comparison structure.

    Output
    ------
    tuple[pd.DataFrame, pd.DataFrame]
        stats_df with descriptive stats and Mann-Whitney results, and
        count_df with per-structure totals and deltas for count metrics.
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
    Build a residue-by-metric delta table for a comparison pair.
    Every residue and every numeric metric is included.  A boolean
    ``in_proximity`` column flags residues inside the proximity zone.
    Rows are sorted by ``abs_delta`` (largest changes first).

    Inputs
    ------
    ref_df : pd.DataFrame
        Reference metrics table.
    cmp_df : pd.DataFrame
        Comparison metrics table.
    ref_label : str
        Label for the reference structure.
    cmp_label : str
        Label for the comparison structure.
    proximity_resi : set[int]
        Residue numbers considered near mutations/ligands.
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


def run_comparison_analysis(
    config_path: Path,
    metrics_dir: Path,
    out_dir: Path,
    bins: int = 30,
    no_histograms: bool = False,
    no_proximity: bool = False,
) -> None:
    """
    Programmatic entry point: run per-pair comparison analysis and write outputs.

    Parameters
    ----------
    config_path : Path
        Path to the grouped_structures TOML config file.
    metrics_dir : Path
        Directory containing ``{PDB_ID}_features.csv`` files.
    out_dir : Path
        Directory where CSV and histogram outputs are written.
    bins : int
        Number of histogram bins.
    no_histograms : bool
        Skip histogram generation.
    no_proximity : bool
        Skip PDB fetching for proximity detection; report all residues.
    """
    entries, pairs, settings = load_config(config_path)

    prefix = settings.get("output_prefix", "")
    cutoff = float(settings.get("proximity_angstroms", 8.0))

    out_dir.mkdir(parents=True, exist_ok=True)
    entry_map = {e.label: e for e in entries}

    # -- Load all metrics ----------------------------------------------------
    metrics_map = load_all_metrics(entries, metrics_dir)
    if not metrics_map:
        sys.exit("ERROR: No metrics CSVs could be loaded.")

    # -- Global histograms (all structures) ----------------------------------
    if not no_histograms:
        generate_histograms(metrics_map, out_dir / "histograms", bins=bins)

    # -- Per-pair analysis ---------------------------------------------------
    all_stats: list[pd.DataFrame] = []
    all_counts: list[pd.DataFrame] = []

    for pair in pairs:
        ref_lbl = pair.reference
        cmp_lbl = pair.comparison
        desc = pair.description.replace(" ", "_").replace("/", "-")

        ref_df = metrics_map[ref_lbl]
        cmp_df = metrics_map[cmp_lbl]
        ref_entry = entry_map[ref_lbl]
        cmp_entry = entry_map[cmp_lbl]

        stats_df, count_df = analyze_global(ref_df, cmp_df, ref_lbl, cmp_lbl)
        stats_df["pair"] = pair.description
        count_df["pair"] = pair.description
        all_stats.append(stats_df)
        all_counts.append(count_df)

        proximity_resi: set[int] = set()
        if not no_proximity:
            has_mutation = bool(cmp_entry.mutations)
            has_ligand = cmp_entry.ligand is not None or ref_entry.ligand is not None
            if has_mutation or has_ligand:
                proximity_resi = get_proximity_residues(ref_entry, cmp_entry, cutoff)

        local_df = analyze_local(ref_df, cmp_df, ref_lbl, cmp_lbl, proximity_resi)
        if not local_df.empty:
            local_dir = out_dir / "local"
            local_dir.mkdir(exist_ok=True)
            local_path = local_dir / f"{prefix}{desc}_local_diffs.csv"
            local_df.to_csv(local_path, index=False)

    # -- Write combined global outputs ---------------------------------------
    if all_stats:
        pd.concat(all_stats, ignore_index=True).to_csv(
            out_dir / f"{prefix}global_stats.csv", index=False
        )
    if all_counts:
        pd.concat(all_counts, ignore_index=True).to_csv(
            out_dir / f"{prefix}count_differences.csv", index=False
        )


