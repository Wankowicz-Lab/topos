"""
Generate PyMOL visualisation scripts from structural annotation CSVs.

Given any annotation CSV produced by this pipeline (multi-structure or
comparison mode), this script:
  1. Writes a PyMOL .pml script that colours residues by the chosen metric.
  2. Writes a B-factor-mapped PDB file where each residue's B-factor is
     replaced by the metric value (allows coloring in any structure viewer).

Coloring modes
--------------
spectrum  : continuous colour gradient (blue → white → red)
           Mapped to the full range of the metric, or --vmin/--vmax.
groups    : categorical colour (one colour per unique group/class label).
           Use for columns like variability_class, sasa_class, change_class.

Usage examples
--------------
# Color 4AKE by variability score (from multi-structure analysis)
python map_to_pymol.py \\
    --csv structural_annotations_multi.csv \\
    --metric variability_score \\
    --pdb 4AKE \\
    --output viz/4AKE_variability

# Color by SASA class (categorical)
python map_to_pymol.py \\
    --csv structural_annotations_multi.csv \\
    --metric sasa_class \\
    --pdb 4AKE \\
    --mode groups \\
    --output viz/4AKE_sasa_class

# Colour 4AKE by composite_change_score from a comparison
python map_to_pymol.py \\
    --csv comparison_annotations_WT_vs_mutant.csv \\
    --metric composite_change_score \\
    --pdb 4AKE \\
    --output viz/4AKE_comparison

Opening in PyMOL
----------------
  pymol viz/4AKE_variability.pml
  # or drag the .pml file into PyMOL

The B-factor PDB can be opened in PyMOL, ChimeraX, VMD, etc. and coloured
with the built-in "colour by B-factor" option.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import io

import biotite.database.rcsb as rcsb
import biotite.structure.io.pdb as pdbio

# ── Categorical colour palettes ───────────────────────────────────────────────
# _GROUP_COLORS maps group/category labels to hex color codes for PyMOL
_GROUP_COLORS = {
    # variability_class
    "conserved":         "0x4C72B0",   # blue
    "moderate":          "0xCCB974",   # yellow
    "variable":          "0xC44E52",   # red
    # sasa_class
    "buried":            "0x2C7BB6",   # deep blue
    "partially_buried":  "0xFDBD62",   # orange
    "exposed":           "0xD7191C",   # red
    # change_class
    "major":             "0xC44E52",   # red
    "moderate_change":   "0xDD8452",   # orange  (renamed to avoid clash)
    "minor":             "0xCCB974",   # yellow
    "minimal":           "0x4C72B0",   # blue
    "unknown":           "0x999999",   # grey
}

# Spectrum palettes available in PyMOL
_SPECTRUM_PALETTE = "blue_white_red"

def parse_args() -> argparse.Namespace:
    """
    Parse command line arguments for the script.
    Returns an argparse.Namespace with all arguments.
    """
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--csv", required=True, type=Path,
        help="Annotation CSV file (from export_dms_annotations.py or any pipeline output).",
    )
    parser.add_argument(
        "--metric", required=True,
        help="Column name to visualise (e.g. variability_score, sasa_class, composite_change_score).",
    )
    parser.add_argument(
        "--pdb", required=True,
        help="PDB ID (fetched from RCSB) or local .pdb/.cif file path.",
    )
    parser.add_argument(
        "--chain", default="A",
        help="Chain to colour (default: A).",
    )
    parser.add_argument(
        "--output", required=True,
        help="Output file prefix (directory + base name, e.g. viz/4AKE_variability).",
    )
    parser.add_argument(
        "--mode", choices=["spectrum", "groups", "both"], default="spectrum",
        help=(
            "spectrum: continuous gradient (numeric metrics). "
            "groups: categorical colours (class columns). "
            "both: generate both types. "
            "Default: spectrum."
        ),
    )
    parser.add_argument(
        "--resi-col", default="resi",
        help="Name of the residue-number column in the CSV (default: resi).",
    )
    parser.add_argument(
        "--vmin", type=float, default=None,
        help="Minimum value for spectrum colour scale (default: data minimum).",
    )
    parser.add_argument(
        "--vmax", type=float, default=None,
        help="Maximum value for spectrum colour scale (default: data maximum).",
    )
    parser.add_argument(
        "--no-bfactor-pdb", action="store_true",
        help="Skip writing the B-factor-mapped PDB file.",
    )
    return parser.parse_args()


# ── Helpers ───────────────────────────────────────────────────────────────────

def _load_annotation(csv_path: Path, metric: str, resi_col: str) -> pd.DataFrame:
    """
    Load the annotation CSV, validate that the residue and metric columns exist, 
    and return a cleaned table with 'resi' and metric columns only.

    High value: 
      - Defensive checks for required columns and clear error messages.
      - Only non-NA residue indices.
      - Ensures residue indices are integers (for mapping to PDB).
    """
    df = pd.read_csv(csv_path)
    if resi_col not in df.columns:
        sys.exit(f"ERROR: column '{resi_col}' not found in {csv_path}.\n"
                 f"Available columns: {list(df.columns)}")
    if metric not in df.columns:
        sys.exit(f"ERROR: metric column '{metric}' not found in {csv_path}.\n"
                 f"Available columns: {list(df.columns)}")
    df = df[[resi_col, metric]].dropna(subset=[resi_col])
    df[resi_col] = df[resi_col].astype(int)
    df = df.rename(columns={resi_col: "resi"})
    return df



# ── PML generation ────────────────────────────────────────────────────────────

def _write_spectrum_pml(
    df: pd.DataFrame,
    metric: str,
    pdb_source: str,
    chain: str,
    out_prefix: Path,
    vmin: float,
    vmax: float,
    pdb_path: str | None,
) -> None:
    """
    Write a PyMOL script for continuous (spectrum) colouring.

    High value:
      - Generates .pml that colours residues on a continuous spectrum based on metric.
      - Uses B-factor property (b) for easy PyMOL colouring.
      - Handles PDB load/local vs remote fetch.
      - Exports a publication-quality PNG of the view.
    """
    lines = ["# PyMOL script generated by map_to_pymol.py"]
    lines.append("# Metric: " + metric)
    lines.append("")
    # Handle local PDB load or fetch from RCSB
    if pdb_path:
        lines.append(f'load {pdb_path}, {pdb_source}')
    else:
        lines.append(f'fetch {pdb_source}, async=0')

    lines += [
        "",
        "hide everything",
        "show cartoon",
        f"color grey80, {pdb_source}",
        "",
        "# Set B-factors to metric values",
    ]

    # Set all B-factors for the chain to zero before setting specific values
    lines.append(f"alter {pdb_source} and chain {chain}, b=0.0")

    # Iterate over DataFrame, set B-factor of each residue to metric value
    for _, row in df.iterrows():
        resi = int(row["resi"])
        val  = row[metric]
        if not pd.isna(val):
            try:
                fval = float(val)
                lines.append(
                    f"alter {pdb_source} and chain {chain} and resi {resi}, b={fval:.4f}"
                )
            except (TypeError, ValueError):
                pass # skip if value can't be parsed to float

    # Add spectrum coloring command and legend/ramp for user reference
    lines += [
        "",
        "# Apply spectrum colouring based on B-factor (metric values)",
        f"spectrum b, {_SPECTRUM_PALETTE}, {pdb_source} and chain {chain}, "
        f"minimum={vmin:.4f}, maximum={vmax:.4f}",
        "",
        "# Colour residues with unset B-factor (b=0.0) as grey80",
        f"color grey80, {pdb_source} and chain {chain} and b=0.0",
        "",
        "# Ramp legend (requires PyMOL ≥ 2.0)",
        f"ramp_new colorbar, none, [{vmin:.4f}, {(vmin+vmax)/2:.4f}, {vmax:.4f}], "
        f"[blue, white, red]",
        "",
        "# Set a default orientation and perform ray tracing for a nice image",
        "set_view [\\",
        "     1.0,  0.0,  0.0,\\",
        "     0.0,  1.0,  0.0,\\",
        "     0.0,  0.0,  1.0,\\",
        "     0.0,  0.0,  -200.0,\\",
        "     0.0,  0.0,  0.0,  40.0, 200.0, -20.0 ]",
        "",
        "ray 1200, 900",
        f"png {out_prefix.name}_spectrum.png, dpi=150",
    ]

    # Write out the PML script
    pml_path = Path(str(out_prefix) + "_spectrum.pml")
    pml_path.parent.mkdir(parents=True, exist_ok=True)
    pml_path.write_text("\n".join(lines))


def _write_groups_pml(
    df: pd.DataFrame,
    metric: str,
    pdb_source: str,
    chain: str,
    out_prefix: Path,
    pdb_path: str | None,
) -> None:
    """
    Write a PyMOL script for categorical (group) colouring.

    High value:
      - Each unique metric value (group) gets a distinct color.
      - Handles auto-legend and color assignment.
      - Robust even if non-standard group names in input.
    """
    lines = ["# PyMOL script generated by map_to_pymol.py"]
    lines.append("# Metric (categorical): " + metric)
    lines.append("")

    # Handle local or remote PDB
    if pdb_path:
        lines.append(f'load {pdb_path}, {pdb_source}')
    else:
        lines.append(f'fetch {pdb_source}, async=0')

    lines += [
        "",
        "hide everything",
        "show cartoon",
        f"color grey80, {pdb_source}",
        "",
    ]

    # Group residues by group label and build legend
    groups: dict[str, list[int]] = {}
    for _, row in df.iterrows():
        key = str(row[metric]) if not pd.isna(row[metric]) else "unknown"
        groups.setdefault(key, []).append(int(row["resi"]))

    legend_lines: list[str] = []
    for group_name, residues in sorted(groups.items()):
        resi_str = "+".join(str(r) for r in sorted(residues))
        sel_name = f"grp_{group_name.replace(' ', '_')}"
        color_hex = _GROUP_COLORS.get(group_name, "0xAAAAAA")
        color_name = f"col_{group_name.replace(' ', '_')}"

        # Set the group color in PyMOL
        lines.append(f"set_color {color_name}, [{color_hex}]")
        # Select residues in this group
        lines.append(
            f"select {sel_name}, {pdb_source} and chain {chain} and resi {resi_str}"
        )
        # Color the selection
        lines.append(f"color {color_name}, {sel_name}")
        lines.append("")
        legend_lines.append(f"# {group_name} ({len(residues)} residues): {color_hex}")

    lines += ["# Legend:"] + legend_lines
    lines += [
        "",
        "ray 1200, 900",
        f"png {out_prefix.name}_groups.png, dpi=150",
    ]

    # Write out group-colouring PyMOL script
    pml_path = Path(str(out_prefix) + "_groups.pml")
    pml_path.parent.mkdir(parents=True, exist_ok=True)
    pml_path.write_text("\n".join(lines))


# ── B-factor PDB ──────────────────────────────────────────────────────────────

def _write_bfactor_pdb(
    df: pd.DataFrame,
    metric: str,
    pdb_id: str,
    chain: str,
    out_prefix: Path,
    vmin: float,
    vmax: float,
    pdb_path_local: str | None,
) -> None:
    """
    Download or read the PDB file and replace B-factors with metric values.
    Writes a new PDB file that can be opened in any structure viewer.

    High value:
      - Makes it easy to color by metric in any molecular viewer, not just PyMOL.
      - Fills in missing residues with vmin so default color is neutral/grey.
      - Supports both local and remote PDB fetch seamlessly.
    """

    # Build mapping: residue index (PDB numbering) to metric value
    resi_to_val: dict[int, float] = {}
    for _, row in df.iterrows():
        try:
            resi_to_val[int(row["resi"])] = float(row[metric])
        except (TypeError, ValueError):
            pass

    # Load PDB structure, supports both local file and RCSB fetch
    if pdb_path_local:
        pdb_file = pdbio.PDBFile.read(pdb_path_local)
    else:
        # Fetch raw PDB contents from RCSB and read directly from string
        raw = rcsb.fetch(pdb_id, format="pdb")
        pdb_file = pdbio.PDBFile()
        pdb_file.read(io.StringIO(raw.getvalue()))
    arr = pdb_file.get_structure(model=1, extra_fields=["b_factor"])

    # Iterate over atoms, set B-factor for correct chain/resi, fallback for missing
    for i, atom in enumerate(arr):
        if atom.chain_id == chain:
            val = resi_to_val.get(atom.res_id, float("nan"))
            if not np.isnan(val):
                arr.b_factor[i] = val
            else:
                arr.b_factor[i] = vmin  # missing residues get minimum value

    # Write the modified structure to an output file for coloring in external tools
    out_pdb = Path(str(out_prefix) + "_bfactor.pdb")
    out_pdb.parent.mkdir(parents=True, exist_ok=True)
    out_pdb_file = pdbio.PDBFile()
    out_pdb_file.set_structure(arr)
    out_pdb_file.write(str(out_pdb))


def main():
    """
    Entry point for command-line usage.
    High value:
      - Orchestrates data loading, decision on visualization mode, and output.
      - Handles both numeric and categorical metrics gracefully.
    """
    args = parse_args()
    df = _load_annotation(args.csv, args.metric, args.resi_col)

    # Determine if input PDB is a local file or remote (RCSB)
    pdb_path_arg = args.pdb
    is_local = Path(pdb_path_arg).exists()
    pdb_source = Path(pdb_path_arg).stem if is_local else pdb_path_arg.upper()
    pdb_local  = pdb_path_arg if is_local else None

    out_prefix = Path(args.output)

    # Compute numeric value range for color scale (only for numeric metrics)
    numeric_vals = pd.to_numeric(df[args.metric], errors="coerce").dropna()
    is_numeric   = len(numeric_vals) > 0 and numeric_vals.dtype.kind in "fi"

    vmin = args.vmin if args.vmin is not None else (float(numeric_vals.min()) if is_numeric else 0.0)
    vmax = args.vmax if args.vmax is not None else (float(numeric_vals.max()) if is_numeric else 1.0)

    # Smartly determine whether to use spectrum or group coloring automatically
    is_categorical = not is_numeric or df[args.metric].nunique() <= 10

    # Decide what PyMOL scripts to make based on mode and metric type
    do_spectrum = args.mode in ("spectrum", "both") and is_numeric
    do_groups   = (args.mode == "groups") or \
                  (args.mode == "both" and is_categorical) or \
                  (not is_numeric)


    if do_spectrum:
        # Generate continuous spectrum .pml script
        _write_spectrum_pml(df, args.metric, pdb_source, args.chain,
                            out_prefix, vmin, vmax, pdb_local)

    if do_groups:
        # Generate categorical (group/color) .pml script
        _write_groups_pml(df, args.metric, pdb_source, args.chain,
                          out_prefix, pdb_local)

    # Generate B-factor mapped PDB for use in any viewer (unless switched off)
    if not args.no_bfactor_pdb and is_numeric:
        _write_bfactor_pdb(df, args.metric, pdb_source, args.chain,
                           out_prefix, vmin, vmax, pdb_local)


if __name__ == "__main__":
    main()
