#!/usr/bin/env python3
"""
Map any numeric column from a metrics CSV to B-factors in a PDB file.

Usage
-----
python map_metric_to_bfactor.py \
    --pdb input.pdb \
    --metrics metrics.csv \
    --column OP_Diff \
    [--chain_col chain] \
    [--resi_col resi] \
    [--output output.pdb]

Residues not found in the metrics CSV receive the median value of the column.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import biotite.structure as struc
from biotite.structure.io.pdb import PDBFile


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pdb",      required=True, help="Input PDB file")
    ap.add_argument("--metrics",  required=True, help="CSV file with per-residue metrics")
    ap.add_argument("--column",   required=True, help="Column name in the CSV to map to B-factor")
    ap.add_argument("--chain_col", default="chain", help="Column name for chain ID (default: chain)")
    ap.add_argument("--resi_col",  default="resi",  help="Column name for residue number (default: resi)")
    ap.add_argument("--output",   default=None,
                    help="Output PDB filename. Defaults to <stem>_<column>.pdb")
    return ap.parse_args()


def derive_output_name(pdb_path: Path, column: str) -> Path:
    safe_col = column.replace(" ", "_").replace("/", "_").replace("\\", "_")
    return pdb_path.parent / f"{pdb_path.stem}_{safe_col}.pdb"


def load_metrics(
    metrics_path: Path, column: str, chain_col: str, resi_col: str
) -> tuple[dict[tuple[str, int], float], float]:
    df = pd.read_csv(metrics_path)

    missing = [c for c in [column, chain_col, resi_col] if c not in df.columns]
    if missing:
        sys.exit(
            f"ERROR: Column(s) not found in metrics file: {missing}\n"
            f"Available columns: {list(df.columns)}"
        )

    df[resi_col] = df[resi_col].astype(int)
    median_val = float(df[column].median())

    lookup: dict[tuple[str, int], float] = {
        (str(row[chain_col]), int(row[resi_col])): float(row[column])
        for _, row in df.iterrows()
    }

    return lookup, median_val


def map_to_bfactor(
    arr: struc.AtomArray,
    lookup: dict[tuple[str, int], float],
    median_val: float,
) -> struc.AtomArray:
    """Return a copy of arr with b_factor set from lookup."""
    arr = arr.copy()
    b_vals = np.full(arr.array_length(), median_val, dtype=float)

    res_starts = struc.get_residue_starts(arr)
    # append sentinel so we can slice [start:end] for the last residue too
    bounds = np.append(res_starts, arr.array_length())

    for start, end in zip(bounds[:-1], bounds[1:]):
        chain   = arr.chain_id[start]
        resi    = int(arr.res_id[start])
        resname = arr.res_name[start].strip()
        hetero  = bool(arr.hetero[start])

        if hetero:
            val = median_val
        else:
            val = lookup.get((chain, resi), median_val)

        b_vals[start:end] = val

    arr.b_factor = b_vals
    return arr


def main():
    args = parse_args()

    pdb_path = Path(args.pdb)
    if not pdb_path.exists():
        sys.exit(f"ERROR: PDB file not found: {pdb_path}")

    metrics_path = Path(args.metrics)
    if not metrics_path.exists():
        sys.exit(f"ERROR: Metrics file not found: {metrics_path}")

    output_path = Path(args.output) if args.output else derive_output_name(pdb_path, args.column)

    lookup, median_val = load_metrics(metrics_path, args.column, args.chain_col, args.resi_col)

    pdb_file = PDBFile.read(str(pdb_path))
    arr = pdb_file.get_structure(model=1, extra_fields=["b_factor", "occupancy"])

    arr = map_to_bfactor(arr, lookup, median_val)

    out_pdb = PDBFile()
    out_pdb.set_structure(arr)
    out_pdb.write(str(output_path))


if __name__ == "__main__":
    main()
