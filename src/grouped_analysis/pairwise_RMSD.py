#!/usr/bin/env python3
"""
Pairwise sequence-aligned superposition + CA-RMSD for all structures defined
in a grouped_structures.toml.


Outputs
-------
  <prefix>pairwise_rmsd.csv      — one row per pair
  <prefix>pairwise_rmsd_hist.png — histogram colored by comparison type

Usage
-----
python pairwise_rmsd.py --config grouped_structures.toml [--output-dir results/]
"""
from __future__ import annotations

import argparse
import itertools
import sys
from pathlib import Path
from tempfile import NamedTemporaryFile

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import biotite.database.rcsb as rcsb
import biotite.sequence as bseq
import biotite.sequence.align as balign
import biotite.structure as struc
from biotite.structure.io.pdb import PDBFile
from biotite.structure.io.pdbx import CIFFile, get_structure as cif_get_structure

plt.rcParams.update({
    "font.size": 18,
    "axes.labelsize": 18,
    "axes.labelweight": "bold",
    "xtick.labelsize": 16,
    "ytick.labelsize": 16,
    "legend.fontsize": 14,
})

_STATE_COLORS = {
    "apo-apo":     "#4C72B0",
    "apo-bound":   "#DD8452",
    "bound-apo":   "#DD8452",
    "bound-bound": "#55A868",
}
_GENO_COLORS = {
    "wt-wt":         "#4C72B0",
    "wt-mutant":     "#DD8452",
    "mutant-wt":     "#DD8452",
    "mutant-mutant": "#C44E52",
}


_THREE_TO_ONE: dict[str, str] = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
    "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I",
    "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
    "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
}


def _load_arr(entry) -> struc.AtomArray:
    """Load a biotite AtomArray for a StructureEntry."""
    extra = ["b_factor", "occupancy"]

    if entry.pdb_path is not None:
        path = Path(entry.pdb_path)
        if not path.exists():
            sys.exit(f"ERROR: PDB file not found for '{entry.label}': {path}")
        ext = path.suffix.lstrip(".").lower()
        if ext in ("cif", "mmcif"):
            cif = CIFFile.read(str(path))
            return cif_get_structure(cif, model=1, extra_fields=extra, altloc="occupancy")
        else:
            pdb = PDBFile.read(str(path))
            return pdb.get_structure(model=1, extra_fields=extra, altloc="occupancy")
    else:
        # Fetch CIF from RCSB into a temp file
        obj = rcsb.fetch(entry.pdb_id, format="cif")
        tmp = NamedTemporaryFile(delete=False, suffix=".cif")
        tmp.write(obj.getvalue().encode("utf-8"))
        tmp.close()
        cif = CIFFile.read(tmp.name)
        return cif_get_structure(cif, model=1, extra_fields=extra, altloc="occupancy")


def extract_ca(arr: struc.AtomArray, chain_ids: list[str], label: str) -> struc.AtomArray:
    """Return CA-only AtomArray for standard amino acids in the given chains."""
    aa_mask    = struc.filter_amino_acids(arr)
    ca_mask    = arr.atom_name == "CA"
    chain_mask = np.isin(arr.chain_id, chain_ids)
    ca = arr[aa_mask & ca_mask & chain_mask]
    if ca.array_length() == 0:
        print(f"  WARNING: no CA atoms found for {label} in chains {chain_ids}")
    return ca


def _to_protein_seq(ca: struc.AtomArray) -> bseq.ProteinSequence:
    letters = "".join(_THREE_TO_ONE.get(r.strip(), "X") for r in ca.res_name)
    return bseq.ProteinSequence(letters)


def _matched_ca(ca1: struc.AtomArray, ca2: struc.AtomArray) -> tuple[struc.AtomArray, struc.AtomArray]:
    """
    Global BLOSUM62 alignment of the two CA sequences; return only the
    aligned (non-gap) positions from each structure.
    """
    seq1 = _to_protein_seq(ca1)
    seq2 = _to_protein_seq(ca2)
    matrix = balign.SubstitutionMatrix.std_protein_matrix()
    alignments = balign.align_optimal(
        seq1, seq2, matrix,
        gap_penalty=(-10, -1),
        terminal_penalty=False,
    )
    trace = alignments[0].trace          # shape (N, 2); -1 = gap
    valid = (trace[:, 0] != -1) & (trace[:, 1] != -1)
    idx1  = trace[valid, 0]
    idx2  = trace[valid, 1]
    return ca1[idx1], ca2[idx2]


def align_rmsd(ca1: struc.AtomArray, ca2: struc.AtomArray) -> tuple[float, int]:
    """
    Sequence-align ca1 vs ca2, superimpose, return (ca_rmsd_Å, n_aligned).

    ca1 is the fixed reference; ca2 is rotated/translated onto ca1.
    """
    fixed, mobile = _matched_ca(ca1, ca2)
    n_aligned = fixed.array_length()
    if n_aligned < 3:
        raise ValueError(f"Too few aligned residues ({n_aligned}) to superimpose.")

    fitted, _ = struc.superimpose(fixed, mobile)
    diff = fixed.coord - fitted.coord
    rmsd = float(np.sqrt(np.mean(np.sum(diff ** 2, axis=1))))
    return round(rmsd, 4), n_aligned

def state_comparison(e1, e2) -> str:
    return f"{e1.state}-{e2.state}"

def genotype_comparison(e1, e2) -> str:
    return f"{e1.genotype}-{e2.genotype}"

def parse_args():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config",     required=True, help="Path to grouped_structures.toml")
    ap.add_argument("--output-dir", default=None,
                    help="Output directory (default: taken from config or '.')")
    ap.add_argument("--color-by",   choices=["state", "genotype"], default="state",
                    help="Color histogram by state or genotype comparison (default: state)")
    ap.add_argument("--bins",       type=int, default=20, help="Histogram bins (default: 20)")
    return ap.parse_args()


def main():
    args = parse_args()

    config_path = Path(args.config)
    # Allow running from any directory
    sys.path.insert(0, str(config_path.resolve().parents[2]))
    from src.grouped_analysis.load_grouped_config import load_config

    entries, global_settings = load_config(config_path)
    if len(entries) < 2:
        sys.exit("ERROR: Need at least 2 structures for pairwise comparison.")

    out_dir = Path(args.output_dir or global_settings.get("output_dir", "."))
    out_dir.mkdir(parents=True, exist_ok=True)
    prefix = global_settings.get("output_prefix", "")


    ca_arrays: dict[str, struc.AtomArray] = {}
    for entry in entries:
        print(f"  Loading: {entry.label} ...", end=" ", flush=True)
        try:
            arr = _load_arr(entry)
        except Exception as exc:
            print(f"FAILED ({exc})")
            continue
        chains = [entry.chain] if isinstance(entry.chain, str) else list(entry.chain)
        ca = extract_ca(arr, chains, entry.label)
        if ca.array_length() > 0:
            ca_arrays[entry.label] = ca
            print(f"{ca.array_length()} CA atoms")

    valid_entries = [e for e in entries if e.label in ca_arrays]
    if len(valid_entries) < 2:
        sys.exit("ERROR: Fewer than 2 structures loaded successfully.")

    # ---- Pairwise alignment --------------------------------------- #
    pairs = list(itertools.combinations(valid_entries, 2))
    print(f"\nRunning pairwise alignment on {len(pairs)} pair(s)...\n")

    rows = []
    for e1, e2 in pairs:
        ca1 = ca_arrays[e1.label]
        ca2 = ca_arrays[e2.label]
        print(f"  {e1.label}  vs  {e2.label} ...", end=" ", flush=True)
        try:
            rmsd, n_aln = align_rmsd(ca1, ca2)
            print(f"RMSD = {rmsd:.3f} Å  ({n_aln} residues aligned)")
        except Exception as exc:
            print(f"FAILED ({exc})")
            rmsd, n_aln = float("nan"), 0

        rows.append({
            "label_1":             e1.label,
            "label_2":             e2.label,
            "state_1":             e1.state,
            "state_2":             e2.state,
            "genotype_1":          e1.genotype,
            "genotype_2":          e2.genotype,
            "ligand_1":            e1.ligand or "",
            "ligand_2":            e2.ligand or "",
            "mutations_1":         e1.mutation_summary,
            "mutations_2":         e2.mutation_summary,
            "state_comparison":    state_comparison(e1, e2),
            "genotype_comparison": genotype_comparison(e1, e2),
            "n_aligned_residues":  n_aln,
            "ca_rmsd_A":           rmsd,
        })

    df = pd.DataFrame(rows)

    csv_path = out_dir / f"{prefix}pairwise_rmsd.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nCSV saved  : {csv_path}")

    color_col = "state_comparison"    if args.color_by == "state" else "genotype_comparison"
    color_map = _STATE_COLORS         if args.color_by == "state" else _GENO_COLORS

    valid_df = df[df["ca_rmsd_A"].notna()].copy()
    comparison_types = sorted(valid_df[color_col].unique())

    fig, ax = plt.subplots(figsize=(10, 6))
    for ctype in comparison_types:
        subset = valid_df[valid_df[color_col] == ctype]["ca_rmsd_A"]
        ax.hist(
            subset,
            bins=args.bins,
            alpha=0.65,
            color=color_map.get(ctype, "#888888"),
            edgecolor="black",
            linewidth=0.8,
            label=ctype,
        )

    overall_median = valid_df["ca_rmsd_A"].median()
    ax.axvline(overall_median, color="black", linestyle="--", linewidth=1.5,
               label=f"median = {overall_median:.2f} Å")

    ax.set_xlabel("CA RMSD (Å)")
    ax.set_ylabel("Number of Pairs")
    ax.set_title(f"Pairwise CA RMSD — colored by {args.color_by}")
    ax.legend(frameon=True)
    ax.yaxis.grid(True, linestyle="--", alpha=0.5)
    ax.set_axisbelow(True)
    plt.tight_layout()

    hist_path = out_dir / f"{prefix}pairwise_rmsd_hist.png"
    plt.savefig(hist_path, dpi=300)
    plt.close()


if __name__ == "__main__":
    main()
