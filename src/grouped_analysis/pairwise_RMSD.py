#!/usr/bin/env python3
"""
Pairwise sequence-aligned superposition + CA-RMSD for all structures defined
in a grouped_structures.toml.

Outputs
-------
  <prefix>pairwise_rmsd.csv      — one row per pair

"""
from __future__ import annotations

import itertools
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from grouped_analysis.load_group_config import load_config

import biotite.database.rcsb as rcsb
import biotite.sequence as bseq
import biotite.sequence.align as balign
import biotite.structure as struc
from biotite.structure.io.pdb import PDBFile
from biotite.structure.io.pdbx import CIFFile, get_structure as cif_get_structure

import io

# Standard 3-letter to 1-letter amino acid code conversion
_THREE_TO_ONE: dict[str, str] = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
    "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I",
    "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
    "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
}

def _load_arr(entry) -> struc.AtomArray:
    """Load a biotite AtomArray for a StructureEntry.
    
    Tries to read structure file from provided path; if not available, fetches from RCSB.
    Handles both PDB and CIF/MMCIF formats.
    """
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
        # Fetch CIF from RCSB 
        obj = rcsb.fetch(entry.pdb_id, format="cif")
        buf = io.StringIO(obj.getvalue())
        cif = CIFFile.read(buf)
        return cif_get_structure(cif, model=1, extra_fields=extra, altloc="occupancy")


def extract_ca(arr: struc.AtomArray, chain_ids: list[str], label: str) -> struc.AtomArray:
    """
    Return CA-only AtomArray for standard amino acids in the given chains.

    Filters for C-alpha atoms (CA) in specified chains and for standard amino acids.
    """
    aa_mask    = struc.filter_amino_acids(arr)
    ca_mask    = arr.atom_name == "CA"
    chain_mask = np.isin(arr.chain_id, chain_ids)
    ca = arr[aa_mask & ca_mask & chain_mask]
    return ca

def _to_protein_seq(ca: struc.AtomArray) -> bseq.ProteinSequence:
    """
    Convert a CA-only AtomArray to a biotite ProteinSequence using 3-to-1 letter codes.
    Residues not in the known dictionary become 'X'.
    """
    letters = "".join(_THREE_TO_ONE.get(r.strip(), "X") for r in ca.res_name)
    return bseq.ProteinSequence(letters)

def _matched_ca(ca1: struc.AtomArray, ca2: struc.AtomArray) -> tuple[struc.AtomArray, struc.AtomArray]:
    """
    Given two AtomArrays with only C-alpha atoms, globally align their sequences
    (using BLOSUM62) and return atom arrays containing only the aligned, non-gap CA positions.
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
    Sequence-align ca1 vs ca2, superimpose their structures, and return the CA RMSD (Å)
    along with number of aligned residues.
    CA1 is treated as the reference; ca2 is superimposed onto it.
    Raises ValueError if <3 aligned residues.
    """
    fixed, mobile = _matched_ca(ca1, ca2)
    n_aligned = fixed.array_length()
    if n_aligned < 3:
        raise ValueError(f"Too few aligned residues ({n_aligned}) to superimpose.")

    # Superimpose ca2 onto ca1 based on matched CAs, then compute pairwise distance RMSD
    fitted, _ = struc.superimpose(fixed, mobile)
    diff = fixed.coord - fitted.coord
    rmsd = float(np.sqrt(np.mean(np.sum(diff ** 2, axis=1))))
    return round(rmsd, 4), n_aligned

def state_comparison(e1, e2) -> str:
    """Return a hyphen-joined label describing the state pair, e.g. 'apo-bound'."""
    return f"{e1.state}-{e2.state}"

def genotype_comparison(e1, e2) -> str:
    """Return a hyphen-joined label describing the genotype pair, e.g. 'wt-mutant'."""
    return f"{e1.genotype}-{e2.genotype}"

def compute_pairwise_rmsd(config_path: Path, output_dir: Path | None = None) -> pd.DataFrame:
    """
    Compute pairwise sequence-aligned CA-RMSD for all structures defined in a config TOML.

    Parameters
    ----------
    config_path : Path
        Path to the grouped_structures TOML config file.
    output_dir : Path or None
        Directory to write ``pairwise_rmsd.csv``.  Falls back to ``output_dir``
        in the config, then the current directory.

    Returns
    -------
    pd.DataFrame
        One row per structure pair with RMSD and metadata columns.
    """
    entries, _pairs, global_settings = load_config(config_path)
    if len(entries) < 2:
        sys.exit("ERROR: Need at least 2 structures for pairwise comparison.")

    out_dir = Path(output_dir or global_settings.get("output_dir", "."))
    out_dir.mkdir(parents=True, exist_ok=True)
    prefix = global_settings.get("output_prefix", "")

    ca_arrays: dict[str, struc.AtomArray] = {}
    for entry in entries:
        try:
            arr = _load_arr(entry)
        except Exception:
            continue
        chains = [entry.chain] if isinstance(entry.chain, str) else list(entry.chain)
        ca = extract_ca(arr, chains, entry.label)
        if ca.array_length() > 0:
            ca_arrays[entry.label] = ca

    valid_entries = [e for e in entries if e.label in ca_arrays]
    if len(valid_entries) < 2:
        sys.exit("ERROR: Fewer than 2 structures loaded successfully.")

    rows = []
    for e1, e2 in itertools.combinations(valid_entries, 2):
        ca1 = ca_arrays[e1.label]
        ca2 = ca_arrays[e2.label]
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
    df.to_csv(out_dir / f"{prefix}pairwise_rmsd.csv", index=False)
    return df
