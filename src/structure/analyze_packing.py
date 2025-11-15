#!/usr/bin/env python3
"""
Compute residue packing for a PDB/mmCIF using Biotite (all altlocs).

Metric = number of *distinct neighboring residues* that have at least one heavy atom
within a cutoff (default 5.0 Å) of any heavy atom in the residue (excluding self).
"""

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd
from biotite.structure import filter_amino_acids, AtomArray
from biotite.structure.io.pdb import PDBFile
from biotite.structure.io.mmcif import MMCIFFile

def parse_args():
    ap = argparse.ArgumentParser(description="Residue packing analysis with Biotite (ignoring water/ligands).")
    ap.add_argument("--pdb", required=True, help="Input PDB/mmCIF file")
    ap.add_argument("--cutoff", type=float, default=5.0, help="Distance cutoff in Å (default: 5.0)")
    ap.add_argument("--model", type=int, default=0, help="Model index to use (0-based; default: 0)")
    ap.add_argument("--out", type=str, default=None, help="Output CSV path (default: <pdb_stem>_packing.csv)")
    return ap.parse_args()

def load_structure_any(path: str, model_index: int) -> AtomArray:
    ext = os.path.splitext(path)[1].lower()
    if ext in (".cif", ".mmcif"):
        mm = MMCIFFile.read(path)
        return mm.get_structure(model=model_index + 1, extra_fields=["b_factor", "occupancy", "altloc_id"])
    else:
        pdb = PDBFile.read(path)
        return pdb.get_structure(model=model_index + 1, extra_fields=["b_factor", "occupancy", "altloc_id"])

def is_heavy(atom_name: str) -> bool:
    n = atom_name.strip()
    return not (n.startswith("H") or n.startswith("D"))

def residue_key(chain_id, res_id):
    return f"{chain_id}:{int(res_id)}"

def main():
    args = parse_args()
    pdb_path = Path(args.pdb)
    out_csv = Path(args.out) if args.out else pdb_path.with_suffix("").with_name(f"{pdb_path.stem}_packing.csv")

    arr = load_structure_any(str(pdb_path), model_index=args.model)
    if arr is None or arr.array_length() == 0:
        raise RuntimeError("No atoms loaded from file/model.")

    # Filter to amino acids only
    aa_mask = filter_amino_acids(arr)
    arr = arr[aa_mask]

    # Keep only heavy atoms
    heavy_mask = np.array([is_heavy(n) for n in arr.atom_name], dtype=bool)
    arr = arr[heavy_mask]

    if arr.array_length() == 0:
        raise RuntimeError("No heavy atoms found in protein residues.")

    # Identify residues
    residue_ids = np.array([residue_key(c, r) for c, r in zip(arr.chain_id, arr.res_id)])
    unique_residues = np.unique(residue_ids)

    # Prepare coordinate array for neighbor search
    coords = arr.coord.astype(float)
    cutoff2 = args.cutoff ** 2

    rows = []
    for res_uid in unique_residues:
        idxs = np.where(residue_ids == res_uid)[0]
        if len(idxs) == 0:
            continue
        res_atoms = arr[idxs]
        chain = res_atoms.chain_id[0]
        resi = int(res_atoms.res_id[0])
        resn = res_atoms.res_name[0]

        # Compute neighbor residues within cutoff
        rcoords = res_atoms.coord
        diff = rcoords[:, None, :] - coords[None, :, :]
        d2 = np.einsum("ijk,ijk->ij", diff, diff)
        within_cutoff = (d2 <= cutoff2)

        close_atom_idxs = np.where(within_cutoff.any(axis=0))[0]
        neighbor_res_keys = set(residue_ids[close_atom_idxs].tolist())

        # exclude self
        if res_uid in neighbor_res_keys:
            neighbor_res_keys.remove(res_uid)

        n_atoms = len(res_atoms)
        n_neighbors = len(neighbor_res_keys)
        contact_density = n_neighbors / max(1, n_atoms)
        avg_b = float(np.mean(res_atoms.b_factor))
        occ = np.array([1.0 if o is None else float(o) for o in res_atoms.occupancy])
        avg_occ = float(np.mean(occ))

        rows.append({
            "chain": chain,
            "resi": resi,
            "resn": resn,
            "n_atoms": n_atoms,
            "n_neighbor_residues": n_neighbors,
            "contact_density": round(contact_density, 4),
            "avg_b": round(avg_b, 2),
            "avg_occupancy": round(avg_occ, 2),
        })

    df = pd.DataFrame(rows).sort_values(["chain", "resi"]).reset_index(drop=True)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)

if __name__ == "__main__":
    main()
