#!/usr/bin/env python3

import os, glob, argparse, warnings
from collections import defaultdict

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
import ot

import biotite.structure as struc
import biotite.structure.io as strucio


# ───────────────────────── Pocket parsing ──────────────────────────────
ELEMENTS = [
    "C", "N", "O", "S", "P",
    "F", "CL", "BR", "I",           # halogens
    "ZN", "MG", "CA", "MN", "FE",   # common metals
    "OTHER"
]

# Ignore Water & Common buffer/salt/cryoprotectant/etc
_IGNORE_RESNAMES = {
    "HOH",
    "MEA","CIR","FMT","BR","PEG","PG4","ZN","ABA","PTR","KCX",
    "MGF","OCS","DTT","CL","B3L","PFF","NH2","NA","ALY","SO4",
    "SGM","PO4","TBS","NO3","MG","EDO","ACE","GOL","DMS","TPO"
}

def _one_hot_element(elem: str) -> np.ndarray:
    e = (elem or "").upper()
    if e in ELEMENTS[:-1]:
        idx = ELEMENTS.index(e)
    else:
        idx = len(ELEMENTS) - 1
    vec = np.zeros(len(ELEMENTS), dtype=float)
    vec[idx] = 1.0
    return vec

def _largest_nonprotein_hetero_residue(array: struc.AtomArray) -> np.ndarray:
    """
    Return a boolean mask for atoms belonging to the largest (by heavy atoms)
    non-amino-acid, non-solvent residue, excluding common buffer components.
    Raises ValueError if none found.
    """
    # Heavy atoms only for counting
    heavy_mask = (array.element != "H")

    # Identify residue spans
    starts = struc.get_residue_starts(array)
    ends   = struc.get_residue_ends(array)

    # Classify atoms
    aa_mask      = struc.filter_amino_acids(array, standard=True)
    solvent_mask = struc.filter_solvent(array) 
    hetero_mask  = ~aa_mask & ~solvent_mask

    best_span = None
    best_count = -1

    for s, e in zip(starts, ends):
        res_slice = slice(s, e)
        if not np.any(hetero_mask[res_slice]):
            continue
        resname = array.res_name[s]
        if (resname or "").upper() in _IGNORE_RESNAMES:
            continue
        count_heavy = int(np.sum(heavy_mask[res_slice]))
        if count_heavy > best_count:
            best_count = count_heavy
            best_span = (s, e)

    if best_span is None or best_count <= 0:
        raise ValueError("no ligand found")

    s, e = best_span
    lig_mask = np.zeros(array.array_length(), dtype=bool)
    lig_mask[s:e] = True
    return lig_mask

def pocket_from_pdb(pdb_file: str, shell: float = 4.0, dcut: float = 5.0):
    """
    Build pocket representation around the largest hetero ligand.

    Returns:
        C : (n × n) geometry matrix  (Å)
        F : (n × d) feature matrix   (one-hot element + ligand flag)
        w : (n,)   uniform mass      (probability)
    """
    # Load as a single AtomArray (or concatenate first model)
    struct = strucio.load_structure(pdb_file)
    if isinstance(struct, list):
        # In case multiple models returned, take first
        struct = struct[0]
        
    array = struct.copy()

    # Find ligand residue
    lig_mask = _largest_nonprotein_hetero_residue(array)
    lig_idx = np.nonzero(lig_mask)[0]
    if lig_idx.size == 0:
        raise ValueError(f"{pdb_file}: no ligand atoms after filtering")

    # Build a mask of pocket atoms: within `shell` Å of any ligand atom
    all_xyz = array.coord
    lig_xyz = all_xyz[lig_mask]
    # Distance from all atoms to any ligand atom
    dists = cdist(all_xyz, lig_xyz)
    within_shell = (dists <= shell).any(axis=1)

    # Pocket atoms: heavy atoms only
    heavy_mask = (array.element != "H")
    pocket_mask = within_shell & heavy_mask
    pocket_idx = np.nonzero(pocket_mask)[0]
    if pocket_idx.size == 0:
        raise ValueError(f"{pdb_file}: empty pocket")

    # Sort by original index to have deterministic ordering
    pocket_idx.sort()

    # Geometry matrix C
    xyz = all_xyz[pocket_idx]
    C = cdist(xyz, xyz)  # Å; full matrix (no cutoff)

    # Feature matrix F: one-hot element + ligand flag
    F_rows = []
    # ligand flag for pocket atoms:
    lig_set = set(np.nonzero(lig_mask)[0])
    for idx in pocket_idx:
        elem = (array.element[idx] or "").upper()
        onehot = _one_hot_element(elem)
        lig_flag = 1.0 if idx in lig_set else 0.0
        F_rows.append(np.concatenate([onehot, [lig_flag]]))
    F = np.vstack(F_rows)

    # Uniform weights
    n = len(pocket_idx)
    w = np.ones(n, dtype=float) / n
    return C, F, w


# ───────────────────────── FGW distance ────────────────────────────────
def fgw_distance2(Ca, Cb, Fa, Fb, pa, pb,
                  alpha=0.7, epsilon=5e-3, iters=100, tol=1e-9):
    """
    Squared Fused GW distance between two pockets using POT.
    Returns scalar (squared distance) from the optimizer log.
    """
    M = ot.dist(Fa, Fb, metric="sqeuclidean")
    T, log = ot.gromov.fused_gromov_wasserstein(
        M, Ca, Cb, pa, pb,
        loss_fun='square_loss',
        alpha=alpha,
        epsilon=epsilon,
        max_iter=iters,
        tol=tol,
        verbose=False, log=True
    )
    return log['fgw_dist']  # squared distance


def main(args):
    pdbs = sorted(glob.glob(os.path.join(args.pdb_dir, "*.pdb")))
    if not pdbs:
        raise SystemExit(f"no PDBs found in {args.pdb_dir}")

    pockets = {}
    for p in pdbs:
        try:
            pockets[p] = pocket_from_pdb(p, args.shell, args.dcut)
        except ValueError as e:
            warnings.warn(str(e))

    names = list(pockets.keys())
    n = len(names)
    if n == 0:
        raise SystemExit("no valid pockets constructed")

    D2 = np.zeros((n, n), dtype=float)
    for i in range(n):
        Ci, Fi, pi = pockets[names[i]]
        for j in range(i+1, n):
            Cj, Fj, pj = pockets[names[j]]
            d2 = fgw_distance2(
                Ci, Cj, Fi, Fj, pi, pj,
                alpha=args.alpha,
                epsilon=args.epsilon,
                iters=args.max_iter,
                tol=args.tol
            )
            D2[i, j] = D2[j, i] = d2

    # Save
    np.save(args.npy_out, D2)
    labels = [os.path.basename(nm) for nm in names]
    pd.DataFrame(D2, index=labels, columns=labels).to_csv(args.csv_out)


if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=UserWarning)
    ap = argparse.ArgumentParser(description="FGW distances between binding pockets (Biotite version)")
    ap.add_argument("pdb_dir", help="folder with .pdb files")
    ap.add_argument("--shell",   type=float, default=4.0,
                    help="pocket radius Å around ligand (default 4)")
    ap.add_argument("--dcut",    type=float, default=5.0,
                    help="(reserved) edge cutoff Å for geometry matrix (default 5)")
    ap.add_argument("--alpha",   type=float, default=0.7,
                    help="FGW balance: 1=geometry only, 0=chemistry only (default 0.7)")
    ap.add_argument("--epsilon", type=float, default=5e-3,
                    help="entropic reg ε (default 5e-3)")
    ap.add_argument("--max_iter", type=int,   default=100,
                    help="max FGW iterations (default 100)")
    ap.add_argument("--tol",     type=float, default=1e-9,
                    help="convergence tol (default 1e-9)")
    ap.add_argument("--npy_out", default="fgw_dist.npy",
                    help="output .npy filename")
    ap.add_argument("--csv_out", default="fgw_dist.csv",
                    help="output .csv filename")
    args = ap.parse_args()
    main(args)
