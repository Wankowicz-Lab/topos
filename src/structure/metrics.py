# metrics_impl.py
from __future__ import annotations
from typing import Optional
import numpy as np
import pandas as pd
import biotite.structure as struc
from .structure_context import Context, register_metric

def calculate_sasa(array: struc.AtomArray, vdw_radii: str = "ProtOr") -> np.ndarray:
    """
    Calculate solvent accessible surface area (SASA) per residue.
    
    Parameters:
    -----------
    array : AtomArray
        Structure array (amino acids only recommended)
    vdw_radii : str
        Van der Waals radii set. Use "ProtOr" (default) for structures without hydrogens,
        or "Single" for structures with hydrogen atoms resolved.
    
    Returns:
    --------
    np.ndarray
        Per-residue SASA values in Angstroms squared
    """
    # Calculate atom-wise SASA
    atom_sasa = struc.sasa(array, vdw_radii=vdw_radii)
    # Sum up SASA for each residue
    res_sasa = struc.apply_residue_wise(array, atom_sasa, np.sum)
    return res_sasa


def calculate_secondary_structure(array: struc.AtomArray) -> np.ndarray:
    """
    Calculate secondary structure assignment per residue.
    
    Parameters:
    -----------
    array : AtomArray
        Structure array (amino acids only recommended)
    
    Returns:
    --------
    np.ndarray
        Per-residue secondary structure assignment.
        'a' = alpha-helix, 'b' = beta-sheet, 'c' = coil
    """
    return struc.annotate_sse(array)
    

def calculate_kyte_doolittle(array: struc.AtomArray) -> np.ndarray:
    """
    Calculate Kyte–Doolittle hydropathy per residue.

    Parameters
    ----------
    array : AtomArray
        Structure array (amino acids recommended). Non-standard residues
        receive NaN.

    Returns
    -------
    np.ndarray
        Per-residue Kyte–Doolittle hydropathy values in the residue order
        implied by `struc.apply_residue_wise`.
    """
    kd_scale = {
        "ILE": 4.5, "VAL": 4.2, "LEU": 3.8, "PHE": 2.8, "CYS": 2.5,
        "MET": 1.9, "ALA": 1.8, "GLY": -0.4, "THR": -0.7, "SER": -0.8,
        "TRP": -0.9, "TYR": -1.3, "PRO": -1.6, "HIS": -3.2, "GLU": -3.5,
        "GLN": -3.5, "ASP": -3.5, "ASN": -3.5, "LYS": -3.9, "ARG": -4.5
    }

    # Assign KD score per atom based on its residue name
    atom_vals = np.array([kd_scale.get(rn.upper(), np.nan) for rn in array.res_name], dtype=float)

    # Collapse to per-residue (mean of identical values == the same value)
    kd_per_res = struc.apply_residue_wise(array, atom_vals, func=lambda x: np.nanmean(x))
    return kd_per_res

def calculate_hbond_metrics(array: struc.AtomArray) -> dict[str, np.ndarray]:
    """
    Compute several per-residue hydrogen-bond metrics using an
    altloc-aware donor/acceptor model.

    Metrics (all per residue, aligned to `struc.get_residue_starts(array)`):
    - 'bb_hbond_count'   : number of H-bonds where this residue participates
                           via a backbone atom (as donor or acceptor)
    - 'sc_hbond_count'   : number of H-bonds where this residue participates
                           via a sidechain atom
    - 'total_hbond_count': total number of H-bonds this residue participates in
                           (backbone + sidechain) == weighted degree
    - 'weighted_degree'  : alias for total_hbond_count

    Parameters
    ----------
    array : AtomArray
        Structure array with at least coordinates, chain_id, res_id,
        res_name, atom_name, and altloc_id.

    Returns
    -------
    dict[str, np.ndarray]
        Dictionary mapping metric name to a 1D NumPy array of length
        n_residues.
    """
    # Build donors/acceptors and H-bond list
    donors, acceptors = _build_sites_biotite(array)
    hbonds = _detect_hbonds(donors, acceptors)

    res_starts = struc.get_residue_starts(array)
    chains = array.chain_id[res_starts]
    res_ids = array.res_id[res_starts]
    resnames = array.res_name[res_starts]

    n_res = len(res_starts)
    bb_counts = np.zeros(n_res, dtype=float)
    sc_counts = np.zeros(n_res, dtype=float)
    total_counts = np.zeros(n_res, dtype=float)

    # Map "chain:resi:resname" -> residue index
    key_to_idx = {
        f"{ch}:{int(ri)}:{rn}": i
        for i, (ch, ri, rn) in enumerate(zip(chains, res_ids, resnames))
    }

    def _is_backbone_for_role(category: str, role: str) -> bool:
        donor_cat, acceptor_cat = category.split("-")
        if role == "donor":
            return donor_cat == "backbone"
        else:
            return acceptor_cat == "backbone"

    # Accumulate counts per residue
    for h in hbonds:
        cat = h["category"]

        # Donor
        d_key = f"{h['donor_chain']}:{h['donor_resi']}:{h['donor_resname']}"
        d_idx = key_to_idx.get(d_key, None)
        if d_idx is not None:
            total_counts[d_idx] += 1
            if _is_backbone_for_role(cat, "donor"):
                bb_counts[d_idx] += 1
            else:
                sc_counts[d_idx] += 1

        # Acceptor
        a_key = f"{h['acceptor_chain']}:{h['acceptor_resi']}:{h['acceptor_resname']}"
        a_idx = key_to_idx.get(a_key, None)
        if a_idx is not None:
            total_counts[a_idx] += 1
            if _is_backbone_for_role(cat, "acceptor"):
                bb_counts[a_idx] += 1
            else:
                sc_counts[a_idx] += 1

    return {
        "bb_hbond_count": bb_counts,
        "sc_hbond_count": sc_counts,
        "total_hbond_count": total_counts,
        "weighted_degree": weighted_degree,
    }

def calculate_residue_packing(
    array: struc.AtomArray,
    cutoff: float = 5.0,
) -> dict[str, np.ndarray]:
    """
    Compute residue packing metrics using Biotite.

    Metric definition:
      For each residue, count the number of *distinct neighboring residues*
      that have at least one heavy atom within `cutoff` Å of any heavy atom
      in the residue (excluding self).

    Parameters
    ----------
    array : AtomArray
        Full structure array (may include solvent/ligands; they are ignored).
    cutoff : float
        Distance cutoff in Å (default: 5.0).

    Returns
    -------
    dict[str, np.ndarray]
        Dictionary of per-residue arrays aligned to
        `struc.get_residue_starts(array)`:
          - "packing_n_atoms"
          - "packing_n_neighbor_residues"
          - "packing_contact_density"
        Non-amino-acid residues receive zeros/NaN.
    """
    # Residue indexing for original array
    res_starts = struc.get_residue_starts(array)
    chains = array.chain_id[res_starts]
    res_ids = array.res_id[res_starts]
    n_res = len(res_starts)

    # Initialize output arrays
    n_atoms = np.zeros(n_res, dtype=int)
    n_neighbors = np.zeros(n_res, dtype=int)
    contact_density = np.full(n_res, np.nan, dtype=float)

    full_keys = np.array(
        [residue_key(ch, ri) for ch, ri in zip(chains, res_ids)],
        dtype=object,
    )
    key_to_idx = {k: i for i, k in enumerate(full_keys)}

    # Filter to heavy amino-acid atoms
    aa_mask = struc.filter_amino_acids(array)
    heavy_mask = np.array([is_heavy(n) for n in array.atom_name], dtype=bool)
    mask = aa_mask & heavy_mask
    arr = array[mask]

    if arr.array_length() == 0:
        return {
            "packing_n_atoms": n_atoms,
            "packing_n_neighbor_residues": n_neighbors,
            "packing_contact_density": contact_density,
        }

    # Residue keys for filtered array
    residue_ids = np.array(
        [residue_key(c, r) for c, r in zip(arr.chain_id, arr.res_id)],
        dtype=object,
    )
    unique_res = np.unique(residue_ids)

    coords = arr.coord.astype(float)
    cutoff2 = cutoff * cutoff

    # Compute per-residue packing
    for res_uid in unique_res:
        idxs = np.where(residue_ids == res_uid)[0]
        if len(idxs) == 0:
            continue

        res_atoms = arr[idxs]
        res_n_atoms = len(res_atoms)

        # Neighbor detection
        rcoords = res_atoms.coord  # (k, 3)
        diff = rcoords[:, None, :] - coords[None, :, :]  
        d2 = np.einsum("ijk,ijk->ij", diff, diff)         
        within_cutoff = d2 <= cutoff2

        close_atom_idxs = np.where(within_cutoff.any(axis=0))[0]
        neighbor_res_keys = set(residue_ids[close_atom_idxs].tolist())
        neighbor_res_keys.discard(res_uid)  # remove self

        # record metrics
        idx = key_to_idx.get(res_uid)
        if idx is not None:
            n_atoms[idx] = res_n_atoms
            n_neighbors[idx] = len(neighbor_res_keys)
            contact_density[idx] = len(neighbor_res_keys) / max(1, res_n_atoms)

    return {
        "packing_n_atoms": n_atoms,
        "packing_n_neighbor_residues": n_neighbors,
        "packing_contact_density": contact_density,
    }
    



