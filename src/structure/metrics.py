# metrics_impl.py
from __future__ import annotations
from typing import Optional
import numpy as np
import pandas as pd
import biotite.structure as struc
from .structure_context import Context, register_metric
from .utils import *
from . import pdbtm
from src.sequence import utils


# TODO: move helper functions to separate file so only @registered_metric functions remain here
@register_metric(name='define_secondary_structure',
                 provides=['ss_group', 'ss_domains'],
                 tags={'structure'})
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


@register_metric(name='sasa', provides=['sasa'], tags={'structure'})
def calculate_sasa(array: struc.AtomArray, vdw_radii: str = "ProtOr") -> np.ndarray:
    """
    Calculate solvent accessible surface area (SASA) per residue.
    
    Parameters:
    -----------
    context.array : AtomArray
        Structure array (amino acids only recommended)
    context.vdw_radii : str
        Van der Waals radii set. Use "ProtOr" (default) for structures without hydrogens,
        or "Single" for structures with hydrogen atoms resolved.
    
    Returns:
    --------
    pd.DataFrame
        DataFrame with a column 'sasa' containing per-residue SASA values in Å².
    """
    # Calculate atom-wise SASA
    array, vdw_radii = context.array, context.vdw_radii
    atom_sasa = struc.sasa(array=array, vdw_radii=vdw_radii)

    # Sum up SASA for each residue
    res_sasa = struc.apply_residue_wise(array, atom_sasa, np.sum)

    # Attach to metadata DataFrame
    metadata_df = utils.get_metadata_cols(array)
    metadata_df['sasa'] = res_sasa

    return metadata_df


@register_metric(name='kyte_doolittle', provides=['kyte_doolittle'], tags={'structure'})
def calculate_kyte_doolittle(array: struc.AtomArray) -> np.ndarray:
    """
    Calculate Kyte–Doolittle hydropathy per residue.

    Parameters
    ----------
    context.array : AtomArray
        Structure array (amino acids recommended). Non-standard residues
        receive NaN.

    Returns
    -------
    pd.DataFrame
        DataFrame with a column 'kyte_doolittle' containing per-residue
        Kyte–Doolittle hydropathy scores.
    """
    kd_scale = {
        "ILE": 4.5, "VAL": 4.2, "LEU": 3.8, "PHE": 2.8, "CYS": 2.5,
        "MET": 1.9, "ALA": 1.8, "GLY": -0.4, "THR": -0.7, "SER": -0.8,
        "TRP": -0.9, "TYR": -1.3, "PRO": -1.6, "HIS": -3.2, "GLU": -3.5,
        "GLN": -3.5, "ASP": -3.5, "ASN": -3.5, "LYS": -3.9, "ARG": -4.5
    }
    array = context.array

    # Assign KD score per atom based on its residue name
    atom_vals = np.array([kd_scale.get(rn.upper(), np.nan) for rn in array.res_name], dtype=float)

    # Collapse to per-residue (mean of identical values == the same value)
    kd_per_res = struc.apply_residue_wise(array, atom_vals, np.nanmean)

    metadata_df = utils.get_metadata_cols(array)
    metadata_df['kyte_doolittle'] = kd_per_res

    return metadata_df


@register_metric(name='membrane_distance', provides=['distance_from_membrane_edge'], tags={'structure', 'membrane'})
def calculate_membrane_distance(array: struc.AtomArray) -> pd.DataFrame:
    """
    Calculate distance of each residue from the edge of the membrane along the z-axis.

    Parameters:
    -----------
    context.array : AtomArray
        Structure array (amino acids only recommended)
    context.membrane_thickness : float
        Half-thickness of the membrane in Angstroms

    Returns:
    ---------
    pd.DataFrame
        DataFrame with a column 'distance_from_membrane_edge' containing
        per-residue distances in Angstroms.
    """

    # Calculate z-coordinate of each residue (mean of atom z-coordinates)
    array, membrane_thickness = context.array, context.membrane_thickness
    atom_z = array.coord[:, 2]
    res_z = struc.apply_residue_wise(array, atom_z, np.mean)

    # Calculate distance from membrane edge
    distance_from_edge = np.abs(res_z) - membrane_thickness

    metadata_df = utils.get_metadata_cols(array)
    metadata_df['distance_from_membrane_edge'] = distance_from_edge

    return metadata_df


def define_secondary_structure(array: struc.AtomArray) -> pd.DataFrame:
    """Calculate secondary structure and merge adjacent regions based on heuristics or membrane information"""

    res_starts = struc.get_residue_starts(context.array)
    chains = context.array.chain_id[res_starts]
    resi = context.array.res_id[res_starts]
    resn = context.array.res_name[res_starts]
    sse_vals = calculate_secondary_structure(context.array)

    ss_df = pd.DataFrame({
        "chain": chains,
        "resi": resi,
        "resn": resn,
        "sse": sse_vals
    })

    if context.membrane_protein:
        ss_output = pdbtm.define_secondary_structure(context.residue_table, ss_df)
    else:
        # TODO: decide if we want to do any merging of secondary structure regions for non-membrane proteins
        ss_output = ss_df.copy()
        ss_output['ss_group'] = pdbtm.make_contiguous_group_labels(ss_output['sse'].tolist())

    return ss_output

def calculate_hbond_metrics(array: struc.AtomArray) -> dict[str, np.ndarray]:
    """
    Compute several per-residue hydrogen-bond metrics using an altloc-aware donor/acceptor model.     
    Metrics (all per residue, aligned to `struc.get_residue_starts(array)`)
    """
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
    }


def calculate_residue_packing(array: struc.AtomArray, cutoff: float = 5.0) -> dict[str, np.ndarray]:
    """
    Compute residue packing values.
    """
    
    # Residue indexing for original array
    res_starts = struc.get_residue_starts(array)
    chains = array.chain_id[res_starts]
    res_ids = array.res_id[res_starts]
    resnames = array.res_name[res_starts]
    n_res = len(res_starts)

    # Initialize output arrays
    n_atoms = np.zeros(n_res, dtype=int)
    n_neighbors = np.zeros(n_res, dtype=int)
    contact_density = np.full(n_res, np.nan, dtype=float)

    key_to_idx = {
            f"{ch}:{int(ri)}:{rn}": i
            for i, (ch, ri, rn) in enumerate(zip(chains, res_ids, resnames))
        }

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
            [f"{c}:{int(r)}:{n}" for c, r, n in zip(arr.chain_id, arr.res_id, arr.res_name)],
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
    
    





