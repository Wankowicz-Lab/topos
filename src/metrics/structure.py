"""
Structure metrics for protein analysis.

This module provides metric calculation functions for protein structures,
including SASA, hydropathy, membrane distance, secondary structure,
hydrogen bonds, and packing metrics.
"""
from __future__ import annotations
import logging
import numpy as np
import pandas as pd
import biotite.structure as struc
from src.pipeline.context import Context
from src.metrics.registry import register_metric
from src.databases import pdbtm
from src.structure.utils import residue_key, is_heavy, get_metadata_cols, is_backbone_atom
from src.structure.utils import build_sites_biotite as _build_sites_biotite, detect_hbonds as _detect_hbonds

logger = logging.getLogger(__name__)


@register_metric(name='sasa', provides=['sasa', 'sasa_backbone', 'sasa_sidechain', 'sasa_polar', 'sasa_nonpolar'], tags={'structure'})
def calculate_sasa(context: Context) -> pd.DataFrame:
    """
    Calculate solvent accessible surface area (SASA) per residue.

    Computes total SASA per residue and averaged SASA for backbone atoms,
    sidechain atoms, polar atoms (N, O, S), and nonpolar atoms (C, H).

    Parameters
    ----------
    context : Context
        Context object containing residue metadata, structural information, and mutation information.

    Returns
    -------
    pd.DataFrame
        DataFrame with 'sasa' (total), 'sasa_backbone', 'sasa_sidechain',
        'sasa_polar', and 'sasa_nonpolar' along with residue metadata.
    """
    logger.info("Calculating SASA")
    
    # Calculate atom-wise SASA
    array = context.aa
    atom_sasa = struc.sasa(array=array, vdw_radii="ProtOr")

    # Sum up SASA for each residue (total)
    res_sasa = struc.apply_residue_wise(array, atom_sasa, np.sum)
    
    # Create masks for atom categories
    # Backbone atoms
    backbone_mask = np.array([is_backbone_atom(name) for name in array.atom_name], dtype=bool)
    # Sidechain atoms (not backbone)
    sidechain_mask = ~backbone_mask
    
    # Polar atoms (N, O, S)
    elements = array.element
    polar_mask = np.array([elem in ('N', 'O', 'S') if elem else False for elem in elements], dtype=bool)
    # Nonpolar atoms (C)
    nonpolar_mask = np.array(elements == 'C', dtype=bool)
    
    # Calculate sum SASA for each category per residue
    # For each category, set non-matching atoms to 0, then compute sum
    def _sum_for_mask(mask):
        """Helper to compute sum SASA for atoms matching a mask."""
        masked_sasa = np.where(mask, atom_sasa, 0)
        return struc.apply_residue_wise(array, masked_sasa, np.sum)
    
    res_sasa_backbone = _sum_for_mask(backbone_mask)
    res_sasa_sidechain = _sum_for_mask(sidechain_mask)
    res_sasa_polar = _sum_for_mask(polar_mask)
    res_sasa_nonpolar = _sum_for_mask(nonpolar_mask)
    
    # Attach to metadata DataFrame
    metadata_df = get_metadata_cols(array)
    metadata_df['sasa'] = res_sasa
    metadata_df['sasa_backbone'] = res_sasa_backbone
    metadata_df['sasa_sidechain'] = res_sasa_sidechain
    metadata_df['sasa_polar'] = res_sasa_polar
    metadata_df['sasa_nonpolar'] = res_sasa_nonpolar
    
    return metadata_df


@register_metric(name='kyte_doolittle', provides=['kyte_doolittle'], tags={'structure'})
def calculate_kyte_doolittle(context: Context) -> pd.DataFrame:
    """
    Calculate Kyte-Doolittle hydropathy per residue.

    Parameters
    ----------
    context : Context
        Context object containing residue metadata, structural information, and mutation information.

    Returns
    -------
    pd.DataFrame
        DataFrame with 'kyte_doolittle' along with residue metadata.
    """
    logger.info("Calculating Kyte-Doolittle hydropathy")

    kd_scale = {
        "ILE": 4.5, "VAL": 4.2, "LEU": 3.8, "PHE": 2.8, "CYS": 2.5,
        "MET": 1.9, "ALA": 1.8, "GLY": -0.4, "THR": -0.7, "SER": -0.8,
        "TRP": -0.9, "TYR": -1.3, "PRO": -1.6, "HIS": -3.2, "GLU": -3.5,
        "GLN": -3.5, "ASP": -3.5, "ASN": -3.5, "LYS": -3.9, "ARG": -4.5
    }
    array = context.aa

    # Assign KD score per atom based on its residue name
    atom_vals = np.array([kd_scale.get(rn.upper(), np.nan) for rn in array.res_name], dtype=float)

    # Collapse to per-residue (mean of identical values == the same value)
    kd_per_res = struc.apply_residue_wise(array, atom_vals, function=np.nanmean)
    
    # Attach to metadata DataFrame
    metadata_df = get_metadata_cols(array)
    metadata_df['kyte_doolittle'] = kd_per_res
    
    return metadata_df

@register_metric(name='membrane_distance', provides=['distance_from_membrane_edge'], tags={'structure', 'membrane'})
def calculate_membrane_distance(context: Context) -> pd.DataFrame:
    """
    Calculate distance of each residue from the edge of the membrane.

    Uses the z-axis to determine membrane position, assuming the membrane plane is oriented horizontally.

    Parameters
    ----------
    context : Context
        Context object containing residue metadata, structural information, and mutation information.

    Returns
    -------
    pd.DataFrame
        DataFrame with 'distance_from_membrane_edge' along with residue metadata.
    """
    logger.info("Calculating membrane distance")

    # Calculate z-coordinate of each residue (mean of atom z-coordinates)
    array, membrane_thickness = context.array, context.config.membrane_thickness

    atom_z = array.coord[:, 2]
    res_z = struc.apply_residue_wise(array, atom_z, np.mean)

    # Calculate distance from membrane edge
    distance_from_edge = np.abs(res_z) - membrane_thickness
    
    metadata_df = get_metadata_cols(array)
    metadata_df['distance_from_membrane_edge'] = distance_from_edge

    return metadata_df


@register_metric(name='calculate_hbond_metrics', provides=['bb_hbond_count', 'sc_hbond_count', 'total_hbond_count'], tags={'structure', 'interaction'})
def calculate_hbond_metrics(context: Context) -> pd.DataFrame:
    """
    Compute per-residue hydrogen bond metrics.

    Uses an altloc-aware donor/acceptor model to detect hydrogen bonds and count them per residue.

    Parameters
    ----------
    context : Context
        Context object containing residue metadata, structural information, and mutation information.

    Returns
    -------
    pd.DataFrame
        DataFrame with 'bb_hbond_count', 'sc_hbond_count', 'total_hbond_count' along with residue metadata.
    """
    logger.info("Calculating hydrogen bond metrics")
    
    array = context.array
    res_starts = struc.get_residue_starts(array)
    donors, acceptors = _build_sites_biotite(array)
    hbonds = _detect_hbonds(donors, acceptors)
    
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

    #TO DO: MOVE TO UTILS
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
    
    metadata_df = get_metadata_cols(array)
    metadata_df['bb_hbond_count'] = bb_counts
    metadata_df['sc_hbond_count'] = sc_counts
    metadata_df['total_hbond_count'] = total_counts
    return metadata_df


@register_metric(name='calculate_packing_metrics', provides=['packing_n_atoms', 'packing_n_neighbor_residues', 'packing_contact_density'], tags={'structure', 'interaction'})
def calculate_residue_packing(context: Context, cutoff: float = 5.0) -> pd.DataFrame:
    """
    Compute residue packing metrics.

    Calculates the number of heavy atoms per residue, the number of neighboring residues within a distance cutoff,
    and the contact density (neighbors per atom).

    Parameters
    ----------
    context : Context
        Context object containing residue metadata, structural information, and mutation information.
    cutoff : float, optional
        Distance cutoff in Angstroms for neighbor detection. Default is 5.0.

    Returns
    -------
    pd.DataFrame
        DataFrame with 'packing_n_atoms', 'packing_n_neighbor_residues', 'packing_contact_density' along with residue metadata.
    """
    logger.info("Calculating packing metrics")
    
    array = context.array
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

    # TODO: handle empty case so that it still returns a DF
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
    
    metadata_df = get_metadata_cols(array)
    metadata_df['packing_n_atoms'] = n_atoms
    metadata_df['packing_n_neighbor_residues'] = n_neighbors
    metadata_df['packing_contact_density'] = contact_density
    return metadata_df
