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
from src.structure.utils import res_key, is_heavy, get_metadata_cols, is_backbone_atom

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
    
    # Calculate atom-wise SASA
    array = context.aa
    
    # Filter by structural_feature_chains if specified
    if context.config.structural_feature_chains is not None:
        chain_mask = np.isin(array.chain_id, context.config.structural_feature_chains)
        array = array[chain_mask]
    
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


# Reference (max) SASA per residue type, Tien et al. 2013 PLOS ONE empirical values (Å²)
_REF_SASA = {
    "ALA": 121.0, "ARG": 265.0, "ASN": 187.0, "ASP": 187.0, "CYS": 148.0,
    "GLN": 214.0, "GLU": 214.0, "GLY": 97.0, "HIS": 216.0, "ILE": 195.0,
    "LEU": 191.0, "LYS": 230.0, "MET": 203.0, "PHE": 228.0, "PRO": 154.0,
    "SER": 143.0, "THR": 163.0, "TRP": 264.0, "TYR": 255.0, "VAL": 165.0,
}


@register_metric(name='distance_to_surface', provides=['distance_to_nearest_surface_residue'], tags={'structure'})
def calculate_distance_to_surface(context: Context, sasa_threshold: float = 0.25) -> pd.DataFrame:
    """
    Calculate distance from each residue to the nearest surface residue.

    Surface is defined as residues whose relative SASA exceeds sasa_threshold.
    Relative SASA = (observed per-residue SASA) / (reference SASA for that residue type).
    Distance is residue centroid to residue centroid.

    Parameters
    ----------
    context : Context
        Context object containing residue metadata and structural information.
    sasa_threshold : float, optional
        Minimum relative SASA for a residue to be considered surface. Default is 0.25.

    Returns
    -------
    pd.DataFrame
        DataFrame with 'distance_to_nearest_surface_residue' along with residue metadata.
    """
    # Per-residue SASA (same setup as calculate_sasa): atom-wise SASA, then sum per residue
    array = context.aa

    # Filter by structural_feature_chains if specified
    if context.config.structural_feature_chains is not None:
        chain_mask = np.isin(array.chain_id, context.config.structural_feature_chains)
        array = array[chain_mask]

    atom_sasa = struc.sasa(array=array, vdw_radii="ProtOr")
    res_sasa = struc.apply_residue_wise(array, atom_sasa, np.sum)

    # Relative SASA = observed / reference; surface = residues with relative SASA above threshold
    res_starts = struc.get_residue_starts(array)
    res_names = array.res_name[res_starts]
    ref = np.array([_REF_SASA.get(str(r).strip(), np.nan) for r in res_names], dtype=float)
    relative_sasa = np.where(ref > 0, res_sasa / ref, np.nan)
    is_surface = relative_sasa > sasa_threshold

    # Residue centroids (mean x, y, z of atoms per residue) for distance calculations
    cx = struc.apply_residue_wise(array, array.coord[:, 0], np.mean)
    cy = struc.apply_residue_wise(array, array.coord[:, 1], np.mean)
    cz = struc.apply_residue_wise(array, array.coord[:, 2], np.mean)
    centroids = np.stack([cx, cy, cz], axis=1)

    # Min centroid-to-centroid distance from each residue to any surface residue; nan if no surface
    n_res = len(res_starts)
    distances = np.full(n_res, np.nan, dtype=float)
    surface_centroids = centroids[is_surface]
    if surface_centroids.size > 0:
        diff = centroids[:, None, :] - surface_centroids[None, :, :]
        d2 = (diff ** 2).sum(axis=2)
        distances = np.sqrt(np.min(d2, axis=1))

    metadata_df = get_metadata_cols(array)
    metadata_df['distance_to_nearest_surface_residue'] = distances
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

    kd_scale = {
        "ILE": 4.5, "VAL": 4.2, "LEU": 3.8, "PHE": 2.8, "CYS": 2.5,
        "MET": 1.9, "ALA": 1.8, "GLY": -0.4, "THR": -0.7, "SER": -0.8,
        "TRP": -0.9, "TYR": -1.3, "PRO": -1.6, "HIS": -3.2, "GLU": -3.5,
        "GLN": -3.5, "ASP": -3.5, "ASN": -3.5, "LYS": -3.9, "ARG": -4.5
    }
    array = context.aa
    
    # Filter by structural_feature_chains if specified
    if context.config.structural_feature_chains is not None:
        chain_mask = np.isin(array.chain_id, context.config.structural_feature_chains)
        array = array[chain_mask]

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

    # Calculate z-coordinate of each residue (mean of atom z-coordinates)
    array, membrane_thickness = context.array, context.config.membrane_thickness
    
    # Filter by structural_feature_chains if specified
    if context.config.structural_feature_chains is not None:
        chain_mask = np.isin(array.chain_id, context.config.structural_feature_chains)
        array = array[chain_mask]

    atom_z = array.coord[:, 2]
    res_z = struc.apply_residue_wise(array, atom_z, np.mean)

    # Calculate distance from membrane edge
    distance_from_edge = np.abs(res_z) - membrane_thickness
    
    metadata_df = get_metadata_cols(array)
    metadata_df['distance_from_membrane_edge'] = distance_from_edge

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
    
    array = context.array
    
    # Filter by structural_feature_chains if specified
    if context.config.structural_feature_chains is not None:
        chain_mask = np.isin(array.chain_id, context.config.structural_feature_chains)
        array = array[chain_mask]
    
    # Residue indexing for original array
    res_starts = struc.get_residue_starts(array)
    chains = array.chain_id[res_starts]
    res_ids = array.res_id[res_starts]
    res_names = array.res_name[res_starts]
    n_res = len(res_starts)

    # Initialize output arrays
    n_atoms = np.zeros(n_res, dtype=int)
    n_neighbors = np.zeros(n_res, dtype=int)
    contact_density = np.full(n_res, np.nan, dtype=float)

    full_keys = np.array(
        [res_key(ch, ri, rn) for ch, ri, rn in zip(chains, res_ids, res_names)],
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
        [res_key(c, r, rn) for c, r, rn in zip(arr.chain_id, arr.res_id, arr.res_name)],
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


@register_metric(name='center_of_mass_distance', provides=['distance_to_center_of_mass'], tags={'structure'})
def calculate_center_of_mass_distance(context: Context) -> pd.DataFrame:
    """
    Calculate distance of each residue to the center of mass of the structure.

    Computes the center of mass of amino acid atoms in the structure, then calculates
    the Euclidean distance from each residue's center (mean of atom coordinates)
    to the structure's center of mass.

    Parameters
    ----------
    context : Context
        Context object containing residue metadata, structural information, and mutation information.

    Returns
    -------
    pd.DataFrame
        DataFrame with 'distance_to_center_of_mass' along with residue metadata.
    """
    
    array = context.aa

    # Filter by structural_feature_chains if specified
    if context.config.structural_feature_chains is not None:
        chain_mask = np.isin(array.chain_id, context.config.structural_feature_chains)
        array = array[chain_mask]

    coords = array.coord.astype(float)

    # Calculate center of mass of the entire structure
    com = np.mean(coords, axis=0)
    
    # Calculate center of each residue (mean of atom coordinates) using apply_residue_wise
    # Apply to each coordinate dimension separately
    res_center_x = struc.apply_residue_wise(array, coords[:, 0], np.mean)
    res_center_y = struc.apply_residue_wise(array, coords[:, 1], np.mean)
    res_center_z = struc.apply_residue_wise(array, coords[:, 2], np.mean)
    res_centers = np.column_stack([res_center_x, res_center_y, res_center_z])
    
    # Calculate Euclidean distance from each residue center to center of mass
    distances = np.linalg.norm(res_centers - com, axis=1)
    
    # Attach to metadata DataFrame
    metadata_df = get_metadata_cols(array)
    metadata_df['distance_to_center_of_mass'] = distances
    
    return metadata_df


DSSP_METRIC_COLUMNS = [
    "dssp_acc",
    "dssp_nh_o_1_relidx",
    "dssp_nh_o_1_energy",
    "dssp_o_nh_1_relidx",
    "dssp_o_nh_1_energy",
    "dssp_nh_o_2_relidx",
    "dssp_nh_o_2_energy",
    "dssp_o_nh_2_relidx",
    "dssp_o_nh_2_energy",
    "dssp_tco",
    "dssp_kappa",
    "dssp_alpha",
    "dssp_phi",
    "dssp_psi",
]


@register_metric(name="dssp_metrics", provides=DSSP_METRIC_COLUMNS, tags={"structure", "dssp"})
def calculate_dssp_metrics(context: Context) -> pd.DataFrame:
    """Return residue-level DSSP metrics parsed during secondary-structure annotation."""
    if context.extras["ss_backend"] != "mkdssp":
        raise ValueError("dssp_metrics can only run when mkdssp secondary structure is enabled.")

    array = context.aa
    if context.config.structural_feature_chains is not None:
        chain_mask = np.isin(array.chain_id, context.config.structural_feature_chains)
        array = array[chain_mask]

    metadata_df = get_metadata_cols(array)
    dssp_df = context.extras["dssp_output"]

    merge_cols = ["chain", "resi_struct"]
    dssp_subset = dssp_df.rename(columns={"resi": "resi_struct"})
    out = metadata_df.merge(dssp_subset[merge_cols + DSSP_METRIC_COLUMNS], on=merge_cols, how="left")
    
    return out
