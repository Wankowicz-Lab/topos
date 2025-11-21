# metrics_impl.py
from __future__ import annotations
from typing import Optional
import numpy as np
import pandas as pd
import biotite.structure as struc
from .structure_context import Context, register_metric
from . import pdbtm
from src.sequence import utils


# TODO: move helper functions to separate file so only @registered_metric functions remain here
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
def calculate_sasa(context: Context) -> pd.DataFrame:
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
def calculate_kyte_doolittle(context: Context) -> pd.DataFrame:
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
def calculate_membrane_distance(context: Context) -> pd.DataFrame:
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


@register_metric(name='define_secondary_structure', provides=['ss_group', 'ss_domains'], tags={'structure'})
def define_secondary_structure(context: Context) -> pd.DataFrame:
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
