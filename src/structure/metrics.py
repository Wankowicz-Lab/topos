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


def calculate_membrane_distance(array: struc.AtomArray, membrane_thickness: float = 15) -> np.ndarray:
    """
    Calculate distance of each residue from the edge of the membrane along the z-axis.

    Parameters:
    -----------
    array : AtomArray
        Structure array (amino acids only recommended)
    membrane_thickness : float
        Half-thickness of the membrane in Angstroms (default: 15 Å)

    Returns:
    --------
    np.ndarray
        Per-residue distance from the membrane edge in Angstroms.
        Negative values indicate positions inside the membrane.
    """

    # Calculate z-coordinate of each residue (mean of atom z-coordinates)
    atom_z = array.coord[:, 2]
    res_z = struc.apply_residue_wise(array, atom_z, np.mean)

    # Calculate distance from membrane edge
    distance_from_edge = np.abs(res_z) - membrane_thickness

    return distance_from_edge



from structure.analyze_hbonds import is_backbone_atom, angle_deg

def calculate_sidechain_angle_from_center(array: struc.AtomArray) -> np.ndarray:
    """
    Calculate angle between sidechain and vector towards centroid of protein for each residue.

    Parameters:
    -----------
    array : AtomArray
        Structure array (amino acids only recommended)

    Returns:
    --------
    np.ndarray
        Per-residue sidechain angle in degrees.
        NaN for residues without defined sidechains (e.g., Glycine).
    """

    # Process each chain separately
    res_starts = struc.get_residue_starts(array)
    chain_ids = array.chain_id[res_starts]
    unique_chains = np.unique(chain_ids)

    angles = np.full(len(res_starts), np.nan)

    for chain in unique_chains:
        # subset to current chain
        chain_mask = (chain_ids == chain)
        chain_res_starts = res_starts[chain_mask]
        chain_array = array[chain_res_starts]

        # get centroid
        centroid = struc.centroid(chain_array)

        for i, res_start in enumerate(chain_res_starts):
            # get all atoms for current residue
            res_atom_mask = struc.get_residue_masks(array, [res_start])[0]
            res_atoms = array[res_atom_mask]

            # get atoms in backbone and define centroid
            backbone_mask = np.array([is_backbone_atom(name) for name in res_atoms.atom_name])
            backbone_atoms = res_atoms[backbone_mask]
            backbone_centroid = struc.centroid(backbone_atoms)

            # vector from backbone centroid to overall chain centroid
            bb_chain_vector = centroid - backbone_centroid

            # get sidechain atoms and define centroid
            sidechain_atoms = res_atoms[~backbone_mask]
            sidechain_centroid = struc.centroid(sidechain_atoms)

            # vector from backbone centroid to sidechain centroid
            bb_sc_vector = sidechain_centroid - backbone_centroid

            # calculate angle between vectors
            angles[np.where(res_starts == res_start)[0][0]] = angle_deg(bb_chain_vector, bb_sc_vector)

    return angles
