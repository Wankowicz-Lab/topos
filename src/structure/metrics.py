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

def calculate_hbond_degree(array: struc.AtomArray) -> np.ndarray:
    



