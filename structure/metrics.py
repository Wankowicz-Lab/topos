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


