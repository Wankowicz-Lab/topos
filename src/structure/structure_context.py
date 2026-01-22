"""
Structure loading utilities for protein analysis.

This module provides helper functions for loading and processing protein structures,
including structure file loading, residue table creation, and alternate location handling.
"""
from __future__ import annotations
import logging
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Optional, Literal, Union
import numpy as np
import pandas as pd
import biotite.structure as struc
from biotite.database import rcsb
from biotite.structure.io.pdb import PDBFile
from biotite.structure.io.pdbx import CIFFile, get_structure

logger = logging.getLogger(__name__)


def residue_table(array: struc.AtomArray) -> pd.DataFrame:
    """
    Create a residue table from an AtomArray.

    Parameters
    ----------
    array : struc.AtomArray
        Biotite AtomArray containing protein structure data.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns 'chain', 'resi', 'resn', 'altloc' for each residue.
    """
    res_starts = struc.get_residue_starts(array)
    chains = array.chain_id[res_starts]
    resi   = array.res_id[res_starts]
    resn   = array.res_name[res_starts]
    altloc = array.altloc[res_starts]
    
    return pd.DataFrame({"chain": chains, "resi": resi, "resn": resn, "altloc": altloc})


def load_structure(
    path: Optional[Union[str, Path]] = None,
    pdb_id: Optional[str] = None,
    model: Optional[int] = 1,
    altloc_policy: Literal["highest", "all"] = "highest",
) -> struc.AtomArray:
    """
    Load a protein structure from a PDB or mmCIF file, or fetch from RCSB by PDB ID.

    Parameters
    ----------
    path : str or Path, optional
        Path to the structure file (PDB or mmCIF format). If not provided, pdb_id must be provided.
    pdb_id : str, optional
        PDB identifier for fetching structure from RCSB. If not provided, path must be provided.
    model : int, optional
        Model number to load. Default is 1. Use None to load all models.
    altloc_policy : {'highest', 'all'}, optional
        Policy for handling alternate locations. 'highest' keeps the
        highest occupancy conformer, 'all' keeps all conformers.
        Default is 'highest'.

    Returns
    -------
    struc.AtomArray
        Loaded protein structure. The 'altloc' annotation contains the alternate
        location identifier for each atom (empty string if no alternate location).
    """
    extra_fields = ["b_factor", "occupancy"]
    
    # Handle PDB ID fetching if path is not provided
    if path is None and pdb_id is not None:
        logger.info("Fetching PDB structure from RCSB")
        obj = rcsb.fetch(pdb_id, format="cif")
        tmp_file = NamedTemporaryFile(delete=False, suffix=".cif")
        tmp_file.write(obj.getvalue().encode("utf-8"))
        tmp_file.close()
        path = Path(tmp_file.name)
        pdb_ext = "cif"
    elif path is not None:
        path = Path(path)
        pdb_ext = path.suffix.lstrip(".")
    else:
        raise ValueError("Either pdb_id or path must be provided")

    # Rename 'highest' to 'occupancy' to match biotite convention
    altloc_policy = "occupancy" if altloc_policy == 'highest' else altloc_policy
    
    # Load structure using appropriate parser
    if pdb_ext in ("cif", "mmcif"):
        cif = CIFFile.read(str(path))
        arr = get_structure(cif, model=model or 1, extra_fields=extra_fields, altloc=altloc_policy)
    else:
        pdb = PDBFile.read(str(path))
        models = pdb.get_model_count()
        if model is None and models > 1:
            arr = pdb.get_structure(model=None, extra_fields=extra_fields, altloc=altloc_policy)
        else:
            arr = pdb.get_structure(model=model or 1, extra_fields=extra_fields, altloc=altloc_policy)
    
    return arr


def ensure_altloc_annotation(array: struc.AtomArray) -> struc.AtomArray:
    """
    Ensure the array has an 'altloc' annotation.
    
    If 'altloc_id' exists (from PDB file), copy it to 'altloc'.
    Otherwise, create an empty 'altloc' annotation.
    
    Parameters
    ----------
    array : struc.AtomArray
        Input atom array.
    
    Returns
    -------
    struc.AtomArray
        Array with 'altloc' annotation guaranteed to exist.
    """
    if "altloc" not in array.get_annotation_categories():
        if "altloc_id" in array.get_annotation_categories():
            # Copy altloc_id to altloc, normalizing empty values
            altloc_vals = np.array([
                str(a).strip() if a is not None else '' 
                for a in array.altloc_id
            ])
            array.set_annotation("altloc", altloc_vals)
        else:
            # No altloc information available, set empty strings
            array.set_annotation("altloc", np.array([''] * array.array_length()))
    return array
