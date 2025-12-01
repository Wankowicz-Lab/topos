import warnings

import biotite.structure as struc
import pandas as pd

# Bi-directional mapping between amino acid codes
AA_3_TO_1 = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D",
    "CYS": "C", "GLN": "Q", "GLU": "E", "GLY": "G",
    "HIS": "H", "ILE": "I", "LEU": "L", "LYS": "K",
    "MET": "M", "PHE": "F", "PRO": "P", "SER": "S",
    "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V"
}

# Automatically create the reverse mapping
AA_1_TO_3 = {v: k for k, v in AA_3_TO_1.items()}


def convert_amino_acid(code):
    """
    Convert between 1-letter and 3-letter amino acid codes.

    Parameters
    ----------
    code : str
        Either a 1-letter or 3-letter amino acid code.

    Returns
    -------
    str
        The corresponding 3-letter or 1-letter code.

    Raises
    ------
    ValueError
        If the provided code is not recognized.
    """
    code = code.upper().strip()
    if len(code) == 1:
        if code in AA_1_TO_3:
            return AA_1_TO_3[code]
        else:
            warnings.warn(f"Unknown 1 letter code: {code}")
            return code

    elif len(code) == 3:
        if code in AA_3_TO_1:
            return AA_3_TO_1[code]
        else:
            warnings.warn(f"Unknown 3 letter code: {code}")
            return code

    warnings.warn(f"Unexpected amino acid code length: {code}")
    return code


def get_metadata_cols(array):
    """Gets metadata columns from an AtomArray or AtomArrayStack."""

    res_starts = struc.get_residue_starts(array)
    chains = array.chain_id[res_starts]
    resi = array.res_id[res_starts]
    resn = array.res_name[res_starts]

    return pd.DataFrame({
        "chain": chains,
        "resi": resi,
        "resn": resn
    })

