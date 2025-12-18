"""
Sequence utility functions for amino acid code conversion.

This module provides utilities for working with amino acid sequences,
including conversion between 1-letter and 3-letter amino acid codes.
"""

import warnings

# Bi-directional mapping between amino acid codes
AA_3_TO_1 = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D",
    "CYS": "C", "GLN": "Q", "GLU": "E", "GLY": "G",
    "HIS": "H", "ILE": "I", "LEU": "L", "LYS": "K",
    "MET": "M", "PHE": "F", "PRO": "P", "SER": "S",
    "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V", 
    "DEL": "DEL", "*": "*", "del": "del"
}

# Automatically create the reverse mapping
AA_1_TO_3 = {v: k for k, v in AA_3_TO_1.items()}


def convert_amino_acid(code: str) -> str:
    """
    Convert between 1-letter and 3-letter amino acid codes.

    Parameters
    ----------
    code : str
        Either a 1-letter or 3-letter amino acid code.

    Returns
    -------
    str
        The corresponding 3-letter or 1-letter code. If the input is a
        1-letter code, returns the 3-letter code, and vice versa.

    Warns
    -----
    UserWarning
        If the provided code is not recognized or has an unexpected length.

    Examples
    --------
    >>> convert_amino_acid('A')
    'ALA'
    >>> convert_amino_acid('ALA')
    'A'
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

