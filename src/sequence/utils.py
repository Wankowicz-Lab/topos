"""
Sequence utility functions for amino acid code conversion.

This module provides directional helpers for working with amino acid codes and
mutation tokens used by the mutation-score loader.
"""

import warnings

STANDARD_AA_3_TO_1 = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D",
    "CYS": "C", "GLN": "Q", "GLU": "E", "GLY": "G",
    "HIS": "H", "ILE": "I", "LEU": "L", "LYS": "K",
    "MET": "M", "PHE": "F", "PRO": "P", "SER": "S",
    "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
    "*": "*",
}

# Mutation shorthands are valid mutant tokens but collapse to X when a single
# residue character is needed for alignment or substitution matrices.
SPECIAL_TOKEN_3_TO_1 = {
    "DEL": "X",
    "DEL1": "X",
    "DEL2": "X",
    "DEL3": "X",
    "INS1": "X",
    "INS2": "X",
    "INS3": "X",
}

AA_3_TO_1 = {
    **STANDARD_AA_3_TO_1,
    **SPECIAL_TOKEN_3_TO_1,
}
AA_1_TO_3 = {v: k for k, v in STANDARD_AA_3_TO_1.items()}

VALID_RESN_3_CODES = frozenset(STANDARD_AA_3_TO_1)
VALID_RESM_3_CODES = frozenset(AA_3_TO_1)
VALID_1_CODES = frozenset(AA_1_TO_3)


def convert_amino_acid_1to3(code: str, force_convert: bool = False) -> str:
    """
    Convert a 1-letter amino acid code to a 3-letter code.

    Unknown 1-letter tokens warn and round-trip unchanged unless
    ``force_convert`` is set, in which case the single character is repeated.
    """
    code = code.upper().strip()
    if len(code) != 1:
        if force_convert:
            raise ValueError(f"Cannot convert 1-letter amino acid code: {code}")
        warnings.warn(f"Unexpected 1 letter amino acid code length: {code}")
        return code

    if code in AA_1_TO_3:
        return AA_1_TO_3[code]

    warnings.warn(f"Unknown 1 letter code: {code}")
    if force_convert:
        return code * 3
    return code


def convert_amino_acid_3to1(code: str, force_convert: bool = False) -> str:
    """
    Convert a 3-letter residue code or mutation token to a 1-letter code.

    Known deletion/insertion shorthand tokens are treated as valid inputs and
    collapse to ``X`` so downstream alignment and sequence-matrix code can keep
    a single-character representation.
    """
    code = code.upper().strip()
    if code in AA_3_TO_1:
        return AA_3_TO_1[code]

    if len(code) == 3:
        warnings.warn(f"Unknown 3 letter code: {code}")
    else:
        warnings.warn(f"Unexpected amino acid code length: {code}")

    if force_convert:
        return "X"
    return code


def invalid_codes(codes: set[str], valid_codes: frozenset[str]) -> set[str]:
    """Return normalized codes that are not present in the allowed set."""
    return {code for code in codes if code not in valid_codes}

