import pandas as pd
import warnings


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


def load_dms_scores(path: str, residue_col_name: str = "wildtype",
                    residue_idx_name : str = "position",
                    mutation_col_name: str = "mutation",
                    score_col_name: str = "effect") -> pd.DataFrame:
    """
    Load DMS scores from a CSV file and standardize column names.

    Parameters:
    -----------
    path : str
        Path to the CSV file containing DMS scores.

    residue_col_name : str
        Name of the column containing wildtype residues (default: "wildtype").

    residue_idx_name : str
        Name of the column containing residue positions (default: "position").

    mutation_col_name : str
        Name of the column containing mutant residues (default: "mutation").

    score_col_name : str
        Name of the column containing mutation effect scores (default: "effect").

    Returns:
    --------
    pd.DataFrame
        DataFrame with standardized column names: 'resn', 'resi', 'resm', 'effect'.
    """

    df = pd.read_csv(path)
    df = df.rename(columns={
        residue_col_name: "resn",
        residue_idx_name: "resi",
        mutation_col_name: "resm",
        score_col_name: "effect"
    })

    # make sure that residue and mutation columns are valid
    residue_lens = df['resn'].str.len().unique()
    if len(residue_lens) != 1 or residue_lens[0] not in (1, 3):
        raise ValueError("Residue column must contain either 1-letter or 3-letter amino acid codes.")

    # mutation column may contain a range of possible lengths because of stops and indels
    mutation_lens = df['resm'].str.len().unique()

    # convert to 3-letter codes if necessary
    if residue_lens[0] == 1:
        df['resn'] = df['resn'].apply(convert_amino_acid)

    if 1 in mutation_lens:
        df['resm'] = df['resm'].apply(convert_amino_acid)

    return df


