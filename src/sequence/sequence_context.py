import warnings

import pandas as pd
from src.sequence.utils import convert_amino_acid


def load_dms_scores(path: str, residue_col_name: str = "wildtype",
                    residue_idx_name: str = "position",
                    mutation_col_name: str = "mutation",
                    mutation_type_col_name: str = "type",
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

    mutation_type_col_name : str
        Name of the column containing mutation types (default: "type").

    score_col_name : str
        Name of the column containing mutation effect scores (default: "effect").

    Returns:
    --------
    pd.DataFrame
        DataFrame with standardized column names: 'resn', 'resi', 'resm', 'effect'.
    """

    df = pd.read_csv(path)

    for input_col_name in [residue_col_name, residue_idx_name, mutation_col_name, mutation_type_col_name, score_col_name]:
        if input_col_name not in df.columns:
            raise ValueError(f"Column '{input_col_name}' not found in DMS scores file.")

    df = df.rename(columns={
        residue_col_name: "resn",
        residue_idx_name: "resi",
        mutation_col_name: "resm",
        mutation_type_col_name: "type",
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

    # check that mutation types are named in a standard way
    valid_types = {'missense', 'nonsense', 'silent', 'insertion', 'deletion', 'synonymous', 'indel', 'del', 'ins'}
    found_types = set(df['type'].unique())
    if not found_types.issubset(valid_types):
        invalid_types = found_types - valid_types
        warnings.warn(f"Mutation types contain unexpected values. Expected types include {valid_types}. "
                      f"Found invalid types: {invalid_types}.")

    return df


def merge_dms_scores(dms_scores: pd.DataFrame, residue_table: pd.DataFrame, chain: str) -> pd.DataFrame:
    """
    Merge DMS scores with structural context based on residue positions.

    Parameters:
    -----------
    dms_scores : pd.DataFrame
        DataFrame containing DMS scores with 'resi' and 'resn' columns.

    residue_table : pd.DataFrame
        DataFrame containing structural residue information with 'chain', 'resi', and 'resn' columns.

    chain : str
        Chain identifier to filter structural context.

    Returns:
    --------
    pd.DataFrame
        Merged DataFrame with DMS scores and structural features.
    """

    # Extract residue information from context
    res_table = residue_table.copy()

    # test merge to make sure sequence is aligned with structure
    res_test = res_table.loc[res_table['chain'] == chain, ["resn", "resi"]].reset_index(drop=True)
    dms_test = dms_scores[['resn', 'resi']].drop_duplicates().reset_index(drop=True)

    merge_test = pd.merge(res_test, dms_test,
                              left_on=['resi', 'resn'],
                              right_on=['resi', 'resn'],
                              how='outer')

    # If sequence is aligned between DMS and structure, there should only be one row for each unique residue index
    if len(merge_test) > len(merge_test.resi.unique()):
        raise ValueError(f"Mismatch between DMS scores and structure residues for chain {chain}. "
                      f"Check that the sequence used for DMS matches the structure.")

    res_table['struct_info'] = True
    res_table_chain = res_table[res_table['chain'] == chain].reset_index(drop=True)
    dms_scores['seq_info'] = True

    # Merge DMS scores with structural residue table
    merged_df = pd.merge(dms_scores, res_table_chain,
                         left_on=['resi', 'resn'],
                         right_on=['resi', 'resn'],
                         how='outer')

    merged_df.loc[merged_df['struct_info'].isna(), 'struct_info'] = False
    merged_df.loc[merged_df['seq_info'].isna(), 'seq_info'] = False
    merged_df['chain'] = chain

    # Remove previous rows from residue table and update with merged data
    res_table = res_table[res_table['chain'] != chain]
    res_table['seq_info'] = False
    res_table = pd.concat([res_table, merged_df], axis=0).reset_index(drop=True)

    # drop extra columns if present
    res_table = res_table[['chain', 'resi', 'resn', 'resm', 'type', 'effect', 'seq_info', 'struct_info']]

    return res_table