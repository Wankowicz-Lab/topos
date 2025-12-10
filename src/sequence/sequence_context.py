import numpy as np
import warnings
from pathlib import Path
from typing import Union

import pandas as pd
from src.sequence.utils import convert_amino_acid


def load_mutation_scores(
    path: Union[str, Path],
    residue_col_name: str = "wildtype",
    residue_idx_name: str = "position",
    mutation_col_name: str = "mutation",
    mutation_type_col_name: str = "type",
    score_col_name: str = "effect",
) -> pd.DataFrame:
    """
    Load mutation scores from a CSV file and standardize column names.

    Parameters
    ----------
    path : str or Path
        Path to the CSV file containing mutation scores.
    residue_col_name : str, optional
        Name of the column containing wildtype residues. Default is "wildtype".
    residue_idx_name : str, optional
        Name of the column containing residue positions. Default is "position".
    mutation_col_name : str, optional
        Name of the column containing mutant residues. Default is "mutation".
    mutation_type_col_name : str, optional
        Name of the column containing mutation types. Default is "type".
    score_col_name : str, optional
        Name of the column containing mutation effect scores. Default is "effect".

    Returns
    -------
    pd.DataFrame
        DataFrame with standardized column names: 'resn', 'resi', 'resm',
        'type', and 'effect'.

    Raises
    ------
    ValueError
        If required columns are missing or if the residue column contains
        codes that are neither 1-letter nor 3-letter amino acid codes.

    Warns
    -----
    UserWarning
        If mutation types contain unexpected values.
    """
    df = pd.read_csv(path)

    required_cols = [residue_col_name, residue_idx_name, mutation_col_name, mutation_type_col_name, score_col_name]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Columns {missing_cols} not found in mutation scores file at {path}.")
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


def alignment_to_index_map(alignment):
    """
    Convert alignment.coordinates into explicit per-residue index mapping to allow indexing into padndas df.

    Parameters
    ----------
    alignment : Alignment object
        Alignment object containing coordinates attribute.

    Returns:
    -------
    list of tuples
        list of (idx1, idx2) where either may be None for gaps
    """

    coords = alignment.coordinates  # shape (2, n_segments+1)
    map_list = []

    for col in range(coords.shape[1] - 1):
        start1, end1 = coords[0, col], coords[0, col + 1]
        start2, end2 = coords[1, col], coords[1, col + 1]

        len1 = end1 - start1
        len2 = end2 - start2

        if len1 == len2:
            # Match/substitution block
            for i in range(len1):
                map_list.append((start1 + i, start2 + i))
        elif len1 > len2:
            # Deletion in seq2 (seq1 has extra)
            for i in range(len1):
                # seq2 gap for positions beyond its end
                map_list.append((start1 + i, start2 + i if i < len2 else None))
        else:
            # Insertion in seq2 (seq2 has extra)
            for i in range(len2):
                map_list.append((start1 + i if i < len1 else None, start2 + i))

    return map_list


def merge_sequence_dfs(df1: pd.DataFrame, df2: pd.DataFrame, mapping: list) -> pd.DataFrame:
    """
    Merge two sequence DataFrames based on a provided index mapping.

    Parameters
    ----------
    df1: pd.DataFrame
        First DataFrame containing sequence information.
    df2: pd.DataFrame
        Second DataFrame containing sequence information.
    mapping: list of tuples
        List of (idx1, idx2) tuples mapping indices from df1 to df2. Either idx may be None for gaps.

    Returns
    -------
    pd.DataFrame
        Merged DataFrame containing combined sequence information from both DataFrames.
    """

    map_df = pd.DataFrame(mapping, columns=["i1", "i2"])

    # add sequential index to each df for merging
    df1['seq_idx'] = range(len(df1))
    df2['seq_idx'] = range(len(df2))

    merged = map_df \
        .merge(df1, how="left", left_on="i1", right_on="seq_idx", suffixes=("", "_df1")) \
        .merge(df2, how="left", left_on="i2", right_on="seq_idx", suffixes=("", "_df2"))

    merged.drop(columns=["i1", "i2", "seq_idx", "seq_idx_df2"], inplace=True)
    merged.rename(columns={'resi': 'resi_df1', 'resn': 'resn_df1'}, inplace=True)

    return merged


def evaluate_sequence_alignment(merged: pd.DataFrame) -> None:
    """
    Evaluate the quality of a sequence alignment by summarizing mismatches, indels, and gaps at termini.

    Parameters
    ----------
    merged : pd.DataFrame
        Merged DataFrame containing combined sequence information from both sequences.

    Returns
    -------
    None
        Prints a summary of alignment quality metrics.
    """
    total_residues = len(merged)
    mismatch_mask = (merged['resn_df1'].notna()) & (merged['resn_df2'].notna()) & (merged['resn_df1'] != merged['resn_df2'])
    indel_mask = ((merged['resn_df1'].isna()) | (merged['resn_df2'].isna())).to_numpy()
    termini_mask = [False] * total_residues

    # Check for contiguous blocks of indels at beginning or end
    if indel_mask[0] or indel_mask[-1]:
        for i in range(total_residues):
            if indel_mask[i]:
                termini_mask[i] = True
            else:
                break
        for i in range(total_residues - 1, -1, -1):
            if indel_mask[i]:
                termini_mask[i] = True
            else:
                break

        # Exclude terminal gaps from indel count
        indel_mask = indel_mask & (~pd.Series(termini_mask))

    if mismatches := mismatch_mask.sum():
        warnings.warn(f"Found {mismatches} mismatches out of {total_residues} residues "
              f"({(mismatches / total_residues) * 100:.2f}%) \n"
              f" Mismatches found at the following positions in df1: {merged.loc[mismatch_mask, 'resi_df1'].tolist()}.")

    if indels := indel_mask.sum():
        warnings.warn(f"Found {indels} residues with indels out of {total_residues} residues "
              f"({(indels / total_residues) * 100:.2f}%) \n"
              f" Indels found at the following positions in df1 {merged.loc[indel_mask, 'resi_df1'].tolist()}"
                      f" and df2 {merged.loc[indel_mask, 'resi_df2'].tolist()}.")

    if sum(termini_mask):
        warnings.warn(f"Found gaps at the termini of the sequence alignment, "
                       f" at positions {merged.loc[termini_mask, 'resi_df1'].tolist()} in df1 "
                       f" and {merged.loc[termini_mask, 'resi_df2'].tolist()} in df2.")



    # print(f"Alignment Summary:")
    # print(f"Total residues compared: {total_residues}")
    # print(f"Mismatches: {mismatches} ({(mismatches / total_residues) * 100:.2f}%)")
    # print(f"Indels: {indels} ({(indels / total_residues) * 100:.2f}%)")
    # print(f"N-terminal gaps - Seq1: {n_term_gaps_seq1}, Seq2: {n_term_gaps_seq2}")
    # print(f"C-terminal gaps - Seq1: {c_term_gaps_seq1}, Seq2: {c_term_gaps_seq2}")


def merge_mutation_scores(mutation_scores: pd.DataFrame, residue_table: pd.DataFrame, chain: str) -> pd.DataFrame:
    """
    Merge mutation scores with structural context based on residue positions.

    Parameters
    ----------
    mutation_scores : pd.DataFrame
        DataFrame containing mutation scores with 'resi' and 'resn' columns.
    residue_table : pd.DataFrame
        DataFrame containing structural residue information with 'chain',
        'resi', and 'resn' columns.
    chain : str
        Chain identifier to filter structural context.

    Returns
    -------
    pd.DataFrame
        Merged DataFrame with mutation scores and structural features. Contains
        columns for chain, resi, resn, resm, type, effect, seq_info, and
        struct_info.

    Raises
    ------
    ValueError
        If there is a mismatch between mutation scores and structure residues
        for the specified chain.
    """
    # Extract residue information from context
    res_table = residue_table.copy()

    # test merge to make sure sequence is aligned with structure
    res_test = res_table.loc[res_table['chain'] == chain, ["resn", "resi"]].reset_index(drop=True)
    mutation_test = mutation_scores[['resn', 'resi']].drop_duplicates().reset_index(drop=True)

    merge_test = pd.merge(res_test, mutation_test,
                              left_on=['resi', 'resn'],
                              right_on=['resi', 'resn'],
                              how='outer')

    # If sequence is aligned between mutation data and structure, there should only be one row for each unique residue index
    if len(merge_test) > len(merge_test.resi.unique()):
        raise ValueError(f"Mismatch between mutation scores and structure residues for chain {chain}. "
                      f"Check that the sequence used for mutation data matches the structure.")

    res_table['struct_info'] = True
    res_table_chain = res_table[res_table['chain'] == chain].reset_index(drop=True)
    # Copy to avoid modifying the caller's DataFrame when adding seq_info column
    mutation_scores = mutation_scores.copy()
    mutation_scores['seq_info'] = True

    # Merge mutation scores with structural residue table
    merged_df = pd.merge(mutation_scores, res_table_chain,
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