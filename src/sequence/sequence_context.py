
import warnings
from pathlib import Path
from typing import Union

from Bio.Align import PairwiseAligner
import pandas as pd
from src.sequence.utils import convert_amino_acid


def load_mutation_scores(
    path: Union[str, Path],
    residue_col_name: str,
    residue_idx_name: str,
    mutation_col_name: str,
    mutation_type_col_name: str,
    score_col_name: str,
) -> pd.DataFrame:
    """
    Load mutation scores from a CSV file and standardize column names.

    Parameters
    ----------
    path : str or Path
        Path to the CSV file containing mutation scores.
    residue_col_name : str
        Name of the column containing wildtype residues.
    residue_idx_name : str
        Name of the column containing residue positions.
    mutation_col_name : str
        Name of the column containing mutant residues.
    mutation_type_col_name : str
        Name of the column containing mutation types.
    score_col_name : str
        Name of the column containing mutation effect scores.

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
        raise ValueError(
            f"Columns {missing_cols} not found in mutation scores file at {path}.\n"
            f"Expected columns based on config settings:\n"
            f"  - Wildtype residue column: '{residue_col_name}'\n"
            f"  - Residue position column: '{residue_idx_name}'\n"
            f"  - Mutant residue column: '{mutation_col_name}'\n"
            f"  - Mutation type column: '{mutation_type_col_name}'\n"
            f"  - Mutation score column: '{score_col_name}'\n"
            f"Available columns in file: {list(df.columns)}"
        )
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
    valid_types = {'missense', 'nonsense', 'silent', 'insertion', 'deletion', 'synonymous', 'indel', 'del', 'ins', 'stop'}
    found_types = set(df['type'].unique())
    if not found_types.issubset(valid_types):
        invalid_types = found_types - valid_types
        warnings.warn(f"Mutation types contain unexpected values. Expected types include {valid_types}. "
                      f"Found invalid types: {invalid_types}.")

    return df


def alignment_to_index_map(alignment):
    """
    Convert alignment.coordinates into explicit per-residue index mapping to allow indexing into pandas df.

    Parameters
    ----------
    alignment : Alignment object
        Alignment object containing coordinates attribute.

    Returns:
    -------
    list of tuples
        list of (align_pos, idx1, idx2) where either idx may be None for gaps
    """

    coords = alignment.coordinates  # shape (2, n_segments+1)
    map_list = []
    align_pos = 0

    for col in range(coords.shape[1] - 1):
        start1, end1 = coords[0, col], coords[0, col + 1]
        start2, end2 = coords[1, col], coords[1, col + 1]

        len1 = end1 - start1
        len2 = end2 - start2

        if len1 == len2:
            # Match/substitution block
            for i in range(len1):
                map_list.append((align_pos, start1 + i, start2 + i))
                align_pos += 1
        elif len1 > len2:
            # Deletion in seq2 (seq1 has extra)
            for i in range(len1):
                # seq2 gap for positions beyond its end
                map_list.append((align_pos, start1 + i, start2 + i if i < len2 else None))
                align_pos += 1
        else:
            # Insertion in seq2 (seq2 has extra)
            for i in range(len2):
                map_list.append((align_pos, start1 + i if i < len1 else None, start2 + i))
                align_pos += 1

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
        List of (align_pos, idx1, idx2) tuples mapping indices from df1 to df2. Either idx may be None for gaps.

    Returns
    -------
    pd.DataFrame
        Merged DataFrame containing combined sequence information from both DataFrames.
    """

    map_df = pd.DataFrame(mapping, columns=["align_pos", "i1", "i2"])

    # copy dfs to avoid modifying originals
    df1 = df1.copy()
    df2 = df2.copy()

    # add sequential index to each df for merging
    df1['seq_idx'] = range(len(df1))
    df2['seq_idx'] = range(len(df2))

    merged = (
        map_df
        .merge(df1, how="left", left_on="i1", right_on="seq_idx", suffixes=("", "_df1"))
        .merge(df2, how="left", left_on="i2", right_on="seq_idx", suffixes=("", "_df2"))
    )

    merged.drop(columns=["i1", "i2", "seq_idx", "seq_idx_df2"], inplace=True)
    merged.rename(columns={'resi': 'resi_df1', 'resn': 'resn_df1'}, inplace=True)

    # guarantee stable alignment ordering
    merged = merged.sort_values("align_pos", kind="mergesort").reset_index(drop=True)
    return merged


def evaluate_sequence_alignment(merged: pd.DataFrame, alignment_cutoff: float) -> None:
    """
    Evaluate the quality of a sequence alignment by summarizing mismatches, indels, and gaps at termini.

    Parameters
    ----------
    merged : pd.DataFrame
        Merged DataFrame containing combined sequence information from both sequences.
    alignment_cutoff : float
        Quality cutoff for the alignment. If the proportion of alignment is below this cutoff,
        a warning is issued.

    Returns
    -------
    None
        Issues warnings for alignment quality metrics.
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

    # determine if alignment quality is below cutoff, excluding terminal gaps
    error_mask = mismatch_mask | indel_mask
    error_mask = error_mask[~pd.Series(termini_mask)]

    if (error_mask.sum() / len(error_mask)) > 1 - alignment_cutoff:
        warnings.warn(f"Alignment quality below cutoff of {alignment_cutoff:.2f}. "
                      f"Found {(error_mask.sum() / len(error_mask)) * 100:.2f}% errors "
                      f"({error_mask.sum()} out of {len(error_mask)} residues) "
                      f"excluding terminal gaps.")

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


def merge_mutation_scores(mutation_scores: pd.DataFrame, residue_table: pd.DataFrame,
                          chain: str, alignment_cutoff: float) -> pd.DataFrame:
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
    alignment_cutoff : float
        Quality cutoff for the alignment. If the proportion of alignment is below this cutoff,
        a warning is issued.

    Returns
    -------
    pd.DataFrame
        Merged DataFrame with mutation scores and structural features. Contains
        columns for chain, resi_mut, resn_mut, resm, resi_struct, resn_struct,
        type, effect, mut_info, and struct_info.
    """
    aligner = PairwiseAligner()

    # Create copies to avoid modifying original DataFrames
    residue_table = residue_table.copy()
    mutation_scores = mutation_scores.copy()

    # Subset residue table to the specified chain
    residue_table_chain = residue_table[residue_table['chain'] == chain]

    # Subset mutation scores to only the wildtype sequence
    mutation_scores_subset = mutation_scores[['resi', 'resn']].drop_duplicates()

    # Prepare sequences for alignment, a single string of single-letter amino acids
    mut_seq_short = mutation_scores_subset['resn'].apply(convert_amino_acid)
    mut_seq = "".join(mut_seq_short.tolist())
    res_seq_short = residue_table_chain['resn'].apply(convert_amino_acid)
    res_seq = "".join(res_seq_short.tolist())

    # Perform alignment
    alignment = aligner.align(mut_seq, res_seq)[0]

    # Create mapping to link dataframes based on alignment
    index_map = alignment_to_index_map(alignment)

    # Merge mutation scores and residue table based on alignment mapping
    merged_df = merge_sequence_dfs(df1=mutation_scores_subset, df2=residue_table_chain, mapping=index_map)

    # Evaluate alignment quality
    evaluate_sequence_alignment(merged=merged_df, alignment_cutoff=alignment_cutoff)

    # Add chain information and rename columns
    merged_df['chain'] = chain
    merged_df.rename(columns={'resn_df1': 'resn_mut', 'resi_df1': 'resi_mut', 'resn_df2': 'resn_struct', 'resi_df2': 'resi_struct'}, inplace=True)

    # Add mutation information into merged_df
    merged_df = merged_df.merge(mutation_scores, how='left', left_on=['resi_mut', 'resn_mut'], right_on=['resi', 'resn'])
    
    # Drop duplicate columns from the merge (resi and resn are duplicates of resi_mut and resn_mut)
    merged_df.drop(columns=['resi', 'resn'], inplace=True, errors='ignore')

    # Remove rows from mutation chain from residue table, update with merged rows
    residue_table = residue_table[residue_table['chain'] != chain]
    residue_table.rename(columns={'resn': 'resn_struct', 'resi': 'resi_struct'}, inplace=True)
    residue_table = pd.concat([residue_table, merged_df], axis=0).reset_index(drop=True)

    # Determine which rows have sequence and structure info
    residue_table['mut_info'] = ~residue_table['resn_mut'].isna()
    residue_table['struct_info'] = ~residue_table['resn_struct'].isna()

    # drop extra columns if present
    keep_cols = ['chain', 'resi_mut', 'resn_mut', 'resm', 'resi_struct', 'resn_struct', 'type', 'effect', 'mut_info', 'struct_info', 'align_pos']
    keep_cols += ['pdbtm_region', 'pdbtm_region_detailed'] if 'pdbtm_region' in residue_table.columns else []
    residue_table = residue_table[keep_cols]

    return residue_table
