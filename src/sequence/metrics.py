import pandas as pd
import numpy as np

import blosum as bl

from src.sequence.sequence_context import convert_amino_acid
from src.structure.structure_context import Context, register_metric

# columns to keep for sequence feature calculation to enable merging back to full table
KEEP_COLS = ['chain', 'resi', 'resn', 'resm']

@register_metric(name='position_effect_quartiles', provides='effect_quartile', tags={'sequence', 'dms'})
def calculate_position_effect_quartiles(context: Context, percentiles: list = [25, 50, 75]) -> pd.DataFrame:
    """
    Calculate quartiles of position effect scores.

    Parameters:
    -----------
    context : Context
        Context object containing residue metadata and DMS scores

    percentiles : list
        List of percentiles to calculate (default: [25, 50, 75])

    Returns:
    --------
    pd.DataFrame
        DataFrame with quartile column and residue information
    """
    # subset to only include positions with DMS data
    seq_data = context.residue_table.loc[context.residue_table.seq_info, :]

    # Determine if position effects are already calculated or need to be computed from data
    if 'pos_effect' in seq_data.columns:
        # exclude synonymous mutations, which have undefined position effects, and subset
        pos_scores = seq_data[['resi', 'pos_effect', 'type']]
        pos_scores = pos_scores.loc[pos_scores.type != 'synonymous', ['resi', 'pos_effect']]
        pos_scores = pos_scores.drop_duplicates()

    else:
        # compute position effects from individual mutation effects, removing synonymous mutations
        pos_counts = seq_data[['resi', 'effect', 'type']]
        pos_counts = pos_counts.loc[pos_counts.type != 'synonymous', ['resi', 'effect']]
        pos_scores = pos_counts.groupby('resi')['effect'].mean().reset_index()
        pos_scores.rename(columns={'effect': 'pos_effect'}, inplace=True)

    # define cutoffs for position effects
    cutoffs = np.percentile(pos_scores.pos_effect.dropna(), percentiles)

    # label positions based on quartiles
    pos_scores['effect_quartile'] = pd.cut(
        pos_scores['pos_effect'],
        bins=[-np.inf, cutoffs[0], cutoffs[1], cutoffs[2], np.inf],
        labels=['Q1', 'Q2', 'Q3', 'Q4']
    )

    # map quartile labels and raw effect scores back to original residues
    # TODO: decide whether this should be per residue or per mutation, drop duplicates as needed
    pos_scores = pd.merge(seq_data[['resi', 'resn']], pos_scores, on='resi', how='left')

    return pos_scores


def calculate_aaindex_scores(residue_table: pd.DataFrame, aaindex_data: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate AAIndex scores for each mutation in the scores DataFrame.

    Parameters:
    -----------
    residue_table : pd.DataFrame
        DataFrame containing residue metadata

    aaindex_data : pd.DataFrame
        DataFrame containing AAIndex values with amino acids columns and scores as rows.

    Returns:
    --------
    pd.DataFrame
        DataFrame with additional AAIndex score columns for wildtype and mutant amino acids.
    """

    # subset to relevant columns for output
    keep_cols = [col for col in KEEP_COLS if col in residue_table.columns]
    aaindex_scores = residue_table[keep_cols].copy()

    # Create a dictionary mapping AAIndex feature to its values, truncating first two metadata columns
    feature_dict = {f: aaindex_data.loc[aaindex_data.accession == f].iloc[0][2:]
                    for f in aaindex_data.accession.unique()}

    for aa_feature, feature_values in feature_dict.items():
        aaindex_scores[f'AAIndex_{aa_feature}_wt'] = aaindex_scores['resn'].map(feature_values)

        # calculate for mutant only if mutation column exists
        if 'resm' in keep_cols:
            aaindex_scores[f'AAIndex_{aa_feature}_mut'] = aaindex_scores['resm'].map(feature_values)
            aaindex_scores[f'AAIndex_{aa_feature}_diff'] = (
                aaindex_scores[f'AAIndex_{aa_feature}_mut'] - aaindex_scores[f'AAIndex_{aa_feature}_wt']
            )

    return aaindex_scores

def calculate_blosum_score(residue_table: pd.DataFrame, blosum_threshold: int = 90) -> pd.DataFrame:
    """
    Calculate BLOSUM scores for each mutation in the scores DataFrame.

    Parameters:
    -----------
    residue_table : pd.DataFrame
        DataFrame containing residue metadata

    blosum_threshold : int
        BLOSUM matrix threshold to use (default: 90).

    Returns:
    --------
    pd.DataFrame
        DataFrame with additional BLOSUM score column.
    """

    blosum_scores = residue_table.loc[residue_table.seq_info, KEEP_COLS].copy()
    b_matrix = bl.BLOSUM(blosum_threshold)

    def get_blosum_score(row):
        wt = convert_amino_acid(row['resn'])
        mut = convert_amino_acid(row['resm'])
        return b_matrix[wt][mut]

    blosum_scores[f'blosum{blosum_threshold}'] = blosum_scores.apply(get_blosum_score, axis=1)

    return blosum_scores
