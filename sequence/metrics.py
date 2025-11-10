import pandas as pd
import numpy as np

def calculate_position_effect_quartiles(scores: pd.DataFrame, percentiles: list = [25, 50, 75]) -> pd.DataFrame:
    """
    Calculate quartiles of position effect scores.

    Parameters:
    -----------
    scores : pd.DataFrame
        DataFrame containing DMS scores

    percentiles : list
        List of percentiles to calculate (default: [25, 50, 75])

    Returns:
    --------
    pd.DataFrame
        DataFrame with quartile column and residue information
    """

    # Determine if position effects are already calculated or need to be computed from data
    if 'pos_effect' in scores.columns:
        # exclude synonymous mutations, which have undefined position effects, and subset
        pos_scores = scores[['position', 'pos_effect', 'type']]
        pos_scores = pos_scores.loc[pos_scores.type != 'synonymous', ['position', 'pos_effect']]
        pos_scores = pos_scores.drop_duplicates()

    else:
        # compute position effects from individual mutation effects, removing synonymous mutations
        pos_counts = scores[['position', 'effect', 'type']]
        pos_counts = pos_counts.loc[pos_counts.type != 'synonymous', ['position', 'effect']]
        pos_scores = pos_counts.groupby('position')['effect'].mean().reset_index()
        pos_scores.rename(columns={'effect': 'pos_effect'}, inplace=True)

    # define cutoffs for position effects
    cutoffs = np.percentile(pos_scores.pos_effect.dropna(), percentiles)

    # label positions based on quartiles
    pos_scores['effect_quartile'] = pd.cut(
        pos_scores['pos_effect'],
        bins=[-np.inf, cutoffs[0], cutoffs[1], cutoffs[2], np.inf],
        labels=['Q1', 'Q2', 'Q3', 'Q4']
    )

    return pos_scores


def calculate_aaindex_scores(scores: pd.DataFrame, aaindex_data: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate AAIndex scores for each mutation in the scores DataFrame.

    Parameters:
    -----------
    scores : pd.DataFrame
        DataFrame containing mutation data with 'wildtype' and 'mutant' columns.

    aaindex_data : pd.DataFrame
        DataFrame containing AAIndex values with amino acids columns and scores as rows.

    Returns:
    --------
    pd.DataFrame
        DataFrame with additional AAIndex score columns for wildtype and mutant amino acids.
    """

    aaindex_scores = scores.copy()

    # Create a dictionary mapping AAIndex feature to its values, truncating first two metadata columns
    feature_dict = {f: aaindex_data.loc[aaindex_data.accession == f].iloc[0][2:]
                    for f in aaindex_data.accession.unique()}

    for aa_feature, feature_values in feature_dict.items():
        aaindex_scores[f'AAIndex_{aa_feature}_wt'] = scores['wildtype'].map(feature_values)
        aaindex_scores[f'AAIndex_{aa_feature}_mut'] = scores['mutation'].map(feature_values)
        aaindex_scores[f'AAIndex_{aa_feature}_diff'] = (
            aaindex_scores[f'AAIndex_{aa_feature}_mut'] - aaindex_scores[f'AAIndex_{aa_feature}_wt']
        )

    return aaindex_scores

