import pandas as pd
import numpy as np
import logging

import blosum as bl

from src.sequence.utils import convert_amino_acid
from src.structure.structure_context import Context, register_metric

from typing import List, Optional

logger = logging.getLogger(__name__)

# columns to keep for sequence feature calculation to enable merging back to full table
KEEP_COLS = ['chain', 'resi_mut', 'resn_mut', 'resm']

@register_metric(name='position_effect_quartiles', provides=['effect_quartile', 'pos_effect'],
                 requires={'resm'}, tags={'sequence'})
def calculate_position_effect_quartiles(context: Context, percentiles: Optional[List[float]] = None) -> pd.DataFrame:
    """
    Calculate quartiles of position effect scores.

    Parameters
    ----------
    context : Context
        Context object containing residue metadata, structural information, and mutation information.
    percentiles : list of float, optional
        List of percentiles to calculate. Default is [25, 50, 75].

    Returns
    -------
    pd.DataFrame
        DataFrame with 'effect_quartile', 'pos_effect' along with residue metadata.

    Raises
    ------
    ValueError
        If the residue table contains data from more than one chain.
    """
    if percentiles is None:
        percentiles = [25, 50, 75]
    # subset to only include positions with mutation data
    seq_data = context.residue_table.loc[context.residue_table.mut_info, :]
    num_positions = len(seq_data['resi_mut'].unique())
    logger.info(f"Calculating position effect quartiles for {num_positions} positions")

    # ensure that only a single chain is provided
    if len(seq_data.chain.unique()) > 1:
        raise ValueError("calculate_position_effect_quartiles only supports single chain mutation data.")

    # Determine if position effects are already calculated or need to be computed from data
    if 'pos_effect' in seq_data.columns:
        # exclude synonymous mutations, which have undefined position effects, and subset
        pos_scores = seq_data[['resi_mut', 'pos_effect', 'type']]
        pos_scores = pos_scores.loc[pos_scores.type != 'synonymous', ['resi_mut', 'pos_effect']]
        pos_scores = pos_scores.drop_duplicates()

    else:
        # compute position effects from individual mutation effects, removing synonymous mutations
        pos_counts = seq_data[['resi_mut', 'effect', 'type']]
        pos_counts = pos_counts.loc[pos_counts.type != 'synonymous', ['resi_mut', 'effect']]
        pos_scores = pos_counts.groupby('resi_mut')['effect'].mean().reset_index()
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
    pos_scores = pd.merge(seq_data[KEEP_COLS], pos_scores, on='resi_mut', how='left')

    logger.info(f"Position effect quartiles calculation completed for {num_positions} positions")
    return pos_scores

@register_metric(name='aaindex_scores', provides={'AAIndex_{acc}_wt', 'AAIndex_{acc}_mut', 'AAIndex_{acc}_diff'},
                 tags={'sequence'})
def calculate_aaindex_scores(context: Context) -> pd.DataFrame:
    """
    Calculate AAIndex scores for each mutation in the scores DataFrame.

    Parameters
    ----------
    context : Context
        Context object containing residue metadata, structural information, and mutation information.

    Returns
    -------
    pd.DataFrame
        DataFrame with 'AAIndex_{acc}_wt', 'AAIndex_{acc}_mut', 'AAIndex_{acc}_diff' along with residue metadata.
    """
    # extract params
    residue_table, aaindex_data = context.residue_table, context.extras['aaindex']
    num_residues = len(residue_table)
    num_features = len(aaindex_data.accession.unique())
    logger.info(f"Calculating AAIndex scores for {num_residues} residues with {num_features} features")

    # remove resm if not present
    keep_cols = [col for col in KEEP_COLS if col in residue_table.columns]
    aaindex_scores = residue_table[keep_cols].copy()

    # Create a dictionary mapping AAIndex feature to its values, truncating first two metadata columns
    feature_dict = {f: aaindex_data.loc[aaindex_data.accession == f].iloc[0][2:]
                    for f in aaindex_data.accession.unique()}

    for aa_feature, feature_values in feature_dict.items():
        aaindex_scores[f'AAIndex_{aa_feature}_wt'] = aaindex_scores['resn_mut'].map(feature_values)

        # calculate for mutant only if mutation column exists
        if 'resm' in keep_cols:
            aaindex_scores[f'AAIndex_{aa_feature}_mut'] = aaindex_scores['resm'].map(feature_values)
            aaindex_scores[f'AAIndex_{aa_feature}_diff'] = (
                aaindex_scores[f'AAIndex_{aa_feature}_mut'] - aaindex_scores[f'AAIndex_{aa_feature}_wt']
            )

    logger.info(f"AAIndex scores calculation completed for {num_residues} residues")
    return aaindex_scores


@register_metric(name='kidera_factors', provides={'kidera_{factornum}_wt', 'kidera_{factornum}_mut',
                                                  'kidera_{factornum}_diff'}, tags={'sequence'})
def calculate_kidera_factor_scores(context: Context) -> pd.DataFrame:
    """
    Calculate kidera factor scores for each mutation in the scores DataFrame.

    Parameters
    ----------
    context : Context
        Context object containing residue metadata, structural information, and mutation information.

    Returns
    -------
    pd.DataFrame
        DataFrame with 'kidera_{factornum}_wt', 'kidera_{factornum}_mut', 'kidera_{factornum}_diff', along with residue metadata.
    """
    # extract params
    residue_table, kidera_data = context.residue_table, context.extras['kidera']
    num_residues = len(residue_table)
    num_factors = len(kidera_data['factor'].unique())
    logger.info(f"Calculating Kidera factors for {num_residues} residues with {num_factors} factors")

    # remove resm if not present
    keep_cols = [col for col in KEEP_COLS if col in residue_table.columns]
    kidera_scores = residue_table[keep_cols].copy()

    # Create a dictionary mapping kidera feature to its values, truncating first two metadata columns
    feature_dict = {f: kidera_data.loc[kidera_data['factor'] == f].iloc[0][2:]
                    for f in kidera_data['factor'].unique()}

    for kidera_feature, feature_values in feature_dict.items():
        kidera_scores[f'kidera_{kidera_feature}_wt'] = kidera_scores['resn_mut'].map(feature_values)

        # calculate for mutant only if mutation column exists
        if 'resm' in keep_cols:
            kidera_scores[f'kidera_{kidera_feature}_mut'] = kidera_scores['resm'].map(feature_values)
            kidera_scores[f'kidera_{kidera_feature}_diff'] = (
                kidera_scores[f'kidera_{kidera_feature}_mut'] - kidera_scores[f'kidera_{kidera_feature}_wt']
            )

    logger.info(f"Kidera factors calculation completed for {num_residues} residues")
    return kidera_scores


@register_metric(name='blosum_score', provides=['blosum90'], requires={'resm'}, tags={'sequence'})
def calculate_blosum_score(context: Context) -> pd.DataFrame:
    """
    Calculate BLOSUM scores for each mutation in the scores DataFrame.

    Parameters
    ----------
    context : Context
        Context object containing residue metadata, structural information, and mutation information.

    Returns
    -------
    pd.DataFrame
        DataFrame with 'blosum90' along with residue metadata.
    """
    residue_table, blosum_threshold = context.residue_table, 90
    blosum_scores = residue_table.loc[residue_table.mut_info, KEEP_COLS].copy()
    num_mutations = len(blosum_scores)
    logger.info(f"Calculating BLOSUM scores for {num_mutations} mutations")
    
    b_matrix = bl.BLOSUM(blosum_threshold)

    def get_blosum_score(row):
        wt = convert_amino_acid(row['resn_mut'])
        mut = convert_amino_acid(row['resm'])
        return b_matrix[wt][mut]

    blosum_scores[f'blosum{blosum_threshold}'] = blosum_scores.apply(get_blosum_score, axis=1)

    logger.info(f"BLOSUM scores calculation completed for {num_mutations} mutations")
    return blosum_scores
