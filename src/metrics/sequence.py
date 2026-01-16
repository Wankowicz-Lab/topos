import pandas as pd
import numpy as np
import logging

import blosum as bl

from src.sequence.utils import convert_amino_acid
from src.pipeline.context import Context
from src.metrics.registry import register_metric
from Bio.Align import substitution_matrices

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
    logger.info("Calculating position effect quartiles")
    
    if percentiles is None:
        percentiles = [25, 50, 75]
    # subset to only include positions with mutation data
    seq_data = context.residue_table.loc[context.residue_table.mut_info, :]

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
    logger.info("Calculating AAIndex scores")
    
    # extract params
    residue_table, aaindex_data = context.residue_table, context.extras['aaindex']

    # remove resm if not present
    keep_cols = [col for col in KEEP_COLS if col in residue_table.columns]
    aaindex_scores = residue_table.loc[residue_table.mut_info, keep_cols].copy()

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
    logger.info("Calculating Kidera factors")
    
    # extract params
    residue_table, kidera_data = context.residue_table, context.extras['kidera']

    # remove resm if not present
    keep_cols = [col for col in KEEP_COLS if col in residue_table.columns]
    kidera_scores = residue_table.loc[residue_table.mut_info, keep_cols].copy()

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
    logger.info("Calculating BLOSUM scores")
    
    residue_table, blosum_threshold = context.residue_table, 90
    blosum_scores = residue_table.loc[residue_table.mut_info, KEEP_COLS].copy()
    b_matrix = bl.BLOSUM(blosum_threshold)

    def get_blosum_score(row):
        wt = convert_amino_acid(row['resn_mut'])
        mut = convert_amino_acid(row['resm'])
        return b_matrix[wt][mut]

    blosum_scores[f'blosum{blosum_threshold}'] = blosum_scores.apply(get_blosum_score, axis=1)

    return blosum_scores


def make_phat75_73():
    """
    Make Phat 75/73 substitution matrix.

    Returns
    -------
    substitution_matrices.Array
        Phat 75/73 substitution matrix.
    """

    PHAT_ALPHABET = "ARNDCQEGHILKMFPSTWYV"

    # PHAT 75/73 scores (symmetric). Values transcribed from the PHAT paper figure.
    _PHAT_LOWER_TRI = [
    [  5],
    [ -6,  9],
    [ -2, -3, 11],
    [ -5, -7,  2, 12],
    [  1, -8, -2, -7,  7],
    [ -3, -2,  2,  0, -5,  9],
    [ -5, -6,  0,  6, -7,  1, 12],
    [  1, -5, -1, -2, -2, -2, -3,  9],
    [ -3, -4,  4, -1, -7,  2, -1, -4, 11],
    [  0, -6, -3, -5, -3, -3, -5, -2, -5,  5],
    [ -1, -6, -3, -5, -2, -3, -5, -2, -4,  2,  4],
    [ -7, -1, -2, -5,-10, -1, -4, -5, -5, -7, -7,  5],
    [ -1, -6, -2, -5, -2, -1, -5, -1, -4,  3,  2, -6,  6],
    [ -1, -7, -1, -5,  0, -2, -5, -2, -2,  0,  1, -7,  0,  6],
    [ -3, -7, -4, -5, -8, -3, -5, -3, -6, -4, -5, -4, -5, -5, 13],
    [  2, -6,  1, -4,  1, -1, -3,  1, -2, -2, -2, -5, -2, -2, -3,  6],
    [  0, -6, -1, -5, -1, -3, -5, -1, -4, -1, -1, -6,  0, -2, -4,  1,  3],
    [ -4, -7, -5, -7, -4,  1, -7, -5, -3, -4, -3, -8, -4,  0, -6, -5, -7, 11],
    [ -3, -6,  2, -4, -1,  0, -2, -3,  3, -3, -2, -4, -2,  4, -5, -2, -3,  1, 11],
    [  1, -7, -3, -5, -2, -3, -5, -2, -5,  3,  1, -8,  1, -1, -4, -2,  0, -4, -3,  4],
    ]

    n = len(PHAT_ALPHABET)
    mat = np.zeros((n, n), dtype=int)
    # fill lower triangle + diagonal
    for i, row in enumerate(_PHAT_LOWER_TRI):
        mat[i, :i+1] = row
    # symmetrize
    mat = mat + mat.T - np.diag(np.diag(mat))

    return substitution_matrices.Array(alphabet=PHAT_ALPHABET, data=mat)


@register_metric(name='phat_score', provides=['phat_score'], requires={'resm'}, tags={'sequence'})
def calculate_phat_score(context: Context) -> pd.DataFrame:
    """
    Calculate Phat score for each mutation in the scores DataFrame.

    Parameters
    ----------
    context : Context
        Context object containing residue metadata, structural information, and mutation information.

    Returns
    -------
    pd.DataFrame
        DataFrame with 'phat_score' along with residue metadata.
    """    
    residue_table = context.residue_table
    phat_scores = residue_table.loc[residue_table.mut_info, KEEP_COLS].copy()

    PHAT75_73 = make_phat75_73()

    def get_phat_score(row):
        wt = convert_amino_acid(row['resn_mut'])
        mut = convert_amino_acid(row['resm'])
        if wt not in PHAT75_73.alphabet or mut not in PHAT75_73.alphabet:
            return np.inf
        return PHAT75_73[wt][mut]

    phat_scores['phat_score'] = phat_scores.apply(get_phat_score, axis=1)

    return phat_scores


@register_metric(name='aa_groupings', provides=['wildtype_aa_group', 'mut_aa_group', 'wildtype_mut_aa_group'],
                 requires={'resm'}, tags={'sequence'})
def calculate_aa_groupings(context: Context) -> pd.DataFrame:
    """
    Calculate amino acid groupings for each mutation.

    Parameters
    ----------
    context : Context
        Context object containing residue metadata, structural information, and mutation information.

    Returns
    -------
    pd.DataFrame
        DataFrame with 'wildtype_aa_group', 'mut_aa_group', 'wildtype_mut_aa_group' along with residue metadata.
    """
    logger.info("Calculating amino acid groupings")
    
    # Define amino acid groups
    aa_groups_3letter = {
        "Nonpolar_Aliphatic": ["ALA", "VAL", "LEU", "ILE", "MET"],
        "Aromatic": ["PHE", "TRP", "TYR"],
        "Polar_Uncharged": ["SER", "THR", "ASN", "GLN", "CYS"],
        "Positively_Charged": ["LYS", "ARG", "HIS"],
        "Negatively_Charged": ["ASP", "GLU"],
        "Special": ["PRO", "GLY"]
    }
    
    # Create reverse mapping from amino acid to group
    aa_to_group = {}
    for group_name, aa_list in aa_groups_3letter.items():
        for aa in aa_list:
            aa_to_group[aa] = group_name
    
    # Extract residue table and subset to KEEP_COLS
    residue_table = context.residue_table
    keep_cols = [col for col in KEEP_COLS if col in residue_table.columns]
    aa_groupings = residue_table[keep_cols].copy()
    
    # Map wildtype amino acid to group
    aa_groupings['wildtype_aa_group'] = aa_groupings['resn_mut'].map(aa_to_group)
    
    # Map mutant amino acid to group if resm column exists
    if 'resm' in keep_cols:
        aa_groupings['mut_aa_group'] = aa_groupings['resm'].map(aa_to_group)
        
        # Create concatenated column: wildtype_group_mut_group
        # Handle NaN values - if either is NaN, result should be NaN
        mask = aa_groupings['wildtype_aa_group'].notna() & aa_groupings['mut_aa_group'].notna()
        aa_groupings['wildtype_mut_aa_group'] = pd.Series(dtype='object')
        aa_groupings.loc[mask, 'wildtype_mut_aa_group'] = (
            aa_groupings.loc[mask, 'wildtype_aa_group'].astype(str) + '_' + 
            aa_groupings.loc[mask, 'mut_aa_group'].astype(str)
        )
    
    return aa_groupings
