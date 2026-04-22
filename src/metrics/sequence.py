import logging
from pathlib import Path
from typing import List, Optional

import blosum as bl
import numpy as np
import pandas as pd
from Bio.Align import substitution_matrices

from src.metrics.aaindex_schema import AAINDEX_AA_COLUMNS
from src.metrics.mutation_category_gmm import (
    MUTATION_CATEGORY_CENTRAL_INTERVAL,
    classify_stop,
    classify_synonymous,
    fit_mutation_category_reference,
    save_mutation_category_diagnostic_png,
)
from src.metrics.registry import register_metric
from src.pipeline.context import Context
from src.sequence.utils import convert_amino_acid_3to1

logger = logging.getLogger(__name__)

# columns to keep for sequence feature calculation to enable merging back to full table
KEEP_COLS = ['chain', 'resi_mut', 'resn_mut', 'resm']


def _empty_mutation_category_frame(seq_data: pd.DataFrame) -> pd.DataFrame:
    keep_cols = [col for col in KEEP_COLS if col in seq_data.columns]
    empty = seq_data[keep_cols].copy()
    empty['mutation_category'] = pd.Series(pd.NA, index=empty.index, dtype='object')
    empty['total_lof'] = np.nan
    empty['total_gof'] = np.nan
    return empty


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


@register_metric(name='effect_variance', provides=['effect_variance', 'effect_variance_rank'], tags={'sequence'})
def calculate_effect_variance(context: Context) -> pd.DataFrame:
    """
    Calculate the standard error of the mean for the effect scores at each position.

    Parameters
    ----------
    context : Context
        Context object containing residue metadata, structural information, and mutation information.

    Returns
    -------
    pd.DataFrame
        DataFrame with 'effect_variance' and 'effect_variance_rank' along with residue metadata.
    """
    
    # Calculate SEM for each position from rows that actually have mutation context.
    seq_data = context.residue_table.loc[context.residue_table.mut_info, :].copy()
    position_cols = ['chain', 'resi_mut', 'resn_mut']
    effect_variance = (
        seq_data
        .groupby(position_cols, dropna=False)['effect']
        .sem()
        .reset_index(name='effect_variance')
    )
    
    # Rank positions based on variance
    effect_variance['effect_variance_rank'] = effect_variance['effect_variance'].rank(method='min')
    effect_variance['effect_variance_rank'] = effect_variance['effect_variance_rank'] / np.max(effect_variance['effect_variance_rank'])
    
    # Add relevant metadata from original table
    keep_cols = [col for col in KEEP_COLS if col in seq_data.columns]
    effect_variance = pd.merge(
        seq_data[keep_cols],
        effect_variance,
        on=position_cols,
        how='left'
    )
    
    return effect_variance


@register_metric(name='effect_ranking', provides=['effect', 'effect_ranking'], tags={'sequence'})
def calculate_effect_ranking(context: Context) -> pd.DataFrame:
    """
    Calculate the ranking of the effects.

    Parameters
    ----------
    context : Context
        Context object containing residue metadata, structural information, and mutation information.

    Returns
    -------
    pd.DataFrame
        DataFrame with 'effect_ranking' along with residue metadata.
    """
    
    # Calculate ranking for each position
    effect_ranking = context.residue_table.loc[context.residue_table.mut_info, :]
    keep_cols = [col for col in KEEP_COLS if col in context.residue_table.columns]
    effect_ranking = effect_ranking[keep_cols + ['effect']]

    effect_ranking['effect_ranking'] = effect_ranking['effect'].rank(method='min')
    effect_ranking['effect_ranking'] = effect_ranking['effect_ranking'] / np.max(effect_ranking['effect_ranking'])
    
    return effect_ranking


@register_metric(
    name='mutation_category',
    provides=['mutation_category', 'total_lof', 'total_gof'],
    tags={'sequence'},
)
def calculate_mutation_category(context: Context) -> pd.DataFrame:
    """
    Classify mutations from a 2-component Gaussian mixture reference and count LOF/GOF per position.

    Synonymous effects use a mixture-sampled equal-tail interval; stop effects use the lower-mean
    component when well-separated, otherwise the combined mixture.
    """
    seq_data = context.residue_table.loc[context.residue_table.mut_info, :].copy()
    if seq_data.empty:
        logger.info(
            "mutation_category: no mutation rows (mut_info); leaving mutation_category, total_lof, total_gof unset."
        )
        return _empty_mutation_category_frame(seq_data)

    central_interval = MUTATION_CATEGORY_CENTRAL_INTERVAL
    fit = fit_mutation_category_reference(seq_data, central_interval)
    if fit is None:
        logger.info(
            "mutation_category: reference fit failed; leaving mutation_category, total_lof, total_gof unset "
            "(see warnings above)."
        )
        return _empty_mutation_category_frame(seq_data)

    effect_values = seq_data['effect']
    if fit.reference_type == 'synonymous':
        mutation_category = classify_synonymous(fit.lower_bound, fit.upper_bound, effect_values)
    else:
        mutation_category = classify_stop(fit.upper_bound, effect_values)

    keep_cols = [col for col in KEEP_COLS if col in seq_data.columns]
    mutation_categories = seq_data[keep_cols].copy()
    mutation_categories['mutation_category'] = mutation_category

    position_cols = ['chain', 'resi_mut', 'resn_mut']
    position_counts = seq_data[position_cols].copy()
    position_counts['total_lof'] = mutation_category.eq('LOF').fillna(False).astype(int)
    position_counts['total_gof'] = mutation_category.eq('GOF').fillna(False).astype(int)
    position_counts = (
        position_counts
        .groupby(position_cols, dropna=False)[['total_lof', 'total_gof']]
        .sum()
        .reset_index()
    )

    mutation_categories = pd.merge(
        mutation_categories,
        position_counts,
        on=position_cols,
        how='left',
    )

    cfg = context.config
    base = Path(cfg.output_dir)
    parts = []
    if cfg.output_prefix:
        p = str(cfg.output_prefix).strip().strip('_')
        if p:
            parts.append(p)
    if cfg.name:
        parts.append(str(cfg.name))
    elif cfg.pdb_id:
        parts.append(str(cfg.pdb_id))
    stem = '_'.join(parts) if parts else 'run'
    out_path = base / 'logs' / f'{stem}_mutation_category_gmm_fit.png'
    save_mutation_category_diagnostic_png(fit, central_interval, out_path)

    return mutation_categories


@register_metric(name='aaindex_scores', provides={'{accession}_{category}_wt', '{accession}_{category}_mut',
                                                   '{accession}_{category}_diff'},
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
        DataFrame with ``{accession}_{category}_wt``, ``{accession}_{category}_mut``,
        and ``{accession}_{category}_diff`` for each index row, along with residue metadata.
    """
    # extract params
    residue_table, aaindex_data = context.residue_table, context.extras['aaindex']

    # remove resm if not present
    keep_cols = [col for col in KEEP_COLS if col in residue_table.columns]
    aaindex_scores = residue_table.loc[residue_table.mut_info, keep_cols].copy()

    for _, row in aaindex_data.iterrows():
        accession = row.iloc[0]
        category = row.iloc[2]
        base = f'{accession}_{category}'
        feature_values = row.loc[list(AAINDEX_AA_COLUMNS)]

        aaindex_scores[f'{base}_wt'] = aaindex_scores['resn_mut'].map(feature_values)

        # calculate for mutant only if mutation column exists
        if 'resm' in keep_cols:
            aaindex_scores[f'{base}_mut'] = aaindex_scores['resm'].map(feature_values)
            aaindex_scores[f'{base}_diff'] = (
                aaindex_scores[f'{base}_mut'] - aaindex_scores[f'{base}_wt']
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
    
    residue_table, blosum_threshold = context.residue_table, 90
    blosum_scores = residue_table.loc[residue_table.mut_info, KEEP_COLS].copy()
    b_matrix = bl.BLOSUM(blosum_threshold)

    def get_blosum_score(row):
        wt = convert_amino_acid_3to1(row['resn_mut'])
        mut = convert_amino_acid_3to1(row['resm'])
        return b_matrix[wt][mut]

    blosum_scores[f'blosum{blosum_threshold}'] = blosum_scores.apply(get_blosum_score, axis=1)

    return blosum_scores


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
        wt = convert_amino_acid_3to1(row['resn_mut'])
        mut = convert_amino_acid_3to1(row['resm'])
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
    aa_groupings = residue_table.loc[residue_table.mut_info, keep_cols].copy()
    
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
