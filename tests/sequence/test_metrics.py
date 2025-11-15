import numpy as np
import pandas as pd
import random

from src.sequence import metrics
from src.sequence.utils import convert_amino_acid

from tests.test_utils import _random_AA_seq, _make_residue_table

def test_calculate_position_effect_quartiles_with_pos_effect():
    # create test residue table with pos_effect column
    residue_table = _make_residue_table(num_residues=10, num_chains=1, make_muts=True)

    # compute position effects
    pos_effects = residue_table.groupby('resi')['effect'].mean().reset_index()
    pos_effects.rename(columns={'effect': 'pos_effect'}, inplace=True)
    residue_table = pd.merge(residue_table, pos_effects, on='resi', how='left')

    # calculate quartiles
    quartile_df = metrics.calculate_position_effect_quartiles(residue_table)

    # check that quartile labels are correct
    assert 'effect_quartile' in quartile_df.columns
    assert set(quartile_df['effect_quartile'].dropna().unique()).issubset({'Q1', 'Q2', 'Q3', 'Q4'})


def test_calculate_position_effect_quartiles_without_pos_effect():
    # create test residue table without pos_effect column
    residue_table = _make_residue_table(num_residues=10, num_chains=1, make_muts=True)

    # calculate quartiles
    quartile_df = metrics.calculate_position_effect_quartiles(residue_table)

    # check that quartile labels are correct
    assert 'effect_quartile' in quartile_df.columns
    assert set(quartile_df['effect_quartile'].dropna().unique()).issubset({'Q1', 'Q2', 'Q3', 'Q4'})


# create test aaindex data
aaindex_data = pd.DataFrame({
    'accession': ['ANDN920101', 'ARGP820101'],
    'description': ['Hydrophobicity', 'Volume'],
    'ALA': [0.5, 1.0],
    'CYS': [1.5, 2.0],
    'ASP': [2.5, 3.0],
    'GLU': [3.5, 4.0],
    'PHE': [4.5, 5.0],
    'GLY': [5.5, 6.0],
    'HIS': [6.5, 7.0],
    'ILE': [7.5, 8.0],
    'LYS': [8.5, 9.0],
    'LEU': [9.5, 10.0],
    'MET': [10.5, 11.0],
    'ASN': [11.5, 12.0],
    'PRO': [12.5, 13.0],
    'GLN': [13.5, 14.0],
    'SER': [14.5, 15.0],
    'THR': [15.5, 16.0],
    'VAL': [16.5, 17.0],
    'TRP': [17.5, 18.0],
    'TYR': [18.5, 19.0],
    'ARG': [19.5, 20.0],
})

def test_calculate_aaindex_scores_no_muts(aaindex_data=aaindex_data):
    # create test residue table
    residue_table = _make_residue_table(num_residues=5, num_chains=1, make_muts=False)

    # calculate aaindex scores
    aaindex_df = metrics.calculate_aaindex_scores(residue_table, aaindex_data)

    # check that aaindex scores are added
    output_cols = [f'AAIndex_{acc}_wt' for acc in aaindex_data['accession']]

    for col in output_cols:
        assert col in aaindex_df.columns

    # verify that values are correct for wildtype
    for acc in aaindex_data['accession']:
        feature_values = aaindex_data.set_index('accession').loc[acc].iloc[1:]
        for idx, row in aaindex_df.iterrows():
            expected_value = feature_values.get(row['resn'], np.nan)
            assert aaindex_df.at[idx, f'AAIndex_{acc}_wt'] == expected_value


def test_calculate_aaindex_scores_with_muts(aaindex_data=aaindex_data):
    # create test residue table
    residue_table = _make_residue_table(num_residues=5, num_chains=1, make_muts=True)

    # calculate aaindex scores
    aaindex_df = metrics.calculate_aaindex_scores(residue_table, aaindex_data)

    # check that aaindex scores are added
    output_cols = []
    for acc in aaindex_data['accession']:
        output_cols.extend([f'AAIndex_{acc}_wt', f'AAIndex_{acc}_mut', f'AAIndex_{acc}_diff'])

    for col in output_cols:
        assert col in aaindex_df.columns

    # verify that values are correct for wildtype, mutant, and diff
    for acc in aaindex_data['accession']:
        feature_values = aaindex_data.set_index('accession').loc[acc].iloc[1:]
        for idx, row in aaindex_df.iterrows():
            print(row)
            expected_wt = feature_values.get(row['resn'], np.nan)
            expected_mut = feature_values.get(row['resm'], np.nan)
            expected_diff = expected_mut - expected_wt if not (np.isnan(expected_wt) or np.isnan(expected_mut)) else np.nan

            assert aaindex_df.at[idx, f'AAIndex_{acc}_wt'] == expected_wt
            assert aaindex_df.at[idx, f'AAIndex_{acc}_mut'] == expected_mut
            if np.isnan(expected_diff):
                assert np.isnan(aaindex_df.at[idx, f'AAIndex_{acc}_diff'])
            else:
                assert aaindex_df.at[idx, f'AAIndex_{acc}_diff'] == expected_diff


def test_calculate_blosum_score():
    # create test residue table
    residue_table = _make_residue_table(num_residues=5, num_chains=1, make_muts=True)

    # calculate blosum scores
    blosum_df = metrics.calculate_blosum_score(residue_table, blosum_threshold=90)

    # check that blosum score column is added
    assert 'blosum90' in blosum_df.columns

    # verify that values are correct
    b_matrix = metrics.bl.BLOSUM(90)
    for idx, row in blosum_df.iterrows():
        expected_score = b_matrix[convert_amino_acid(row['resn'])][convert_amino_acid(row['resm'])]
        assert blosum_df.at[idx, 'blosum90'] == expected_score