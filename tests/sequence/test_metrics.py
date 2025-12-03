"""Tests for sequence metrics module."""
import numpy as np
import pandas as pd
import random

from src.sequence import metrics
from src.sequence.utils import convert_amino_acid

from tests.test_utils import _random_AA_seq, _make_residue_table, _make_aaindex_data

# Seed RNGs for deterministic tests
np.random.seed(42)
random.seed(42)


def test_calculate_position_effect_quartiles_with_pos_effect():

    # create test residue table with pos_effect column
    residue_table = _make_residue_table(num_residues=10, num_chains=1, make_muts=True)

    # remove mutation data for last residue to test handling of missing data
    residue_table = residue_table[residue_table['resi'] != 10]
    new_row = pd.DataFrame({'chain': ['A'], 'resi': [10], 'resn': ['ALA'], 'resm': [np.nan],
                            'effect': [np.nan], 'type': [np.nan], 'struct_info': [True], 'seq_info': [False]})
    residue_table = pd.concat([residue_table, new_row], ignore_index=True)

    # compute position effects
    pos_effects = residue_table.groupby('resi')['effect'].mean().reset_index()
    pos_effects.rename(columns={'effect': 'pos_effect'}, inplace=True)
    residue_table = pd.merge(residue_table, pos_effects, on='resi', how='left')

    # create mock context
    class MockContext:
        def __init__(self, residue_table):
            self.residue_table = residue_table
    context = MockContext(residue_table)

    # calculate quartiles
    quartile_df = metrics.calculate_position_effect_quartiles(context)

    # check that quartile labels are correct
    assert 'effect_quartile' in quartile_df.columns
    assert set(quartile_df['effect_quartile'].dropna().unique()).issubset({'Q1', 'Q2', 'Q3', 'Q4'})
    assert 10 not in quartile_df.resi.values


def test_calculate_position_effect_quartiles_without_pos_effect():
    # create test residue table without pos_effect column
    residue_table = _make_residue_table(num_residues=10, num_chains=1, make_muts=True)

    # remove mutation data for last residue to test handling of missing data
    residue_table = residue_table[residue_table['resi'] != 10]
    new_row = pd.DataFrame({'chain': ['A'], 'resi': [10], 'resn': ['ALA'], 'resm': [np.nan],
                            'effect': [np.nan], 'type': [np.nan], 'struct_info': [True], 'seq_info': [False]})
    residue_table = pd.concat([residue_table, new_row], ignore_index=True)

    # create mock context
    class MockContext:
        def __init__(self, residue_table):
            self.residue_table = residue_table

    context = MockContext(residue_table)

    # calculate quartiles
    quartile_df = metrics.calculate_position_effect_quartiles(context)

    # check that quartile labels are correct
    assert 'effect_quartile' in quartile_df.columns
    assert set(quartile_df['effect_quartile'].dropna().unique()).issubset({'Q1', 'Q2', 'Q3', 'Q4'})
    assert 10 not in quartile_df.resi.values


def test_calculate_position_effect_quartiles_custom_percentiles():
    """Test that custom percentiles can be passed without affecting default behavior."""
    # create test residue table
    residue_table = _make_residue_table(num_residues=10, num_chains=1, make_muts=True)

    class MockContext:
        def __init__(self, residue_table):
            self.residue_table = residue_table

    context = MockContext(residue_table)

    # Test with custom percentiles (must be 3 values for quartile bins)
    custom_percentiles = [20, 50, 80]
    quartile_df = metrics.calculate_position_effect_quartiles(context, percentiles=custom_percentiles)

    assert 'effect_quartile' in quartile_df.columns
    assert set(quartile_df['effect_quartile'].dropna().unique()).issubset({'Q1', 'Q2', 'Q3', 'Q4'})


def test_calculate_position_effect_quartiles_default_not_mutated():
    """Test that mutable default argument is not mutated across calls."""
    # This test ensures the fix for mutable default argument is working
    residue_table = _make_residue_table(num_residues=10, num_chains=1, make_muts=True)

    class MockContext:
        def __init__(self, residue_table):
            self.residue_table = residue_table

    context = MockContext(residue_table)

    # Call the function twice with default percentiles
    # If the mutable default was not fixed, subsequent calls could be affected
    result1 = metrics.calculate_position_effect_quartiles(context)
    result2 = metrics.calculate_position_effect_quartiles(context)

    # Results should be identical (same default percentiles used)
    assert len(result1) == len(result2)
    assert result1['effect_quartile'].equals(result2['effect_quartile'])


def test_calculate_aaindex_scores_no_muts():
    # create test residue table
    residue_table = _make_residue_table(num_residues=5, num_chains=1, make_muts=False)
    accessions = ['AA1', 'AA2']
    aaindex_data = _make_aaindex_data(accessions=accessions)

    class MockContext:
        def __init__(self, residue_table, aaindex_data):
            self.residue_table = residue_table
            self.extras = {'aaindex': aaindex_data}

    context = MockContext(residue_table, aaindex_data)

    # calculate aaindex scores
    aaindex_df = metrics.calculate_aaindex_scores(context)

    # check that aaindex scores are added
    output_cols = [f'AAIndex_{acc}_wt' for acc in accessions]

    for col in output_cols:
        assert col in aaindex_df.columns

    # verify that values are correct for wildtype
    for acc in accessions:
        feature_values = aaindex_data.set_index('accession').loc[acc].iloc[1:]
        for idx, row in aaindex_df.iterrows():
            expected_value = feature_values.get(row['resn'], np.nan)
            assert aaindex_df.at[idx, f'AAIndex_{acc}_wt'] == expected_value


def test_calculate_aaindex_scores_with_muts():
    # create test residue table
    residue_table = _make_residue_table(num_residues=5, num_chains=1, make_muts=True)
    accessions = ['AA1', 'AA2']
    aaindex_data = _make_aaindex_data(accessions=accessions)

    class MockContext:
        def __init__(self, residue_table, aaindex_data):
            self.residue_table = residue_table
            self.extras = {'aaindex': aaindex_data}

    context = MockContext(residue_table, aaindex_data)

    # calculate aaindex scores
    aaindex_df = metrics.calculate_aaindex_scores(context)

    # check that aaindex scores are added
    output_cols = []
    for acc in accessions:
        output_cols.extend([f'AAIndex_{acc}_wt', f'AAIndex_{acc}_mut', f'AAIndex_{acc}_diff'])

    for col in output_cols:
        assert col in aaindex_df.columns

    # verify that values are correct for wildtype, mutant, and diff
    for acc in accessions:
        feature_values = aaindex_data.set_index('accession').loc[acc].iloc[1:]
        for idx, row in aaindex_df.iterrows():
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

    class MockContext:
        def __init__(self, residue_table):
            self.residue_table = residue_table

    context = MockContext(residue_table)

    # calculate blosum scores
    blosum_df = metrics.calculate_blosum_score(context)

    # check that blosum score column is added
    assert 'blosum90' in blosum_df.columns

    # verify that values are correct
    b_matrix = metrics.bl.BLOSUM(90)
    for idx, row in blosum_df.iterrows():
        expected_score = b_matrix[convert_amino_acid(row['resn'])][convert_amino_acid(row['resm'])]
        assert blosum_df.at[idx, 'blosum90'] == expected_score