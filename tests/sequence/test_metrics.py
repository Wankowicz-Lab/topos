"""Tests for sequence metrics module."""
import numpy as np
import pandas as pd
import pytest
import random

from src.sequence import metrics
from src.sequence.utils import convert_amino_acid

from tests.test_utils import _make_residue_table, _make_aaindex_data, AA_LIST

# Seed RNGs for deterministic tests
np.random.seed(42)
random.seed(42)


def test_calculate_position_effect_quartiles_with_pos_effect():

    # create test residue table with pos_effect column
    residue_table = _make_residue_table(num_residues=10, num_chains=1, make_muts=True)

    # remove mutation data for last residue to test handling of missing data
    residue_table = residue_table[residue_table['resi_mut'] != 10]
    new_row = pd.DataFrame({'chain': ['A'], 'resi_mut': [10], 'resn_mut': ['ALA'], 'resi_struct': [10], 'resn_struct': ['ALA'], 'resm': [np.nan],
                            'effect': [np.nan], 'type': [np.nan], 'struct_info': [True], 'mut_info': [False]})
    residue_table = pd.concat([residue_table, new_row], ignore_index=True)

    # compute position effects
    pos_effects = residue_table.groupby('resi_mut')['effect'].mean().reset_index()
    pos_effects.rename(columns={'effect': 'pos_effect'}, inplace=True)
    residue_table = pd.merge(residue_table, pos_effects, on='resi_mut', how='left')

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
    assert 10 not in quartile_df.resi_mut.values


def test_calculate_position_effect_quartiles_without_pos_effect():
    # create test residue table without pos_effect column
    residue_table = _make_residue_table(num_residues=10, num_chains=1, make_muts=True)

    # remove mutation data for last residue to test handling of missing data
    residue_table = residue_table[residue_table['resi_mut'] != 10]
    new_row = pd.DataFrame({'chain': ['A'], 'resi_mut': [10], 'resn_mut': ['ALA'], 'resi_struct': [10], 'resn_struct': ['ALA'], 'resm': [np.nan],
                            'effect': [np.nan], 'type': [np.nan], 'struct_info': [True], 'mut_info': [False]})
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
    assert 10 not in quartile_df.resi_mut.values


def test_calculate_position_effect_quartiles_custom_percentiles():
    """Test that custom percentiles produce different quartile assignments than defaults."""
    # create test residue table
    residue_table = _make_residue_table(num_residues=10, num_chains=1, make_muts=True)

    class MockContext:
        def __init__(self, residue_table):
            self.residue_table = residue_table

    context = MockContext(residue_table)

    # Get results with default percentiles
    default_df = metrics.calculate_position_effect_quartiles(context)

    # Use extreme custom percentiles - with [90, 95, 99], most values fall below the 90th percentile, ending up in Q1
    custom_percentiles = [90, 95, 99]
    custom_df = metrics.calculate_position_effect_quartiles(context, percentiles=custom_percentiles)

    assert 'effect_quartile' in custom_df.columns
    assert set(custom_df['effect_quartile'].dropna().unique()).issubset({'Q1', 'Q2', 'Q3', 'Q4'})

    # With extreme percentiles, the distribution should be different from default
    default_q1_count = (default_df['effect_quartile'] == 'Q1').sum()
    custom_q1_count = (custom_df['effect_quartile'] == 'Q1').sum()
    # With [90, 95, 99] cutoffs, most values should fall into Q1
    assert custom_q1_count > default_q1_count


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
            expected_value = feature_values.get(row['resn_mut'], np.nan)
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
            expected_wt = feature_values.get(row['resn_mut'], np.nan)
            expected_mut = feature_values.get(row['resm'], np.nan)
            expected_diff = expected_mut - expected_wt if not (np.isnan(expected_wt) or np.isnan(expected_mut)) else np.nan

            assert aaindex_df.at[idx, f'AAIndex_{acc}_wt'] == expected_wt
            assert aaindex_df.at[idx, f'AAIndex_{acc}_mut'] == expected_mut
            if np.isnan(expected_diff):
                assert np.isnan(aaindex_df.at[idx, f'AAIndex_{acc}_diff'])
            else:
                assert aaindex_df.at[idx, f'AAIndex_{acc}_diff'] == expected_diff


def test_calculate_kidera_factor_scores_no_muts():
    # create test residue table
    residue_table = _make_residue_table(num_residues=5, num_chains=1, make_muts=False)

    # create mock kidera data
    kidera_data = pd.DataFrame({
        'factor': ['f' + str(i) for i in range(1, 11)],
        'description': ['desc' + str(i) for i in range(1, 11)],
        **{f'{aa}': np.random.rand(10) for aa in AA_LIST}
    })

    class MockContext:
        def __init__(self, residue_table, kidera_data):
            self.residue_table = residue_table
            self.extras = {'kidera': kidera_data}

    context = MockContext(residue_table, kidera_data)

    # calculate kidera factor scores
    kidera_df = metrics.calculate_kidera_factor_scores(context)

    # check that kidera scores are added
    output_cols = [f'kidera_f{i}_wt' for i in range(1, 11)]
    for col in output_cols:
        assert col in kidera_df.columns

    # verify that values are correct for wildtype
    for i in range(1, 11):
        factor_values = kidera_data.set_index('factor').loc[f'f{i}']
        for idx, row in kidera_df.iterrows():
            expected_value = factor_values.get(row['resn_mut'], np.nan)
            assert kidera_df.at[idx, f'kidera_f{i}_wt'] == expected_value


def test_calculate_kidera_factor_scores_with_muts():
    # create test residue table
    residue_table = _make_residue_table(num_residues=5, num_chains=1, make_muts=True)

    # create mock kidera data
    kidera_data = pd.DataFrame({
        'factor': ['f' + str(i) for i in range(1, 11)],
        'description': ['desc' + str(i) for i in range(1, 11)],
        **{f'{aa}': np.random.rand(10) for aa in AA_LIST}
    })

    class MockContext:
        def __init__(self, residue_table, kidera_data):
            self.residue_table = residue_table
            self.extras = {'kidera': kidera_data}
    context = MockContext(residue_table, kidera_data)

    # calculate kidera factor scores
    kidera_df = metrics.calculate_kidera_factor_scores(context)

    # check that kidera scores are added
    output_cols = []
    for i in range(1, 11):
        output_cols.extend([f'kidera_f{i}_wt', f'kidera_f{i}_mut', f'kidera_f{i}_diff'])

    for col in output_cols:
        assert col in kidera_df.columns

    # verify that values are correct for wildtype, mutant, and diff
    for i in range(1, 11):
        factor_values = kidera_data.set_index('factor').loc[f'f{i}']
        for idx, row in kidera_df.iterrows():
            expected_wt = factor_values.get(row['resn_mut'], np.nan)
            expected_mut = factor_values.get(row['resm'], np.nan)
            expected_diff = expected_mut - expected_wt if not (np.isnan(expected_wt) or np.isnan(expected_mut)) else np.nan

            assert kidera_df.at[idx, f'kidera_f{i}_wt'] == expected_wt
            assert kidera_df.at[idx, f'kidera_f{i}_mut'] == expected_mut
            if np.isnan(expected_diff):
                assert np.isnan(kidera_df.at[idx, f'kidera_f{i}_diff'])
            else:
                assert kidera_df.at[idx, f'kidera_f{i}_diff'] == expected_diff


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
        expected_score = b_matrix[convert_amino_acid(row['resn_mut'])][convert_amino_acid(row['resm'])]
        assert blosum_df.at[idx, 'blosum90'] == expected_score


def test_calculate_position_effect_quartiles_multichain_error():
    """Test that ValueError is raised when residue table contains data from more than one chain."""
    # Create test residue table with multiple chains
    residue_table = _make_residue_table(num_residues=5, num_chains=2, make_muts=True)
    
    class MockContext:
        def __init__(self, residue_table):
            self.residue_table = residue_table
    
    context = MockContext(residue_table)
    
    # Verify that ValueError is raised with appropriate message
    with pytest.raises(ValueError, match="calculate_position_effect_quartiles only supports single chain mutation data"):
        metrics.calculate_position_effect_quartiles(context)


def test_make_phat75_73():
    """Test that make_phat75_73 returns Phat 75/73 substitution matrix."""
    phat75_73 = metrics.make_phat75_73()
    assert phat75_73['A']['A'] == 5
    assert phat75_73['V']['V'] == 4

    assert phat75_73.alphabet == "ARNDCQEGHILKMFPSTWYV"
    assert phat75_73.shape == (20, 20)
    

def test_calculate_phat_score():
    """Test that calculate_phat_score returns a DataFrame with 'phat_score' column."""
    residue_table = _make_residue_table(num_residues=5, num_chains=1, make_muts=True)

    class MockContext:
        def __init__(self, residue_table):
            self.residue_table = residue_table

    context = MockContext(residue_table)
    phat_df = metrics.calculate_phat_score(context)
    assert 'phat_score' in phat_df.columns
    assert phat_df['phat_score'].notna().all()

    # Test for amino acids not in alphabet
    residue_table.at[0, 'resn_mut'] = 'XXX' 
    context = MockContext(residue_table)
    phat_df = metrics.calculate_phat_score(context)
    assert phat_df['phat_score'].iloc[0] == np.inf