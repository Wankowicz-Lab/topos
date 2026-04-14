"""Tests for sequence metrics module."""
import random

import numpy as np
import pandas as pd
import pytest

from src.metrics import sequence as metrics
from src.metrics.aaindex_schema import AAINDEX_AA_COLUMNS
from src.sequence.utils import convert_amino_acid_3to1
from tests.test_utils import AA_LIST, _make_aaindex_data, _make_residue_table

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


def test_calculate_effect_variance():
    # create test residue table
    residue_table = _make_residue_table(num_residues=100, num_chains=1, make_muts=True)
    
    class MockContext:
        def __init__(self, residue_table):
            self.residue_table = residue_table
    
    context = MockContext(residue_table)

    # calculate effect variance
    effect_variance_df = metrics.calculate_effect_variance(context)

    # check that effect variance column is added
    assert 'effect_variance' in effect_variance_df.columns

    # verify that values are correct
    assert effect_variance_df['effect_variance'].sum() > 0
    assert 'effect_variance_rank' in effect_variance_df.columns

    # verify that effect variance rank is between 0 and 1
    assert effect_variance_df['effect_variance_rank'].min() <= (1 / 100)
    assert effect_variance_df['effect_variance_rank'].max() == 1

    # verify that effect variance rank is correct
    min_rank_idx = effect_variance_df['effect_variance_rank'].idxmin()
    max_rank_idx = effect_variance_df['effect_variance_rank'].idxmax()
    min_effect_idx = effect_variance_df['effect_variance'].idxmin()
    max_effect_idx = effect_variance_df['effect_variance'].idxmax()
    assert min_rank_idx == min_effect_idx
    assert max_rank_idx == max_effect_idx
    
    # maximum ranking should correspond to a normalized rank of 1.0
    assert effect_variance_df.at[max_rank_idx, 'effect_variance_rank'] == 1


def test_calculate_effect_ranking():
    # create test residue table
    residue_table = _make_residue_table(num_residues=100, num_chains=1, make_muts=True)
    
    class MockContext:
        def __init__(self, residue_table):
            self.residue_table = residue_table
    
    context = MockContext(residue_table)

    # calculate effect ranking
    effect_ranking_df = metrics.calculate_effect_ranking(context)

    # check that effect ranking column is added
    assert 'effect_ranking' in effect_ranking_df.columns

    # verify that values are correct
    assert effect_ranking_df['effect_ranking'].sum() > 0

    # verify that effect ranking rank is between 0 and 1
    assert effect_ranking_df['effect_ranking'].min() <= (1 / 100)
    assert effect_ranking_df['effect_ranking'].max() == 1

    # verify that ranking reflects the ordering of effect values
    min_rank_idx = effect_ranking_df['effect_ranking'].idxmin()
    max_rank_idx = effect_ranking_df['effect_ranking'].idxmax()
    min_effect_idx = effect_ranking_df['effect'].idxmin()
    max_effect_idx = effect_ranking_df['effect'].idxmax()
    assert min_rank_idx == min_effect_idx
    assert max_rank_idx == max_effect_idx


def test_calculate_mutation_category_from_synonymous_reference():
    residue_table = pd.DataFrame({
        'chain': ['A'] * 11,
        'resi_mut': [1, 1, 1, 1, 1, 2, 2, 3, 3, 4, 4],
        'resn_mut': ['ALA'] * 5 + ['GLY', 'GLY', 'SER', 'SER', 'THR', 'THR'],
        'resi_struct': [1, 1, 1, 1, 1, 2, 2, 3, 3, 4, 4],
        'resn_struct': ['ALA'] * 5 + ['GLY', 'GLY', 'SER', 'SER', 'THR', 'THR'],
        'resm': ['ALA', 'GLY', 'SER', 'THR', 'VAL', 'LEU', 'ILE', 'PHE', 'TYR', 'ASN', 'GLN'],
        'effect': [-0.2, -0.1, 0.0, 0.1, 0.2, -2.0, 1.7, 0.05, 0.2, -1.5, 2.2],
        'type': ['synonymous'] * 5 + ['missense'] * 6,
        'struct_info': [True] * 11,
        'mut_info': [True] * 11,
    })

    class MockConfig:
        mutation_category_central_interval = 0.90
        mutation_category_logs_base = None
        output_dir = None
        output_prefix = ''
        name = None
        pdb_id = None

    class MockContext:
        def __init__(self, residue_table):
            self.residue_table = residue_table
            self.config = MockConfig()

    result = metrics.calculate_mutation_category(MockContext(residue_table))
    category_map = dict(zip(result['resm'], result['mutation_category']))

    assert category_map['LEU'] == 'LOF'
    assert category_map['ILE'] == 'GOF'
    assert category_map['PHE'] == 'neutral'
    assert category_map['ASN'] == 'LOF'
    assert category_map['GLN'] == 'GOF'

    pos2 = result.loc[result['resi_mut'] == 2, ['total_lof', 'total_gof']].drop_duplicates()
    pos3 = result.loc[result['resi_mut'] == 3, ['total_lof', 'total_gof']].drop_duplicates()
    pos4 = result.loc[result['resi_mut'] == 4, ['total_lof', 'total_gof']].drop_duplicates()
    assert pos2.iloc[0].to_dict() == {'total_lof': 1, 'total_gof': 1}
    assert pos3.iloc[0].to_dict() == {'total_lof': 0, 'total_gof': 0}
    assert pos4.iloc[0].to_dict() == {'total_lof': 1, 'total_gof': 1}


def test_calculate_mutation_category_falls_back_to_stop_reference():
    residue_table = pd.DataFrame({
        'chain': ['A'] * 10,
        'resi_mut': [1, 1, 1, 1, 1, 2, 2, 3, 3, 4],
        'resn_mut': ['ALA'] * 5 + ['GLY', 'GLY', 'SER', 'SER', 'THR'],
        'resi_struct': [1, 1, 1, 1, 1, 2, 2, 3, 3, 4],
        'resn_struct': ['ALA'] * 5 + ['GLY', 'GLY', 'SER', 'SER', 'THR'],
        'resm': ['*', '*', '*', '*', '*', 'LEU', 'ILE', 'PHE', 'TYR', 'ASN'],
        'effect': [-2.2, -2.0, -1.8, -1.6, -1.4, -1.9, 0.2, -1.5, 0.6, -0.1],
        'type': ['stop'] * 5 + ['missense'] * 5,
        'struct_info': [True] * 10,
        'mut_info': [True] * 10,
    })

    class MockConfig:
        mutation_category_central_interval = 0.90
        mutation_category_logs_base = None
        output_dir = None
        output_prefix = ''
        name = None
        pdb_id = None

    class MockContext:
        def __init__(self, residue_table):
            self.residue_table = residue_table
            self.config = MockConfig()

    with pytest.warns(UserWarning, match='no synonymous mutations'):
        result = metrics.calculate_mutation_category(MockContext(residue_table))

    category_map = dict(zip(result['resm'], result['mutation_category']))
    assert category_map['LEU'] == 'LOF'
    assert category_map['ILE'] == 'neutral'
    assert category_map['PHE'] == 'neutral'
    assert category_map['TYR'] == 'neutral'
    assert category_map['ASN'] == 'neutral'

    pos2 = result.loc[result['resi_mut'] == 2, ['total_lof', 'total_gof']].drop_duplicates()
    pos3 = result.loc[result['resi_mut'] == 3, ['total_lof', 'total_gof']].drop_duplicates()
    assert pos2.iloc[0].to_dict() == {'total_lof': 1, 'total_gof': 0}
    assert pos3.iloc[0].to_dict() == {'total_lof': 0, 'total_gof': 0}


def test_calculate_mutation_category_rejects_implausibly_narrow_synonymous_fit():
    residue_table = pd.DataFrame({
        'chain': ['A'] * 10,
        'resi_mut': [1, 1, 1, 1, 1, 2, 2, 3, 3, 4],
        'resn_mut': ['ALA'] * 5 + ['GLY', 'GLY', 'SER', 'SER', 'THR'],
        'resi_struct': [1, 1, 1, 1, 1, 2, 2, 3, 3, 4],
        'resn_struct': ['ALA'] * 5 + ['GLY', 'GLY', 'SER', 'SER', 'THR'],
        'resm': ['ALA', 'GLY', 'SER', 'THR', 'VAL', 'LEU', 'ILE', 'PHE', 'TYR', 'ASN'],
        'effect': [0.00000, 0.00001, -0.00001, 0.00002, -0.00002, -2.0, 1.8, -1.4, 1.1, 0.3],
        'type': ['synonymous'] * 5 + ['missense'] * 5,
        'struct_info': [True] * 10,
        'mut_info': [True] * 10,
    })

    class MockConfig:
        mutation_category_central_interval = 0.90
        mutation_category_logs_base = None
        output_dir = None
        output_prefix = ''
        name = None
        pdb_id = None

    class MockContext:
        def __init__(self, residue_table):
            self.residue_table = residue_table
            self.config = MockConfig()

    with pytest.warns(UserWarning, match='normalized to synonymous'):
        result = metrics.calculate_mutation_category(MockContext(residue_table))

    assert result['mutation_category'].isna().all()
    assert result['total_lof'].isna().all()
    assert result['total_gof'].isna().all()


def test_mutation_category_diagnostic_png(tmp_path):
    residue_table = pd.DataFrame({
        'chain': ['A'] * 11,
        'resi_mut': [1, 1, 1, 1, 1, 2, 2, 3, 3, 4, 4],
        'resn_mut': ['ALA'] * 5 + ['GLY', 'GLY', 'SER', 'SER', 'THR', 'THR'],
        'resi_struct': [1, 1, 1, 1, 1, 2, 2, 3, 3, 4, 4],
        'resn_struct': ['ALA'] * 5 + ['GLY', 'GLY', 'SER', 'SER', 'THR', 'THR'],
        'resm': ['ALA', 'GLY', 'SER', 'THR', 'VAL', 'LEU', 'ILE', 'PHE', 'TYR', 'ASN', 'GLN'],
        'effect': [-0.2, -0.1, 0.0, 0.1, 0.2, -2.0, 1.7, 0.05, 0.2, -1.5, 2.2],
        'type': ['synonymous'] * 5 + ['missense'] * 6,
        'struct_info': [True] * 11,
        'mut_info': [True] * 11,
    })

    class MockConfig:
        mutation_category_central_interval = 0.90
        mutation_category_logs_base = None
        output_dir = tmp_path
        output_prefix = 'pfx'
        name = 'MyProt'
        pdb_id = None

    class MockContext:
        def __init__(self, residue_table):
            self.residue_table = residue_table
            self.config = MockConfig()

    metrics.calculate_mutation_category(MockContext(residue_table))
    out = tmp_path / 'logs' / 'pfx_MyProt_mutation_category_gmm_fit.png'
    assert out.is_file()


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
    output_cols = [f"{r['accession']}_{r['category']}_wt" for _, r in aaindex_data.iterrows()]

    for col in output_cols:
        assert col in aaindex_df.columns

    # verify that values are correct for wildtype
    for _, r in aaindex_data.iterrows():
        base = f"{r['accession']}_{r['category']}"
        feature_values = r.loc[list(AAINDEX_AA_COLUMNS)]
        for idx, row in aaindex_df.iterrows():
            expected_value = feature_values.get(row['resn_mut'], np.nan)
            assert aaindex_df.at[idx, f'{base}_wt'] == expected_value


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
    for _, r in aaindex_data.iterrows():
        base = f"{r['accession']}_{r['category']}"
        output_cols.extend([f'{base}_wt', f'{base}_mut', f'{base}_diff'])

    for col in output_cols:
        assert col in aaindex_df.columns

    # verify that values are correct for wildtype, mutant, and diff
    for _, r in aaindex_data.iterrows():
        base = f"{r['accession']}_{r['category']}"
        feature_values = r.loc[list(AAINDEX_AA_COLUMNS)]
        for idx, row in aaindex_df.iterrows():
            expected_wt = feature_values.get(row['resn_mut'], np.nan)
            expected_mut = feature_values.get(row['resm'], np.nan)
            expected_diff = expected_mut - expected_wt if not (np.isnan(expected_wt) or np.isnan(expected_mut)) else np.nan

            assert aaindex_df.at[idx, f'{base}_wt'] == expected_wt
            assert aaindex_df.at[idx, f'{base}_mut'] == expected_mut
            if np.isnan(expected_diff):
                assert np.isnan(aaindex_df.at[idx, f'{base}_diff'])
            else:
                assert aaindex_df.at[idx, f'{base}_diff'] == expected_diff


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
        expected_score = b_matrix[convert_amino_acid_3to1(row['resn_mut'])][convert_amino_acid_3to1(row['resm'])]
        assert blosum_df.at[idx, 'blosum90'] == expected_score


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


def test_calculate_aa_groupings_with_muts():
    """Test amino acid groupings with mutations."""
    # Define expected groups for verification
    aa_groups = {
        "Nonpolar_Aliphatic": ["ALA", "VAL", "LEU", "ILE", "MET"],
        "Aromatic": ["PHE", "TRP", "TYR"],
        "Polar_Uncharged": ["SER", "THR", "ASN", "GLN", "CYS"],
        "Positively_Charged": ["LYS", "ARG", "HIS"],
        "Negatively_Charged": ["ASP", "GLU"],
        "Special": ["PRO", "GLY"]
    }

    # Create reverse mapping for verification
    aa_to_group = {}
    for group_name, aa_list in aa_groups.items():
        for aa in aa_list:
            aa_to_group[aa] = group_name

    # create test residue table with mutations
    residue_table = _make_residue_table(num_residues=5, num_chains=1, make_muts=True)

    class MockContext:
        def __init__(self, residue_table):
            self.residue_table = residue_table

    context = MockContext(residue_table)

    # calculate aa groupings
    aa_groupings_df = metrics.calculate_aa_groupings(context)

    # check that all three output columns exist
    assert 'wildtype_aa_group' in aa_groupings_df.columns
    assert 'mut_aa_group' in aa_groupings_df.columns
    assert 'wildtype_mut_aa_group' in aa_groupings_df.columns

    # verify that values are correctly assigned
    for idx, row in aa_groupings_df.iterrows():
        expected_wt_group = aa_to_group.get(row['resn_mut'], np.nan)
        expected_mut_group = aa_to_group.get(row['resm'], np.nan)

        assert aa_groupings_df.at[idx, 'wildtype_aa_group'] == expected_wt_group
        assert aa_groupings_df.at[idx, 'mut_aa_group'] == expected_mut_group

        expected_concatenated = f"{expected_wt_group}_{expected_mut_group}"
        assert aa_groupings_df.at[idx, 'wildtype_mut_aa_group'] == expected_concatenated


def test_calculate_aa_groupings_no_muts():
    """Test amino acid groupings without mutations."""
    # create test residue table without mutations
    residue_table = _make_residue_table(num_residues=5, num_chains=1, make_muts=False)

    class MockContext:
        def __init__(self, residue_table):
            self.residue_table = residue_table

    context = MockContext(residue_table)

    # calculate aa groupings
    aa_groupings_df = metrics.calculate_aa_groupings(context)

    # check that wildtype_aa_group column exists
    assert 'wildtype_aa_group' in aa_groupings_df.columns

    # check that mut columns do not exist when resm is missing
    # Note: The function should still create mut columns if resm is in KEEP_COLS but empty
    # But if resm column doesn't exist at all, the columns shouldn't be created
    if 'resm' not in residue_table.columns:
        assert 'mut_aa_group' not in aa_groupings_df.columns
        assert 'wildtype_mut_aa_group' not in aa_groupings_df.columns