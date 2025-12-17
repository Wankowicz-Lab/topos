import pandas as pd
import os
import pytest
import warnings

from src.sequence.sequence_context import load_mutation_scores, merge_mutation_scores


def test_load_mutation_scores(tmp_path):
    # Create a mutation_scores df
    test_df = pd.DataFrame({
        'resn_rename': ['ARG', 'THR', 'GLU'],
        'resi_rename': [1, 2, 3],
        'resm_rename': ['ALA', 'CYS', 'ASP'],
        'type_rename': ['missense', 'missense', 'nonsense'],
        'effect_rename': [0.5, -1.2, 0.3]
    })

    test_file_valid_path = os.path.join(tmp_path, 'test_mutation_scores.csv')
    test_df.to_csv(test_file_valid_path, index=False)

    # Load using the function (note: defaults are now in Config, so we must pass all parameters)
    loaded_df = load_mutation_scores(
        path=test_file_valid_path,
        residue_col_name='resn_rename',
        residue_idx_name='resi_rename',
        mutation_col_name='resm_rename',
        mutation_type_col_name='type_rename',
        score_col_name='effect_rename'
    )

    assert list(loaded_df.columns) == ['resn', 'resi', 'resm', 'type', 'effect']

    # check that invalid residue column raises error
    test_df['resn_rename'] = ['AR', 'THR', 'GLU']  # invalid length
    test_file_invalid_path_res = os.path.join(tmp_path, 'test_mutation_scores_invalid_res.csv')
    test_df.to_csv(test_file_invalid_path_res, index=False)

    with pytest.raises(ValueError, match="Residue column must contain either 1-letter or 3-letter amino acid codes."):
        load_mutation_scores(
            path=test_file_invalid_path_res,
            residue_col_name='resn_rename',
            residue_idx_name='resi_rename',
            mutation_col_name='resm_rename',
            mutation_type_col_name='type_rename',
            score_col_name='effect_rename'
        )


def test_merge_mutation_scores(tmp_path):
    # Create a mutation_scores df
    mutation_scores_df = pd.DataFrame({
        'resn': ['ARG', 'ARG', 'THR', 'THR', 'GLU', 'GLU', 'ASP'],
        'resi': [2, 2, 3, 3, 4, 4, 5],
        'resm': ['ALA', 'GLY', 'VAL', 'GLU', 'CYS', 'ASP', 'MET'],
        'type': ['missense', 'missense', 'nonsense', 'missense', 'missense', 'nonsense', 'missense'],
        'effect': [0.5, -1.2, 0.3, -0.7, 1.0, -0.4, -0.1],
        'extra_col': [10, 20, 30, 40, 50, 60, 70]
    })

    # Create a residue_table
    residue_table = pd.DataFrame({'chain': ['A', 'A', 'A', 'A', 'B'], 'resi': [1, 2, 3, 4, 1],
                                  'resn': ['LEU', 'ARG', 'THR', 'GLU', 'TYR']})

    # Merge using the function
    merged_df = merge_mutation_scores(mutation_scores=mutation_scores_df, residue_table=residue_table, chain='A')

    assert len(merged_df) == 9 # 1 row for residues 1 and 5, two rows for residues 2, 3, and 4 each, 1 row for chain B
    assert set(merged_df.columns) == {'resn', 'resi', 'resm', 'type', 'effect', 'chain', 'seq_info', 'struct_info'}
    assert merged_df['seq_info'].tolist() == [False, False, True, True, True, True, True, True, True]
    assert merged_df['struct_info'].tolist() == [True, True, True, True, True, True, True, True, False]

    # Check that incorrect indices raise an error
    mutation_scores_invalid_df = mutation_scores_df.copy()
    mutation_scores_invalid_df.loc[[4,5], 'resn'] = 'LYS'  # change GLU to cause mismatch

    with pytest.raises(ValueError, match="Mismatch between mutation scores and structure residues"):
        merge_mutation_scores(mutation_scores_invalid_df, residue_table, chain='A')

    residue_table_invalid = residue_table.copy()
    residue_table_invalid.at[1, 'resn'] = 'LYS'  # change second residue to cause mismatch

    with pytest.raises(ValueError, match="Mismatch between mutation scores and structure residues"):
        merge_mutation_scores(mutation_scores_df, residue_table_invalid, chain='A')


def test_load_mutation_scores_1letter_conversion(tmp_path):
    """Test conversion of 1-letter mutation codes to 3-letter codes."""
    # Create mutation scores with 1-letter amino acid codes
    test_df = pd.DataFrame({
        'resn_col': ['R', 'T', 'E'],  # 1-letter wildtype codes
        'resi_col': [1, 2, 3],
        'resm_col': ['A', 'C', 'D'],  # 1-letter mutant codes
        'type_col': ['missense', 'missense', 'nonsense'],
        'effect_col': [0.5, -1.2, 0.3]
    })
    
    test_file_path = os.path.join(tmp_path, 'test_mutation_1letter.csv')
    test_df.to_csv(test_file_path, index=False)
    
    # Load using the function
    loaded_df = load_mutation_scores(
        path=test_file_path,
        residue_col_name='resn_col',
        residue_idx_name='resi_col',
        mutation_col_name='resm_col',
        mutation_type_col_name='type_col',
        score_col_name='effect_col'
    )
    
    # Verify that 1-letter codes are converted to 3-letter codes
    assert loaded_df['resn'].tolist() == ['ARG', 'THR', 'GLU']
    assert loaded_df['resm'].tolist() == ['ALA', 'CYS', 'ASP']


def test_load_mutation_scores_1letter_mutation_conversion(tmp_path):
    """Test conversion of 1-letter mutation codes to 3-letter codes (lines 87-88)."""
    # Create mutation scores with 3-letter wildtype and 1-letter mutant codes
    test_df = pd.DataFrame({
        'resn_col': ['ARG', 'THR', 'GLU'],  # 3-letter wildtype codes
        'resi_col': [1, 2, 3],
        'resm_col': ['A', 'C', 'D'],  # All 1-letter mutant codes
        'type_col': ['missense', 'missense', 'nonsense'],
        'effect_col': [0.5, -1.2, 0.3]
    })
    
    test_file_path = os.path.join(tmp_path, 'test_mutation_1letter_mut.csv')
    test_df.to_csv(test_file_path, index=False)
    
    # Load using the function
    loaded_df = load_mutation_scores(
        path=test_file_path,
        residue_col_name='resn_col',
        residue_idx_name='resi_col',
        mutation_col_name='resm_col',
        mutation_type_col_name='type_col',
        score_col_name='effect_col'
    )
    
    # Verify that 1-letter mutant codes are converted to 3-letter codes
    # This tests lines 87-88 where convert_amino_acid is applied to mutation column
    assert loaded_df['resm'].tolist() == ['ALA', 'CYS', 'ASP']


def test_load_mutation_scores_invalid_mutation_type_warning(tmp_path):
    """Test that UserWarning is issued when mutation types contain invalid values."""
    # Create mutation scores with invalid mutation types
    test_df = pd.DataFrame({
        'resn_col': ['ARG', 'THR', 'GLU'],
        'resi_col': [1, 2, 3],
        'resm_col': ['ALA', 'CYS', 'ASP'],
        'type_col': ['missense', 'invalid_type', 'another_invalid'],  # Invalid types
        'effect_col': [0.5, -1.2, 0.3]
    })
    
    test_file_path = os.path.join(tmp_path, 'test_mutation_invalid_types.csv')
    test_df.to_csv(test_file_path, index=False)
    
    # Verify that warning is issued
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        loaded_df = load_mutation_scores(
            path=test_file_path,
            residue_col_name='resn_col',
            residue_idx_name='resi_col',
            mutation_col_name='resm_col',
            mutation_type_col_name='type_col',
            score_col_name='effect_col'
        )
        
        # Verify that a warning was issued
        assert len(w) == 1
        assert issubclass(w[0].category, UserWarning)
        
        # Verify that the warning message contains the invalid types
        warning_message = str(w[0].message)
        assert 'invalid_type' in warning_message
        assert 'another_invalid' in warning_message
        assert 'Mutation types contain unexpected values' in warning_message
