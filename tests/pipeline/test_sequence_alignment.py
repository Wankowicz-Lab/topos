import os
import warnings

import pandas as pd
import pytest
from Bio.Align import PairwiseAligner

from src.pipeline.sequence_alignment import (
    alignment_to_index_map,
    evaluate_sequence_alignment,
    load_mutation_scores,
    merge_mutation_scores,
    merge_sequence_dfs,
)


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
    
    # Verify that 'resn' is still 3-letter codes
    assert all(len(resn) == 3 for resn in loaded_df['resn'])
    
    # Verify that 1-letter mutant codes are converted to 3-letter codes
    # This tests lines 87-88 where convert_amino_acid is applied to mutation column
    assert loaded_df['resm'].tolist() == ['ALA', 'CYS', 'ASP']


def test_alignment_to_index_map():
    # seq1 has a W insertion at position 4 and a Q mutation at position 8, seq2 has ZZ at the end
    seq1 = 'ABCWDEQGH'
    seq2 = 'ABCDEFGHZZ'

    aligner = PairwiseAligner()
    alignment = aligner.align(seq1, seq2)[0]
    index_map = alignment_to_index_map(alignment)

    expected_map = [(0, 0, 0), (1, 1, 1), (2, 2, 2), (3, 3, None), (4, 4, 3), (5, 5, 4), (6, 6, 5), (7, 7, 6), (8, 8, 7), (9, None, 8), (10, None, 9)]
    assert index_map == expected_map


def test_merge_sequence_dfs():
    # Create two sequence dfs
    df1 = pd.DataFrame({
        'resi': range(1, 6),
        'resn': ['A', 'R', 'N', 'D', 'C'],
        'feature1': [0.1, 0.2, 0.3, 0.4, 0.5]
    })

    # Create second sequence df with 'R' deleted and 'E' added at the end
    df2 = pd.DataFrame({
        'resi': range(3, 8),
        'resn': ['A', 'N', 'D', 'C', 'E'],
        'feature2': [1.0, 1.1, 1.2, 1.3, 1.4]
    })

    # None for the deleted 'R' and for the added 'E'
    mapping = [(0, 0, 0), (1, 1, None), (2, 2, 1), (3, 3, 2), (4, 4, 3), (5, None, 4)]
    merged_df = merge_sequence_dfs(df1=df1, df2=df2, mapping=mapping)

    assert len(merged_df) == 6
    assert set(merged_df.columns) == {'align_pos', 'resi_df1', 'resn_df1', 'feature1', 'resi_df2', 'resn_df2', 'feature2'}

    # Check that resn values are correctly aligned
    assert merged_df['resn_df1'].equals(pd.Series(['A', 'R', 'N', 'D', 'C', None]))
    assert merged_df['resn_df2'].equals(pd.Series(['A', None, 'N', 'D', 'C', 'E']))

    # Check that feature values are correctly aligned
    assert merged_df['feature1'].equals(pd.Series([0.1, 0.2, 0.3, 0.4, 0.5, None]))
    assert merged_df['feature2'].equals(pd.Series([1.0, None, 1.1, 1.2, 1.3, 1.4]))
    
    # Check that align_pos is correctly set
    assert merged_df['align_pos'].equals(pd.Series([0, 1, 2, 3, 4, 5]))


def test_evaluate_sequence_alignment():
    # check that warning is raised for poor alignment
    threshold_merged_df = pd.DataFrame({
        'resn_df1': ['A', 'K', 'N', 'D', 'A', 'A', 'A', 'A', 'I', 'M'],
        'resi_df1': list(range(1, 11)),
        'resn_df2': ['A', 'K', 'N', 'D', 'C', 'E', 'G', 'H', 'I', 'M'],
        'resi_df2': list(range(1, 11))
    })
    pd.options.mode.chained_assignment = 'raise'

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        # Call the function that should issue the warning
        evaluate_sequence_alignment(merged=threshold_merged_df, alignment_cutoff=0.8)

        assert len(w) == 2 # poor alignment and mismatches
        assert issubclass(w[-2].category, UserWarning)
        assert "Alignment quality below cutoff of 0.80. Found 40.00%" in str(w[-2].message)

    # check that warning is raised for mismatched residues
    mismatch_merged_df = pd.DataFrame({
        'resn_df1': ['A', 'R', 'N', 'D', 'C', 'E'],
        'resi_df1': [1, 2, 3, 4, 5, 6],
        'resn_df2': ['A', 'K', 'N', 'D', 'C', 'E'],
        'resi_df2': [3, 4, 5, 6, 7, 8]
    })

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        # Call the function that should issue the warning
        evaluate_sequence_alignment(merged=mismatch_merged_df, alignment_cutoff=0.8)

        assert len(w) == 1
        assert issubclass(w[-1].category, UserWarning)
        assert "Found 1 mismatches out of 6 residues" in str(w[-1].message)

    # check that warning is raised for indels
    indel_merged_df = pd.DataFrame({
        'resn_df1': ['A', 'G', 'R', 'N', 'D', 'C', 'E'],
        'resi_df1': [1, 2, 3, 4, 5, 6, 7],
        'resn_df2': ['A', None, 'R', 'N', None, 'C', 'E'],
        'resi_df2': [3, 4, 5, 6, 7, 8, 9]
    })

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        # Call the function that should issue the warning
        evaluate_sequence_alignment(merged=indel_merged_df, alignment_cutoff=0.6)

        assert len(w) == 1
        assert issubclass(w[-1].category, UserWarning)
        assert "Found 2 residues with indels out of 7 residues" in str(w[-1].message)


    # check that warning is raised for terminal gaps, and that they don't count towards alignment quality
    termini_merged_df = pd.DataFrame({
        'resn_df1': ['A', 'R', 'N', 'D', 'C'],
        'resi_df1': [1, 2, 3, 4, 5],
        'resn_df2': [None, None, None, 'D', 'C'],
        'resi_df2': [2, 3, 4, 5, 6]
    })

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        # Call the function that should issue the warning
        evaluate_sequence_alignment(merged=termini_merged_df, alignment_cutoff=0.9)

        assert len(w) == 1
        assert issubclass(w[-1].category, UserWarning)
        assert "Found gaps at the termini of the sequence alignment" in str(w[-1].message)


def test_merge_mutation_scores(tmp_path):
    # Create a mutation_scores df
    mutation_scores_df = pd.DataFrame({
        'resn': ['ARG', 'ARG', 'THR', 'THR', 'GLU', 'GLU', 'ASP'],
        'resi': [12, 12, 13, 13, 14, 14, 15],
        'resm': ['ALA', 'GLY', 'VAL', 'GLU', 'CYS', 'ASP', 'MET'],
        'type': ['missense', 'missense', 'nonsense', 'missense', 'missense', 'nonsense', 'missense'],
        'effect': [0.5, -1.2, 0.3, -0.7, 1.0, -0.4, -0.1],
        'extra_col': [10, 20, 30, 40, 50, 60, 70]
    })

    # Create a residue_table
    residue_table = pd.DataFrame({'chain': ['A', 'A', 'A', 'A', 'B'], 'resi': [1, 2, 3, 4, 1],
                                  'resn': ['LEU', 'THR', 'GLU', 'ASP', 'TYR'],
                                  'ss_group': ['TM1', 'TM1', 'TM1', 'TM1', 'TM1'],
                                  'ss_domains': ['TM1_start', 'TM1_mid', 'TM1_mid', 'TM1_end', 'TM1_start']})

    # Merge using the function
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        # Call the function that should issue the warning
        merged_df = merge_mutation_scores(mutation_scores=mutation_scores_df, residue_table=residue_table, chain='A',
                                          alignment_cutoff=0.7)

        assert len(w) == 1
        assert issubclass(w[-1].category, UserWarning)
        assert "Found 1 mismatches out of 4 residues" in str(w[-1].message)

        assert len(merged_df) == 8 # 1 row for each mutant, plus 1 for residue in chain B
        assert set(merged_df.columns) == {'resn_struct', 'resi_struct', 'resn_mut', 'resi_mut', 'resm', 'type',
                                          'effect', 'chain', 'mut_info', 'struct_info', 'ss_group', 'ss_domains', 'align_pos'}
        assert merged_df['mut_info'].tolist() == [False, True, True, True, True, True, True, True]
        assert merged_df['struct_info'].tolist() == [True, True, True, True, True, True, True, True]
