import pandas as pd
import os
import pytest

from sequence.sequence_context import load_dms_scores, merge_dms_scores


def test_load_dms_scores(tmp_path):
    # Create a dms_scores df
    test_df = pd.DataFrame({
        'resn_rename': ['ARG', 'THR', 'GLU'],
        'resi_rename': [1, 2, 3],
        'resm_rename': ['ALA', 'CYS', 'ASP'],
        'type_rename': ['missense', 'missense', 'nonsense'],
        'effect_rename': [0.5, -1.2, 0.3]
    })

    test_file_valid_path = os.path.join(tmp_path, 'test_dms_scores.csv')
    test_df.to_csv(test_file_valid_path, index=False)

    # Load using the function
    loaded_df = load_dms_scores(
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
    test_file_invalid_path_res = os.path.join(tmp_path, 'test_dms_scores_invalid_res.csv')
    test_df.to_csv(test_file_invalid_path_res, index=False)

    with pytest.raises(ValueError, match="Residue column must contain either 1-letter or 3-letter amino acid codes."):
        load_dms_scores(
            path=test_file_invalid_path_res,
            residue_col_name='resn_rename',
            residue_idx_name='resi_rename',
            mutation_col_name='resm_rename',
            mutation_type_col_name='type_rename',
            score_col_name='effect_rename'
        )


def test_merge_dms_scores(tmp_path):
    # Create a dms_scores df
    dms_scores_df = pd.DataFrame({
        'resn': ['ARG', 'ARG', 'THR', 'THR', 'GLU', 'GLU'],
        'resi': [1, 1, 2, 2, 3, 3],
        'resm': ['ALA', 'GLY', 'VAL', 'GLU', 'CYS', 'ASP'],
        'type': ['missense', 'missense', 'nonsense', 'missense', 'missense', 'nonsense'],
        'effect': [0.5, -1.2, 0.3, -0.7, 1.0, -0.4],
        'extra_col': [10, 20, 30, 40, 50, 60]
    })

    # Create a mock sequence context with res_keys
    class MockSequenceContext:
        def __init__(self):
            self.res_keys = pd.DataFrame({
                'chain': ['A', 'A', 'A'],
                'resi': [1, 2, 3],
                'resn': ['ARG', 'THR', 'GLU']
            })

    ctx = MockSequenceContext()

    # Merge using the function
    merged_df = merge_dms_scores(dms_scores_df, ctx, chain='A').res_keys

    assert len(merged_df) == 6
    assert set(merged_df.columns) == {'resn', 'resi', 'resm', 'type', 'effect', 'chain', 'seq_info', 'struct_info'}

    # Check that incorrect indices raise an error
    dms_scores_invalid_df = dms_scores_df.copy()
    dms_scores_invalid_df.at[0, 'resi'] = 99  # invalid index

    with pytest.raises(ValueError, match="Mismatch between DMS scores and structure residues"):
        merge_dms_scores(dms_scores_invalid_df, ctx, chain='A')

    # TODO: Check that function works with multiple chains, that seq_info and struct_info are set correctly, that non-redundant seq and structure information are handled appropriately