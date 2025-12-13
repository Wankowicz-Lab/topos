import pandas as pd
import pytest

from src.pipeline import runner
from src.pipeline.batch_processing import batch_process, expand_batch_arguments


def test_batch_process(monkeypatch, tmp_path):
    # Create batch file with two entries
    batch_df = pd.DataFrame({
                'name': ['protein1', 'protein2'],
                'pdb_id': ['1abc|123', '8smv'],
                'membrane_protein': [False, True],
                'mutation_data_path': [pd.NA, 'mut_data.csv'],
                'config_path': ['config.toml1', 'config.toml2']
            })
    batch_file_path = tmp_path / 'batch_file.csv'
    batch_df.to_csv(batch_file_path, index=False)

    # Mock Runner to track calls and return a simple dataframe
    calls = []
    class FakeRunner:
        def __init__(self, **kwargs):
            calls.append(kwargs)

        def run(self):
            return pd.DataFrame({'result_summary': ['custom']})


    monkeypatch.setattr('src.pipeline.batch_processing.Runner', FakeRunner)
    processed = batch_process(batch_file_path)

    for idx, call in enumerate(calls):
        assert call['pdb_id'] == ['1abc', '123', '8smv'][idx]

        # 1st and 2nd entries of all other params are the same, with third being from 2nd row
        duped_idx = 0 if idx < 2 else 1
        assert call['membrane_protein'] == batch_df.iloc[duped_idx]['membrane_protein']
        assert call['config_path'] == batch_df.iloc[duped_idx]['config_path']
        assert call['name'] == batch_df.iloc[duped_idx]['name']

        mut_path = batch_df.iloc[duped_idx]['mutation_data_path']
        if pd.isna(mut_path):
            assert call['mutation_data_path'] is None
        else:
            assert call['mutation_data_path'] == mut_path

    with pytest.raises(FileNotFoundError, match="Batch file not found at"):
        batch_process(tmp_path / 'non_existent_file.csv')


def test_expand_batch_arguments_single_value():
    batch_df = pd.DataFrame({
        'name': ['protein1', 'protein2'],
        'pdb_id': ['1abc', '2xyz'],
        'membrane_protein': [False, True],
        'mutation_data_path': [None, 'mut_data.csv'],
        'config_path': ['config.toml1', 'config.toml2']
    })
    expanded_args = expand_batch_arguments(batch_df)

    assert len(expanded_args) == 2
    for i, row in batch_df.iterrows():
        assert expanded_args[i]['pdb_id'] == row['pdb_id']
        assert expanded_args[i]['membrane_protein'] == row['membrane_protein']
        assert expanded_args[i]['mutation_data_path'] == row['mutation_data_path']
        assert expanded_args[i]['config_path'] == row['config_path']


    # Check that missing columns raise errors
    incomplete_batch_df = batch_df.drop(columns=['pdb_id'])
    with pytest.raises(ValueError, match="Batch file is missing required column: pdb_id"):
        expand_batch_arguments(incomplete_batch_df)


def test_expand_batch_arguments_empty_vals():
    batch_df = pd.DataFrame({
        'name': ['protein1'],
        'pdb_id': ['1abc'],
        'membrane_protein': [False],
        'mutation_data_path': [pd.NA],
        'config_path': ['config.toml1']
    })
    expanded_args = expand_batch_arguments(batch_df)

    assert len(expanded_args) == 1
    assert expanded_args[0]['pdb_id'] == '1abc'
    assert expanded_args[0]['membrane_protein'] is False
    assert expanded_args[0]['mutation_data_path'] is None
    assert expanded_args[0]['config_path'] == 'config.toml1'


def test_expand_batch_arguments_multiple_values():
    # Create batch dataframe with multiple PDB IDs in some entries
    batch_df_multiple_pdb = pd.DataFrame({
        'name': ['protein1', 'protein2', 'protein1'],
        'pdb_id': ['1abc|1def', '2xyz', '3ghi|3jkl|3lmo'],
        'membrane_protein': [False, True, False],
        'mutation_data_path': ['mut_data.csv1', 'mut_data.csv2', 'mut_data.csv3'],
        'config_path': ['config.toml', 'config.toml', 'config.toml']
    })

    expanded_args = expand_batch_arguments(batch_df_multiple_pdb)

    assert len(expanded_args) == 6  # protein1 expands to 2 entries, protein2 is 1 entry, protein3 expands to 3 entries

    # Check that each expanded entry matches the correct PDB ID and other parameters
    output_idx = 0
    for i, row in batch_df_multiple_pdb.iterrows():
        pdb_ids = row['pdb_id'].split('|')
        for pdb_id in pdb_ids:
            assert expanded_args[output_idx]['pdb_id'] == pdb_id
            assert expanded_args[output_idx]['membrane_protein'] == row['membrane_protein']
            assert expanded_args[output_idx]['mutation_data_path'] == row['mutation_data_path']
            assert expanded_args[output_idx]['config_path'] == row['config_path']
            output_idx += 1

    # Create batch dataframe with multiple mutation data paths in some entries
    batch_df_multiple_mut = pd.DataFrame({
        'name': ['protein1', 'protein2', 'protein1'],
        'pdb_id': ['1abc', '2xyz', '3ghi'],
        'membrane_protein': [False, True, False],
        'mutation_data_path': ['mut1.csv|mut2.csv', 'mut3.csv', 'mut4.csv|mut5.csv|mut6.csv'],
        'config_path': ['config.toml', 'config.toml', 'config.toml']
    })

    expanded_args_mut = expand_batch_arguments(batch_df_multiple_mut)
    assert len(expanded_args_mut) == 6  # protein1 expands to 2 entries, protein2 is 1 entry, protein3 expands to 3 entries

    # Check that each expanded entry matches the correct mutation data path and other parameters
    output_idx = 0
    for i, row in batch_df_multiple_mut.iterrows():
        mut_paths = row['mutation_data_path'].split('|')
        for mut_path in mut_paths:
            assert expanded_args_mut[output_idx]['pdb_id'] == row['pdb_id']
            assert expanded_args_mut[output_idx]['membrane_protein'] == row['membrane_protein']
            assert expanded_args_mut[output_idx]['mutation_data_path'] == mut_path
            assert expanded_args_mut[output_idx]['config_path'] == row['config_path']
            output_idx += 1


def test_expand_batch_arguments_product_expansion():
    # Create batch dataframe with multiple PDB IDs and mutation data paths
    batch_df_product = pd.DataFrame({
        'name': ['protein1'],
        'pdb_id': ['1abc|1def'],
        'membrane_protein': [False],
        'mutation_data_path': ['mut1.csv|mut2.csv'],
        'config_path': ['config.toml']
    })

    expanded_args = expand_batch_arguments(batch_df_product)

    assert len(expanded_args) == 4  # 2 PDB IDs x 2 mutation data paths = 4 combinations

    expected_combinations = [
        ('1abc', 'mut1.csv'),
        ('1abc', 'mut2.csv'),
        ('1def', 'mut1.csv'),
        ('1def', 'mut2.csv')
    ]

    for i, (expected_pdb, expected_mut) in enumerate(expected_combinations):
        assert expanded_args[i]['pdb_id'] == expected_pdb
        assert expanded_args[i]['mutation_data_path'] == expected_mut
        assert expanded_args[i]['membrane_protein'] is False
        assert expanded_args[i]['config_path'] == 'config.toml'