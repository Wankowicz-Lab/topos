"""Tests for the pipeline runner module."""
import numpy as np
import pandas as pd
import pytest
import random
import tomli_w

from tests.test_utils import _make_residue_table, _write_mmcif_file, _make_aaindex_data, _make_config_file
from src.pipeline import runner

# import files containing metrics to register them in _REGISTRY
import src.sequence.metrics
import src.structure.metrics
from src.structure.structure_context import _REGISTRY, Config

# Seed RNGs for deterministic tests
np.random.seed(42)
random.seed(42)


def test_runner_initialization_from_config(tmp_path):

    config_path = tmp_path / 'config.toml'
    _make_config_file(config_path, mutation_data_chain='A', mutation_data_path="")

    base_runner = runner.Runner(
        config_path=config_path
    )

    assert base_runner.context.array is not None
    assert base_runner.context is not None
    assert base_runner.context.config is not None
    assert base_runner.context.config.pdb_id == '8smv'
    assert base_runner.context.config.pdb_path is not None
    assert base_runner.context.config.pdb_ext == 'cif'
    assert base_runner.context.config.mutation_data_path is None

    with pytest.raises(ValueError, match="Either pdb_id or config_path must be provided."):
        _ = runner.Runner()


def test_runner_initialization_from_pdb_id():
    pdb_id = '8smv'

    id_runner = runner.Runner(
        pdb_id=pdb_id)

    assert id_runner.context.array is not None
    assert id_runner.context is not None
    assert id_runner.context.config is not None
    assert id_runner.context.config.pdb_id == pdb_id
    assert id_runner.context.config.pdb_path is not None
    assert id_runner.context.config.pdb_ext == 'cif'
    assert id_runner.context.config.mutation_data_path is None


def test_runner_initialization_overrides_membrane(tmp_path):

    config_path = tmp_path / 'config.toml'
    _make_config_file(config_path, mutation_data_chain='A')

    membrane_runner = runner.Runner(
        pdb_id='9DMS',
        pdb_path=None,
        membrane_protein=True,
        mutation_data_path=None,
        config_path=config_path
    )
    assert membrane_runner.context.config.pdb_id == '9DMS'
    assert membrane_runner.context.config.membrane_protein is True
    assert 'pdbtm_region' in membrane_runner.context.residue_table.columns.tolist()
    assert 'pdbtm_region_detailed' in membrane_runner.context.residue_table.columns.tolist()


def test_runner_membrane_protein_not_in_pdbtm(tmp_path):
    """Test that pipeline handles PDB not in PDBTM gracefully."""
    # Create a synthetic mmcif file for a non-membrane protein
    residues = ['ALA', 'VAL', 'GLY', 'SER', 'THR']
    mmcif_path = tmp_path / "test_structure.cif"
    _write_mmcif_file(file_path=mmcif_path, pdb_id="TEST", chains={"A": residues})
    
    # Use a fake PDB ID that will never be in PDBTM
    pdb_id = 'FAKE'
    
    with pytest.warns(UserWarning, match="Failed to fetch PDBTM annotation"):
        membrane_runner = runner.Runner(
            pdb_id=pdb_id,
            pdb_path=mmcif_path,
            membrane_protein=True
        )
    
    # Should have set membrane_protein to False after failure
    assert membrane_runner.context.config.membrane_protein is False
    
    # Should not have PDBTM columns since fetch failed
    assert 'pdbtm_region' not in membrane_runner.context.residue_table.columns.tolist()
    assert 'pdbtm_region_detailed' not in membrane_runner.context.residue_table.columns.tolist()
    
    # Should still have a valid context and be able to continue
    assert membrane_runner.context.array is not None
    assert membrane_runner.context.residue_table is not None


def test_runner_initialization_overrides_mutation_data(tmp_path):
    """Test that mutation data can be loaded with custom column names specified in config."""
    # Create a mock mutation dataset with CUSTOM column names
    residue_table = _make_residue_table(
        num_chains=1,
        num_residues=10,
        start_resis=1,
        make_muts=True
    )

    # Use custom column names different from defaults
    mut_dataset = residue_table[['resn', 'resi', 'resm', 'effect', 'type']]
    mut_dataset = mut_dataset.rename(columns={
        'resn': 'wt_residue',  # custom name instead of 'wildtype'
        'resi': 'res_position',  # custom name instead of 'position'
        'resm': 'mut_residue',  # custom name instead of 'mutation'
        'effect': 'fitness_score',  # custom name instead of 'effect'
        'type': 'mut_type'  # custom name instead of 'type'
    })

    mut_data_path = tmp_path / 'mut_data.csv'
    mut_dataset.to_csv(mut_data_path, index=False)

    # Create synthetic mmcif file to match mutation data
    residues = residue_table[['resn', 'resi']].drop_duplicates()['resn']
    mmcif_path = tmp_path / "test_structure.cif"
    _write_mmcif_file(file_path=mmcif_path, pdb_id="TEST", chains={"A": residues.tolist()})

    # Create config file WITH custom column names
    config_path = tmp_path / 'config.toml'
    config_dict = {
        'pdb_id': '8SMV',
        'mutation_data_chain': 'A',
        'mutation_data_path': str(mut_data_path),
        'mutation_residue_col_name': 'wt_residue',
        'mutation_residue_idx_name': 'res_position',
        'mutation_col_name': 'mut_residue',
        'mutation_type_col_name': 'mut_type',
        'mutation_score_col_name': 'fitness_score'
    }
    with config_path.open("wb") as f:
        tomli_w.dump(config_dict, f)

    mut_runner = runner.Runner(
        pdb_path=mmcif_path,
        config_path=config_path
    )
    assert mut_runner.context.config.pdb_path == mmcif_path
    assert mut_runner.context.config.mutation_data_path == mut_data_path
    # Check that 'effect' column exists (standardized name after loading)
    assert 'effect' in mut_runner.context.residue_table.columns.tolist()
    # Verify the config has the custom column names
    assert mut_runner.context.config.mutation_residue_col_name == 'wt_residue'
    assert mut_runner.context.config.mutation_score_col_name == 'fitness_score'


def test_runner_initialization_mutation_data_incorrect_columns(tmp_path):
    """Test that an appropriate error is raised when column names are incorrect."""
    # Create a mock mutation dataset
    residue_table = _make_residue_table(
        num_chains=1,
        num_residues=10,
        start_resis=1,
        make_muts=True
    )

    # Create mutation data with standard column names
    mut_dataset = residue_table[['resn', 'resi', 'resm', 'effect', 'type']]
    mut_dataset = mut_dataset.rename(columns={
        'resn': 'wildtype',
        'resi': 'position',
        'resm': 'mutation',
        'effect': 'effect',
        'type': 'type'
    })

    mut_data_path = tmp_path / 'mut_data.csv'
    mut_dataset.to_csv(mut_data_path, index=False)

    # Create synthetic mmcif file to match mutation data
    residues = residue_table[['resn', 'resi']].drop_duplicates()['resn']
    mmcif_path = tmp_path / "test_structure.cif"
    _write_mmcif_file(file_path=mmcif_path, pdb_id="TEST", chains={"A": residues.tolist()})

    # Create config file with INCORRECT column names
    config_path = tmp_path / 'config.toml'
    config_dict = {
        'pdb_id': '8SMV',
        'mutation_data_chain': 'A',
        'mutation_data_path': str(mut_data_path),
        'mutation_residue_col_name': 'wrong_wt_col',  # incorrect column name
        'mutation_residue_idx_name': 'position',
        'mutation_col_name': 'mutation',
        'mutation_type_col_name': 'type',
        'mutation_score_col_name': 'effect'
    }
    with config_path.open("wb") as f:
        tomli_w.dump(config_dict, f)

    # Verify that appropriate error is raised with detailed message
    with pytest.raises(ValueError, match=r"Columns \['wrong_wt_col'\] not found in mutation scores file"):
        runner.Runner(
            pdb_path=mmcif_path,
            config_path=config_path
        )


def test_runner__merge_config(tmp_path):
    config_path = tmp_path / 'config.toml'
    _make_config_file(config_path, mutation_data_path=None)
    myrunner = runner.Runner(config_path=config_path)

    # make placeholder files
    pdb_file_path = tmp_path / '1abc.cif'
    pdb_file_path.touch()
    aaindex_file_path = tmp_path / 'aaindex.csv'
    aaindex_file_path.touch()

    base_config = Config(
        pdb_id='1abc',
        pdb_path=pdb_file_path,
        membrane_protein=False,
        mutation_data_path=None,
        aaindex_path=aaindex_file_path
    )

    overrides = {
        'pdb_id': '2xyz',
        'membrane_protein': True
    }

    merged_config = myrunner._merge_config(base=base_config, overrides=overrides)

    assert merged_config.pdb_id == '2xyz'
    assert merged_config.pdb_path == pdb_file_path
    assert merged_config.membrane_protein is True
    assert merged_config.mutation_data_path is None
    assert merged_config.aaindex_path == aaindex_file_path


def test_runner_run_metric(tmp_path):
    # Create a mock mutation dataset
    residue_table = _make_residue_table(
        num_chains=1,
        num_residues=10,
        start_resis=1,
        make_muts=True
    )
    # add membrane region columns so that we don't need to integrate with PDBTM to test
    residue_table['pdbtm_region'] = 'membrane_spanning'
    residue_table['pdbtm_region_detailed'] = 'TM1'

    mut_dataset = residue_table[['resn', 'resi', 'resm', 'effect', 'type']]
    mut_dataset = mut_dataset.rename(columns={
        'resn': 'wildtype',
        'resi': 'position',
        'resm': 'mutation',
        'effect': 'effect',
        'type': 'type'
    })

    mut_data_path = tmp_path / 'mut_data.csv'
    mut_dataset.to_csv(mut_data_path, index=False)

    # Create synthetic mmcif file to match mutation data
    residues = residue_table[['resn', 'resi']].drop_duplicates()['resn']
    mmcif_path = tmp_path / "test_structure.cif"
    _write_mmcif_file(file_path=mmcif_path, pdb_id="TEST", chains={"A": residues.tolist()})

    # make aaindex data
    aaindex_path = tmp_path / 'aaindex.csv'
    aaindex_data = _make_aaindex_data(accessions=['AA1', 'AA2'])
    aaindex_data.to_csv(aaindex_path, index=False)

    # Make config file
    config_path = tmp_path / 'config.toml'
    _make_config_file(config_path, mutation_data_path=mut_data_path, mutation_data_chain='A',
                      aaindex_path=aaindex_path)

    myrunner = runner.Runner(
        pdb_id='test',
        pdb_path=mmcif_path,
        membrane_protein=False,
        mutation_data_path=mut_data_path,
        config_path=config_path
    )

    # modify arguments for downstream metrics
    myrunner.context.config.membrane_protein = True
    myrunner.context.residue_table = residue_table

    # get metrics that are registered
    metrics = _REGISTRY.keys()

    # run each metric individually to ensure 'provides' columns are present
    for metric in metrics:
        meta, func = _REGISTRY[metric]
        provides, requires = meta.provides, meta.requires
        result = myrunner.run(metrics=[metric])

        returned_cols = result.columns.tolist()
        expected_cols = ['chain', 'resi', 'resn', 'resm']

        if metric == 'aaindex_scores':
            # aaindex scores add columns for each index
            for acc in ['AA1', 'AA2']:
                expected_cols.extend([f'AAIndex_{acc}_wt', f'AAIndex_{acc}_mut', f'AAIndex_{acc}_diff'])
        else:
            expected_cols.extend(provides)

        assert set(expected_cols) == set(returned_cols)

    # run all metrics
    all_result = myrunner.run(metrics=list(metrics))
    assert len(all_result) == len(residue_table)

    all_result_default = myrunner.run()
    pd.testing.assert_frame_equal(all_result, all_result_default)


def test_runner_run_metric_no_mutations(tmp_path):
    pdb_id = '8smv'

    myrunner = runner.Runner(
        pdb_id=pdb_id,
        pdb_path=None,
        membrane_protein=False,
        mutation_data_path=None
    )

    # run all metrics
    all_result = myrunner.run(metrics=['define_secondary_structure', 'sasa', 'kyte_doolittle'])

    # Check that all residues are present and no 'resm' column
    assert len(all_result) == len(myrunner.context.residue_table)
    assert 'resm' not in all_result.columns.tolist()


def test_runner__merge_features():
    pdb_id = '8smv'

    myrunner = runner.Runner(
        pdb_id=pdb_id,
        pdb_path=None,
        membrane_protein=False,
        mutation_data_path=None
    )

    residue_table = _make_residue_table(num_residues=6, num_chains=2, start_resis=[1,8], make_muts=False)

    # df1 has residue level features from a subset of the residues
    df1_keep_resis = np.random.choice(residue_table['resi'], size=7, replace=False)
    df1 = residue_table[residue_table['resi'].isin(df1_keep_resis)].copy()
    df1 = df1[['chain', 'resi', 'resn']]
    df1['feature1'] = np.random.rand(len(df1))

    # df2 has residue-level features from a subset of positions in chain A
    df2_keep_resis = np.random.choice(residue_table[residue_table['chain']=='A']['resi'], size=3, replace=False)
    df2 = residue_table[(residue_table['chain']=='A') & (residue_table['resi'].isin(df2_keep_resis))].copy()
    df2 = df2[['chain', 'resi', 'resn']]
    df2['feature2'] = np.random.rand(len(df2))

    # df3 has residue level features from residues in Chain B
    df3_keep_resis = np.random.choice(residue_table.loc[residue_table['chain']=='B']['resi'],
                                      size=5, replace=False)
    df3 = residue_table[residue_table['resi'].isin(df3_keep_resis)].copy()
    df3 = df3[['chain', 'resi', 'resn']]
    df3['feature3'] = np.random.rand(len(df3))

    result_frames = [df1, df2, df3]

    class MockContext:
        def __init__(self, residue_table):
            self.residue_table = residue_table
    myrunner.context = MockContext(residue_table=residue_table)

    merged_df = myrunner._merge_features(result_frames, mutations=False)

    assert 'resm' not in merged_df.columns.tolist()
    assert len(merged_df) == len(residue_table)

    # Check that df values are correctly merged
    df1_mask = merged_df['resi'].isin(df1_keep_resis)
    assert not merged_df.loc[df1_mask, 'feature1'].isnull().all()
    assert merged_df.loc[~df1_mask, 'feature1'].isnull().all()

    df2_mask = (merged_df['chain']=='A') & (merged_df['resi'].isin(df2_keep_resis))
    assert not merged_df.loc[df2_mask, 'feature2'].isnull().all()
    assert merged_df.loc[~df2_mask, 'feature2'].isnull().all()

    df3_mask = merged_df['resi'].isin(df3_keep_resis)
    assert not merged_df.loc[df3_mask, 'feature3'].isnull().all()
    assert merged_df.loc[~df3_mask, 'feature3'].isnull().all()


def test_runner__merge_features_with_muts():
    pdb_id = '8smv'

    myrunner = runner.Runner(
        pdb_id=pdb_id,
        pdb_path=None,
        membrane_protein=False,
        mutation_data_path=None
    )

    residue_table = _make_residue_table(num_residues=6, num_chains=2, start_resis=[1,8], make_muts=[True, False])
    residue_table_no_muts = residue_table[['chain', 'resi', 'resn']].drop_duplicates().reset_index(drop=True)

    # df1 has residue level features from a subset of the residues
    df1_keep_resis = np.random.choice(residue_table_no_muts['resi'], size=7, replace=False)
    df1 = residue_table_no_muts[residue_table_no_muts['resi'].isin(df1_keep_resis)].copy()
    df1 = df1[['chain', 'resi', 'resn']]
    df1['feature1'] = np.random.rand(len(df1))

    # df2 has mutation-level features for all mutations from a subset of positions in chain A
    df2_keep_resis = np.random.choice(residue_table[residue_table['chain']=='A']['resi'], size=3, replace=False)
    df2 = residue_table[(residue_table['chain']=='A') & (residue_table['resi'].isin(df2_keep_resis))].copy()
    df2 = df2[['chain', 'resi', 'resn', 'resm']]
    df2['feature2'] = np.random.rand(len(df2))

    # df3 has residue level features from residues in Chain B
    df3_keep_resis = np.random.choice(residue_table_no_muts.loc[residue_table_no_muts['chain']=='B']['resi'],
                                      size=5, replace=False)
    df3 = residue_table_no_muts[residue_table_no_muts['resi'].isin(df3_keep_resis)].copy()
    df3 = df3[['chain', 'resi', 'resn']]
    df3['feature3'] = np.random.rand(len(df3))

    # df4 has a subset of mutations at each position
    df4_keep_resis = np.random.choice(len(residue_table), size=40, replace=False)
    df4 = residue_table.iloc[df4_keep_resis].copy()
    df4 = df4[['chain', 'resi', 'resn', 'resm']]
    df4['feature4'] = np.random.rand(len(df4))

    result_frames = [df1, df2, df3, df4]

    class MockContext:
        def __init__(self, residue_table):
            self.residue_table = residue_table
    myrunner.context = MockContext(residue_table=residue_table)

    merged_df = myrunner._merge_features(result_frames, mutations=True)

    assert len(merged_df) == len(residue_table)
    assert 'resm' in merged_df.columns.tolist()

    # Check that df values are correctly merged
    df1_mask = merged_df['resi'].isin(df1_keep_resis)
    assert not merged_df.loc[df1_mask, 'feature1'].isnull().all()
    assert merged_df.loc[~df1_mask, 'feature1'].isnull().all()

    df2_mask = (merged_df['chain']=='A') & (merged_df['resi'].isin(df2_keep_resis))
    assert not merged_df.loc[df2_mask, 'feature2'].isnull().all()
    assert merged_df.loc[~df2_mask, 'feature2'].isnull().all()

    df3_mask = merged_df['resi'].isin(df3_keep_resis)
    assert not merged_df.loc[df3_mask, 'feature3'].isnull().all()
    assert merged_df.loc[~df3_mask, 'feature3'].isnull().all()

    df4 = df4.set_index(['chain', 'resi', 'resn', 'resm'])
    merged_df = merged_df.set_index(['chain', 'resi', 'resn', 'resm'])
    df4_mask = merged_df.index.isin(df4.index)
    assert not merged_df.loc[df4_mask, 'feature4'].isnull().all()
    assert merged_df.loc[~df4_mask,  'feature4'].isnull().all()



def test_runner_expand_batch_arguments_single_value():
    batch_df = pd.DataFrame({
        'name': ['protein1', 'protein2'],
        'pdb_id': ['1abc', '2xyz'],
        'membrane_protein': [False, True],
        'mutation_data_path': [None, 'mut_data.csv'],
        'config_path': ['config.toml1', 'config.toml2']
    })
    batch_runner = runner.Runner(pdb_id='8smv')
    expanded_args = batch_runner.expand_batch_arguments(batch_df)

    assert len(expanded_args) == 2
    for i, row in batch_df.iterrows():
        assert expanded_args[i]['pdb_id'] == row['pdb_id']
        assert expanded_args[i]['membrane_protein'] == row['membrane_protein']
        assert expanded_args[i]['mutation_data_path'] == row['mutation_data_path']
        assert expanded_args[i]['config_path'] == row['config_path']


def test_runner_expand_arguments_misssing_vals():
    batch_df = pd.DataFrame({
        'name': ['protein1'],
        'pdb_id': ['1abc'],
        'membrane_protein': [False],
        'mutation_data_path': [pd.NA],
        'config_path': ['config.toml1']
    })
    batch_runner = runner.Runner(pdb_id='8smv')
    expanded_args = batch_runner.expand_batch_arguments(batch_df)

    assert len(expanded_args) == 1
    assert expanded_args[0]['pdb_id'] == '1abc'
    assert expanded_args[0]['membrane_protein'] is False
    assert expanded_args[0]['mutation_data_path'] is None
    assert expanded_args[0]['config_path'] == 'config.toml1'


def test_runner_expand_batch_arguments_multiple_values():
    # Create batch dataframe with multiple PDB IDs in some entries
    batch_df_multiple_pdb = pd.DataFrame({
        'name': ['protein1', 'protein2', 'protein1'],
        'pdb_id': ['1abc|1def', '2xyz', '3ghi|3jkl|3lmo'],
        'membrane_protein': [False, True, False],
        'mutation_data_path': ['mut_data.csv1', 'mut_data.csv2', 'mut_data.csv3'],
        'config_path': ['config.toml', 'config.toml', 'config.toml']
    })

    batch_runner = runner.Runner(pdb_id='8smv')
    expanded_args = batch_runner.expand_batch_arguments(batch_df_multiple_pdb)

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

    expanded_args_mut = batch_runner.expand_batch_arguments(batch_df_multiple_mut)
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


def test_runner_expand_batch_arguments_product_expansion():
    # Create batch dataframe with multiple PDB IDs and mutation data paths
    batch_df_product = pd.DataFrame({
        'name': ['protein1'],
        'pdb_id': ['1abc|1def'],
        'membrane_protein': [False],
        'mutation_data_path': ['mut1.csv|mut2.csv'],
        'config_path': ['config.toml']
    })

    batch_runner = runner.Runner(pdb_id='8smv')
    expanded_args = batch_runner.expand_batch_arguments(batch_df_product)

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