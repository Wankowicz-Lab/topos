"""Tests for the pipeline runner module."""
import numpy as np
import pandas as pd
import pytest
import random

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
    config_path = tmp_path / 'config.toml'
    _make_config_file(config_path, mutation_data_chain='A')

    # Create a mock mutation dataset
    residue_table = _make_residue_table(
        num_chains=1,
        num_residues=10,
        start_resis=1,
        make_muts=True
    )

    mut_dataset = residue_table[['resn_mut', 'resi_mut', 'resm', 'effect', 'type']]
    mut_dataset = mut_dataset.rename(columns={
        'resn_mut': 'wildtype',
        'resi_mut': 'position',
        'resm': 'mutation',
        'effect': 'effect',
        'type': 'type'
    })

    mut_data_path = tmp_path / 'mut_data.csv'
    mut_dataset.to_csv(mut_data_path, index=False)

    # Create synthetic mmcif file to match mutation data
    residues = residue_table[['resn_mut', 'resi_mut']].drop_duplicates()['resn_mut']
    mmcif_path = tmp_path / "test_structure.cif"
    _write_mmcif_file(file_path=mmcif_path, pdb_id="TEST", chains={"A": residues.tolist()})

    mut_runner = runner.Runner(
        pdb_id='8SMV',
        pdb_path=mmcif_path,
        membrane_protein=True,
        mutation_data_path=mut_data_path,
        config_path=config_path
    )
    assert mut_runner.context.config.pdb_path == mmcif_path
    assert mut_runner.context.config.mutation_data_path == mut_data_path
    assert 'effect' in mut_runner.context.residue_table.columns.tolist()


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

    mut_dataset = residue_table[['resn_mut', 'resi_mut', 'resm', 'effect', 'type']]
    mut_dataset = mut_dataset.rename(columns={
        'resn_mut': 'wildtype',
        'resi_mut': 'position',
        'resm': 'mutation',
        'effect': 'effect',
        'type': 'type'
    })

    mut_data_path = tmp_path / 'mut_data.csv'
    mut_dataset.to_csv(mut_data_path, index=False)

    # Create synthetic mmcif file to match mutation data
    residues = residue_table[['resn_mut', 'resi_mut']].drop_duplicates()['resn_mut']
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
        expected_cols = ['chain', 'resi_mut', 'resn_mut', 'resi_struct', 'resn_struct', 'resm']

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

    # Check that resi_mut and resn_mut columns exist and equal resi_struct and resn_struct
    assert 'resi_mut' in myrunner.context.residue_table.columns
    assert 'resn_mut' in myrunner.context.residue_table.columns
    assert (myrunner.context.residue_table['resi_mut'] == myrunner.context.residue_table['resi_struct']).all()
    assert (myrunner.context.residue_table['resn_mut'] == myrunner.context.residue_table['resn_struct']).all()

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
    df1_keep_resis = np.random.choice(residue_table['resi_struct'], size=7, replace=False)
    df1 = residue_table[residue_table['resi_struct'].isin(df1_keep_resis)].copy()
    df1 = df1[['chain', 'resi_struct', 'resn_struct']]
    df1['feature1'] = np.random.rand(len(df1))

    # df2 has residue-level features from a subset of positions in chain A
    df2_keep_resis = np.random.choice(residue_table[residue_table['chain']=='A']['resi_struct'], size=3, replace=False)
    df2 = residue_table[(residue_table['chain']=='A') & (residue_table['resi_struct'].isin(df2_keep_resis))].copy()
    df2 = df2[['chain', 'resi_struct', 'resn_struct']]
    df2['feature2'] = np.random.rand(len(df2))

    # df3 has residue level features from residues in Chain B
    df3_keep_resis = np.random.choice(residue_table.loc[residue_table['chain']=='B']['resi_struct'],
                                      size=5, replace=False)
    df3 = residue_table[residue_table['resi_struct'].isin(df3_keep_resis)].copy()
    df3 = df3[['chain', 'resi_struct', 'resn_struct']]
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
    df1_mask = merged_df['resi_struct'].isin(df1_keep_resis)
    assert not merged_df.loc[df1_mask, 'feature1'].isnull().all()
    assert merged_df.loc[~df1_mask, 'feature1'].isnull().all()

    df2_mask = (merged_df['chain']=='A') & (merged_df['resi_struct'].isin(df2_keep_resis))
    assert not merged_df.loc[df2_mask, 'feature2'].isnull().all()
    assert merged_df.loc[~df2_mask, 'feature2'].isnull().all()

    df3_mask = merged_df['resi_struct'].isin(df3_keep_resis)
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
    residue_table_no_muts = residue_table[['chain', 'resi_struct', 'resn_struct', 'resi_mut', 'resn_mut']].drop_duplicates().reset_index(drop=True)

    # df1 has residue level features from a subset of the residues  
    df1_keep_resis = np.random.choice(residue_table_no_muts['resi_struct'], size=7, replace=False)
    df1 = residue_table_no_muts[residue_table_no_muts['resi_struct'].isin(df1_keep_resis)].copy()
    df1 = df1[['chain', 'resi_struct', 'resn_struct']]
    df1['feature1'] = np.random.rand(len(df1))

    # df2 has mutation-level features for all mutations from a subset of positions in chain A
    df2_keep_resis = np.random.choice(residue_table[residue_table['chain']=='A']['resi_mut'], size=3, replace=False)
    df2 = residue_table[(residue_table['chain']=='A') & (residue_table['resi_mut'].isin(df2_keep_resis))].copy()
    df2 = df2[['chain', 'resi_mut', 'resn_mut', 'resm']]
    df2['feature2'] = np.random.rand(len(df2))

    # df3 has residue level features from residues in Chain B
    df3_keep_resis = np.random.choice(residue_table_no_muts.loc[residue_table_no_muts['chain']=='B']['resi_struct'],
                                      size=5, replace=False)
    df3 = residue_table_no_muts[residue_table_no_muts['resi_struct'].isin(df3_keep_resis)].copy()
    df3 = df3[['chain', 'resi_struct', 'resn_struct']]
    df3['feature3'] = np.random.rand(len(df3))

    # df4 has a subset of mutations at each position
    df4_keep_resis = np.random.choice(len(residue_table), size=40, replace=False)
    df4 = residue_table.iloc[df4_keep_resis].copy()
    df4 = df4[['chain', 'resi_mut', 'resn_mut', 'resm']]
    df4['feature4'] = np.random.rand(len(df4))

    result_frames = [df1, df2, df3, df4]

    class MockContext:
        def __init__(self, residue_table):
            self.residue_table = residue_table
    myrunner.context = MockContext(residue_table=residue_table)

    merged_df = myrunner._merge_features(result_frames, mutations=True)

    assert len(merged_df) == len(residue_table)
    assert 'resm' in merged_df.columns.tolist()

    # Check that df values are correctly merged for structure-based features
    df1_mask = merged_df['resi_struct'].isin(df1_keep_resis)
    assert not merged_df.loc[df1_mask, 'feature1'].isnull().all()
    assert merged_df.loc[~df1_mask, 'feature1'].isnull().all()

    # Check sequence-based features
    df2_mask = (merged_df['chain']=='A') & (merged_df['resi_mut'].isin(df2_keep_resis))
    assert not merged_df.loc[df2_mask, 'feature2'].isnull().all()
    assert merged_df.loc[~df2_mask, 'feature2'].isnull().all()

    df3_mask = merged_df['resi_struct'].isin(df3_keep_resis)
    assert not merged_df.loc[df3_mask, 'feature3'].isnull().all()
    assert merged_df.loc[~df3_mask, 'feature3'].isnull().all()

    df4 = df4.set_index(['chain', 'resi_mut', 'resn_mut', 'resm'])
    merged_df = merged_df.set_index(['chain', 'resi_mut', 'resn_mut', 'resm'])
    df4_mask = merged_df.index.isin(df4.index)
    assert not merged_df.loc[df4_mask, 'feature4'].isnull().all()
    assert merged_df.loc[~df4_mask,  'feature4'].isnull().all()
