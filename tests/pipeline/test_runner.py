"""Tests for the pipeline runner module."""

import numpy as np
import pandas as pd
import pytest
import random
import tomli_w

import biotite.structure as struc
from tests.test_utils import _make_residue_table, _write_mmcif_file, _make_aaindex_data, _make_config_file
from src.pipeline import runner
from src.structure.structure_context import load_structure
from src.structure.utils import res_key

# import files containing metrics to register them in _REGISTRY
import src.metrics.sequence
import src.metrics.structure
from src.metrics.registry import _REGISTRY
from tests.test_utils import _make_residue, _make_atoms

from src.pipeline.context import Config

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
    assert base_runner.context.config.mutation_data_path is None
    assert base_runner.context.config.name == 'test_protein'

    with pytest.raises(ValueError, match="Either pdb_id or config_path must be provided."):
        _ = runner.Runner()

    with pytest.raises(ValueError, match="Either name or config_path must be provided."):
        _ = runner.Runner(pdb_id='test')


def test_runner_initialization_altloc_policy(tmp_path):
    """Test that altloc policy is set correctly."""
    config_path = tmp_path / 'config.toml'
    _make_config_file(config_path, altloc_policy='all', pdb_id='5C1M')

    altloc_runner = runner.Runner(config_path=config_path)
    assert altloc_runner.context.config.altloc_policy == 'all'
    assert set(np.unique(altloc_runner.context.array.altloc_id)) == {'.', 'A', 'B'}

    # Altlocs in body of residue don't show up in residue table, only those at the start of the residue (res starts). Need to decide how to handle this
    assert set(np.unique(altloc_runner.context.residue_table.altloc)) == {'.', 'A'}


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
    
    with pytest.raises(RuntimeError, match="Failed to fetch PDBTM annotation"):
        membrane_runner = runner.Runner(
            pdb_id=pdb_id,
            name='test_membrane_protein',
            pdb_path=mmcif_path,
            membrane_protein=True
        )


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
    mut_dataset = residue_table[['resn_mut', 'resi_mut', 'resm', 'effect', 'type']]
    mut_dataset = mut_dataset.rename(columns={
        'resn_mut': 'wt_residue',  # custom name instead of 'wildtype'
        'resi_mut': 'res_position',  # custom name instead of 'position'
        'resm': 'mut_residue',  # custom name instead of 'mutation'
        'effect': 'fitness_score',  # custom name instead of 'effect'
        'type': 'mut_type'  # custom name instead of 'type'
    })

    mut_data_path = tmp_path / 'mut_data.csv'
    mut_dataset.to_csv(mut_data_path, index=False)

    # Create synthetic mmcif file to match mutation data
    residues = residue_table[['resn_mut', 'resi_mut']].drop_duplicates()['resn_mut']
    mmcif_path = tmp_path / "test_structure.cif"
    _write_mmcif_file(file_path=mmcif_path, pdb_id="TEST", chains={"A": residues.tolist()})

    # Create config file with custom column names
    config_path = tmp_path / 'config.toml'
    config_dict = {
        'pdb_id': '8SMV',
        'name': 'test_protein',
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
    
    # Check that residue_table has align_pos column and is sorted appropriately
    assert 'align_pos' in mut_runner.context.residue_table.columns, "residue_table should have align_pos column"
    
    # Check that mutation chain A comes first
    chains = mut_runner.context.residue_table['chain'].tolist()
    unique_chains = []
    for c in chains:
        if not unique_chains or unique_chains[-1] != c:
            unique_chains.append(c)
    
    assert unique_chains[0] == 'A', "Mutation chain A should come first"
    
    # Check that within each chain, align_pos is sorted
    for chain in mut_runner.context.residue_table['chain'].unique():
        chain_df = mut_runner.context.residue_table[mut_runner.context.residue_table['chain'] == chain]
        align_pos_values = chain_df['align_pos'].tolist()
        assert align_pos_values == sorted(align_pos_values), f"Within chain {chain}, align_pos should be sorted"

    # Now test with invalid chain specified in config
    bad_config = config_dict.copy()
    bad_config['mutation_data_chain'] = 'B'  # chain not in mmcif
    bad_config_path = tmp_path / 'bad_config.toml'

    with bad_config_path.open("wb") as f:
        tomli_w.dump(bad_config, f)

    with pytest.raises(ValueError, match="Specified mutation_data_chain 'B' not found in structure chains"):
        _ = runner.Runner(
            pdb_path=mmcif_path,
            config_path=bad_config_path
        )


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

    # Create config file with INCORRECT column names
    config_path = tmp_path / 'config.toml'
    config_dict = {
        'pdb_id': '8SMV',
        'name': 'test_protein',
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


def test_runner_initialization_bad_config(tmp_path):
    """Test that FileNotFoundError is raised when config file doesn't exist."""
    # Path to a non-existent config file
    config_path = tmp_path / 'nonexistent_config.toml'

    with pytest.raises(FileNotFoundError, match="Configuration file not found at"):
        runner.Runner(config_path=config_path)


def test_runner_initialization_invalid_config(tmp_path):
    """Test that ValueError is raised when config file has invalid TOML syntax."""
    # Create a config file with invalid TOML syntax
    config_path = tmp_path / 'invalid_config.toml'
    with config_path.open("w") as f:
        f.write("pdb_id = 8smv\n")  # Invalid: missing quotes around string value
        f.write("name = invalid toml syntax\n")  # Invalid: missing quotes around string value
        f.write("[broken section\n")  # Invalid: missing closing bracket

    with pytest.raises(ValueError, match="Invalid TOML in configuration file"):
        runner.Runner(config_path=config_path)


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
        name='test_protein',
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

    # Test missing pdb_id
    empty_config = Config(name='test')
    with pytest.raises(ValueError, match="'pdb_id' must be provided"):
        myrunner._merge_config(base=empty_config, overrides={})
    with pytest.raises(ValueError, match="'pdb_id' must be provided"):
        myrunner._merge_config(base=empty_config,
                               overrides={'membrane_protein': True})

    # Test missing name
    empty_config = Config(pdb_id='1abc')
    with pytest.raises(ValueError, match="'name' must be provided"):
        myrunner._merge_config(base=empty_config, overrides={})
    with pytest.raises(ValueError, match="'name' must be provided"):
        myrunner._merge_config(base=empty_config,
                               overrides={'membrane_protein': True})




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
        returned_features = myrunner.run_metrics(metrics=[metric], mutations=True)

        returned_cols = returned_features.columns.tolist()
        expected_cols = ['chain', 'resi_mut', 'resn_mut', 'resi_struct', 'resn_struct', 'resm', 'name']

        if metric == 'aaindex_scores':
            # aaindex scores add columns for each index
            for acc in ['AA1', 'AA2']:
                expected_cols.extend([f'AAIndex_{acc}_wt', f'AAIndex_{acc}_mut', f'AAIndex_{acc}_diff'])
        elif metric == 'kidera_factors':
            for factornum in range(1, 11):
                expected_cols.extend([f'kidera_f{factornum}_wt', f'kidera_f{factornum}_mut', f'kidera_f{factornum}_diff'])
        else:
            expected_cols.extend(provides)

        assert set(expected_cols) == set(returned_cols)

    # run all metrics
    returned_features = myrunner.run_metrics(metrics=list(metrics), mutations=True)
    assert len(returned_features) == len(residue_table)


def test_runner_run_metric_no_mutations(tmp_path):
    pdb_id = '8smv'

    myrunner = runner.Runner(
        pdb_id=pdb_id,
        name='test_no_mutations',
        pdb_path=None,
        membrane_protein=False,
        mutation_data_path=None
    )

    # run multiple metrics
    returned_features = myrunner.run_metrics(metrics=['sasa', 'kyte_doolittle'])

    # Check that all residues are present and no 'resm' column
    assert len(returned_features) == len(myrunner.context.residue_table)
    assert 'resm' not in returned_features.columns.tolist()


def test_runner_run(tmp_path):
    pdb_id = '8efo'

    myrunner = runner.Runner(
        pdb_id=pdb_id,
        name='test_run',
        pdb_path=None,
        membrane_protein=False,
        mutation_data_path=None
    )

    # run multiple metrics
    myrunner.run()
    returned_features = myrunner.features

    # Check that output has same length as residue table.
    assert len(returned_features) == len(myrunner.context.residue_table)

    # Verify integration outputs are present and populated.
    ss_cols = [c for c in returned_features.columns if c.startswith('ss_')]
    neighborhood_cols = [c for c in returned_features.columns if c.startswith('n_')]
    ligand_cols = [
        c for c in returned_features.columns
        if c.startswith('ligand_') and c.endswith('_interactions')
    ]
    graph_cols = [c for c in returned_features.columns if c.startswith('graph_')]
    expected_graph_cols = {
        f'graph_{bond_type}_{metric_col}'
        for bond_type in ['all', 'vdw_contact', 'hbond']
        for metric_col in [
            'graph_betweenness_centrality',
            'graph_closeness_centrality',
            'graph_eigenvector_centrality',
            'graph_core_number',
            'graph_community_id',
            'graph_in_lcc',
        ]
    }

    assert ss_cols, "Expected at least one secondary-structure column (prefix 'ss_')."
    assert neighborhood_cols, "Expected at least one neighborhood column (prefix 'n_')."
    assert ligand_cols, "Expected at least one ligand interaction column (pattern 'ligand_*_interactions')."
    assert graph_cols, "Expected at least one graph column (prefix 'graph_')."
    assert expected_graph_cols.issubset(set(returned_features.columns.tolist())), (
        "Expected graph columns from all/vdw_contact/hbond graph metric passes."
    )

    assert returned_features[ss_cols].notna().any().any(), "Secondary-structure columns are all null."
    assert returned_features[neighborhood_cols].notna().any().any(), "Neighborhood columns are all null."
    assert returned_features[ligand_cols].notna().any().any(), "Ligand interaction columns are all null."
    all_graph_cols = [c for c in returned_features.columns if c.startswith('graph_all_')]
    assert returned_features[all_graph_cols].notna().any().any(), "graph_all_* columns are all null."


def test_runner__merge_features():
    pdb_id = '8smv'

    myrunner = runner.Runner(
        pdb_id=pdb_id,
        name='test_merge_features',
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

    class MockConfig:
        structural_feature_chains = None

    class MockContext:
        def __init__(self, residue_table):
            self.residue_table = residue_table
            self.config = MockConfig()
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
        name='test_merge_features_with_muts',
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

    class MockConfig:
        def __init__(self):
            self.mutation_data_chain = 'A'
            self.structural_feature_chains = None

    class MockContext:
        def __init__(self, residue_table):
            self.residue_table = residue_table
            self.config = MockConfig()

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
    
    # Reset index for sorting checks
    merged_df = merged_df.reset_index()
    
    # Check that output is sorted appropriately (chains in order)
    chains = merged_df['chain'].tolist()
    unique_chains = []
    for c in chains:
        if not unique_chains or unique_chains[-1] != c:
            unique_chains.append(c)
    # Should be alphabetically sorted since no mutation_chain
    assert unique_chains == sorted(unique_chains), "Chains should be alphabetically sorted"


def test_runner_save_results(tmp_path):
    # Create data to save
    features_df = pd.DataFrame({
        'chain': ['A', 'A', 'B'],
        'resi_struct': [1, 2, 8],
        'resn_struct': ['ALA', 'VAL', 'GLY'],
        'resi_mut': [1, 2, np.nan],
        'resn_mut': ['ALA', 'VAL', np.nan],
        'resm': ['ARG', 'SER', np.nan],
        'struct_info': [True, True, True],
        'mut_info': [True, True, False],
        'feature1': [0.1, 0.2, 0.3],
        'feature2': [0.4, 0.5, 0.6]
    })

    residue_table = _make_residue_table()
    residue_table['ss_domains'] = 'membrane_spanning'
    residue_table['ss_group'] = 'TM1'

    # Create runner
    pdb_id = '8smv'

    save_runner = runner.Runner(
        pdb_id=pdb_id,
        name='test_save_results',
        membrane_protein=True)
    save_runner.features = features_df
    save_runner.context.residue_table = residue_table

    manual_output_dir = tmp_path / 'output'
    save_runner.save_results(output_dir=manual_output_dir)

    # Check that files are created
    features_path = manual_output_dir / f"{pdb_id}_features.csv"
    metadata_path = manual_output_dir / f"{pdb_id}_metadata.csv"
    assert features_path.exists()
    assert metadata_path.exists()

    saved_features = pd.read_csv(features_path)
    pd.testing.assert_frame_equal(saved_features, features_df)

    saved_metadata = pd.read_csv(metadata_path)
    assert set(saved_metadata['resi_mut']) == set(residue_table['resi_mut'])
    assert set(saved_metadata['ss_domains']) == set(residue_table['ss_domains'])
    assert set(saved_metadata['ss_group']) == set(residue_table['ss_group'])

    # Check that files are created with custom prefix
    custom_prefix = 'testprefix'
    expected_prefix = custom_prefix + '_' + pdb_id
    save_runner.save_results(output_dir=manual_output_dir, output_prefix=custom_prefix)

    features_path = manual_output_dir / f"{expected_prefix}_features.csv"
    metadata_path = manual_output_dir / f"{expected_prefix}_metadata.csv"
    assert features_path.exists()
    assert metadata_path.exists()

    with pytest.raises(ValueError, match="If output_dir is not provided, config_path must be provided to determine output location."):
        save_runner.save_results(output_dir=None)

    # Now test with config_path provided
    config_path = tmp_path / 'input_dir/config.toml'
    save_runner.config_path = config_path
    save_runner.save_results(output_dir=None)

    features_path = config_path.parent / f"{pdb_id}_features.csv"
    metadata_path = config_path.parent / f"{pdb_id}_metadata.csv"
    assert features_path.exists()
    assert metadata_path.exists()

    # Now test with output_dir provided in config
    output_dir_in_config = tmp_path / 'config_output'
    save_runner.context.config.output_dir = output_dir_in_config
    save_runner.save_results(output_dir=None)

    features_path = output_dir_in_config / f"{pdb_id}_features.csv"
    metadata_path = output_dir_in_config / f"{pdb_id}_metadata.csv"
    assert features_path.exists()
    assert metadata_path.exists()


def test_sort_residue_table():
    """Test that _sort_residue_table works correctly."""
    # Test 1: Sorting with mutation_chain specified
    df_with_mutation = pd.DataFrame({
        'chain': ['C', 'B', 'A', 'B', 'A', 'C'],
        'align_pos': [2, 0, 1, 3, 4, 5],
        'resi_mut': [1, 2, 3, 4, 5, 6],
        'resn_mut': ['ALA', 'GLY', 'SER', 'THR', 'VAL', 'LEU']
    })
    
    sorted_df = runner._sort_residue_table(df_with_mutation, mutation_chain='B')
    
    # Check that chain B comes first
    chains = sorted_df['chain'].tolist()
    assert chains[0] == 'B' and chains[1] == 'B', "Mutation chain B should come first"
    
    # Check that within each chain, residues are sorted by align_pos
    chain_b_align_pos = sorted_df[sorted_df['chain'] == 'B']['align_pos'].tolist()
    assert chain_b_align_pos == sorted(chain_b_align_pos), "Within chain B, should be sorted by align_pos"
    
    # Check that other chains are alphabetically ordered
    assert chains == ['B', 'B', 'A', 'A', 'C', 'C'], "Should be B first, then A, then C"
    
    # Test 2: Sorting without mutation_chain (alphabetical)
    sorted_df_alpha = runner._sort_residue_table(df_with_mutation, mutation_chain=None)
    chains_alpha = sorted_df_alpha['chain'].tolist()
    assert chains_alpha == ['A', 'A', 'B', 'B', 'C', 'C'], "Should be alphabetically sorted when no mutation_chain"
    
    # Check sorting within each chain
    for chain in ['A', 'B', 'C']:
        chain_align_pos = sorted_df_alpha[sorted_df_alpha['chain'] == chain]['align_pos'].tolist()
        assert chain_align_pos == sorted(chain_align_pos), f"Within chain {chain}, should be sorted by align_pos"


def _make_synthetic_ss_fixture(include_na_ss=False, metric_with_nan=False, aa_groups=None):
    """Build synthetic residue_table and features for run_secondary_structure tests.

    Default: 6 residues, 1 chain, ss_domains alpha-helix_1 (2), beta-sheet_1 (2), coil_1 (2).
    merge key: chain, resi_struct, resn_struct, resi_mut, resn_mut.
    """
    residue_table = pd.DataFrame({
        'chain': ['A'] * 6,
        'resi_struct': [1, 2, 3, 4, 5, 6],
        'resn_struct': ['ALA', 'ARG', 'CYS', 'ASP', 'GLU', 'PHE'],
        'resi_mut': [1, 2, 3, 4, 5, 6],
        'resn_mut': ['ALA', 'ARG', 'CYS', 'ASP', 'GLU', 'PHE'],
        'ss_domains': ['alpha-helix_1', 'alpha-helix_1', 'beta-sheet_1', 'beta-sheet_1', 'coil_1', 'coil_1'],
    })
    features = pd.DataFrame({
        'chain': ['A'] * 6,
        'resi_struct': [1, 2, 3, 4, 5, 6],
        'resn_struct': ['ALA', 'ARG', 'CYS', 'ASP', 'GLU', 'PHE'],
        'resi_mut': [1, 2, 3, 4, 5, 6],
        'resn_mut': ['ALA', 'ARG', 'CYS', 'ASP', 'GLU', 'PHE'],
        'metric_a': [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
    })
    if aa_groups is not None:
        features['wildtype_aa_group'] = aa_groups
    if include_na_ss:
        residue_table = pd.concat([
            residue_table,
            pd.DataFrame({
                'chain': ['A'], 'resi_struct': [7], 'resn_struct': ['GLY'],
                'resi_mut': [7], 'resn_mut': ['GLY'], 'ss_domains': [pd.NA],
            }),
        ], ignore_index=True)
        features = pd.concat([
            features,
            pd.DataFrame({
                'chain': ['A'], 'resi_struct': [7], 'resn_struct': ['GLY'],
                'resi_mut': [7], 'resn_mut': ['GLY'], 'metric_a': [7.0],
            }),
        ], ignore_index=True)
    if metric_with_nan:
        features.loc[features['resi_struct'] == 1, 'metric_a'] = np.nan
    return residue_table, features


def test_run_secondary_structure_columns_exist(tmp_path):
    """run_secondary_structure averages features per domain; each residue in a domain gets the same domain mean."""
    config_path = tmp_path / 'config.toml'
    _make_config_file(config_path, mutation_data_chain='A', mutation_data_path="")
    myrunner = runner.Runner(config_path=config_path)
    residue_table, features = _make_synthetic_ss_fixture()
    myrunner.context.residue_table = residue_table
    myrunner.features = features

    out = myrunner.run_secondary_structure(ss_metrics=['metric_a'])

    # Check that output has expected columns
    assert len(out) == len(residue_table)
    assert 'ss_domains' in out.columns and 'ss_domain_length' in out.columns and 'ss_domain_metric_a' in out.columns
    from src.metrics import secondary_structure as ss_metrics
    for g in ss_metrics.AA_GROUPS:
        assert f'ss_domain_log2_aa_group_ratio_{g}' in out.columns

    # Check that each residue in a domain gets the same domain mean
    feat_with_ss = features.merge(
        residue_table[['chain', 'resi_struct', 'resn_struct', 'ss_domains']],
        on=['chain', 'resi_struct', 'resn_struct'],
    )
    merged = feat_with_ss.merge(
        out[['chain', 'resi_struct', 'resn_struct', 'ss_domain_metric_a']],
        on=['chain', 'resi_struct', 'resn_struct'],
    )
    merged['expected'] = merged.groupby(['chain', 'ss_domains'])['metric_a'].transform('mean')
    np.testing.assert_array_equal(merged['ss_domain_metric_a'].values, merged['expected'].values)


def test_run_secondary_structure_na_in_metric(tmp_path):
    """With NA in metric column, output has expected columns and one row per residue."""
    config_path = tmp_path / 'config.toml'
    _make_config_file(config_path, mutation_data_chain='A', mutation_data_path="")
    myrunner = runner.Runner(config_path=config_path)
    residue_table, features = _make_synthetic_ss_fixture(metric_with_nan=True)
    myrunner.context.residue_table = residue_table
    myrunner.features = features

    out = myrunner.run_secondary_structure(ss_metrics=['metric_a'])
    assert len(out) == len(residue_table)
    assert 'ss_domain_metric_a' in out.columns and 'ss_domain_length' in out.columns

    # Check that metric column is not NA
    assert not out['ss_domain_metric_a'].isna().all()


def test_run_secondary_structure_na_ss_domains_excluded(tmp_path):
    """Row with NA ss_domains is excluded; output has expected columns and one row per residue."""
    config_path = tmp_path / 'config.toml'
    _make_config_file(config_path, mutation_data_chain='A', mutation_data_path="")
    myrunner = runner.Runner(config_path=config_path)
    residue_table, features = _make_synthetic_ss_fixture(include_na_ss=True)
    myrunner.context.residue_table = residue_table
    myrunner.features = features

    out = myrunner.run_secondary_structure(ss_metrics=['metric_a'])
    assert len(out) == len(residue_table) - 1 # 1 row with NA ss_domains is excluded
    assert 'ss_domains' in out.columns and 'ss_domain_length' in out.columns and 'ss_domain_metric_a' in out.columns


def test_compute_residue_neighbors_basic():
    """Test _compute_residue_neighbors computes neighbors correctly and stores in extras."""
    myrunner = runner.Runner(
        pdb_id='8smv',
        name='test_neighbors',
        pdb_path=None,
        membrane_protein=False
    )
    # Compute neighbors with a reasonable cutoff
    mapping = myrunner._compute_residue_neighbors(cutoff=10.0)
    
    # Check that result is stored in extras
    assert myrunner.context.extras['residue_neighbors'] == mapping
    
    # Check structure: Dict[str, List[str]]
    assert isinstance(mapping, dict)
    assert len(mapping) > 0
    
    # Check that all residues from structure are present
    rt = myrunner.context.residue_table
    expected_keys = {res_key(row["chain"], row["resi_struct"], row["resn_struct"]) for _, row in rt.iterrows()}
    assert set(mapping.keys()) == expected_keys


def test_compute_residue_neighbors_cutoff_effect():
    """Test that different cutoffs produce different neighbor sets."""
    myrunner = runner.Runner(
        pdb_id='8smv',
        name='test_neighbors',
        pdb_path=None,
        membrane_protein=False
    )
    
    # Small cutoff - fewer neighbors
    mapping_small = myrunner._compute_residue_neighbors(cutoff=5.0)
    
    # Large cutoff - more neighbors
    mapping_large = myrunner._compute_residue_neighbors(cutoff=20.0)
    
    # Check that large cutoff has at least as many neighbors per residue
    for res_key in mapping_small.keys():
        assert res_key in mapping_large
        assert len(mapping_large[res_key]) >= len(mapping_small[res_key])


def test_calculate_neighborhood_features_basic():
    """Test calculate_neighborhood_features loops over functions and aggregates correctly."""
    myrunner = runner.Runner(
        pdb_id='8smv',
        name='test_neighbors',
        pdb_path=None,
        membrane_protein=False
    )
    myrunner.features = myrunner.run_metrics(metrics=['sasa'])
    
    # Set up neighbor mapping in extras
    myrunner._compute_residue_neighbors(cutoff=10.0)
    
    # Call calculate_neighborhood_features
    result = runner.calculate_neighborhood_features(
        myrunner.context,
        myrunner.features,
    )
    
    # Check that result has merge columns
    merge_cols = ['chain', 'resi_struct', 'resn_struct']
    assert all(c in result.columns for c in merge_cols)
    
    # Check that neighborhood metric columns are present
    assert 'n_ala_neighbors' in result.columns
    assert 'neighborhood_sasa' in result.columns
    
    # Check that result has one row per unique (chain, resi_struct, resn_struct) from features
    expected_rows = myrunner.features[merge_cols].drop_duplicates()
    assert len(result) == len(expected_rows)
    
    # Check that merge columns match features
    merged_check = pd.merge(
        expected_rows, result[merge_cols],
        on=merge_cols, how='inner'
    )
    assert len(merged_check) == len(expected_rows)


def test_calculate_neighborhood_features_aggregates_multiple_metrics():
    """Test that calculate_neighborhood_features aggregates multiple metric outputs."""
    myrunner = runner.Runner(
        pdb_id='8smv',
        name='test_neighbors_multi',
        pdb_path=None,
        membrane_protein=False
    )
    myrunner.features = myrunner.run_metrics(metrics=['sasa', 'kyte_doolittle'])
    myrunner._compute_residue_neighbors(cutoff=10.0)

    result = runner.calculate_neighborhood_features(
        myrunner.context,
        myrunner.features,
    )

    merge_cols = ['chain', 'resi_struct', 'resn_struct']
    assert all(c in result.columns for c in merge_cols)
    assert 'n_ala_neighbors' in result.columns
    assert 'neighborhood_sasa' in result.columns
    assert 'neighborhood_kyte_doolittle' in result.columns

    expected_rows = myrunner.features[merge_cols].drop_duplicates()
    assert len(result) == len(expected_rows)


def test_calculate_neighborhood_features_neighbor_averages_deterministic():
    """Neighborhood averages use only mapped neighbors and ignore NaN values."""
    class DummyContext:
        def __init__(self, neighbor_map):
            self.extras = {'residue_neighbors': neighbor_map}

    features = pd.DataFrame({
        'chain': ['A', 'A', 'A', 'A', 'A'],
        'resi_struct': [1, 2, 2, 3, 4],
        'resn_struct': ['ALA', 'VAL', 'VAL', 'GLY', 'SER'],
        'sasa': [1.0, 5.0, 7.0, 3.0, 10.0],
        'kyte_doolittle': [2.0, 4.0, 10.0, 6.0, 8.0],
    })
    neighbor_map = {
        'A:1:ALA': ['A:2:VAL', 'A:3:GLY', 'A:999:UNK'],
        'A:2:VAL': ['A:1:ALA'],
        'A:3:GLY': [],
        'A:4:SER': ['A:2:VAL'],
    }
    context = DummyContext(neighbor_map)

    result = runner.calculate_neighborhood_features(
        context,
        features,
    )
    result = result.set_index(['chain', 'resi_struct', 'resn_struct'])

    # A:2 has two mutation-level rows and is first averaged to sasa=6.0, kd=7.0.
    # A:1 neighbors are A:2 and A:3 (plus one missing key ignored).
    assert result.loc[('A', 1, 'ALA'), 'neighborhood_sasa'] == 4.5
    assert result.loc[('A', 1, 'ALA'), 'neighborhood_kyte_doolittle'] == 6.5
    # A:2 neighbor is A:1
    assert result.loc[('A', 2, 'VAL'), 'neighborhood_sasa'] == 1.0
    # A:3 has no neighbors
    assert pd.isna(result.loc[('A', 3, 'GLY'), 'neighborhood_sasa'])
    # A:4 neighbor is A:2 and should use the residue-level mean of duplicate A:2 rows.
    assert result.loc[('A', 4, 'SER'), 'neighborhood_sasa'] == 6.0


def test_run_secondary_structure_averages_mutation_rows_per_residue(tmp_path):
    """Secondary-structure averaging should first collapse mutation-level duplicate rows per residue."""
    config_path = tmp_path / 'config.toml'
    _make_config_file(config_path, mutation_data_chain='A', mutation_data_path="")
    myrunner = runner.Runner(config_path=config_path)

    residue_table, features = _make_synthetic_ss_fixture()
    # Duplicate residue 1 with a second mutation-level value.
    dup_row = features.loc[features['resi_struct'] == 1].copy()
    dup_row['metric_a'] = 9.0
    features = pd.concat([features, dup_row], ignore_index=True)

    myrunner.context.residue_table = residue_table
    myrunner.features = features

    out = myrunner.run_secondary_structure(ss_metrics=['metric_a'])
    alpha = out[out['ss_domains'] == 'alpha-helix_1']

    # Residue-level means: res1=(1+9)/2=5, res2=2, domain mean=(5+2)/2=3.5.
    assert np.isclose(alpha['ss_domain_metric_a'].iloc[0], 3.5)

def test_run_neighborhood_requires_run_first(tmp_path):
    """run_neighborhood must be called after run(); it raises if self.features is missing."""
    config_path = tmp_path / 'config.toml'
    _make_config_file(config_path)

    base_runner = runner.Runner(config_path=config_path)
    with pytest.raises(ValueError, match="No features to extend"):
        base_runner.run_neighborhood(cutoff=10.0)


def test_run_neighborhood_fills_extras_and_merges(tmp_path):
    """run_neighborhood fills context.extras['residue_neighbors'] and merges n_ala_neighbors into self.features."""
    config_path = tmp_path / 'config.toml'
    _make_config_file(config_path)

    myrunner = runner.Runner(config_path=config_path)
    myrunner.features = myrunner.run_metrics(metrics=['sasa', 'kyte_doolittle'])

    myrunner.run_neighborhood(cutoff=10.0)

    # Extras filled with residue_key -> [residue_key, ...]; no self-neighbors
    mapping = myrunner.context.extras['residue_neighbors']
    assert isinstance(mapping, dict)
    for res_key, neighbors in mapping.items():
        assert isinstance(res_key, str)
        assert ":" in res_key
        assert isinstance(neighbors, list)
        assert res_key not in neighbors, "neighbors must not include self"

    # n_ala_neighbors from count_ala_neighbors is merged into self.features
    assert 'n_ala_neighbors' in myrunner.features.columns
    assert 'neighborhood_sasa' in myrunner.features.columns
    assert 'neighborhood_kyte_doolittle' in myrunner.features.columns
    assert myrunner.features['n_ala_neighbors'].shape[0] == myrunner.features.shape[0]


def test_structural_feature_chains_validation(tmp_path):
    """Test that structural_feature_chains validation works correctly in Runner.__post_init__."""
    # Create a structure with multiple chains
    residues = ['ALA', 'VAL', 'GLY', 'SER', 'THR']
    mmcif_path = tmp_path / "test_structure.cif"
    _write_mmcif_file(file_path=mmcif_path, pdb_id="TEST", chains={"A": residues, "B": residues, "C": residues})
    
    # Test 1: Invalid chain ID should raise ValueError
    config_path = tmp_path / 'config_invalid.toml'
    _make_config_file(config_path, pdb_id='test', structural_feature_chains=['Z'])
    
    with pytest.raises(ValueError, match="Specified structural_feature_chains.*not found in structure chains"):
        runner.Runner(
            pdb_id='test',
            pdb_path=mmcif_path,
            name='test_validation',
            config_path=config_path
        )
    
    # Test 2: Empty list should be treated as None (no error)
    config_path_empty = tmp_path / 'config_empty.toml'
    _make_config_file(config_path_empty, pdb_id='test', structural_feature_chains=[])
    
    runner_empty = runner.Runner(
        pdb_id='test',
        pdb_path=mmcif_path,
        name='test_empty',
        config_path=config_path_empty
    )
    assert runner_empty.context.config.structural_feature_chains is None
    assert runner_empty.context.residue_table.chain.unique().tolist() == ['A', 'B', 'C']
    
    # Test 3: Valid chains should work
    config_path_valid = tmp_path / 'config_valid.toml'
    _make_config_file(config_path_valid, pdb_id='test', structural_feature_chains=['A', 'B'])
    
    runner_valid = runner.Runner(
        pdb_id='test',
        pdb_path=mmcif_path,
        name='test_valid',
        config_path=config_path_valid
    )
    assert runner_valid.context.config.structural_feature_chains == ['A', 'B']
    # residue_table should still contain all chains (filtering happens in metrics, not globally)
    assert set(runner_valid.context.residue_table.chain.unique().tolist()) == {'A', 'B', 'C'}


def test_structural_feature_chains_filtering(tmp_path):
    """Test that structural_feature_chains correctly filters structural metrics."""
    # Create a structure with multiple chains
    residues = ['ALA', 'VAL', 'GLY', 'SER', 'THR']
    mmcif_path = tmp_path / "test_structure.cif"
    _write_mmcif_file(file_path=mmcif_path, pdb_id="TEST", chains={"A": residues, "B": residues, "C": residues})
    
    # Test 1: None (default) - should include all chains
    config_path_all = tmp_path / 'config_all.toml'
    _make_config_file(config_path_all, pdb_id='test', structural_feature_chains=None)
    
    runner_all = runner.Runner(
        pdb_id='test',
        pdb_path=mmcif_path,
        name='test_all',
        config_path=config_path_all
    )
    runner_all.features = runner_all.run_metrics(metrics=['sasa'])
    
    # Should have residues from all 3 chains
    result_chains = set(runner_all.features['chain'].unique())
    assert result_chains == {'A', 'B', 'C'}, f"Expected all chains, got {result_chains}"
    assert len(runner_all.features) == len(residues) * 3
    
    # Test 2: Single chain - should only include chain A
    config_path_single = tmp_path / 'config_single.toml'
    _make_config_file(config_path_single, pdb_id='test', structural_feature_chains=['A'])
    
    runner_single = runner.Runner(
        pdb_id='test',
        pdb_path=mmcif_path,
        name='test_single',
        config_path=config_path_single
    )
    runner_single.features = runner_single.run_metrics(metrics=['sasa'])
    
    # Should only have residues from chain A
    result_chains = set(runner_single.features['chain'].unique())
    assert result_chains == {'A'}, f"Expected only chain A, got {result_chains}"
    assert len(runner_single.features) == len(residues)
    
    # Test 3: Multiple chains - should include chains A and B
    config_path_multi = tmp_path / 'config_multi.toml'
    _make_config_file(config_path_multi, pdb_id='test', structural_feature_chains=['A', 'B'])
    
    runner_multi = runner.Runner(
        pdb_id='test',
        pdb_path=mmcif_path,
        name='test_multi',
        config_path=config_path_multi
    )
    runner_multi.features = runner_multi.run_metrics(metrics=['sasa'])
    
    # Should only have residues from chains A and B
    result_chains = set(runner_multi.features['chain'].unique())
    assert result_chains == {'A', 'B'}, f"Expected chains A and B, got {result_chains}"
    assert len(runner_multi.features) == len(residues) * 2


def test_format_ligand_id():
    """Test format_ligand_id produces canonical ligand ID for matching."""
    assert runner.format_ligand_id("A", 1, "ATP") == "A:1:ATP"
    assert runner.format_ligand_id(" B ", 2, " NAG ") == "B:2:NAG"
    assert runner.format_ligand_id("A", 10, None) == "A:10:"


def test_find_ligands():
    """Test find_ligands identifies ligands when run on PDB 8EFO (requires network)."""
    arr = load_structure(pdb_id="8EFO")
    ligands = runner.find_ligands(arr)
    assert len(ligands) >= 1, "8EFO should have at least one ligand identified"
    
    ligands_with_cholesterol = runner.find_ligands(arr, exclude_cholesterol=False)
    assert len(ligands_with_cholesterol) > len(ligands), "With cholesterol excluded, should have fewer ligands"


def test_find_ligands_empty_when_no_hetero():
    """Test find_ligands returns empty list when structure has no hetero atoms."""
    arr = _make_residue("ALA", res_id=1, chain_id="A")
    if "hetero" in arr.get_annotation_categories():
        arr.del_annotation("hetero")
    ligands = runner.find_ligands(arr)
    assert ligands == []


def test_find_ligands_inclusion_criteria():
    """Test that find_ligands inclusion criteria work correctly."""
    protein = _make_atoms(["N", "CA", "C"], [[0, 0, 0], [1, 0, 0], [2, 0, 0]], res_name="ALA", res_id=1, chain_id="A")
    
    # make a mg ion (res_id 2)
    mg = _make_atoms(["MG"], [[10, 10, 10]], res_name="MG", res_id=2, chain_id="A")
    
    # make a mse protein mod (res_id 3 so it is a distinct residue from MG)
    mse = _make_atoms(["N", "CA", "C"], [[20, 20, 20]] * 3, res_name="MSE", res_id=3, chain_id="A")

    arr = struc.concatenate([protein, mg, mse])
    arr.set_annotation("hetero", np.array([False, False, False, True, True, True, True]))
    permissive_ligands = runner.find_ligands(arr, exclude_ions=False, exclude_protein_mods=False)
    assert (
        ("A", 2, "MG") in permissive_ligands
        and ("A", 3, "MSE") in permissive_ligands
    )
    restrictive_ligands = runner.find_ligands(arr, exclude_ions=True, exclude_protein_mods=True)
    assert (
        ("A", 2, "MG") not in restrictive_ligands
        and ("A", 3, "MSE") not in restrictive_ligands
    )


def test_calculate_protein_ligand_interactions(tmp_path):
    """Test calculate_protein_ligand_interactions with hetero-based ligands and partner_residue_key."""
    residues = ['ALA', 'VAL', 'GLY', 'SER', 'THR']
    mmcif_path = tmp_path / "test_structure.cif"
    _write_mmcif_file(file_path=mmcif_path, pdb_id="TEST", chains={"A": residues, "B": residues})

    config_path = tmp_path / 'config_ligand.toml'
    _make_config_file(config_path, pdb_id='test')

    myrunner = runner.Runner(
        pdb_id='test',
        pdb_path=mmcif_path,
        name='test_ligand',
        config_path=config_path
    )

    # Mark chain B as hetero so find_ligands returns B residues as ligands
    arr = myrunner.context.array

    if "hetero" not in arr.get_annotation_categories():
        hetero = np.zeros(arr.array_length(), dtype=bool)
        hetero[arr.chain_id == 'B'] = True
        arr.set_annotation("hetero", hetero)
    else:
        arr.hetero[arr.chain_id == 'B'] = True

    # Contacting df: protein residue (A, 1, ALA) contacting ligand B:1:ALA; use canonical partner_residue_key
    contacting_df = pd.DataFrame({
        'chain': ['A'],
        'resi_struct': [1],
        'resn_struct': ['ALA'],
        'partner_residue_key': ['B:1:ALA']
    })

    result = runner.calculate_protein_ligand_interactions(myrunner.context, contacting_df)

    assert 'ligand_B_1_ALA_interactions' in result.columns
    assert result.shape[0] == myrunner.context.residue_table.shape[0]
    vals = result['ligand_B_1_ALA_interactions'].dropna().unique()
    assert len(vals) >= 1
    assert set(vals).issubset({'contact', 'binding site', 'second shell'})

    # When no ligands (clear hetero), returns residue_table unchanged
    arr.hetero[:] = False
    out_skip = runner.calculate_protein_ligand_interactions(myrunner.context, contacting_df)
    assert out_skip.shape[0] == myrunner.context.residue_table.shape[0]
    assert 'ligand_B_1_ALA_interactions' not in out_skip.columns