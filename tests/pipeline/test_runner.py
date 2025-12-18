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
    assert base_runner.context.config.name == 'test_protein'

    with pytest.raises(ValueError, match="Either pdb_id or config_path must be provided."):
        _ = runner.Runner()

    with pytest.raises(ValueError, match="Either name or config_path must be provided."):
        _ = runner.Runner(pdb_id='test')


def test_runner_initialization_from_pdb_id():
    pdb_id = '8smv'

    id_runner = runner.Runner(
        pdb_id=pdb_id,
        name='test')

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
            name='test_membrane_protein',
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


def test_runner_initialization_load_pdb_format(tmp_path):
    """Test loading structure with PDB format (not just CIF)."""
    # Create a simple PDB file
    residues = ['ALA', 'VAL', 'GLY', 'SER', 'THR']
    pdb_path = tmp_path / "test_structure.pdb"

    # Write a minimal valid PDB file
    pdb_content = []
    atom_id = 1
    for res_idx, res_name in enumerate(residues, start=1):
        # Add backbone atoms for each residue
        for atom_name in ['N', 'CA', 'C', 'O']:
            x, y, z = float(atom_id), 0.0, 0.0
            pdb_content.append(
                f"ATOM  {atom_id:5d}  {atom_name:4s}{res_name:3s} A{res_idx:4d}    "
                f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00 20.00           {atom_name[0]:>2s}\n"
            )
            atom_id += 1
    pdb_content.append("END\n")

    with pdb_path.open("w") as f:
        f.writelines(pdb_content)

    # Test that both .pdb and .cif extensions work
    test_runner = runner.Runner(
        pdb_id='TEST',
        name='test_pdb_format',
        pdb_path=pdb_path
    )

    assert test_runner.context.array is not None
    assert test_runner.context.config.pdb_ext == 'pdb'
    assert test_runner.context.config.pdb_path == pdb_path


def test_runner_initialization_load_cif_format(tmp_path):
    """Test loading structure with CIF format to verify extension handling."""
    residues = ['ALA', 'VAL', 'GLY', 'SER', 'THR']
    cif_path = tmp_path / "test_structure.cif"
    _write_mmcif_file(file_path=cif_path, pdb_id="TEST", chains={"A": residues})

    test_runner = runner.Runner(
        pdb_id='TEST',
        name='test_cif_format',
        pdb_path=cif_path
    )

    assert test_runner.context.array is not None
    assert test_runner.context.config.pdb_ext == 'cif'
    assert test_runner.context.config.pdb_path == cif_path


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
        myrunner.run(metrics=[metric])

        returned_cols = myrunner.features.columns.tolist()
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
    myrunner.run(metrics=list(metrics))
    all_result = myrunner.features
    assert len(all_result) == len(residue_table)

    # run with default (all metrics)
    myrunner.run()
    all_result_default = myrunner.features
    pd.testing.assert_frame_equal(all_result, all_result_default)


def test_runner_run_metric_no_mutations(tmp_path):
    pdb_id = '8smv'

    myrunner = runner.Runner(
        pdb_id=pdb_id,
        name='test_no_mutations',
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
    myrunner.run(metrics=['define_secondary_structure', 'sasa', 'kyte_doolittle'])

    # Check that all residues are present and no 'resm' column
    assert len(myrunner.features) == len(myrunner.context.residue_table)
    assert 'resm' not in myrunner.features.columns.tolist()



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
    residue_table['pdbtm_region'] = 'membrane_spanning'
    residue_table['pdbtm_region_detailed'] = 'TM1'

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
    assert set(saved_metadata['pdbtm_region']) == set(residue_table['pdbtm_region'])

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


def test_merge_features_output_is_sorted(tmp_path):
    """Test that _merge_features output is sorted appropriately."""
    # Create a residue table with mutation data
    residue_table = _make_residue_table(
        num_chains=2,
        num_residues=5,
        start_resis=[1, 10],
        make_muts=True
    )
    
    # Add mutation data columns
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
    
    # Create synthetic mmcif file
    residues = residue_table[['resn_mut', 'resi_mut']].drop_duplicates()['resn_mut']
    mmcif_path = tmp_path / "test_structure.cif"
    _write_mmcif_file(file_path=mmcif_path, pdb_id="TEST", chains={"A": residues.tolist()[:5], "B": residues.tolist()[:5]})
    
    # Make config file
    config_path = tmp_path / 'config.toml'
    _make_config_file(config_path, mutation_data_path=mut_data_path, mutation_data_chain='A')
    
    myrunner = runner.Runner(
        pdb_id='test',
        pdb_path=mmcif_path,
        membrane_protein=False,
        mutation_data_path=mut_data_path,
        config_path=config_path
    )
    
    # Create some feature dataframes
    feature_df1 = pd.DataFrame({
        'chain': ['A', 'B', 'A', 'B'],
        'resi_mut': [1, 10, 2, 11],
        'resn_mut': ['ALA', 'GLY', 'SER', 'THR'],
        'feature1': [0.1, 0.2, 0.3, 0.4]
    })
    
    feature_df2 = pd.DataFrame({
        'chain': ['A', 'B', 'A', 'B'],
        'resi_mut': [1, 10, 2, 11],
        'resn_mut': ['ALA', 'GLY', 'SER', 'THR'],
        'feature2': [0.5, 0.6, 0.7, 0.8]
    })
    
    # Call _merge_features
    merged = myrunner._merge_features([feature_df1, feature_df2], mutations=True)
    
    # Check that output is sorted with mutation_chain 'A' first
    chains = merged['chain'].tolist()
    # Find where chain changes from A to B
    first_b_index = next((i for i, c in enumerate(chains) if c == 'B'), None)
    if first_b_index is not None:
        # All A's should come before all B's
        assert all(c == 'A' for c in chains[:first_b_index]), "All A chains should come before B chains"
        assert all(c == 'B' for c in chains[first_b_index:]), "All B chains should come after A chains"


def test_residue_table_sorted_after_initialization(tmp_path):
    """Test that residue_table is sorted appropriately after runner initializes."""
    # Test 1: With mutation data
    residue_table = _make_residue_table(
        num_chains=3,
        num_residues=5,
        start_resis=[1, 10, 20],
        make_muts=True
    )
    
    mut_dataset = residue_table[residue_table['chain'] == 'A'][['resn_mut', 'resi_mut', 'resm', 'effect', 'type']].head(10)
    mut_dataset = mut_dataset.rename(columns={
        'resn_mut': 'wildtype',
        'resi_mut': 'position',
        'resm': 'mutation',
        'effect': 'effect',
        'type': 'type'
    })
    
    mut_data_path = tmp_path / 'mut_data.csv'
    mut_dataset.to_csv(mut_data_path, index=False)
    
    # Create synthetic mmcif file
    residues_a = residue_table[residue_table['chain'] == 'A']['resn_mut'].unique()[:5]
    residues_b = residue_table[residue_table['chain'] == 'B']['resn_mut'].unique()[:5]
    residues_c = residue_table[residue_table['chain'] == 'C']['resn_mut'].unique()[:5]
    mmcif_path = tmp_path / "test_structure.cif"
    _write_mmcif_file(file_path=mmcif_path, pdb_id="TEST", 
                     chains={"A": residues_a.tolist(), "B": residues_b.tolist(), "C": residues_c.tolist()})
    
    config_path = tmp_path / 'config.toml'
    _make_config_file(config_path, mutation_data_path=mut_data_path, mutation_data_chain='A')
    
    myrunner = runner.Runner(
        pdb_id='test',
        pdb_path=mmcif_path,
        membrane_protein=False,
        mutation_data_path=mut_data_path,
        config_path=config_path
    )
    
    # Check that residue_table has align_pos column
    assert 'align_pos' in myrunner.context.residue_table.columns, "residue_table should have align_pos column"
    
    # Check that mutation chain A comes first
    chains = myrunner.context.residue_table['chain'].tolist()
    unique_chains = []
    for c in chains:
        if not unique_chains or unique_chains[-1] != c:
            unique_chains.append(c)
    
    assert unique_chains[0] == 'A', "Mutation chain A should come first"
    
    # Check that within each chain, align_pos is sorted
    for chain in myrunner.context.residue_table['chain'].unique():
        chain_df = myrunner.context.residue_table[myrunner.context.residue_table['chain'] == chain]
        align_pos_values = chain_df['align_pos'].tolist()
        assert align_pos_values == sorted(align_pos_values), f"Within chain {chain}, align_pos should be sorted"
    
    # Test 2: Without mutation data (alphabetical sorting)
    mmcif_path_no_mut = tmp_path / "test_structure_no_mut.cif"
    _write_mmcif_file(file_path=mmcif_path_no_mut, pdb_id="TEST2", 
                     chains={"C": ['ALA', 'GLY', 'SER'], "A": ['THR', 'VAL', 'LEU'], "B": ['MET', 'PHE', 'TYR']})
    
    myrunner_no_mut = runner.Runner(
        pdb_id='test2',
        name='test_no_mut',
        pdb_path=mmcif_path_no_mut,
        membrane_protein=False,
        mutation_data_path=None
    )
    
    # Check that chains are alphabetically sorted
    chains_no_mut = myrunner_no_mut.context.residue_table['chain'].tolist()
    unique_chains_no_mut = []
    for c in chains_no_mut:
        if not unique_chains_no_mut or unique_chains_no_mut[-1] != c:
            unique_chains_no_mut.append(c)
    
    assert unique_chains_no_mut == sorted(unique_chains_no_mut), "Chains should be alphabetically sorted when no mutation data"
