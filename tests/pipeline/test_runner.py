import pandas as pd
import pytest

from tests.test_utils import _make_residue_table, _write_mmcif_file, _make_aaindex_data, _make_config_file
from src.pipeline import runner

# import files containing metrics to register them in _REGISTRY
import src.sequence.metrics
import src.structure.metrics
from src.structure.structure_context import _REGISTRY, Config


def test_runner_initialization(tmp_path):

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


def test_runner_run_metric_provides(tmp_path):
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
        if metric == 'aaindex_scores':
            # aaindex scores add columns for each index
            expected_cols = ['chain', 'resi', 'resn', 'resm']
            for acc in ['AA1', 'AA2']:
                expected_cols.extend([f'AAIndex_{acc}_wt', f'AAIndex_{acc}_mut', f'AAIndex_{acc}_diff'])
        else:
            expected_cols = provides + ['chain', 'resi', 'resn']
            if 'resm' in requires:
                expected_cols.append('resm')

        assert set(expected_cols) == set(returned_cols)

    # run all metrics
    all_result = myrunner.run(metrics=list(metrics))


