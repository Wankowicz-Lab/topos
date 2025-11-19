import pandas as pd
import pytest

from tests.test_utils import _make_residue_table, _write_mmcif_file
from src.pipeline import runner


def test_runner_initialization(tmp_path):

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

    pdb_id = '8smv'

    myrunner = runner.Runner(
        pdb_id=pdb_id,
        pdb_path=None,
        membrane_protein=False,
        mutation_data_path=None,
        mutation_data_chain=None
    )

    assert myrunner.pdb_path is not None
    assert myrunner.pdb_ext == 'cif'
    assert myrunner.array is not None

    myrunner_membrane = runner.Runner(
        pdb_id=pdb_id,
        pdb_path=None,
        membrane_protein=True,
        mutation_data_path=None,
        mutation_data_chain=None
    )

    assert 'pdbtm_region' in myrunner_membrane.context.residue_table.columns.tolist()
    assert 'pdbtm_region_detailed' in myrunner_membrane.context.residue_table.columns.tolist()

    # Create synthetic mmcif file to match mutation data
    residues = residue_table[['resn', 'resi']].drop_duplicates()['resn']
    mmcif_path = tmp_path / "test_structure.cif"
    _write_mmcif_file(file_path=mmcif_path, pdb_id="TEST", chains={"A": residues.tolist()})

    myrunner_mut = runner.Runner(
        pdb_id=pdb_id,
        pdb_path=mmcif_path,
        membrane_protein=True,
        mutation_data_path=mut_data_path,
        mutation_data_chain='A'
    )
    assert 'effect' in myrunner_mut.context.residue_table.columns.tolist()


def test_runner_run(tmp_path):
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

    myrunner = runner.Runner(
        pdb_id='test',
        pdb_path=mmcif_path,
        membrane_protein=False,
        mutation_data_path=mut_data_path,
        mutation_data_chain='A'
    )

    # TODO: more systematic testing of all metrics and expected outputs
    # test individual metrics
    metrics = myrunner.run(metrics=['position_effect_quartiles'])
    assert 'effect_quartile' in metrics.columns.tolist()

    metrics1 = myrunner.run(metrics=['define_secondary_structure'])
    assert 'ss_group' in metrics1.columns.tolist()

    # test multiple metrics
    metrics2 = myrunner.run(metrics=['position_effect_quartiles', 'define_secondary_structure'])
    assert 'effect_quartile' in metrics2.columns.tolist()
    assert 'ss_group' in metrics2.columns.tolist()