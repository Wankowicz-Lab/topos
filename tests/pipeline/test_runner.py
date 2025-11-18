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

    from biotite.structure.io.pdbx import CIFFile, get_structure, get_model_count
    import biotite.structure as struc
    pdb_file = CIFFile.read(mmcif_path)
    array = get_structure(pdb_file)
    starts = struc.get_residue_starts(array)
    pdb_names = array.res_name[starts]
    pdb_indices = array.res_id[starts]

    pdb_df = pd.DataFrame({
        "resi": pdb_indices,
        "resn": pdb_names
    })

    print(pdb_df)

    mut_data = pd.read_csv(mut_data_path)
    mut_data = mut_data[['wildtype', 'position']].drop_duplicates()
    print(mut_data)
    #raise "Debug"

    myrunner_mut = runner.Runner(
        pdb_id=pdb_id,
        pdb_path=mmcif_path,
        membrane_protein=True,
        mutation_data_path=mut_data_path,
        mutation_data_chain='A'
    )
    assert myrunner_mut.mutation_data is not None
    assert 'effect' in myrunner_mut.context.residue_table.columns.tolist()





