import pytest
import tomli
import numpy as np
from pathlib import Path
from src.structure.structure_context import Config, Context, residue_table, load_structure
from tests.test_utils import _make_config_file, _make_chain, _make_aaindex_data, _write_mmcif_file

def test_config(tmp_path):
    config_args = {'pdb_id': "1abc", 'membrane_protein': True, 'mutation_data_path': "data/aaindex_parsed_small.csv",
                   'mutation_data_chain': "A", 'aaindex_path': "data/aaindex_parsed_small.csv"}

    _ = Config(**config_args)

    config_file_path = tmp_path / "test_config.toml"
    _make_config_file(config_file_path)

    # Load TOML file
    with config_file_path.open("rb") as f:
        loaded_config_data = tomli.load(f)

    config_from_file = Config(**loaded_config_data)

    with pytest.raises(ValueError, match="Mutation data file not found at nonexistent.csv"):
        bad_config_args = config_args.copy()
        bad_config_args["mutation_data_path"] = "nonexistent.csv"
        Config(**bad_config_args)

    with pytest.raises(ValueError, match="If mutation_data_path is provided, mutation_data_chain must also be provided."):
        bad_config_args = config_args.copy()
        bad_config_args["mutation_data_chain"] = None
        Config(**bad_config_args)

    with pytest.raises(ValueError, match="AA index data file not found at nonexistent.csv"):
        bad_config_args = config_args.copy()
        bad_config_args["aaindex_path"] = "nonexistent.csv"
        Config(**bad_config_args)


def test_context(tmp_path):
    # Create a test chain
    arr = _make_chain(aa_list=['ALA', 'CYS', 'ASP'], chain_id='A')

    context = Context(array=arr)

    assert context.neighbor_cache == {}
    assert context.residue_table is not None
    assert len(context.residue_table) == 3
    assert context.config is not None
    assert context.config.aaindex_path == 'data/aaindex_parsed_small.csv'
    assert context.config.kidera_path == 'data/kidera_factors.csv'
    assert context.extras['kidera'] is not None

    # Test loading AA index data
    aaindex_path = tmp_path / "aaindex.csv"
    aaindex_data = _make_aaindex_data(accessions=['AA1', 'AA2'])
    aaindex_data.to_csv(aaindex_path, index=False)

    config = Config(aaindex_path=aaindex_path, membrane_protein=True)
    context_with_aaindex = Context(array=arr, config=config)

    assert 'aaindex' in context_with_aaindex.extras
    assert context_with_aaindex.extras['aaindex'].equals(aaindex_data)
    assert context_with_aaindex.config.membrane_protein is True

def test_residue_table_altloc():
    """Test that residue_table properly captures altloc information."""
    # Create chain with mixed altloc identifiers
    aa_list = ['ALA', 'CYS', 'ASP', 'GLU']
    altlocs = ['A', '', 'B', '']
    arr = _make_chain(aa_list=aa_list, chain_id='A', altloc=altlocs)
    
    res_table = residue_table(arr)
    
    # Check that altloc column exists
    assert 'altloc' in res_table.columns
    
    # Check that altloc values are correct
    assert res_table['altloc'].iloc[0] == 'A'
    assert res_table['altloc'].iloc[1] == ''
    assert res_table['altloc'].iloc[2] == 'B'
    assert res_table['altloc'].iloc[3] == ''
    
    # Check other columns are correct
    assert list(res_table['resn']) == ['ALA', 'CYS', 'ASP', 'GLU']
    assert list(res_table['resi']) == [1, 2, 3, 4]
    assert all(res_table['chain'] == 'A')


def test_context_with_altloc():
    """Test that Context properly handles arrays with altloc information."""
    aa_list = ['SER', 'THR', 'TYR']
    altlocs = ['A', 'A', '']
    arr = _make_chain(aa_list=aa_list, chain_id='B', altloc=altlocs)
    
    context = Context(array=arr)
    
    # Check residue_table has altloc column
    assert 'altloc' in context.residue_table.columns
    
    # Check altloc values are preserved
    assert context.residue_table['altloc'].iloc[0] == 'A'
    assert context.residue_table['altloc'].iloc[1] == 'A'
    assert context.residue_table['altloc'].iloc[2] == ''


def test_load_structure_from_pdb_id():
    """Test loading structure from PDB ID by fetching from RCSB."""
    pdb_id = '8smv'
    
    arr = load_structure(pdb_id=pdb_id)
    
    assert arr is not None
    assert arr.array_length() > 0


def test_load_structure_pdb_format(tmp_path):
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
    
    # Test loading PDB format
    arr = load_structure(path=pdb_path, pdb_ext="pdb")
    
    assert arr is not None
    assert arr.array_length() > 0


def test_load_structure_cif_format(tmp_path):
    """Test loading structure with CIF format to verify extension handling."""
    residues = ['ALA', 'VAL', 'GLY', 'SER', 'THR']
    cif_path = tmp_path / "test_structure.cif"
    _write_mmcif_file(file_path=cif_path, pdb_id="TEST", chains={"A": residues})
    
    arr = load_structure(path=cif_path, pdb_ext="cif")
    
    assert arr is not None
    assert arr.array_length() > 0
