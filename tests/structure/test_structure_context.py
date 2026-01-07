import pytest
import tomli
import numpy as np
from src.structure.structure_context import Config, Context, residue_table
from tests.test_utils import _make_config_file, _make_chain, _make_aaindex_data

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
