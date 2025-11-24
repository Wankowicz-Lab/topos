import pytest
import tomli
from src.structure.structure_context import Config
from tests.test_utils import _make_config_file

def test_config(tmp_path):
    config_args = {'pdb_id': "1abc", 'membrane_protein': True, 'mutation_data_path': "data/aaindex_parsed_small.csv",
                   'mutation_data_chain': "A", 'aa_index_path': "data/aaindex_parsed_small.csv"}

    _ = Config(**config_args)

    config_file_path = tmp_path / "test_config.toml"
    _make_config_file(config_file_path)

    # 3️⃣ Load TOML file
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
        bad_config_args["aa_index_path"] = "nonexistent.csv"
        Config(**bad_config_args)