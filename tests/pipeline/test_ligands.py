import biotite.structure as struc
import numpy as np
import pandas as pd
import tomli

from topos.pipeline.context import Config, Context
from topos.pipeline.ligands import (
    calculate_protein_ligand_interactions,
    find_ligands,
    format_ligand_id,
)
from topos.structure.structure_context import load_structure
from tests.test_utils import _make_atoms, _make_config_file, _make_residue, _write_mmcif_file


def test_format_ligand_id():
    """Test format_ligand_id produces canonical ligand ID for matching."""
    assert format_ligand_id("A", 1, "ATP") == "A:1:ATP"
    assert format_ligand_id(" B ", 2, " NAG ") == "B:2:NAG"
    assert format_ligand_id("A", 10, None) == "A:10:"


def test_find_ligands():
    """Test find_ligands identifies ligands when run on PDB 8EFO (requires network)."""
    arr = load_structure(pdb_id="8EFO")
    ligands = find_ligands(arr)
    assert len(ligands) >= 1, "8EFO should have at least one ligand identified"

    ligands_with_cholesterol = find_ligands(arr, exclude_cholesterol=False)
    assert len(ligands_with_cholesterol) > len(ligands), "With cholesterol excluded, should have fewer ligands"


def test_find_ligands_empty_when_no_hetero():
    """Test find_ligands returns empty list when structure has no hetero atoms."""
    arr = _make_residue("ALA", res_id=1, chain_id="A")
    if "hetero" in arr.get_annotation_categories():
        arr.del_annotation("hetero")
    ligands = find_ligands(arr)
    assert ligands == []


def test_find_ligands_inclusion_criteria():
    """Test that find_ligands inclusion criteria work correctly."""
    protein = _make_atoms(["N", "CA", "C"], [[0, 0, 0], [1, 0, 0], [2, 0, 0]], res_name="ALA", res_id=1, chain_id="A")
    mg = _make_atoms(["MG"], [[10, 10, 10]], res_name="MG", res_id=2, chain_id="A")
    mse = _make_atoms(["N", "CA", "C"], [[20, 20, 20]] * 3, res_name="MSE", res_id=3, chain_id="A")

    arr = struc.concatenate([protein, mg, mse])
    arr.set_annotation("hetero", np.array([False, False, False, True, True, True, True]))
    permissive_ligands = find_ligands(arr, exclude_ions=False, exclude_protein_mods=False)
    assert (
        ("A", 2, "MG") in permissive_ligands
        and ("A", 3, "MSE") in permissive_ligands
    )
    restrictive_ligands = find_ligands(arr, exclude_ions=True, exclude_protein_mods=True)
    assert (
        ("A", 2, "MG") not in restrictive_ligands
        and ("A", 3, "MSE") not in restrictive_ligands
    )


def test_calculate_protein_ligand_interactions(tmp_path):
    """Test calculate_protein_ligand_interactions with hetero-based ligands and partner_residue_key."""
    residues = ["ALA", "VAL", "GLY", "SER", "THR"]
    mmcif_path = tmp_path / "test_structure.cif"
    _write_mmcif_file(file_path=mmcif_path, pdb_id="TEST", chains={"A": residues, "B": residues})

    config_path = tmp_path / "config_ligand.toml"
    _make_config_file(config_path, pdb_id="test")
    with config_path.open("rb") as f:
        config = Config(**tomli.load(f))
    config.pdb_path = mmcif_path

    arr = load_structure(path=mmcif_path, pdb_id="test")
    if "hetero" not in arr.get_annotation_categories():
        hetero = np.zeros(arr.array_length(), dtype=bool)
        hetero[arr.chain_id == "B"] = True
        arr.set_annotation("hetero", hetero)
    else:
        arr.hetero[arr.chain_id == "B"] = True

    context = Context(array=arr, config=config)
    context.residue_table.rename(
        columns={"resn": "resn_struct", "resi": "resi_struct"},
        inplace=True,
    )

    contacting_df = pd.DataFrame({
        "chain": ["A"],
        "resi_struct": [1],
        "resn_struct": ["ALA"],
        "partner_residue_key": ["B:1:ALA"],
    })

    result = calculate_protein_ligand_interactions(context, contacting_df)

    assert "ligand_B_1_ALA_interactions" in result.columns
    assert result.shape[0] == context.residue_table.shape[0]
    vals = result["ligand_B_1_ALA_interactions"].dropna().unique()
    assert len(vals) >= 1
    assert set(vals).issubset({"contact", "binding site", "second shell"})

    arr.hetero[:] = False
    out_skip = calculate_protein_ligand_interactions(context, contacting_df)
    assert out_skip.shape[0] == context.residue_table.shape[0]
    assert "ligand_B_1_ALA_interactions" not in out_skip.columns
