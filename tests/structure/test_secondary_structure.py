import numpy as np
import pandas as pd
import pytest
from biotite.structure.io.pdbx import CIFFile

from src.pipeline.context import Config, Context
from src.structure.secondary_structure import (
    _write_temp_mmcif,
    define_membrane_secondary_structure,
    define_soluble_secondary_structure,
    get_secondary_structure_annotations,
    make_contiguous_group_labels,
)
from tests.test_utils import AA_LIST, _make_chain, _make_residue_table

np.random.seed(42)
import random


def test_make_contiguous_group_labels():
    input = ['A', 'A', 'B', 'B', 'A', 'A', 'A', 'C', 'C', 'B']
    expected_output = ['A_1', 'A_1', 'B_1', 'B_1', 'A_2', 'A_2', 'A_2', 'C_1', 'C_1', 'B_2']

    output = make_contiguous_group_labels(input)

    assert output == expected_output


def test_get_secondary_structure_annotations():
    aa_list = random.choices(AA_LIST, k=10)
    arr = _make_chain(aa_list=aa_list, chain_id='A')
    context = Context(array=arr, config=Config())
    ss_df = get_secondary_structure_annotations(context)
    assert 'ss_group' in ss_df.columns.tolist()
    assert len(ss_df) == len(aa_list)
    assert all(ss in {'a', 'b', 'c'} for ss in ss_df['sse'].tolist())


def test_get_secondary_structure_annotations_uses_mkdssp_backend(monkeypatch):
    aa_list = random.choices(AA_LIST, k=6)
    arr = _make_chain(aa_list=aa_list, chain_id='A')
    context = Context(array=arr, config=Config())
    context.extras["ss_backend"] = "mkdssp"

    mock_ss_df = pd.DataFrame({
        "chain": ["A"] * 6,
        "resi": [1, 2, 3, 4, 5, 6],
        "sse": ["a", "a", "b", "b", "c", "c"],
    })
    mock_dssp_df = pd.DataFrame({
        "chain": ["A"] * 6,
        "resi": [1, 2, 3, 4, 5, 6],
        "dssp_acc": [10, 20, 30, 40, 50, 60],
    })

    import src.structure.secondary_structure as ss_module
    monkeypatch.setattr(ss_module, "_annotate_with_mkdssp", lambda _ctx: (mock_ss_df, mock_dssp_df))

    ss_df = get_secondary_structure_annotations(context)
    assert "ss_group" in ss_df.columns
    assert "dssp_output" in context.extras


def test_get_secondary_structure_annotations_uses_pydssp_backend(monkeypatch):
    aa_list = random.choices(AA_LIST, k=6)
    arr = _make_chain(aa_list=aa_list, chain_id='A')
    context = Context(array=arr, config=Config())
    context.extras["ss_backend"] = "pydssp"
    context.extras["dssp_output"] = pd.DataFrame({"dummy": [1]})

    mock_ss_df = pd.DataFrame({
        "chain": ["A"] * 6,
        "resi": [1, 2, 3, 4, 5, 6],
        "sse": ["c"] * 6,
    })

    import src.structure.secondary_structure as ss_module
    monkeypatch.setattr(ss_module, "_annotate_with_pydssp", lambda _ctx: mock_ss_df)

    ss_df = get_secondary_structure_annotations(context)
    assert "ss_group" in ss_df.columns
    assert "dssp_output" not in context.extras


def test_get_secondary_structure_annotations_unknown_backend():
    aa_list = random.choices(AA_LIST, k=4)
    arr = _make_chain(aa_list=aa_list, chain_id='A')
    context = Context(array=arr, config=Config())
    context.extras["ss_backend"] = "not_a_backend"
    with pytest.raises(ValueError, match="Unknown secondary-structure backend"):
        get_secondary_structure_annotations(context)


def test_write_temp_mmcif_accepts_long_residue_names():
    """PDB serialization rejects res_name > 3 chars; mmCIF temp input for mkdssp must not."""
    arr = _make_chain(["ALA"], chain_id="A")
    arr.res_name[:] = "TOOLONG"
    context = Context(array=arr, config=Config())
    path = _write_temp_mmcif(context)
    try:
        assert path.suffix == ".cif"
        body = path.read_text(encoding="utf-8")
        assert "atom_site" in body
    finally:
        path.unlink(missing_ok=True)


def test_write_temp_mmcif_includes_polymer_metadata_categories():
    arr = _make_chain(["ALA", "GLY"], chain_id="A")
    context = Context(array=arr, config=Config())
    path = _write_temp_mmcif(context)
    try:
        cif = CIFFile.read(str(path))
        block = next(iter(cif.values()))
        assert "entry" in block
        assert "entity" in block
        assert "entity_poly" in block
        assert "entity_poly_seq" in block
    finally:
        path.unlink(missing_ok=True)


def test_define_membrane_secondary_structure():
    # Create a mock residue table with the following topology:
    # Extracellular length 4, membrane length 5, cytoplasmic length 4, membrane length 6, extracellular length 6.
    # There is a single alphahelix in the first membrane region that overlaps with the extracellular region by 1,
    # and terminates 2 before the end of membrane
    # The second membrane region has a beta sheet of len 2 in the middle of alpha helices

    pdbtm_region = (['extracellular'] * 4 + ['membrane_spanning'] * 5 + ['cytoplasmic'] * 4 +
                    ['membrane_spanning'] * 6 + ['extracellular'] * 6)
    pdbtm_region_detailed = make_contiguous_group_labels(pdbtm_region)

    ss_annotation = (['c'] * 3 + ['a'] * 4 + ['c'] * 6 + ['a'] * 2 + ['b'] * 2 + ['a'] * 3 + ['c'] * 5)

    ss_df = pd.DataFrame({
        'chain': ['A'] * len(ss_annotation),
        'resi': list(range(1, len(ss_annotation) + 1)),
        'sse': ss_annotation,
        'ss_group': make_contiguous_group_labels(ss_annotation)
    })

    residue_table = pd.DataFrame({
        'chain': ['A'] * len(pdbtm_region),
        'resi': list(range(1, len(pdbtm_region) + 1)),
        'resn': ['ALA'] * len(ss_annotation),
        'pdbtm_region': pdbtm_region,
        'pdbtm_region_detailed': pdbtm_region_detailed
    })

    expected_annotation = (['extracellular_loop_1'] * 3 + ['TMD_1'] * 4 + ['cytoplasmic_loop_1'] * 6 +
                           ['TMD_2'] * 7 + ['extracellular_loop_2'] * 5)

    output_df = define_membrane_secondary_structure(residue_table, ss_df)

    assert output_df['ss_domains'].tolist() == expected_annotation


def test_define_soluble_secondary_structure():
    ss_annotation = ['a_1', 'a_1', 'b_1', 'b_1', 'a_2', 'a_2', 'a_2', 'c_1', 'c_1', 'b_2', 'c_2', 'c_2', 'c_2']
    residue_table = _make_residue_table(num_residues=len(ss_annotation), num_chains=1, make_muts=False)
    residue_table.rename(columns={'resi_struct': 'resi'}, inplace=True)
    ss_df = residue_table[['chain', 'resi']].copy()
    ss_df['ss_group'] = ss_annotation
    output_df = define_soluble_secondary_structure(residue_table, ss_df)

    # 'b2' should be merged into 'c1', 'c2' merged into 'c1'
    assert output_df['ss_group'].tolist() == ['a_1', 'a_1', 'b_1', 'b_1', 'a_2', 'a_2', 'a_2', 'c_1', 'c_1', 'c_1', 'c_1', 'c_1', 'c_1']
    assert output_df['ss_domains'].tolist() == ['alpha-helix_1', 'alpha-helix_1', 'beta-sheet_1', 'beta-sheet_1', 'alpha-helix_2', 'alpha-helix_2', 'alpha-helix_2', 'coil_1', 'coil_1', 'coil_1', 'coil_1', 'coil_1', 'coil_1']

def test_define_soluble_secondary_structure_edge_cases():
    # a1 is first in the chain, shouldn't be merged. b_3 is last in the chain, shouldn't be merged. A_3 isn't sandwhiched between two secondary structure elements of the same type, so shouldn't be merged.
    ss_annotation = ['a_1', 'b_1', 'b_1', 'a_2', 'a_2', 'a_2', 'c_1', 'c_1', 'b_2', 'c_2', 'c_2', 'a_3', 'b_3']
    residue_table = _make_residue_table(num_residues=len(ss_annotation), num_chains=1, make_muts=False)
    residue_table.rename(columns={'resi_struct': 'resi'}, inplace=True)
    ss_df = residue_table[['chain', 'resi']].copy()
    ss_df['ss_group'] = ss_annotation
    output_df = define_soluble_secondary_structure(residue_table, ss_df)

    assert output_df['ss_group'].tolist() == ['a_1', 'b_1', 'b_1', 'a_2', 'a_2', 'a_2', 'c_1', 'c_1', 'c_1', 'c_1', 'c_1', 'a_3', 'b_3']
    assert output_df['ss_domains'].tolist() == ['alpha-helix_1', 'beta-sheet_1', 'beta-sheet_1', 'alpha-helix_2', 'alpha-helix_2', 'alpha-helix_2', 'coil_1', 'coil_1', 'coil_1', 'coil_1', 'coil_1', 'alpha-helix_3', 'beta-sheet_3']
