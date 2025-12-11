import numpy as np
import pandas as pd
import pytest

from src.structure import pdbtm
from tests.test_utils import _make_residue_table

def test_describe_pdbtm_region():
    inputs = ['H', '1', '2', 'U', 'not_defined']
    expected = ['membrane_spanning', 'cytoplasmic', 'extracellular', 'unknown', 'not_defined']

    for inp, exp in zip(inputs, expected):
        assert pdbtm.describe_pdbtm_region(inp) == exp


def test_fetch_pdbtm_annotation():
    pdb_id = "8smv"  # Example PDB ID known to be in PDBTM
    df, mat = pdbtm.fetch_pdbtm_annotation(pdb_id)

    # Basic checks
    expected_cols = ['chain', 'type', 'seq_beg', 'seq_end', 'pdb_beg', 'pdb_end']

    for col in expected_cols:
        assert col in df.columns

    assert set(df.type.unique()).issubset({'membrane_spanning', 'cytoplasmic', 'extracellular', 'unknown'})

    assert mat.shape == (4, 4)  # Transformation matrix should be 4x4


def test_transform_coordinates():
    coords = np.array([[1.0, 2.0, 3.0],
                       [4.0, 5.0, 6.0],
                       [7.0, 8.0, 9.0]])

    # Identity matrix should return the same coordinates
    identity_matrix = np.eye(4)
    transformed_identity = pdbtm.transform_coordinates(coords, identity_matrix)
    assert np.allclose(transformed_identity, coords)

    # Translation matrix
    translation_matrix = np.array([[1, 0, 0, 10],
                                   [0, 1, 0, 20],
                                   [0, 0, 1, 30],
                                   [0, 0, 0, 1]])
    transformed_translation = pdbtm.transform_coordinates(coords, translation_matrix)
    expected_translation = coords + np.array([10, 20, 30])
    assert np.allclose(transformed_translation, expected_translation)

    # Rotation matrix (90 degrees around z-axis, x = -y, y = x)
    theta = np.pi / 2  # 90 degrees
    rotation_matrix = np.array([[np.cos(theta), -np.sin(theta), 0, 0],
                                    [np.sin(theta), np.cos(theta), 0, 0],
                                    [0, 0, 1, 0],
                                    [0, 0, 0, 1]])
    transformed_rotation = pdbtm.transform_coordinates(coords, rotation_matrix)
    expected_rotation = np.array([[-2.0, 1.0, 3.0],
                                  [-5.0, 4.0, 6.0],
                                  [-8.0, 7.0, 9.0]])

    assert np.allclose(transformed_rotation, expected_rotation)

    with pytest.raises(ValueError, match="Transformation matrix must be of shape 4x4"):
        invalid_matrix = np.array([[1, 0, 0],
                                    [0, 1, 0],
                                    [0, 0, 1]])
        pdbtm.transform_coordinates(coords, invalid_matrix)

    with pytest.raises(ValueError, match="Coordinates must be of shape Nx3"):
        invalid_coords = np.array([[1.0, 2.0],
                                    [3.0, 4.0]])
        pdbtm.transform_coordinates(invalid_coords, identity_matrix)


def test_annotate_pdbtm_detailed():
    input_chains = ['A'] * 10 + ['B'] * 2
    input_types1 = (['unknown', 'typeA', 'typeB', 'typeA', 'typeC', 'typeA','typeB',
                     'unknown', 'typeB', 'unknown'])
    input_types2 = ['unknown', 'typeD']
    expected1 = (['protein_start', 'typeA_1', 'typeB_1', 'typeA_2', 'typeC_1', 'typeA_3', 'typeB_2',
                  'unknown_2', 'typeB_3', 'protein_end'])
    expected2 = ['unknown_1', 'typeD_1']

    input_df = pd.DataFrame({
        'chain': input_chains,
        'type': input_types1 + input_types2,
    })

    output = pdbtm.annotate_pdbtm_detailed(input_df)
    assert output['detailed_type'].tolist() == expected1 + expected2


def test_add_pdbtm_regions():
    residue_table = _make_residue_table(num_residues=5, num_chains=2, make_muts=False, start_resis=[1, 3])
    pdbtm_regions = pd.DataFrame({
        'chain': ['A', 'A', 'B', 'B'],
        'type': ['membrane_spanning', 'cytoplasmic', 'extracellular', 'extracellular'],
        'pdb_beg': [1, 4, 3, 7],
        'pdb_end': [3, 5, 6, 8]
    })

    expected_region = ['membrane_spanning'] * 3 + ['cytoplasmic'] * 2 + ['extracellular'] * 5
    expected_region_detail = (['membrane_spanning_1'] * 3 + ['cytoplasmic_1'] * 2 +
                              ['extracellular_1'] * 4 + ['extracellular_2'])

    merged = pdbtm.add_pdbtm_regions(residue_table, pdbtm_regions)
    print(merged)

    assert merged['pdbtm_region'].tolist() == expected_region
    assert merged['pdbtm_region_detailed'].tolist() == expected_region_detail


def test_make_contiguous_group_labels():
    input = ['A', 'A', 'B', 'B', 'A', 'A', 'A', 'C', 'C', 'B']
    expected_output = ['A_1', 'A_1', 'B_1', 'B_1', 'A_2', 'A_2', 'A_2', 'C_1', 'C_1', 'B_2']

    output = pdbtm.make_contiguous_group_labels(input)

    assert output == expected_output


def test_define_secondary_structure():
    # Create a mock residue table with the following topology:
    # Extracellular length 4, membrane length 5, cytoplasmic length 4, membrane length 6, extracellular length 6.
    # There is a single alphahelix in the first membrane region that overlaps with the extracellular region by 1,
    # and terminates 2 before the end of membrane
    # The second membrane region has a beta sheet of len 2 in the middle of alpha helices

    pdbtm_region = (['extracellular'] * 4 + ['membrane_spanning'] * 5 + ['cytoplasmic'] * 4 +
                    ['membrane_spanning'] * 6 + ['extracellular'] * 6)
    pdbtm_region_detailed = (pdbtm.make_contiguous_group_labels(pdbtm_region))

    ss_annotation = (['c'] * 3 + ['a'] * 4 + ['c'] * 6 + ['a'] * 2 + ['b'] * 2 + ['a'] * 3 + ['c'] * 5)

    ss_df = pd.DataFrame({
        'chain': ['A'] * len(ss_annotation),
        'resi_struct': list(range(1, len(ss_annotation) + 1)),
        'sse': ss_annotation
    })

    residue_table = pd.DataFrame({
        'chain': ['A'] * len(pdbtm_region),
        'resi_struct': list(range(1, len(pdbtm_region) + 1)),
        'resn_struct': ['ALA'] * len(ss_annotation),
        'pdbtm_region': pdbtm_region,
        'pdbtm_region_detailed': pdbtm_region_detailed
    })

    expected_annotation = (['extracellular_loop_1'] * 3 + ['TMD_1'] * 4 + ['cytoplasmic_loop_1'] * 6 +
                           ['TMD_2'] * 7 + ['extracellular_loop_2'] * 5)

    output_df = pdbtm.define_secondary_structure(residue_table, ss_df)

    assert output_df['ss_domains'].tolist() == expected_annotation