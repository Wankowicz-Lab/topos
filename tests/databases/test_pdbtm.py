import numpy as np
import pandas as pd
import pytest

from src.databases import pdbtm
from tests.test_utils import _make_residue_table


@pytest.mark.parametrize(
    ("code", "expected"),
    [
        ("H", "transmembrane_helix"),
        ("1", "side1"),
        ("2", "side2"),
        ("U", "unknown"),
        ("I", "beta_barrel_interior"),
        ("N", "beta_barrel_interior"),
    ],
)
def test_describe_pdbtm_region_known_codes(code, expected):
    assert pdbtm.describe_pdbtm_region(code) == expected


def test_describe_pdbtm_region_unmapped_code():
    assert pdbtm.describe_pdbtm_region("not_defined") == "NOT_DEFINED"


def test_fetch_pdbtm_annotation():
    pdb_id = "8smv"  # Example PDB ID known to be in PDBTM
    df, mat = pdbtm.fetch_pdbtm_annotation(pdb_id)

    # Basic checks
    expected_cols = ['chain', 'type', 'seq_beg', 'seq_end', 'pdb_beg', 'pdb_end']

    for col in expected_cols:
        assert col in df.columns

    assert set(df.type.unique()).issubset(
        {'transmembrane_helix', 'side1', 'side2', 'unknown'}
    )

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
    # rename columns, resi_struct hasn't been created in pipeline yet
    residue_table.rename(columns={'resi_struct': 'resi', 'resn_struct': 'resn'}, inplace=True)
    pdbtm_regions = pd.DataFrame({
        'chain': ['A', 'A', 'B', 'B'],
        'type': ['transmembrane_helix', 'side1', 'side2', 'side2'],
        'pdb_beg': [1, 4, 3, 7],
        'pdb_end': [3, 5, 6, 8]
    })

    expected_region = ['transmembrane_helix'] * 3 + ['side1'] * 2 + ['side2'] * 5
    expected_region_detail = (['transmembrane_helix_1'] * 3 + ['side1_1'] * 2 +
                              ['side2_1'] * 4 + ['side2_2'])

    merged = pdbtm.add_pdbtm_regions(residue_table, pdbtm_regions)

    assert merged['pdbtm_region'].tolist() == expected_region
    assert merged['pdbtm_region_detailed'].tolist() == expected_region_detail
