import numpy as np
import pandas as pd
import random

from src.structure import metrics
from tests.test_utils import _make_chain, AA_LIST

import biotite.structure as struc

np.random.seed(42)

def test_calculate_membrane_distance():
    # Create a test chain with varying z-coordinates
    z_values = list(range(-25, 25, 5))
    coords = [[np.random.randint(10), np.random.randint(10), z] for z in z_values]
    aa_list = random.choices(AA_LIST, k=len(z_values))
    arr = _make_chain(aa_list=aa_list, coords=coords, chain_id='A')
    distances = metrics.calculate_membrane_distance(arr, membrane_thickness=15.0)

    # Expected distance is absolute z minus membrane thickness
    expected_distances = np.abs(np.array(z_values)) - 15.0

    assert np.allclose(distances, expected_distances)


def test_calculate_sidechain_angle_from_center():
    # TODO: correcntess test with known angles
    input_arr = _make_chain(aa_list=['ALA', 'ARG', 'ASN', 'ASP', 'CYS', 'GLU', 'GLN', 'THR'])

    angles = metrics.calculate_sidechain_angle_from_center(input_arr)

    assert len(angles) == len(struc.get_residue_starts(input_arr))

