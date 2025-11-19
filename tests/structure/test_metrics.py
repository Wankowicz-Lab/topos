import numpy as np
import pandas as pd
import random

from src.structure import metrics
from tests.test_utils import _make_chain, AA_LIST, _make_residue_table

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


def test_define_secondary_structure():
    # Create input data
    residue_table = _make_residue_table(num_chains=1, make_muts=False)
    residue_table['pdbtm_region'] = 'membrane_spanning'
    residue_table['pdbtm_region_detailed'] = 'TM1'
    aa_list = residue_table.resn.tolist()
    arr = _make_chain(aa_list=aa_list, chain_id='A')

    context = metrics.Context(array=arr)
    context.residue_table = residue_table

    output = metrics.define_secondary_structure(context)
    assert 'ss_domains' not in output.columns.tolist()
    assert 'ss_group' in output.columns.tolist()

    context.membrane_protein = True
    output = metrics.define_secondary_structure(context)
    assert 'ss_domains' in output.columns.tolist()
    assert 'ss_group' in output.columns.tolist()
