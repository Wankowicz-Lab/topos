import numpy as np
import pandas as pd
import random

from src.structure import metrics
from src.structure.structure_context import Config
from tests.test_utils import _make_chain, AA_LIST, _make_residue_table

import biotite.structure as struc

np.random.seed(42)


def test_calculate_secondary_structure():
    # Create a test chain with random coordinates
    aa_list = random.choices(AA_LIST, k=10)
    arr = _make_chain(aa_list=aa_list, chain_id='A')

    sse = metrics.calculate_secondary_structure(arr)

    assert len(sse) == len(aa_list)
    assert all(ss in {'a', 'b', 'c'} for ss in sse)


def test_calculate_sasa():
    # Create a test chain with random coordinates
    aa_list = random.choices(AA_LIST, k=10)
    arr = _make_chain(aa_list=aa_list, chain_id='A')

    context = metrics.Context(array=arr)
    context.vdw_radii = "ProtOr"

    output = metrics.calculate_sasa(context)

    assert 'sasa' in output.columns.tolist()
    assert len(output) == len(aa_list)
    assert all(output['sasa'] >= 0)


def test_calculate_kyte_doolittle():
    # Create a test chain with random coordinates
    aa_list = random.choices(AA_LIST, k=10)
    arr = _make_chain(aa_list=aa_list, chain_id='A')

    context = metrics.Context(array=arr)

    output = metrics.calculate_kyte_doolittle(context)

    assert 'kyte_doolittle' in output.columns.tolist()
    assert len(output) == len(aa_list)


def test_calculate_membrane_distance():
    # Create a test chain with varying z-coordinates
    z_values = list(range(-25, 25, 5))
    coords = [[np.random.randint(10), np.random.randint(10), z] for z in z_values]
    aa_list = random.choices(AA_LIST, k=len(z_values))
    arr = _make_chain(aa_list=aa_list, coords=coords, chain_id='A')

    class MockContext:
        def __init__(self, array):
            self.array = array
            self.config = Config()

    context = MockContext(array=arr)

    distances = metrics.calculate_membrane_distance(context)

    # Expected distance is absolute z minus membrane thickness
    expected_distances = np.abs(np.array(z_values)) - 15.0

    assert np.allclose(distances['distance_from_membrane_edge'], expected_distances)


def test_define_secondary_structure():
    # Create input data
    residue_table = _make_residue_table(num_chains=1, make_muts=False)
    residue_table['pdbtm_region'] = 'membrane_spanning'
    residue_table['pdbtm_region_detailed'] = 'TM1'
    aa_list = residue_table.resn.tolist()
    arr = _make_chain(aa_list=aa_list, chain_id='A')

    context = metrics.Context(array=arr, config=Config())
    context.residue_table = residue_table

    output = metrics.define_secondary_structure(context)
    assert 'ss_domains' not in output.columns.tolist()
    assert 'ss_group' in output.columns.tolist()

    context = metrics.Context(array=arr, config=Config(membrane_protein=True))
    context.residue_table = residue_table
    output = metrics.define_secondary_structure(context)
    assert 'ss_domains' in output.columns.tolist()
    assert 'ss_group' in output.columns.tolist()
