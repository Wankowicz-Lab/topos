import numpy as np
import pandas as pd
from numpy.f2py.crackfortran import expectbegin

from src.structure import pdbtm
from tests.test_utils import _make_residue_table

def test_describe_pdbtm_region():
    inputs = ['H', '1', '2', 'U', 'not_defined']
    expected = ['membrane_spanning', 'cytoplasmic', 'extracellular', 'unknown_or_unresolved', 'not_defined']

    for inp, exp in zip(inputs, expected):
        assert pdbtm.describe_pdbtm_region(inp) == exp


def test_fetch_pdbtm_annotation():
    pdb_id = "8smv"  # Example PDB ID known to be in PDBTM
    df, cmap = pdbtm.fetch_pdbtm_annotation(pdb_id)

    # Basic checks
    expected_cols = ['chain', 'type', 'seq_beg', 'seq_end', 'pdb_beg', 'pdb_end']

    for col in expected_cols:
        assert col in df.columns

    assert set(df.type.unique()).issubset({'membrane_spanning', 'cytoplasmic', 'extracellular', 'unknown_or_unresolved'})


def test_annotate_pdbtm_detailed():
    input_chains = ['A'] * 10 + ['B'] * 2
    input_types1 = (['unknown_or_unresolved', 'typeA', 'typeB', 'typeA', 'typeC', 'typeA','typeB',
                     'unknown_or_unresolved', 'typeB', 'unknown_or_unresolved'])
    input_types2 = ['unknown_or_unresolved', 'typeD']
    expected1 = (['protein_start', 'typeA_1', 'typeB_1', 'typeA_2', 'typeC_1', 'typeA_3', 'typeB_2',
                  'unknown_or_unresolved_2', 'typeB_3', 'protein_end'])
    expected2 = ['unknown_or_unresolved_1', 'typeD_1']

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