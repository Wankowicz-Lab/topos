"""Tests for structure metrics module."""
import numpy as np
import pandas as pd
import random
import pytest
from pathlib import Path
import tempfile

from src.structure import metrics
from src.structure.structure_context import Config, Context
from tests.test_utils import _make_chain, AA_LIST, _make_residue_table

import biotite.structure as struc

# Seed RNGs for deterministic tests
np.random.seed(42)
random.seed(42)

# Real protein structure data for testing residue packing
# PDB ATOM records for residues VAL 165, SER 166, SER 167, PHE 168, LEU 169 (chain A)
PDB_ATOM_RECORDS = """ATOM   2461  N   VAL A 165     -33.684 -38.677   3.695  1.00 13.65           N
ATOM   2462  CA  VAL A 165     -32.604 -39.151   4.551  1.00 14.21           C
ATOM   2463  C   VAL A 165     -31.766 -40.195   3.825  1.00 14.75           C
ATOM   2464  O   VAL A 165     -30.534 -40.106   3.781  1.00 13.43           O
ATOM   2465  CB  VAL A 165     -33.182 -39.700   5.867  1.00 14.42           C
ATOM   2466  CG1 VAL A 165     -32.107 -40.416   6.666  1.00 14.42           C
ATOM   2467  CG2 VAL A 165     -33.823 -38.579   6.677  1.00 16.39           C
ATOM   2468  H   VAL A 165     -34.472 -38.879   3.974  1.00 13.65           H
ATOM   2469  HA  VAL A 165     -32.025 -38.404   4.769  1.00 14.21           H
ATOM   2470  HB  VAL A 165     -33.874 -40.346   5.658  1.00 14.42           H
ATOM   2471 HG11 VAL A 165     -32.408 -40.508   7.584  1.00 14.42           H
ATOM   2472 HG12 VAL A 165     -31.953 -41.293   6.280  1.00 14.42           H
ATOM   2473 HG13 VAL A 165     -31.291 -39.892   6.638  1.00 14.42           H
ATOM   2474 HG21 VAL A 165     -34.168 -38.946   7.506  1.00 16.39           H
ATOM   2475 HG22 VAL A 165     -33.152 -37.904   6.867  1.00 16.39           H
ATOM   2476 HG23 VAL A 165     -34.546 -38.190   6.161  1.00 16.39           H
ATOM   2477  N   SER A 166     -32.425 -41.202   3.243  1.00 13.39           N
ATOM   2478  CA  SER A 166     -31.699 -42.335   2.677  1.00 14.13           C
ATOM   2479  C   SER A 166     -30.855 -41.932   1.479  1.00 18.19           C
ATOM   2480  O   SER A 166     -29.848 -42.587   1.187  1.00 19.17           O
ATOM   2481  CB  SER A 166     -32.669 -43.447   2.277  1.00 17.00           C
ATOM   2482  OG  SER A 166     -33.297 -44.014   3.410  1.00 20.45           O
ATOM   2483  H   SER A 166     -33.280 -41.250   3.165  1.00 13.39           H
ATOM   2484  HA  SER A 166     -31.102 -42.692   3.353  1.00 14.13           H
ATOM   2485  HB2 SER A 166     -33.349 -43.076   1.694  1.00 17.00           H
ATOM   2486  HB3 SER A 166     -32.177 -44.141   1.811  1.00 17.00           H
ATOM   2487  HG  SER A 166     -32.720 -44.341   3.925  1.00 20.45           H
ATOM   2488  N   SER A 167     -31.240 -40.872   0.774  1.00 15.41           N
ATOM   2489  CA  SER A 167     -30.525 -40.432  -0.412  1.00 17.41           C
ATOM   2490  C   SER A 167     -29.573 -39.280  -0.139  1.00 18.27           C
ATOM   2491  O   SER A 167     -28.937 -38.788  -1.076  1.00 20.28           O
ATOM   2492  CB  SER A 167     -31.514 -40.033  -1.513  1.00 20.23           C
ATOM   2493  OG  SER A 167     -32.317 -38.937  -1.106  1.00 17.89           O
ATOM   2494  H   SER A 167     -31.923 -40.386   0.967  1.00 15.41           H
ATOM   2495  HA  SER A 167     -29.998 -41.172  -0.750  1.00 17.41           H
ATOM   2496  HB2 SER A 167     -31.017 -39.781  -2.307  1.00 20.23           H
ATOM   2497  HB3 SER A 167     -32.089 -40.789  -1.709  1.00 20.23           H
ATOM   2498  HG  SER A 167     -32.756 -39.142  -0.420  1.00 17.89           H
ATOM   2499  N   PHE A 168     -29.450 -38.841   1.112  1.00 16.12           N
ATOM   2500  CA  PHE A 168     -28.658 -37.655   1.404  1.00 16.88           C
ATOM   2501  C   PHE A 168     -27.171 -37.958   1.311  1.00 18.57           C
ATOM   2502  O   PHE A 168     -26.684 -38.926   1.904  1.00 19.94           O
ATOM   2503  CB  PHE A 168     -28.966 -37.123   2.797  1.00 14.79           C
ATOM   2504  CG  PHE A 168     -28.224 -35.861   3.115  1.00 13.43           C
ATOM   2505  CD1 PHE A 168     -28.593 -34.676   2.507  1.00 14.09           C
ATOM   2506  CD2 PHE A 168     -27.146 -35.852   3.983  1.00 13.15           C
ATOM   2507  CE1 PHE A 168     -27.918 -33.502   2.771  1.00 13.15           C
ATOM   2508  CE2 PHE A 168     -26.465 -34.673   4.250  1.00 14.30           C
ATOM   2509  CZ  PHE A 168     -26.857 -33.498   3.640  1.00 13.88           C
ATOM   2510  H   PHE A 168     -29.811 -39.208   1.801  1.00 16.12           H
ATOM   2511  HA  PHE A 168     -28.869 -36.962   0.759  1.00 16.88           H
ATOM   2512  HB2 PHE A 168     -29.916 -36.936   2.861  1.00 14.79           H
ATOM   2513  HB3 PHE A 168     -28.715 -37.792   3.453  1.00 14.79           H
ATOM   2514  HD1 PHE A 168     -29.312 -34.668   1.917  1.00 14.09           H
ATOM   2515  HD2 PHE A 168     -26.880 -36.642   4.396  1.00 13.15           H
ATOM   2516  HE1 PHE A 168     -28.183 -32.712   2.359  1.00 13.15           H
ATOM   2517  HE2 PHE A 168     -25.746 -34.674   4.840  1.00 14.30           H
ATOM   2518  HZ  PHE A 168     -26.403 -32.706   3.818  1.00 13.88           H
ATOM   2519  N   LEU A 169     -26.445 -37.111   0.594  1.00 19.76           N
ATOM   2520  CA  LEU A 169     -25.008 -37.284   0.473  1.00 23.95           C
ATOM   2521  C   LEU A 169     -24.288 -36.394   1.482  1.00 24.55           C
ATOM   2522  O   LEU A 169     -24.149 -35.186   1.282  1.00 22.90           O
ATOM   2523  CB  LEU A 169     -24.544 -36.966  -0.951  1.00 25.86           C
ATOM   2524  CG  LEU A 169     -23.587 -37.983  -1.580  1.00 29.87           C
ATOM   2525  CD1 LEU A 169     -22.995 -37.437  -2.873  1.00 30.39           C
ATOM   2526  CD2 LEU A 169     -22.491 -38.385  -0.601  1.00 27.94           C
ATOM   2527  OXT LEU A 169     -23.826 -36.865   2.525  1.00 26.83           O
ATOM   2528  H   LEU A 169     -26.759 -36.432   0.170  1.00 19.76           H
ATOM   2529  HA  LEU A 169     -24.786 -38.207   0.670  1.00 23.95           H
ATOM   2530  HB2 LEU A 169     -25.326 -36.913  -1.522  1.00 25.86           H
ATOM   2531  HB3 LEU A 169     -24.091 -36.109  -0.944  1.00 25.86           H
ATOM   2532  HG  LEU A 169     -24.089 -38.783  -1.802  1.00 29.87           H
ATOM   2533 HD11 LEU A 169     -22.405 -38.105  -3.257  1.00 30.39           H
ATOM   2534 HD12 LEU A 169     -23.716 -37.238  -3.491  1.00 30.39           H
ATOM   2535 HD13 LEU A 169     -22.496 -36.629  -2.675  1.00 30.39           H
ATOM   2536 HD21 LEU A 169     -21.747 -38.760  -1.098  1.00 27.94           H
ATOM   2537 HD22 LEU A 169     -22.200 -37.598  -0.114  1.00 27.94           H
ATOM   2538 HD23 LEU A 169     -22.842 -39.045   0.017  1.00 27.94           H"""


def test_calculate_secondary_structure():
    # Create a test chain with random coordinates
    aa_list = random.choices(AA_LIST, k=10)
    arr = _make_chain(aa_list=aa_list, chain_id='A')

    sse = metrics.calculate_secondary_structure(arr)

    assert len(sse) == len(aa_list)
    assert all(ss in {'a', 'b', 'c'} for ss in sse)


def test_calculate_membrane_distance():
    # Create a test chain with varying z-coordinates
    z_values = list(range(-25, 25, 5))
    coords = [[np.random.randint(10), np.random.randint(10), z] for z in z_values]
    aa_list = random.choices(AA_LIST, k=len(z_values))
    arr = _make_chain(aa_list=aa_list, coords=coords, chain_id='A', altloc='')

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
    aa_list = residue_table.resn_struct.tolist()
    arr = _make_chain(aa_list=aa_list, chain_id='A', altloc='')

    context = Context(array=arr, config=Config())
    context.residue_table = residue_table

    output = metrics.define_secondary_structure(context)
    assert 'ss_domains' not in output.columns.tolist()
    assert 'ss_group' in output.columns.tolist()

    context = Context(array=arr, config=Config(membrane_protein=True))
    context.residue_table = residue_table
    output = metrics.define_secondary_structure(context)
    assert 'ss_domains' in output.columns.tolist()
    assert 'ss_group' in output.columns.tolist()

##TO DO MAKE MORE ROBUST WITH REAL PDB FILE
def test_calculate_sasa():  
    # Create a simple chain with a few residues
    aa_list = ['ALA', 'GLY', 'SER']
    arr = _make_chain(aa_list=aa_list, chain_id='A', altloc='')
    context = Context(array=arr)
    
    # Calculate SASA - should return a DataFrame with 'sasa' column
    sasa_df = metrics.calculate_sasa(context)

    # Check that SASA values are non-negative
    sasa_values = sasa_df['sasa']
    assert np.all(sasa_values >= 0), "SASA values should be non-negative"


def test_KD_values():  
    # Create a chain with known hydrophobic and hydrophilic residues
    aa_list = ['ILE', 'VAL', 'ALA', 'ASP', 'GLU', 'LYS']
    arr = _make_chain(aa_list=aa_list, chain_id='A', altloc='')
    context = Context(array=arr)
    
    kd_df = metrics.calculate_kyte_doolittle(context)

    # Check that we get per-residue values
    res_starts = struc.get_residue_starts(arr)
    assert len(kd_df) == len(res_starts)
    
    # Extract values for testing
    kd_values = kd_df['kyte_doolittle']
    
    assert kd_values.iloc[0] > 4.0, "ILE should be highly hydrophobic"
    assert kd_values.iloc[3] < -3.0, "ASP should be hydrophilic"
    assert kd_values.iloc[4] < -3.0, "GLU should be hydrophilic"


##TO DO MAKE MORE ROBUST WITH REAL PDB FILE
def test_calculate_residue_packing():
    # Create a chain with a few residues that can be close together
    aa_list = ['ALA', 'GLY', 'ALA', 'LEU']
    arr = _make_chain(aa_list=aa_list, chain_id='A', altloc='')
    context = Context(array=arr)
    
    # Calculate packing metrics
    packing = metrics.calculate_residue_packing(context, cutoff=5.0)
    
    # Check that all expected keys are present
    expected_keys = ['packing_n_atoms', 'packing_n_neighbor_residues', 'packing_contact_density']
    assert all(key in packing for key in expected_keys)
    
    # Check that arrays have correct length
    res_starts = struc.get_residue_starts(arr)
    n_res = len(res_starts)
    assert len(packing['packing_n_atoms']) == n_res
    assert len(packing['packing_n_neighbor_residues']) == n_res
    assert len(packing['packing_contact_density']) == n_res
    
    # Check that values are reasonable
    assert all(packing['packing_n_atoms'] >= 0), "Number of atoms should be positive"
    assert all(packing['packing_n_neighbor_residues'] >= 0), "Number of neighbors should be non-negative"


##TO DO MAKE MORE ROBUST WITH REAL PDB FILE
def test_calculate_hbond_metrics():
    # Create a chain with residues that can form H-bonds
    aa_list = ['SER', 'GLY', 'ASP', 'ASN']
    arr = _make_chain(aa_list=aa_list, chain_id='A')
    context = Context(array=arr)
    
    hbond_metrics = metrics.calculate_hbond_metrics(context)
        
    # Check that all expected keys are present
    expected_keys = ['bb_hbond_count', 'sc_hbond_count', 'total_hbond_count']
    assert all(key in hbond_metrics for key in expected_keys)
        
    # Check that arrays have correct length
    res_starts = struc.get_residue_starts(arr)
    n_res = len(res_starts)
    for key in ['bb_hbond_count', 'sc_hbond_count', 'total_hbond_count']:
        assert len(hbond_metrics[key]) == n_res
    
    # Check that counts are non-negative
    assert all(hbond_metrics['bb_hbond_count'] >= 0)
    assert all(hbond_metrics['sc_hbond_count'] >= 0)
    assert all(hbond_metrics['total_hbond_count'] >= 0)
    

def test_calculate_hbond_metrics_with_altloc():
    """Test that hbond metrics properly handle altloc information."""
    # Create a chain with altloc identifiers
    aa_list = ['SER', 'GLY', 'ASP']
    altlocs = ['A', '', 'B']  # Mix of altlocs and no altloc
    arr = _make_chain(aa_list=aa_list, chain_id='A', altloc=altlocs)
    context = Context(array=arr)
    
    hbond_metrics = metrics.calculate_hbond_metrics(context)
    
    # Check that altloc column is present in metadata
    assert 'altloc' in hbond_metrics.columns
    
    # Check altloc values are preserved
    assert hbond_metrics['altloc'].iloc[0] == 'A'
    assert hbond_metrics['altloc'].iloc[1] == ''
    assert hbond_metrics['altloc'].iloc[2] == 'B'
