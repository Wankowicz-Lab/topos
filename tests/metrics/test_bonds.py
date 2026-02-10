"""Tests for bonds metrics module."""
import pandas as pd

from src.metrics import bonds
from src.pipeline.context import Context
from src.structure.utils import _res_key
from tests.test_utils import _make_atoms, _make_chain, _make_residue

import biotite.structure as struc




def test_identify_salt_bridges():
    """Test salt bridge identification with ASP-LYS pairs."""
    # Create ASP and LYS residues close together
    # ASP: N, CA, C, O, CB, CG, OD1, OD2 (8 atoms)
    asp_coords = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0], [3.0, 0.0, 0.0],
                  [1.5, 1.0, 0.0], [1.5, 1.5, 0.0], [5.0, 0.0, 0.0], [1.5, 1.5, -0.5]]  # CB, CG, OD1, OD2
    # LYS: N, CA, C, O, CB, CG, CD, CE, NZ (9 atoms)
    lys_coords = [[10.0, 0.0, 0.0], [11.0, 0.0, 0.0], [12.0, 0.0, 0.0], [13.0, 0.0, 0.0],
                  [11.5, 1.0, 0.0], [11.5, 2.0, 0.0], [11.5, 3.0, 0.0], [11.5, 4.0, 0.0],
                  [5.0, 0.0, 3.5]]  # CB, CG, CD, CE, NZ (3.5 A away from OD1)
    
    asp = _make_residue('ASP', res_id=1, chain_id='A', coords=asp_coords)
    lys = _make_residue('LYS', res_id=3, chain_id='A', coords=lys_coords)
    arr = struc.concatenate([asp, lys])
    
    result = bonds.identify_salt_bridges(arr, cutoff=4.0)
    
    assert len(result) == 2
    
    # Check that expected residues are found
    assert result.iloc[0]['chain'] == 'A'
    assert result.iloc[0]['resi_struct'] == 1
    assert result.iloc[0]['resn_struct'] == 'ASP'
    assert result.iloc[0]['partner_chain'] == 'A'
    assert result.iloc[0]['partner_resi'] == 3
    assert result.iloc[0]['partner_resn'] == 'LYS'
    assert result.iloc[0]['bond_type'] == 'salt_bridge'
    assert result.iloc[1]['chain'] == 'A'
    assert result.iloc[1]['resi_struct'] == 3
    assert result.iloc[1]['resn_struct'] == 'LYS'
    assert result.iloc[1]['partner_chain'] == 'A'
    assert result.iloc[1]['partner_resi'] == 1
    assert result.iloc[1]['partner_resn'] == 'ASP'
    assert result.iloc[1]['bond_type'] == 'salt_bridge'


def test_classify_bond_types():
    """Classify bond types correctly flags protein/protein vs protein/ligand."""
    ala = _make_residue('ALA', res_id=1, chain_id='A')
    gly = _make_residue('GLY', res_id=2, chain_id='A')
    lig = _make_atoms(['C1', 'N1'], [[0, 0, 0], [1, 0, 0]], res_name='LIG', res_id=100, chain_id='B')
    arr = struc.concatenate([ala, gly, lig])

    bond_results = pd.DataFrame({
        'residue_key': [_res_key('A', 1, 'ALA'), _res_key('A', 1, 'ALA'), _res_key('A', 2, 'GLY')],
        'partner_residue_key': [_res_key('A', 2, 'GLY'), _res_key('B', 100, 'LIG'), _res_key('B', 100, 'LIG')],
    })

    result = bonds.classify_bond_types(bond_results, arr)

    # Protein–protein (ALA–GLY) is True; protein–ligand rows are False
    assert result['protein_protein'].tolist() == [True, False, False]


def test_identify_hbonds():
    """Identify hbonds returns two rows per bond with extras category (residue_type-partner_type)."""
    aa_list = ['SER', 'GLY', 'ASP', 'ASN']
    arr = _make_chain(aa_list=aa_list, chain_id='A')
    result = bonds.identify_hbonds(arr)

    expected_cols = ['chain', 'resi_struct', 'resn_struct', 'residue_key', 'partner_chain', 'partner_resi', 'partner_resn', 'partner_residue_key', 'bond_type', 'extras']
    for col in expected_cols:
        assert col in result.columns
    assert all(result['bond_type'] == 'hbond')
    # Two rows per hbond
    assert len(result) % 2 == 0
    # Each row has extras['category'] with first part = this row's residue type (backbone or sidechain)
    for _, row in result.iterrows():
        assert 'category' in row['extras']
        parts = row['extras']['category'].split('-')
        assert len(parts) == 2
        assert parts[0] in ('backbone', 'sidechain')
        assert parts[1] in ('backbone', 'sidechain')


def test_calculate_salt_bridges():
    """Test salt bridge metric calculation and bonds_df storage."""
    # Create ASP and LYS close together (same setup as identify test)
    # ASP: N, CA, C, O, CB, CG, OD1, OD2 (8 atoms)
    asp_coords = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0], [3.0, 0.0, 0.0],
                  [1.5, 1.0, 0.0], [1.5, 1.5, 0.0], [5.0, 0.0, 0.0], [1.5, 1.5, -0.5]]  # CB, CG, OD1, OD2
    # LYS: N, CA, C, O, CB, CG, CD, CE, NZ (9 atoms)
    lys_coords = [[10.0, 0.0, 0.0], [11.0, 0.0, 0.0], [12.0, 0.0, 0.0], [13.0, 0.0, 0.0],
                  [11.5, 1.0, 0.0], [11.5, 2.0, 0.0], [11.5, 3.0, 0.0], [11.5, 4.0, 0.0],
                  [5.0, 0.0, 3.5]]  # CB, CG, CD, CE, NZ (3.5 A away from OD1)
    
    asp = _make_residue('ASP', res_id=1, chain_id='A', coords=asp_coords)
    lys = _make_residue('LYS', res_id=3, chain_id='A', coords=lys_coords)
    arr = struc.concatenate([asp, lys])
    context = Context(array=arr)
    
    result = bonds.calculate_salt_bridges(context, cutoff=4.0)
    
    assert isinstance(result, pd.DataFrame)
    assert 'salt_bridge_count' in result.columns
    # Check that we have nonzero counts
    assert result['salt_bridge_count'].sum() > 0
    # Check bonds_df in context
    assert 'bonds_df' in context.extras
    assert isinstance(context.extras['bonds_df'], pd.DataFrame)
    bonds_df = context.extras['bonds_df']
    # Check that bonds_df contains salt_bridge bond type
    salt_bridge_rows = bonds_df[bonds_df['bond_type'] == 'salt_bridge']
    assert len(salt_bridge_rows) > 0


def test_identify_ionic_bonds():
    """Test ionic bond identification with ASP-HIS pairs."""
    # Create ASP and HIS residues close together
    # ASP: N, CA, C, O, CB, CG, OD1, OD2 (8 atoms)
    asp_coords = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0], [3.0, 0.0, 0.0],
                  [1.5, 1.0, 0.0], [1.5, 1.5, 0.0], [5.0, 0.0, 0.0], [1.5, 1.5, -0.5]]  # CB, CG, OD1, OD2
    # HIS: N, CA, C, O, CB, CG, ND1, CD2, CE1, NE2 (10 atoms)
    # ND1 at [5.0, 0.0, 3.5] is 3.5 A from ASP OD1 at [5.0, 0.0, 0.0] (within 4 A cutoff)
    his_coords = [[10.0, 0.0, 0.0], [11.0, 0.0, 0.0], [12.0, 0.0, 0.0], [13.0, 0.0, 0.0],
                  [11.5, 1.0, 0.0], [11.5, 2.0, 0.0], [5.0, 0.0, 3.5], [11.5, 4.0, 0.0],
                  [11.5, 5.0, 0.0], [11.5, 6.0, 0.0]]  # ND1, CD2, CE1, NE2
                 
    asp = _make_residue('ASP', res_id=1, chain_id='A', coords=asp_coords)
    his = _make_residue('HIS', res_id=3, chain_id='A', coords=his_coords)
    arr = struc.concatenate([asp, his])
    
    result = bonds.identify_ionic_bonds(arr, cutoff=4.0)
    
    assert len(result) == 2
    
    # Check that expected residues are found
    assert result.iloc[0]['chain'] == 'A'
    assert result.iloc[0]['resi_struct'] == 1
    assert result.iloc[0]['resn_struct'] == 'ASP'
    assert result.iloc[0]['partner_chain'] == 'A'
    assert result.iloc[0]['partner_resi'] == 3
    assert result.iloc[0]['partner_resn'] == 'HIS'
    assert result.iloc[0]['bond_type'] == 'ionic'
    assert result.iloc[1]['chain'] == 'A'
    assert result.iloc[1]['resi_struct'] == 3
    assert result.iloc[1]['resn_struct'] == 'HIS'
    assert result.iloc[1]['partner_chain'] == 'A'
    assert result.iloc[1]['partner_resi'] == 1
    assert result.iloc[1]['partner_resn'] == 'ASP'
    assert result.iloc[1]['bond_type'] == 'ionic'


def test_calculate_ionic_bond_count():
    """Test ionic bond metric calculation and bonds_df storage."""
    # Create ASP and LYS close together (same setup as identify test)
    # ASP: N, CA, C, O, CB, CG, OD1, OD2 (8 atoms)
    asp_coords = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0], [3.0, 0.0, 0.0],
                  [1.5, 1.0, 0.0], [1.5, 1.5, 0.0], [5.0, 0.0, 0.0], [1.5, 1.5, -0.5]]  # CB, CG, OD1, OD2
    # LYS: N, CA, C, O, CB, CG, CD, CE, NZ (9 atoms)
    lys_coords = [[10.0, 0.0, 0.0], [11.0, 0.0, 0.0], [12.0, 0.0, 0.0], [13.0, 0.0, 0.0],
                  [11.5, 1.0, 0.0], [11.5, 2.0, 0.0], [11.5, 3.0, 0.0], [11.5, 4.0, 0.0],
                  [5.0, 0.0, 3.5]]  # CB, CG, CD, CE, NZ (3.5 A away from OD1)
    
    asp = _make_residue('ASP', res_id=1, chain_id='A', coords=asp_coords)
    lys = _make_residue('LYS', res_id=3, chain_id='A', coords=lys_coords)
    arr = struc.concatenate([asp, lys])
    context = Context(array=arr)
    
    result = bonds.calculate_ionic_bond_count(context, cutoff=4.0)
    
def test_identify_disulfide_bonds():
    """Test disulfide bond identification with CYS residues."""
    # Create two CYS residues with SG atoms close together
    cys1_coords = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0], [3.0, 0.0, 0.0],
                   [1.5, 1.0, 0.0], [1.5, 1.5, 0.0]]  # SG at [1.5, 1.5, 0.0]
    cys2_coords = [[10.0, 0.0, 0.0], [11.0, 0.0, 0.0], [12.0, 0.0, 0.0], [13.0, 0.0, 0.0],
                   [11.5, 1.0, 0.0], [1.5, 1.5, 2.0]]  # SG at [1.5, 1.5, 2.0] (2.0 A away)
    
    cys1 = _make_residue('CYS', res_id=1, chain_id='A', coords=cys1_coords)
    cys2 = _make_residue('CYS', res_id=2, chain_id='A', coords=cys2_coords)
    arr = struc.concatenate([cys1, cys2])
    
    result = bonds.identify_disulfide_bonds(arr, cutoff=2.5)
    
    assert len(result) == 2
    assert result.iloc[0]['chain'] == 'A'
    assert result.iloc[0]['resi_struct'] == 1
    assert result.iloc[0]['resn_struct'] == 'CYS'
    assert result.iloc[0]['partner_chain'] == 'A'
    assert result.iloc[0]['partner_resi'] == 2
    assert result.iloc[0]['partner_resn'] == 'CYS'
    assert result.iloc[0]['bond_type'] == 'disulfide'
    assert result.iloc[1]['chain'] == 'A'

    assert result.iloc[1]['resi_struct'] == 2
    assert result.iloc[1]['resn_struct'] == 'CYS'
    assert result.iloc[1]['partner_chain'] == 'A'
    assert result.iloc[1]['partner_resi'] == 1
    assert result.iloc[1]['partner_resn'] == 'CYS'
    assert result.iloc[1]['bond_type'] == 'disulfide'


def test_calculate_disulfide_bond_count():
    """Test disulfide bond metric calculation and bonds_df storage."""
    # Create two CYS residues with SG atoms close together (same setup as identify test)
    cys1_coords = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0], [3.0, 0.0, 0.0],
                   [1.5, 1.0, 0.0], [1.5, 1.5, 0.0]]  # SG at [1.5, 1.5, 0.0]
    cys2_coords = [[10.0, 0.0, 0.0], [11.0, 0.0, 0.0], [12.0, 0.0, 0.0], [13.0, 0.0, 0.0],
                   [11.5, 1.0, 0.0], [1.5, 1.5, 2.0]]  # SG at [1.5, 1.5, 2.0] (2.0 A away)
    
    cys1 = _make_residue('CYS', res_id=1, chain_id='A', coords=cys1_coords)
    cys2 = _make_residue('CYS', res_id=2, chain_id='A', coords=cys2_coords)
    arr = struc.concatenate([cys1, cys2])
    context = Context(array=arr)
    
    result = bonds.calculate_disulfide_bond_count(context, cutoff=2.5)
    
    assert isinstance(result, pd.DataFrame)
    assert 'disulfide_bond_count' in result.columns
    # Check that we have nonzero counts
    assert result['disulfide_bond_count'].sum() > 0
    # Check bonds_df in context
    assert 'bonds_df' in context.extras
    assert isinstance(context.extras['bonds_df'], pd.DataFrame)
    bonds_df = context.extras['bonds_df']
    # Check that bonds_df contains disulfide bond type
    disulfide_rows = bonds_df[bonds_df['bond_type'] == 'disulfide']
    assert len(disulfide_rows) > 0


def test_identify_pi_stacking():
    """Test pi-stacking identification with aromatic residues."""
    # Create two PHE residues with parallel rings (same plane, close together)
    # PHE ring atoms: CG, CD1, CD2, CE1, CE2, CZ
    # Ring center for PHE1 at approximately [1.5, 3.0, 0.0]
    phe1_coords = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0], [3.0, 0.0, 0.0],
                   [1.5, 1.0, 0.0],  # CB
                   [1.5, 2.0, 0.0],  # CG
                   [0.5, 2.5, 0.0],  # CD1
                   [2.5, 2.5, 0.0],  # CD2
                   [0.5, 3.5, 0.0],  # CE1
                   [2.5, 3.5, 0.0],  # CE2
                   [1.5, 4.0, 0.0]]  # CZ
    
    # Second PHE with parallel ring (same z, shifted in x, ring center at [11.5, 3.0, 0.0])
    # Distance between centers: 10.0 A, but we'll place them closer
    phe2_coords = [[10.0, 0.0, 0.0], [11.0, 0.0, 0.0], [12.0, 0.0, 0.0], [13.0, 0.0, 0.0],
                   [1.5, 1.0, 0.0],  # CB (moved to be close)
                   [1.5, 2.0, 0.0],  # CG (moved to be close)
                   [0.5, 2.5, 0.0],  # CD1 (moved to be close)
                   [2.5, 2.5, 0.0],  # CD2 (moved to be close)
                   [0.5, 3.5, 0.0],  # CE1 (moved to be close)
                   [2.5, 3.5, 0.0],  # CE2 (moved to be close)
                   [1.5, 4.0, 0.0]]  # CZ (moved to be close, parallel ring)
    
    phe1 = _make_residue('PHE', res_id=1, chain_id='A', coords=phe1_coords)
    phe2 = _make_residue('PHE', res_id=2, chain_id='A', coords=phe2_coords)
    arr = struc.concatenate([phe1, phe2])
    
    result = bonds.identify_pi_stacking(arr, distance_cutoff=5.5, angle_cutoff=30.0)
    
    assert len(result) == 2
    
    # Check that expected residues are found
    assert result.iloc[0]['chain'] == 'A'
    assert result.iloc[0]['resi_struct'] == 1
    assert result.iloc[0]['resn_struct'] == 'PHE'
    assert result.iloc[0]['partner_chain'] == 'A'
    assert result.iloc[0]['partner_resi'] == 2
    assert result.iloc[0]['partner_resn'] == 'PHE'
    assert result.iloc[0]['bond_type'] == 'pi_stacking'
    assert 'geometry' in result.iloc[0]['extras']
    assert result.iloc[1]['chain'] == 'A'
    assert result.iloc[1]['resi_struct'] == 2
    assert result.iloc[1]['resn_struct'] == 'PHE'
    assert result.iloc[1]['partner_chain'] == 'A'
    assert result.iloc[1]['partner_resi'] == 1
    assert result.iloc[1]['partner_resn'] == 'PHE'
    assert result.iloc[1]['bond_type'] == 'pi_stacking'
    assert 'geometry' in result.iloc[1]['extras']


def test_calculate_pi_stacking_count():
    """Test pi-stacking metric calculation and bonds_df storage."""
    # Create two PHE residues with parallel rings (same setup as identify test)
    phe1_coords = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0], [3.0, 0.0, 0.0],
                   [1.5, 1.0, 0.0],  # CB
                   [1.5, 2.0, 0.0],  # CG
                   [0.5, 2.5, 0.0],  # CD1
                   [2.5, 2.5, 0.0],  # CD2
                   [0.5, 3.5, 0.0],  # CE1
                   [2.5, 3.5, 0.0],  # CE2
                   [1.5, 4.0, 0.0]]  # CZ
    
    phe2_coords = [[10.0, 0.0, 0.0], [11.0, 0.0, 0.0], [12.0, 0.0, 0.0], [13.0, 0.0, 0.0],
                   [1.5, 1.0, 0.0],  # CB (moved to be close)
                   [1.5, 2.0, 0.0],  # CG (moved to be close)
                   [0.5, 2.5, 0.0],  # CD1 (moved to be close)
                   [2.5, 2.5, 0.0],  # CD2 (moved to be close)
                   [0.5, 3.5, 0.0],  # CE1 (moved to be close)
                   [2.5, 3.5, 0.0],  # CE2 (moved to be close)
                   [1.5, 4.0, 0.0]]  # CZ (moved to be close, parallel ring)
    
    phe1 = _make_residue('PHE', res_id=1, chain_id='A', coords=phe1_coords)
    phe2 = _make_residue('PHE', res_id=2, chain_id='A', coords=phe2_coords)
    arr = struc.concatenate([phe1, phe2])
    context = Context(array=arr)
    
    result = bonds.calculate_pi_stacking_count(context, distance_cutoff=5.5, angle_cutoff=30.0)
    
    assert isinstance(result, pd.DataFrame)
    assert 'pi_stacking_count' in result.columns
    # Check that we have nonzero counts
    assert result['pi_stacking_count'].sum() > 0
    # Check bonds_df in context
    assert 'bonds_df' in context.extras
    assert isinstance(context.extras['bonds_df'], pd.DataFrame)
    bonds_df = context.extras['bonds_df']
    # Check that bonds_df contains pi_stacking bond type
    pi_stacking_rows = bonds_df[bonds_df['bond_type'] == 'pi_stacking']
    assert len(pi_stacking_rows) > 0
    # Check extras column contains geometry
    for _, row in pi_stacking_rows.iterrows():
        assert 'geometry' in row['extras']


def test_identify_cation_pi():
    """Test cation-pi identification with LYS/ARG and aromatic residues."""
    # Create LYS and PHE residues with NZ close to ring center
    # LYS NZ atom
    lys_coords = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0], [3.0, 0.0, 0.0],
                  [1.5, 1.0, 0.0], [1.5, 2.0, 0.0], [1.5, 3.0, 0.0], [1.5, 4.0, 0.0],
                  [1.5, 3.0, 5.0]]  # NZ at [1.5, 3.0, 5.0]
    
    # PHE ring center at [1.5, 3.0, 5.0] (same as NZ, distance = 0)
    phe_coords = [[10.0, 0.0, 0.0], [11.0, 0.0, 0.0], [12.0, 0.0, 0.0], [13.0, 0.0, 0.0],
                  [1.5, 1.0, 5.0],  # CB
                  [1.5, 2.0, 5.0],  # CG
                  [0.5, 2.5, 5.0],  # CD1
                  [2.5, 2.5, 5.0],  # CD2
                  [0.5, 3.5, 5.0],  # CE1
                  [2.5, 3.5, 5.0],  # CE2
                  [1.5, 4.0, 5.0]]  # CZ (ring center at [1.5, 3.0, 5.0])
    
    lys = _make_residue('LYS', res_id=1, chain_id='A', coords=lys_coords)
    phe = _make_residue('PHE', res_id=2, chain_id='A', coords=phe_coords)
    arr = struc.concatenate([lys, phe])
    
    result = bonds.identify_cation_pi(arr, cutoff=6.0)
    
    assert len(result) == 2
    
    # Check that expected residues are found
    # First row should be the cation (LYS)
    cation_row = result[result['resn_struct'] == 'LYS'].iloc[0]
    assert cation_row['chain'] == 'A'
    assert cation_row['resi_struct'] == 1
    assert cation_row['resn_struct'] == 'LYS'
    assert cation_row['partner_chain'] == 'A'
    assert cation_row['partner_resi'] == 2
    assert cation_row['partner_resn'] == 'PHE'
    assert cation_row['bond_type'] == 'cation_pi'
    assert 'role' in cation_row['extras']
    assert cation_row['extras']['role'] == 'cation'
    
    # Second row should be the aromatic (PHE)
    aromatic_row = result[result['resn_struct'] == 'PHE'].iloc[0]
    assert aromatic_row['chain'] == 'A'
    assert aromatic_row['resi_struct'] == 2
    assert aromatic_row['resn_struct'] == 'PHE'
    assert aromatic_row['partner_chain'] == 'A'
    assert aromatic_row['partner_resi'] == 1
    assert aromatic_row['partner_resn'] == 'LYS'
    assert aromatic_row['bond_type'] == 'cation_pi'
    assert 'role' in aromatic_row['extras']
    assert aromatic_row['extras']['role'] == 'aromatic'


def test_calculate_cation_pi_count():
    """Test cation-pi metric calculation and bonds_df storage."""
    # Create LYS and PHE residues with NZ close to ring center (same setup as identify test)
    lys_coords = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0], [3.0, 0.0, 0.0],
                  [1.5, 1.0, 0.0], [1.5, 2.0, 0.0], [1.5, 3.0, 0.0], [1.5, 4.0, 0.0],
                  [1.5, 3.0, 5.0]]  # NZ at [1.5, 3.0, 5.0]
    
    phe_coords = [[10.0, 0.0, 0.0], [11.0, 0.0, 0.0], [12.0, 0.0, 0.0], [13.0, 0.0, 0.0],
                  [1.5, 1.0, 5.0],  # CB
                  [1.5, 2.0, 5.0],  # CG
                  [0.5, 2.5, 5.0],  # CD1
                  [2.5, 2.5, 5.0],  # CD2
                  [0.5, 3.5, 5.0],  # CE1
                  [2.5, 3.5, 5.0],  # CE2
                  [1.5, 4.0, 5.0]]  # CZ (ring center at [1.5, 3.0, 5.0])
    
    lys = _make_residue('LYS', res_id=1, chain_id='A', coords=lys_coords)
    phe = _make_residue('PHE', res_id=2, chain_id='A', coords=phe_coords)
    arr = struc.concatenate([lys, phe])
    context = Context(array=arr)
    
    result = bonds.calculate_cation_pi_count(context, cutoff=6.0)
    
    assert isinstance(result, pd.DataFrame)
    assert 'cation_pi_count' in result.columns
    # Check that we have nonzero counts
    assert result['cation_pi_count'].sum() > 0
    # Check bonds_df in context
    assert 'bonds_df' in context.extras
    assert isinstance(context.extras['bonds_df'], pd.DataFrame)
    bonds_df = context.extras['bonds_df']
    # Check that bonds_df contains cation_pi bond type
    cation_pi_rows = bonds_df[bonds_df['bond_type'] == 'cation_pi']
    assert len(cation_pi_rows) > 0
    # Check extras column contains role
    for _, row in cation_pi_rows.iterrows():
        assert 'role' in row['extras']


def test_identify_vdw_contacts():
    """Test van der Waals contact identification."""
    # Create two ALA residues with atoms close together
    # ALA1: place atoms starting at [0, 0, 0]
    ala1_coords = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0], [3.0, 0.0, 0.0],
                   [1.5, 1.0, 0.0]]  # CB
    
    # ALA2: place atoms close to ALA1 (within vdw contact distance)
    # CB of ALA2 at [1.5, 1.0, 3.0] - within vdw radius + cutoff
    ala2_coords = [[10.0, 0.0, 0.0], [11.0, 0.0, 0.0], [12.0, 0.0, 0.0], [13.0, 0.0, 0.0],
                   [1.5, 1.0, 3.0]]  # CB close to ALA1 CB
    
    ala1 = _make_residue('ALA', res_id=1, chain_id='A', coords=ala1_coords)
    ala2 = _make_residue('ALA', res_id=2, chain_id='A', coords=ala2_coords)
    arr = struc.concatenate([ala1, ala2])
    
    result = bonds.identify_vdw_contacts(arr, cutoff_factor=1.0)
    
    assert len(result) >= 2  # At least one pair, but could be more
    
    # Check that we have vdw contacts
    assert all(result['bond_type'] == 'vdw_contact')
    
    # Check that we have entries for both residues
    res1_rows = result[result['resi_struct'] == 1]
    res2_rows = result[result['resi_struct'] == 2]
    
    assert len(res1_rows) > 0
    assert len(res2_rows) > 0
    
    # Check first row structure
    assert result.iloc[0]['chain'] == 'A'
    assert result.iloc[0]['resi_struct'] in [1, 2]
    assert result.iloc[0]['resn_struct'] == 'ALA'
    assert result.iloc[0]['partner_chain'] == 'A'
    assert result.iloc[0]['partner_resi'] in [1, 2]
    assert result.iloc[0]['partner_resn'] == 'ALA'
    assert result.iloc[0]['bond_type'] == 'vdw_contact'
    assert result.iloc[0]['extras'] == {}


def test_calculate_vdw_contact_count():
    """Test van der Waals contact metric calculation and bonds_df storage."""
    # Create two ALA residues with atoms close together (same setup as identify test)
    ala1_coords = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0], [3.0, 0.0, 0.0],
                   [1.5, 1.0, 0.0]]  # CB
    
    ala2_coords = [[10.0, 0.0, 0.0], [11.0, 0.0, 0.0], [12.0, 0.0, 0.0], [13.0, 0.0, 0.0],
                   [1.5, 1.0, 3.0]]  # CB close to ALA1 CB
    
    ala1 = _make_residue('ALA', res_id=1, chain_id='A', coords=ala1_coords)
    ala2 = _make_residue('ALA', res_id=2, chain_id='A', coords=ala2_coords)
    # Ligand with one atom within vdw distance of ALA2 CB (1.5, 1.0, 3.0)
    lig = _make_atoms(['C1', 'N1'], [[1.5, 1.0, 4.0], [2.0, 1.0, 4.0]], res_name='LIG', res_id=100, chain_id='B')
    arr = struc.concatenate([ala1, ala2, lig])
    context = Context(array=arr)
    
    result = bonds.calculate_vdw_contact_count(context, cutoff_factor=1.0)
    
    assert isinstance(result, pd.DataFrame)
    assert 'vdw_contact_count' in result.columns
    assert result['vdw_contact_count'].sum() > 0
    bonds_df = context.extras['bonds_df']
    vdw_rows = bonds_df[bonds_df['bond_type'] == 'vdw_contact']
    # 2 rows per pair: 1 protein-protein pair (ALA1-ALA2) + 2 protein-ligand pairs (ALA1-LIG, ALA2-LIG)
    assert len(vdw_rows) == 6
    assert (vdw_rows['protein_protein']).sum() == 2


def test_calculate_hbond_metrics():
    """Per-residue hbond counts and bonds_df; with protein + ligand, not all hbonds are protein/ligand."""
    aa_list = ['SER', 'GLY', 'ASP', 'ASN']
    chain = _make_chain(aa_list=aa_list, chain_id='A')
    # Ligand O as acceptor; place O so SER backbone N (at [0,0,0]) can donate: O in -x from N for valid angle
    lig = _make_atoms(['O', 'N'], [[-2.0, 0.0, 0.0], [-3.5, 0.0, 0.0]], res_name='LIG', res_id=100, chain_id='B')
    arr = struc.concatenate([chain, lig])
    context = Context(array=arr)

    hbond_metrics = bonds.calculate_hbond_metrics(context)

    expected_keys = ['bb_hbond_count', 'sc_hbond_count', 'total_hbond_count']
    assert all(key in hbond_metrics for key in expected_keys)

    res_starts = struc.get_residue_starts(arr)
    n_res = len(res_starts)
    for key in ['bb_hbond_count', 'sc_hbond_count', 'total_hbond_count']:
        assert len(hbond_metrics[key]) == n_res

    assert all(hbond_metrics['bb_hbond_count'] >= 0)
    assert all(hbond_metrics['sc_hbond_count'] >= 0)
    assert all(hbond_metrics['total_hbond_count'] >= 0)

    bonds_df = context.extras['bonds_df']
    hbond_rows = bonds_df[bonds_df['bond_type'] == 'hbond']
    assert len(hbond_rows) > 0
    for _, row in hbond_rows.iterrows():
        assert 'category' in row['extras']
        parts = row['extras']['category'].split('-')
        assert parts[0] in ('backbone', 'sidechain') and parts[1] in ('backbone', 'sidechain')

    # With protein + ligand: some hbonds are protein-protein, some are protein-ligand (not all ligands)
    assert hbond_rows['protein_protein'].any(), "expected at least one protein-protein hbond"
    assert (hbond_rows['protein_protein'] == False).any(), "expected at least one protein-ligand hbond"

    # Total count sum = number of protein-protein hbond rows (only those count toward per-residue metrics)
    assert hbond_metrics['total_hbond_count'].sum() == (hbond_rows['protein_protein']).sum()


def test_calculate_hbond_metrics_with_altloc():
    """Hbond metrics handle altloc; bonds_df has hbond rows with category in extras."""
    aa_list = ['SER', 'GLY', 'ASP']
    altlocs = ['A', '', 'B']
    arr = _make_chain(aa_list=aa_list, chain_id='A', altloc=altlocs)
    context = Context(array=arr)

    hbond_metrics = bonds.calculate_hbond_metrics(context)
    assert 'bb_hbond_count' in hbond_metrics.columns
    assert 'sc_hbond_count' in hbond_metrics.columns

    hbond_rows = context.extras['bonds_df'][context.extras['bonds_df']['bond_type'] == 'hbond']
    if len(hbond_rows) > 0:
        assert 'category' in hbond_rows.iloc[0]['extras']


def test_bonds_df_consolidation():
    """Test that multiple metric functions append to same bonds_df."""
    # Create structure with multiple bond types
    asp = _make_residue('ASP', res_id=1, chain_id='A')
    lys = _make_residue('LYS', res_id=2, chain_id='A')
    cys1 = _make_residue('CYS', res_id=3, chain_id='A')
    cys2 = _make_residue('CYS', res_id=4, chain_id='A')
    arr = struc.concatenate([asp, lys, cys1, cys2])
    context = Context(array=arr)
    
    # Call multiple metric functions
    bonds.calculate_salt_bridges(context, cutoff=4.0)
    bonds.calculate_disulfide_bond_count(context, cutoff=2.5)
    
    assert 'bonds_df' in context.extras
    bonds_df = context.extras['bonds_df']
    assert isinstance(bonds_df, pd.DataFrame)
    
    # Check that bonds_df contains multiple bond types
    if len(bonds_df) > 0:
        bond_types = bonds_df['bond_type'].unique()
        assert 'salt_bridge' in bond_types or 'disulfide' in bond_types

