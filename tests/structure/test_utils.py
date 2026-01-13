import numpy as np
import pytest
from src.structure import utils
from tests.test_utils import _make_chain, _make_residue, AA_LIST
import biotite.structure as struc


def test_is_heavy():
    # Test heavy atoms
    assert utils.is_heavy("C") == True
    assert utils.is_heavy("CA") == True
    assert utils.is_heavy("N") == True
    assert utils.is_heavy("O") == True
    
    # Test hydrogen atoms (should return False)
    assert utils.is_heavy("H") == False
    assert utils.is_heavy("HA") == False
    assert utils.is_heavy("H1") == False
    assert utils.is_heavy("HG") == False
    assert utils.is_heavy("HD1") == False
    
    # Test deuterium
    assert utils.is_heavy("D") == False
    assert utils.is_heavy("DA") == False
    
    # Test edge cases
    assert utils.is_heavy("  C  ") == True  # with whitespace
    assert utils.is_heavy("  H  ") == False  # with whitespace


def test_residue_key():
    # Test basic residue key generation
    assert utils.residue_key("A", 1) == "A:1"
    assert utils.residue_key("B", 42) == "B:42"
    assert utils.residue_key("X", 100) == "X:100"
    
    # Test with float residue ID (should convert to int)
    assert utils.residue_key("A", 1.0) == "A:1"
    assert utils.residue_key("A", 1.9) == "A:1"


def test_is_backbone_atom():
    # Test backbone atoms (should return True)
    assert utils.is_backbone_atom("N") == True
    assert utils.is_backbone_atom("CA") == True
    assert utils.is_backbone_atom("C") == True
    assert utils.is_backbone_atom("O") == True
    assert utils.is_backbone_atom("OXT") == True
    assert utils.is_backbone_atom("H") == True
    assert utils.is_backbone_atom("H1") == True
    assert utils.is_backbone_atom("H2") == True
    assert utils.is_backbone_atom("H3") == True
    
    # Test sidechain atoms (should return False)
    assert utils.is_backbone_atom("CB") == False
    assert utils.is_backbone_atom("CG") == False
    assert utils.is_backbone_atom("NZ") == False
    assert utils.is_backbone_atom("OD1") == False


def test_norm_alt():
    # Test normalization of altloc identifiers
    assert utils.norm_alt(None) == ""
    assert utils.norm_alt("") == ""
    assert utils.norm_alt("A") == "A"
    assert utils.norm_alt("a") == "A"  # should uppercase
    assert utils.norm_alt("  A  ") == "A"  # should strip
    assert utils.norm_alt("  a  ") == "A"  # should strip and uppercase


def test_altloc_compatible():
    # Test altloc compatibility
    # Empty or None should be compatible with anything
    assert utils.altloc_compatible(None, None) == True
    assert utils.altloc_compatible("", "") == True
    assert utils.altloc_compatible(None, "A") == True
    assert utils.altloc_compatible("A", None) == True
    assert utils.altloc_compatible("", "A") == True
    assert utils.altloc_compatible("A", "") == True
    
    # Same altloc should be compatible
    assert utils.altloc_compatible("A", "A") == True
    assert utils.altloc_compatible("a", "A") == True  # case insensitive
    
    # Different altloc should not be compatible
    assert utils.altloc_compatible("A", "B") == False
    assert utils.altloc_compatible("A", "b") == False
    assert utils.altloc_compatible("a", "B") == False


def test_build_sites_biotite():
    # Create a simple chain with residues that have donors and acceptors
    aa_list = ['SER', 'GLY']  # SER has sidechain donor/acceptor
    arr = _make_chain(aa_list=aa_list, chain_id='A')
    
    # Build donor and acceptor sites
    donors, acceptors = utils.build_sites_biotite(arr)
    
    # Check that we get some sites
    assert len(donors) > 0 or len(acceptors) > 0
    
    # Check structure of DonorSite
    if len(donors) > 0:
        d = donors[0]
        assert hasattr(d, 'res_key')
        assert hasattr(d, 'resname')
        assert hasattr(d, 'chain')
        assert hasattr(d, 'resi')
        assert hasattr(d, 'atom_name')
        assert hasattr(d, 'coord_donor')
        assert hasattr(d, 'is_backbone')
    
    # Check structure of AcceptorSite
    if len(acceptors) > 0:
        a = acceptors[0]
        assert hasattr(a, 'res_key')
        assert hasattr(a, 'resname')
        assert hasattr(a, 'chain')
        assert hasattr(a, 'resi')
        assert hasattr(a, 'atom_name')
        assert hasattr(a, 'coord')
        assert hasattr(a, 'is_backbone')


def test_build_sites_biotite_with_known_residues():
    # Create a chain with residues known to have specific donors/acceptors
    # SER has sidechain OH (both donor and acceptor)
    # LYS has NZ (donor)
    # ASP has OD1, OD2 (acceptors)
    aa_list = ['SER', 'LYS', 'ASP']
    arr = _make_chain(aa_list=aa_list, chain_id='A')
    
    donors, acceptors = utils.build_sites_biotite(arr)
    
    # Should have backbone donors (N atoms)
    backbone_donors = [d for d in donors if d.is_backbone]
    assert len(backbone_donors) >= len(aa_list)
    
    # Should have backbone acceptors (O atoms)
    backbone_acceptors = [a for a in acceptors if a.is_backbone]
    assert len(backbone_acceptors) >= len(aa_list)
    
    # Should have some sidechain sites
    sidechain_donors = [d for d in donors if not d.is_backbone]
    sidechain_acceptors = [a for a in acceptors if not a.is_backbone]
    # SER should have OG (donor and acceptor), LYS should have NZ (donor), ASP should have OD1/OD2 (acceptors)
    assert len(sidechain_donors) > 0 or len(sidechain_acceptors) > 0


def test_detect_hbonds():
    # Create a simple structure with two residues that could form an H-bond
    # Place them close together with proper geometry
    ser1 = _make_residue("SER", res_id=1, chain_id="A")
    gly2 = _make_residue("GLY", res_id=2, chain_id="A")
    
    # Move GLY closer to SER (simple geometry)
    gly2.coord[:, 0] += 5.0  # Move 5 Angstroms along x-axis
    
    arr = struc.concatenate([ser1, gly2])
    
    # Build sites
    donors, acceptors = utils.build_sites_biotite(arr)
    
    # Detect H-bonds
    hbonds = utils.detect_hbonds(donors, acceptors)
    
    # Should get a list (may be empty if geometry isn't ideal, but function should work)
    assert isinstance(hbonds, list)
    
    # Check structure of H-bond dicts if any found
    if len(hbonds) > 0:
        hb = hbonds[0]
        assert 'donor_chain' in hb
        assert 'donor_resi' in hb
        assert 'donor_resname' in hb
        assert 'donor_atom' in hb
        assert 'acceptor_chain' in hb
        assert 'acceptor_resi' in hb
        assert 'acceptor_resname' in hb
        assert 'acceptor_atom' in hb
        assert 'DA_dist' in hb
        assert 'angle' in hb
        assert 'category' in hb


def test_detect_hbonds_empty_input():
    # Test with empty lists
    hbonds = utils.detect_hbonds([], [])
    assert hbonds == []


def test_detect_hbonds_parameters():
    # Test that custom parameters work
    aa_list = ['SER', 'GLY']
    arr = _make_chain(aa_list=aa_list, chain_id='A')
    donors, acceptors = utils.build_sites_biotite(arr)
    
    # Very strict parameters (should find fewer or no H-bonds)
    hbonds_strict = utils.detect_hbonds(donors, acceptors, da_max=2.0, h_a_max=1.5, angle_min=150.0)
    assert isinstance(hbonds_strict, list)
    
    # More lenient parameters
    hbonds_lenient = utils.detect_hbonds(donors, acceptors, da_max=4.0, h_a_max=3.0, angle_min=100.0)
    assert isinstance(hbonds_lenient, list)
    
    # Lenient should find at least as many as strict
    assert len(hbonds_lenient) >= len(hbonds_strict)


def test_get_metadata_cols():
    """Test that get_metadata_cols extracts correct metadata including altloc."""
    # Create a chain with altloc identifiers
    aa_list = ['ALA', 'CYS', 'ASP']
    arr = _make_chain(aa_list=aa_list, chain_id='A', altloc='')
    
    metadata_df = utils.get_metadata_cols(arr)
    
    # Check that all expected columns are present
    assert 'chain' in metadata_df.columns
    assert 'resi_struct' in metadata_df.columns
    assert 'resn_struct' in metadata_df.columns
    assert 'altloc' in metadata_df.columns
    
    # Check values
    assert len(metadata_df) == 3
    assert list(metadata_df['resn_struct']) == ['ALA', 'CYS', 'ASP']
    assert list(metadata_df['resi_struct']) == [1, 2, 3]
    assert all(metadata_df['chain'] == 'A')



def test_build_sites_biotite_with_altloc():
    """Test that build_sites_biotite properly captures altloc for donor/acceptor sites."""
    aa_list = ['SER', 'GLY']
    altlocs = ['A', '']
    arr = _make_chain(aa_list=aa_list, chain_id='A', altloc=altlocs)
    
    donors, acceptors = utils.build_sites_biotite(arr)
    
    # Check that donors and acceptors have altloc information
    if len(donors) > 0:
        # Donors from SER (with altloc 'A') should have altloc set
        ser_donors = [d for d in donors if d.resi == 1]
        for d in ser_donors:
            assert hasattr(d, 'altloc')
            assert d.altloc == 'A'
        
        # Donors from GLY (no altloc) should have empty altloc
        gly_donors = [d for d in donors if d.resi == 2]
        for d in gly_donors:
            assert hasattr(d, 'altloc')
            assert d.altloc == ''
    
    if len(acceptors) > 0:
        # Similar checks for acceptors
        ser_acceptors = [a for a in acceptors if a.resi == 1]
        for a in ser_acceptors:
            assert hasattr(a, 'altloc')
            assert a.altloc == 'A'


def test_detect_hbonds_with_altloc():
    """Test that detect_hbonds properly includes altloc in results."""
    aa_list = ['SER', 'GLY']
    altlocs = ['A', '']
    arr = _make_chain(aa_list=aa_list, chain_id='A', altloc=altlocs)
    
    donors, acceptors = utils.build_sites_biotite(arr)
    hbonds = utils.detect_hbonds(donors, acceptors)
    
    # Check that altloc fields are present in results
    for hb in hbonds:
        assert 'donor_altloc' in hb
        assert 'acceptor_altloc' in hb
        # Values should be normalized (uppercase or empty string)
        assert isinstance(hb['donor_altloc'], str)
        assert isinstance(hb['acceptor_altloc'], str)
