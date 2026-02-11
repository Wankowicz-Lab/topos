"""Tests for secondary structure domain metrics module."""
import numpy as np
import pandas as pd
import pytest

from src.metrics import secondary_structure


def test_ss_domain_lengths():
    """ss_domain_lengths returns ss_domains and ss_length with correct counts."""
    merged = pd.DataFrame({
        'ss_domains': ['alpha-helix_1', 'alpha-helix_1', 'beta-sheet_1', 'beta-sheet_1', 'coil_1', 'coil_1', 'alpha-helix_1'],
        'chain': ['A'] * 6 + ['B'],
    })
    out = secondary_structure.ss_domain_lengths(merged)
    out = out.sort_values(['chain', 'ss_domains']).reset_index(drop=True)
    assert list(out.columns) == ['chain', 'ss_domains', 'ss_length']
    assert out['ss_length'].tolist() == [2, 2, 2, 1]


def test_ss_domain_log2_aa_group_ratios():
    """ss_domain_log2_aa_group_ratios: 4 residues, d1 (2 Nonpolar_Aliphatic), d2 (1 Nonpolar_Aliphatic, 1 Aromatic)."""
    # ALA -> Nonpolar_Aliphatic, PHE -> Aromatic
    merged = pd.DataFrame({
        'chain': ['A'] * 4 + ['B'],
        'resi_struct': [1, 2, 3, 4, 1],
        'resn_struct': ['ALA', 'ALA', 'ALA', 'PHE', 'ALA'],
        'ss_domains': ['d1', 'd1', 'd2', 'd2', 'b1'],
    })
    out = secondary_structure.ss_domain_log2_aa_group_ratios(merged)
    out = out.sort_values(['chain', 'ss_domains']).reset_index(drop=True)
    d1 = out[out['ss_domains'] == 'd1'].iloc[0]
    d2 = out[out['ss_domains'] == 'd2'].iloc[0]
    b1 = out[out['ss_domains'] == 'b1'].iloc[0]
    # Global: 3 Nonpolar_Aliphatic, 1 Aromatic -> prop 3/4, 1/4
    assert d1['log2_ratio_Nonpolar_Aliphatic'] == pytest.approx(np.log2(1.0 / (3/4)))
    assert d1['log2_ratio_Aromatic'] == -np.inf
    assert d2['log2_ratio_Nonpolar_Aliphatic'] == pytest.approx(np.log2(0.5 / (3/4)))
    assert d2['log2_ratio_Aromatic'] == pytest.approx(np.log2(0.5 / (1/4)))
    
    # B1: 1 Nonpolar_Aliphatic, 0 Aromatic -> prop 1/1, 0/1
    assert b1['log2_ratio_Nonpolar_Aliphatic'] == pytest.approx(np.log2(1.0 / (1/1)))
    assert b1['log2_ratio_Aromatic'] == pytest.approx(np.log2(0.0 / (0/1)))
    
