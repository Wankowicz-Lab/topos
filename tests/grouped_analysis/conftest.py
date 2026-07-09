"""
Shared pytest fixtures for grouped_analysis tests.
"""
import sys
import textwrap
from pathlib import Path

import pandas as pd
import pytest


@pytest.fixture
def minimal_toml(tmp_path):
    """Create a minimal valid pipeline config for testing."""
    cfg_path = tmp_path / "config.toml"
    cfg_path.write_text(textwrap.dedent("""\
        output_dir = "output/"
        reference_pdb = "4AKE"
        chain = "A"
        max_mismatches = 5
        proximity_angstroms = 8.0
        top_n_variable = 20

        [analysis]
        run_multi = true
        run_comparison = true

        [[structures]]
        label = "4AKE"
        pdb_id = "4AKE"
        state = "apo"
        genotype = "wt"
        chain = "A"

        [[structures]]
        label = "1AKE"
        pdb_id = "1AKE"
        state = "bound"
        genotype = "wt"
        chain = "A"

        [[pairs]]
        reference = "4AKE"
        comparison = "1AKE"
        description = "apo vs bound"
    """))
    return cfg_path


@pytest.fixture
def minimal_features_csv(tmp_path):
    """Create a minimal valid features CSV for a single structure."""
    df = pd.DataFrame({
        "chain": ["A"] * 5,
        "resi_struct": [1, 2, 3, 4, 5],
        "resn_struct": ["ALA", "GLY", "VAL", "LEU", "SER"],
        "resi_mut": [1, 2, 3, 4, 5],
        "resn_mut": ["ALA", "GLY", "VAL", "LEU", "SER"],
        "sasa": [10.0, 20.0, 30.0, 25.0, 15.0],
        "total_hbond_count": [2, 3, 1, 2, 3],
        "packing_contact_density": [0.5, 0.6, 0.4, 0.55, 0.45],
        "vdw_contact_count": [8, 10, 6, 9, 7],
        "salt_bridge_count": [1, 0, 0, 1, 0],
    })
    path = tmp_path / "4AKE_features.csv"
    df.to_csv(path, index=False)
    return path


@pytest.fixture
def synthetic_structures_with_metrics(tmp_path):
    """Create a complete synthetic dataset with multiple structures."""
    import numpy as np
    rng = np.random.default_rng(42)
    
    structures = ["4AKE", "1AKE", "3HPQ"]
    
    for pdb_id in structures:
        df = pd.DataFrame({
            "chain": ["A"] * 10,
            "resi_struct": list(range(1, 11)),
            "resn_struct": ["ALA", "GLY", "VAL", "LEU", "SER"] * 2,
            "resi_mut": list(range(1, 11)),
            "resn_mut": ["ALA", "GLY", "VAL", "LEU", "SER"] * 2,
            "sasa": rng.uniform(5, 40, 10).tolist(),
            "total_hbond_count": rng.integers(0, 5, 10).tolist(),
            "packing_contact_density": rng.uniform(0.3, 0.7, 10).tolist(),
            "vdw_contact_count": rng.integers(5, 15, 10).tolist(),
            "dssp_phi": rng.uniform(-180, 180, 10).tolist(),
            "dssp_psi": rng.uniform(-180, 180, 10).tolist(),
        })
        path = tmp_path / f"{pdb_id}_features.csv"
        df.to_csv(path, index=False)
    
    return tmp_path, structures
