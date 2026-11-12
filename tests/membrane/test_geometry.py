"""Tests for helix-axis membrane-frame estimation."""

import biotite.structure as struc
import numpy as np
import pandas as pd
import pytest

from topos.databases.pdbtm import transform_coordinates
from topos.membrane.geometry import (
    InsufficientTransmembraneHelices,
    estimate_membrane_tmatrix,
    helix_axis_unit,
)


def _helix_ca_coords(xy, z_start, n=8, dz=1.5):
    """Straight helix of Cα points parallel to +z at fixed (x, y)."""
    x, y = xy
    return np.array([[x, y, z_start + i * dz] for i in range(n)], dtype=float)


def _atom_array_from_helices(helix_coords_list):
    """Build a minimal AtomArray with one Cα per coordinate, chained as A."""
    coords = np.vstack(helix_coords_list)
    n = len(coords)
    array = struc.AtomArray(n)
    array.coord = coords
    array.chain_id[:] = "A"
    array.res_id = np.arange(1, n + 1)
    array.res_name[:] = "ALA"
    array.atom_name[:] = "CA"
    array.element[:] = "C"
    return array


def test_helix_axis_unit_along_z():
    pts = _helix_ca_coords((0.0, 0.0), z_start=-5.0)
    u = helix_axis_unit(pts)
    assert u is not None
    assert abs(abs(u[2]) - 1.0) < 1e-6


def test_estimate_membrane_tmatrix_recovers_z_normal():
    # Three parallel TM helices along z, midpoints near z=0.
    helices = [
        _helix_ca_coords((0.0, 0.0), z_start=-5.25),
        _helix_ca_coords((8.0, 0.0), z_start=-5.25),
        _helix_ca_coords((4.0, 7.0), z_start=-5.25),
    ]
    array = _atom_array_from_helices(helices)
    # Residues 1-8, 9-16, 17-24
    regions = pd.DataFrame(
        {
            "chain": ["A", "A", "A"],
            "type": ["transmembrane_helix"] * 3,
            "seq_beg": [1, 9, 17],
            "seq_end": [8, 16, 24],
            "pdb_beg": [1, 9, 17],
            "pdb_end": [8, 16, 24],
        }
    )
    tmatrix = estimate_membrane_tmatrix(array, regions)
    transformed = transform_coordinates(array.coord, tmatrix)

    # Membrane normal maps to z; TM Cα midplane near z=0.
    mid_z = float(np.median(transformed[:, 2]))
    assert abs(mid_z) < 0.5
    # Helix axes remain mostly along ±z after transform.
    assert np.std(transformed[:8, 0]) < 0.2
    assert np.std(transformed[:8, 1]) < 0.2


def test_estimate_membrane_tmatrix_too_few_helices():
    helices = [
        _helix_ca_coords((0.0, 0.0), z_start=-5.0),
        _helix_ca_coords((8.0, 0.0), z_start=-5.0),
    ]
    array = _atom_array_from_helices(helices)
    regions = pd.DataFrame(
        {
            "chain": ["A", "A"],
            "type": ["transmembrane_helix"] * 2,
            "seq_beg": [1, 9],
            "seq_end": [8, 16],
            "pdb_beg": [1, 9],
            "pdb_end": [8, 16],
        }
    )
    with pytest.raises(InsufficientTransmembraneHelices, match="≥3"):
        estimate_membrane_tmatrix(array, regions)
