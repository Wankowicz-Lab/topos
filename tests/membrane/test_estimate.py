"""Tests for membrane parameter estimation orchestration."""

import biotite.structure as struc
import numpy as np
import pandas as pd

from topos.databases.pdbtm import transform_coordinates
from topos.membrane.estimate import estimate_membrane_parameters
from topos.pipeline.context import Config, Context


def _toy_membrane_context():
    """Three parallel TM helices with matching residue_table rows."""
    coords_list = []
    for xy in [(0.0, 0.0), (8.0, 0.0), (4.0, 7.0)]:
        x, y = xy
        coords_list.append(
            np.array([[x, y, -5.25 + i * 1.5] for i in range(8)], dtype=float)
        )
    coords = np.vstack(coords_list)
    n = len(coords)
    array = struc.AtomArray(n)
    array.coord = coords
    array.chain_id[:] = "A"
    array.res_id = np.arange(1, n + 1)
    array.res_name[:] = "ALA"
    array.atom_name[:] = "CA"
    array.element[:] = "C"

    ctx = Context(array, config=Config(membrane_protein=True))
    return ctx


def test_estimate_membrane_parameters_applies_tmbed_regions(monkeypatch):
    ctx = _toy_membrane_context()

    def fake_predict(residue_table, chains=None, use_gpu=False):
        return pd.DataFrame(
            {
                "chain": ["A", "A", "A"],
                "type": ["transmembrane_helix"] * 3,
                "seq_beg": [1, 9, 17],
                "seq_end": [8, 16, 24],
                "pdb_beg": [1, 9, 17],
                "pdb_end": [8, 16, 24],
            }
        )

    monkeypatch.setattr(
        "topos.membrane.estimate.predict_tm_regions", fake_predict
    )

    regions, tmatrix = estimate_membrane_parameters(ctx)
    assert len(regions) == 3
    assert ctx.extras["membrane_source"] == "tmbed_estimate"

    transformed = transform_coordinates(ctx.aa.coord, tmatrix)
    assert abs(float(np.median(transformed[:, 2]))) < 0.5
