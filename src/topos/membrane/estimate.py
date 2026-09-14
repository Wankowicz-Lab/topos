"""Orchestrate TMbed spans + helix-axis membrane-frame estimation."""

from __future__ import annotations

from typing import Tuple

import numpy as np
import pandas as pd

from topos.membrane.geometry import estimate_membrane_tmatrix
from topos.membrane.tmbed import predict_tm_regions
from topos.pipeline.context import Context


def estimate_membrane_parameters(context: Context) -> Tuple[pd.DataFrame, np.ndarray]:
    """
    Estimate PDBTM-compatible regions and transform via TMbed + helix geometry.

    Uses ``structural_feature_chains`` when set; otherwise all amino-acid chains.
    Records ``membrane_source="tmbed_estimate"`` in ``context.extras``.
    """
    chains = context.config.structural_feature_chains
    regions = predict_tm_regions(context.residue_table, chains=chains)
    tmatrix = estimate_membrane_tmatrix(context.aa, regions)
    context.extras["membrane_source"] = "tmbed_estimate"
    return regions, tmatrix
