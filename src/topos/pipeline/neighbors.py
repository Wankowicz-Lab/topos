from typing import Dict, List

import biotite.structure as struc
import numpy as np
import pandas as pd

from topos.metrics.neighborhood_metrics import NEIGHBORHOOD_METRIC_FUNCTIONS
from topos.pipeline.context import Context
from topos.structure.utils import is_heavy, res_key


def compute_residue_neighbors(
    context: Context,
    cutoff: float,
) -> Dict[str, List[str]]:
    """Compute residue neighbors from heavy amino-acid atoms within a distance cutoff."""
    array = context.aa
    res_starts = struc.get_residue_starts(array)
    chains = array.chain_id[res_starts]
    res_ids = array.res_id[res_starts]
    res_names = array.res_name[res_starts]
    full_keys = np.array(
        [res_key(ch, ri, rn) for ch, ri, rn in zip(chains, res_ids, res_names)],
        dtype=object,
    )

    aa_mask = struc.filter_amino_acids(array)
    heavy_mask = np.array([is_heavy(name) for name in array.atom_name], dtype=bool)
    mask = aa_mask & heavy_mask
    arr = array[mask]

    if arr.array_length() == 0:
        mapping: Dict[str, List[str]] = {key: [] for key in full_keys.tolist()}
        context.extras["residue_neighbors"] = mapping
        return mapping

    residue_ids = np.array(
        [res_key(c, r, rn) for c, r, rn in zip(arr.chain_id, arr.res_id, arr.res_name)],
        dtype=object,
    )

    unique_res = np.unique(residue_ids)
    coords = arr.coord.astype(float)
    cutoff2 = cutoff * cutoff

    mapping = {}
    for res_uid in unique_res:
        idxs = np.where(residue_ids == res_uid)[0]
        if len(idxs) == 0:
            continue
        res_atoms = arr[idxs]

        rcoords = res_atoms.coord
        diff = rcoords[:, None, :] - coords[None, :, :]
        d2 = np.einsum("ijk,ijk->ij", diff, diff)

        within_cutoff = d2 <= cutoff2
        close_atom_idxs = np.where(within_cutoff.any(axis=0))[0]
        neighbor_res_keys = set(residue_ids[close_atom_idxs].tolist())
        neighbor_res_keys.discard(res_uid)
        mapping[str(res_uid)] = sorted(neighbor_res_keys)

    for key in full_keys.tolist():
        if key not in mapping:
            mapping[key] = []

    context.extras["residue_neighbors"] = mapping
    return mapping


def calculate_neighborhood_features(
    context: Context,
    features: pd.DataFrame,
) -> pd.DataFrame:
    """Run registered neighborhood metric functions and merge their outputs."""
    merge_cols = ["chain", "resi_struct", "resn_struct"]
    base = features[merge_cols].drop_duplicates().reset_index(drop=True)

    for func in NEIGHBORHOOD_METRIC_FUNCTIONS:
        df = func(context, features)
        base = pd.merge(base, df, on=merge_cols, how="left")

    return base
