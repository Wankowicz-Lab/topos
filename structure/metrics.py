# metrics_impl.py
from __future__ import annotations
import numpy as np
import pandas as pd
import biotite.structure as struc
from biotite.structure.sasa import sasa as sr_sasa
from .metrics_core import Context, register_metric

@register_metric(name="packing_density", provides=["packing_density"], tags={"packing"})
def m_packing(ctx: Context, radius: float = 6.0) -> pd.DataFrame:
    aa = ctx.aa
    if ctx.kdtree is None:
        ctx.kdtree = struc.KDTree(aa.coord)
    if radius not in ctx.neighbor_cache:
        ctx.neighbor_cache[radius] = ctx.kdtree.query_radius(aa.coord, radius)
    neigh = ctx.neighbor_cache[radius]
    atom_counts = np.fromiter((len(n)-1 for n in neigh), dtype=float)
    res_idx = struc.get_residue_indices(aa)
    res_counts = np.bincount(res_idx, weights=atom_counts) / np.bincount(res_idx)
    out = ctx.res_keys.copy()
    out["packing_density"] = res_counts
    return out

@register_metric(name="contacts", provides=["n_contacts"], tags={"contacts"})
def m_contacts(ctx: Context, cutoff: float = 4.5) -> pd.DataFrame:
    aa = ctx.aa
    if ctx.kdtree is None:
        ctx.kdtree = struc.KDTree(aa.coord)
    key = ("contacts", cutoff)
    if key not in ctx.neighbor_cache:
        ctx.neighbor_cache[key] = ctx.kdtree.query_radius(aa.coord, cutoff)
    neigh = ctx.neighbor_cache[key]
    atom_counts = np.fromiter((len(n)-1 for n in neigh), dtype=float)
    res_idx = struc.get_residue_indices(aa)
    res_counts = np.bincount(res_idx, weights=atom_counts) / np.bincount(res_idx)
    out = ctx.res_keys.copy()
    out["n_contacts"] = res_counts
    return out
