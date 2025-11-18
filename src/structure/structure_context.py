# metrics_core.py
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Set, Any, Protocol, Literal
import inspect
import numpy as np
import pandas as pd

import biotite.structure as struc
from biotite.structure.io.pdb import PDBFile
from biotite.structure.io.pdbx import CIFFile, get_structure as pdbx_get_structure
# ---------------- Registry ----------------
@dataclass(frozen=True)
class MetricMeta:
    name: str
    provides: List[str]             # column names this metric adds
    tags: Set[str] = frozenset()    # e.g., {"geom","packing","water"}
    requires: Set[str] = frozenset()# dependency on other metric outputs (by column name)

class MetricFunc(Protocol):
    def __call__(self, ctx: "Context", **kwargs: Any) -> pd.DataFrame: ...

_REGISTRY: Dict[str, tuple[MetricMeta, MetricFunc]] = {}

def register_metric(*, name: str, provides: Iterable[str],
                    tags: Iterable[str] = (), requires: Iterable[str] = ()):
    meta = MetricMeta(name=name, provides=list(provides),
                      tags=set(tags), requires=set(requires))
    def _wrap(fn: MetricFunc):
        if name in _REGISTRY:
            raise ValueError(f"Metric '{name}' already registered")
        _REGISTRY[name] = (meta, fn)
        return fn
    return _wrap

def metric_names() -> List[str]:
    return sorted(_REGISTRY.keys())

def metrics_with_tag(tag: str) -> List[str]:
    return sorted(m for m,(meta,_) in _REGISTRY.items() if tag in meta.tags)

# --------------- Context ------------------
@dataclass
class Context:
    array: struc.AtomArray | struc.AtomArrayStack
    aa: Optional[struc.AtomArray] = None        # amino-acid only
    res_keys: Optional[pd.DataFrame] = None     # (chain, resi, resn)
    kdtree: Any = None                          # built on demand
    neighbor_cache: Dict[float, list[np.ndarray]] = None # cutoff -> neighbor lists
    extras: Dict[str, Any] = None               # room for DSSP, graphs, etc.

    def __post_init__(self):
        self.neighbor_cache = {}
        self.extras = {} if self.extras is None else self.extras
        if isinstance(self.array, struc.AtomArray):
            aa = self.array[struc.filter_amino_acids(self.array)]
        else:
            aa0 = self.array[0]
            aa = aa0[struc.filter_amino_acids(aa0)]
        self.aa = aa
        self.res_keys = residue_table(aa)

def residue_table(array: struc.AtomArray) -> pd.DataFrame:
    res_starts = struc.get_residue_starts(array)
    chains = array.chain_id[res_starts]
    resi   = array.res_id[res_starts]
    resn   = array.res_name[res_starts]
    return pd.DataFrame({"chain": chains, "resi": resi, "resn": resn})



