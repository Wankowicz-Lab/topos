# metrics_api.py
from __future__ import annotations
from typing import Iterable, Optional, Dict, Any, List, Tuple, Set
import pandas as pd
from .structure_context import Context, load_structure, load_structure_with_id, metric_names, metrics_with_tag, _REGISTRY

def compute_all(pdb_or_array, *, metrics: Optional[Iterable[str]] = None, tags: Optional[Iterable[str]] = None,
                model: Optional[int] = 1, **shared_kwargs) -> pd.DataFrame:
    """
    Run many metrics and outer-merge by residue keys.
    - metrics: explicit list; if None, use all registered (or filtered by tags).
    - tags: include metrics that have ANY of these tags.
    """
    pdb_id = None
    if not hasattr(pdb_or_array, "coord"):
        arr, pdb_id = load_structure_with_id(pdb_or_array, model=model)
    else:
        arr = pdb_or_array
    ctx = Context(arr, pdb_id=pdb_id)

    wanted: List[str]
    if metrics:
        wanted = list(metrics)
    elif tags:
        names = set()
        for t in tags:
            names.update(metrics_with_tag(t))
        wanted = sorted(names)
    else:
        wanted = metric_names()

    frames: List[pd.DataFrame] = []
    for name in wanted:
        meta, fn = _REGISTRY[name]
        df = fn(ctx, **_filter_kwargs(fn, shared_kwargs))
        frames.append(df)

    out = frames[0]
    for df in frames[1:]:
        out = out.merge(df, on=["chain","resi","ins","resn"], how="outer")
    return out.sort_values(["chain","resi"]).reset_index(drop=True)

def compute_metrics(pdb_or_array, metric_list: Iterable[str], **kwargs) -> pd.DataFrame:
    return compute_all(pdb_or_array, metrics=metric_list, **kwargs)

def _filter_kwargs(fn, kwargs: Dict[str, Any]) -> Dict[str, Any]:
    import inspect
    sig = inspect.signature(fn)
    return {k: v for k, v in kwargs.items() if k in sig.parameters}
