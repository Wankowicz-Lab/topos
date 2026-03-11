"""
Metric registry system for protein analysis.

This module provides the registration system for metric functions,
allowing metrics to be discovered and executed by the pipeline.
"""
from __future__ import annotations

from dataclasses import dataclass

# Forward reference to avoid circular import
# Context will be imported from pipeline.context
from typing import TYPE_CHECKING, Any, Callable, Dict, FrozenSet, Iterable, List, Protocol

import pandas as pd

if TYPE_CHECKING:
    from src.pipeline.context import Context


@dataclass(frozen=True)
class MetricMeta:
    """
    Metadata for a registered metric function.

    Attributes
    ----------
    name : str
        Unique name of the metric.
    provides : List[str]
        Column names this metric adds to the output DataFrame.
    tags : Set[str]
        Tags for categorizing the metric (e.g., 'structure', 'sequence').
    requires : Set[str]
        Dependency on other metric outputs by column name.
    """
    name: str
    provides: List[str]
    tags: FrozenSet[str] = frozenset()
    requires: FrozenSet[str] = frozenset()


class MetricFunc(Protocol):
    """Protocol for metric functions."""
    def __call__(self, ctx: "Context", **kwargs: Any) -> pd.DataFrame: ...


_REGISTRY: Dict[str, tuple[MetricMeta, MetricFunc]] = {}


def register_metric(
    *,
    name: str,
    provides: Iterable[str],
    tags: Iterable[str] = (),
    requires: Iterable[str] = ()
) -> Callable[[MetricFunc], MetricFunc]:
    """
    Decorator to register a metric function.

    Parameters
    ----------
    name : str
        Unique name for the metric.
    provides : Iterable[str]
        Column names this metric provides in its output DataFrame.
    tags : Iterable[str], optional
        Tags for categorizing the metric. Default is empty.
    requires : Iterable[str], optional
        Column names this metric depends on. Default is empty.

    Returns
    -------
    Callable
        Decorator function that registers the metric.

    Raises
    ------
    ValueError
        If a metric with the same name is already registered with a
        different function.
    """
    meta = MetricMeta(
        name=name,
        provides=list(provides),
        tags=frozenset(tags),
        requires=frozenset(requires),
    )
    def _wrap(fn: MetricFunc):
        if name in _REGISTRY:
            existing_meta, existing_fn = _REGISTRY[name]
            if existing_fn is not fn:
                raise ValueError(f"Metric '{name}' already registered with a different function")
        else:
            _REGISTRY[name] = (meta, fn)
        return fn
    return _wrap


def metric_names() -> List[str]:
    """
    Get all registered metric names.

    Returns
    -------
    List[str]
        Sorted list of registered metric names.
    """
    return sorted(_REGISTRY.keys())


def metrics_with_tag(tag: str) -> List[str]:
    """
    Get metric names that have a specific tag.

    Parameters
    ----------
    tag : str
        Tag to filter by.

    Returns
    -------
    List[str]
        Sorted list of metric names with the specified tag.
    """
    return sorted(m for m,(meta,_) in _REGISTRY.items() if tag in meta.tags)
