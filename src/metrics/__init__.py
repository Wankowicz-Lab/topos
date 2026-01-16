"""
Metrics module for protein analysis.

This module provides the metric registry system and metric calculation functions
for both sequence and structure-based metrics.
"""
from .registry import (
    MetricMeta,
    MetricFunc,
    register_metric,
    metric_names,
    metrics_with_tag,
    _REGISTRY,
)

__all__ = [
    "MetricMeta",
    "MetricFunc",
    "register_metric",
    "metric_names",
    "metrics_with_tag",
    "_REGISTRY",
]
