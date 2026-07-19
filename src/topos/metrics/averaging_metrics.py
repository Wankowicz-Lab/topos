"""Shared rules for spatial pooling (neighbor + SS domain) and sequence-window inputs."""

from __future__ import annotations

import pandas as pd
from pandas.api.types import is_bool_dtype, is_numeric_dtype

# Columns that are never pooled (merge keys, labels, config echo, derived SS label on row).
NON_METRIC_COLUMNS: frozenset[str] = frozenset(
    {
        "chain",
        "resi_struct",
        "resn_struct",
        "resi_mut",
        "resn_mut",
        "resm",
        "type",
        "name",
        "align_pos",
        "ss_group",
        "struct_info",
        "mut_info",
        "ss_domains",
    }
)

# Explicit spatial-pool excludes: mutation typing, AA groups, ordinal labels, high-dim biochem.
METRICS_EXCLUDED_FROM_SPATIAL_POOLING: frozenset[str] = frozenset(
    {
        "mutation_category",
        "total_lof",
        "total_gof",
        "avg_effect_quartile",
        "wildtype_aa_group",
        "mut_aa_group",
        "wildtype_mut_aa_group",
    }
)

# Registry gaps: experiment readout columns not always marked requires={'resm'}.
MUTATION_READOUT_COLUMNS: frozenset[str] = frozenset(
    {
        "effect",
        "effect_variance",
        "effect_variance_rank",
        "effect_ranking",
    }
)

_DERIVED_OUTPUT_PREFIXES: tuple[str, ...] = (
    "neighborhood_",
    "ss_domain_",
    "sequence_window_",
    "ligand_",
    "graph_",
)


_resm_provides_columns_cache: frozenset[str] | None = None


def resm_dependent_metric_columns_from_registry() -> frozenset[str]:
    """Column names supplied by metrics that declare dependency on ``resm``."""
    global _resm_provides_columns_cache
    if _resm_provides_columns_cache is None:
        import topos.metrics.bonds  # noqa: F401 — register metric modules before reading registry
        import topos.metrics.sequence  # noqa: F401
        import topos.metrics.structure  # noqa: F401
        from topos.metrics.registry import _REGISTRY

        names: set[str] = set()
        for meta, _ in _REGISTRY.values():
            if "resm" in meta.requires:
                names.update(meta.provides)
        _resm_provides_columns_cache = frozenset(names)
    return _resm_provides_columns_cache


def column_needs_synonym_mask(column: str) -> bool:
    """True if pooled values for synonymous mutation rows must be suppressed (NaN)."""
    if column in MUTATION_READOUT_COLUMNS:
        return True
    if column in resm_dependent_metric_columns_from_registry():
        return True
    if column.endswith(("_mut", "_diff")):
        return True
    return False


def _is_derived_metric_column(name: str) -> bool:
    return any(name.startswith(prefix) for prefix in _DERIVED_OUTPUT_PREFIXES)


def _excluded_biochemical_patterns(name: str) -> bool:
    if name.startswith("kidera_"):
        return True
    if name.endswith("_wt") or name.endswith("_mut") or name.endswith("_diff"):
        return True
    return False


def spatial_pool_metric_columns(features: pd.DataFrame) -> list[str]:
    """Columns eligible for neighbor + SS-domain pooling: exclude metadata, labels, and high-dim biochem."""
    out: list[str] = []
    for column in sorted(features.columns):
        if column in NON_METRIC_COLUMNS:
            continue
        if column in METRICS_EXCLUDED_FROM_SPATIAL_POOLING:
            continue
        if _is_derived_metric_column(column):
            continue
        if _excluded_biochemical_patterns(column):
            continue
        out.append(column)
    return out


def assert_poolable_numeric_columns(columns: list[str], features: pd.DataFrame) -> None:
    """Raise ValueError if any listed column is missing or not poolable numeric (exclude bool)."""
    bad: list[tuple[str, str]] = []
    for column in columns:
        if column not in features.columns:
            bad.append((column, "missing"))
            continue
        series = features[column]
        if is_bool_dtype(series):
            bad.append((column, "bool"))
        elif not is_numeric_dtype(series):
            bad.append((column, str(series.dtype)))
    if bad:
        detail = "; ".join(f"{c} ({reason})" for c, reason in bad)
        raise ValueError(
            "Neighborhood averaging requires poolable numeric columns (non-boolean). Offending: " + detail
        )
