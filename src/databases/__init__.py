"""
External database integrations for protein analysis.

This module provides integrations with external databases such as PDBTM.
"""
from .pdbtm import (
    fetch_pdbtm_annotation,
    transform_coordinates,
    add_pdbtm_regions,
    annotate_pdbtm_detailed,
    describe_pdbtm_region,
)

__all__ = [
    "fetch_pdbtm_annotation",
    "transform_coordinates",
    "add_pdbtm_regions",
    "annotate_pdbtm_detailed",
    "describe_pdbtm_region",
]
