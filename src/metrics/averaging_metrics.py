"""Shared metric lists used by averaging integrations."""

from __future__ import annotations

# Feature columns eligible for both ss-domain and neighborhood averaging.
METRICS_TO_AVERAGE: list[str] = [
    "pos_effect",
    "effect_variance",
    "effect_variance_rank",
    "effect",
    "effect_ranking",
    "sasa",
    "sasa_backbone",
    "sasa_sidechain",
    "sasa_polar",
    "sasa_nonpolar",
    "kyte_doolittle",
    "distance_from_membrane_edge",
    "bb_hbond_count",
    "sc_hbond_count",
    "total_hbond_count",
    "packing_n_atoms",
    "packing_n_neighbor_residues",
    "packing_contact_density",
    "blosum90",
    "phat_score",
]
