"""
Configuration models for grouped structure comparison analysis.
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Literal

import tomli
from pydantic import BaseModel, model_validator


# ---------------------------------------------------------------------------
# Metric category definitions
# ---------------------------------------------------------------------------
# Each category maps to a list of (match_type, pattern) tuples.
# match_type: "exact" | "prefix" | "suffix" | "contains"
METRIC_CATEGORY_PATTERNS: dict[str, list[tuple[str, str]]] = {
    "structural": [
        ("prefix", "sasa"),
        ("prefix", "distance_"),
        ("prefix", "kyte_"),
        ("prefix", "packing_"),
        ("exact", "distance_to_center_of_mass"),
        ("exact", "distance_to_nearest_surface_residue"),
        ("exact", "distance_from_membrane_edge"),
    ],
    "bonds": [
        ("suffix", "_count"),
    ],
    "graph": [
        ("contains", "_betweenness_centrality"),
        ("contains", "_closeness_centrality"),
        ("contains", "_eigenvector_centrality"),
        ("contains", "_core_number"),
        ("contains", "_in_lcc"),
        # Exclude community_id — handled by graph_diff
    ],
    "sequence": [
        ("exact", "blosum90"),
        ("exact", "phat_score"),
        ("prefix", "AAIndex_"),
        ("prefix", "kidera_"),
    ],
    "ligand": [
        ("suffix", "_interactions"),
    ],
}

# Columns that should never be included in numeric comparisons
EXCLUDED_COLUMNS = {
    "chain", "resi_struct", "resn_struct", "resi_mut", "resn_mut", "resm",
    "name", "ss_domains", "wildtype_aa_group", "mut_aa_group",
    "wildtype_mut_aa_group", "effect_quartile", "effect_ranking",
    "_label", "_group",
    # community IDs are categorical — excluded from numeric comparison
    "graph_all_graph_community_id",
    "graph_vdw_contact_graph_community_id",
    "graph_hbond_graph_community_id",
}


class StructureEntry(BaseModel):
    """A single structure's features CSV file with a label and group."""
    path: Path
    label: str
    group: str

    @model_validator(mode="after")
    def path_exists(self) -> "StructureEntry":
        if not self.path.is_file():
            raise FileNotFoundError(
                f"Structure features file not found: {self.path}"
            )
        return self


class ComparisonConfig(BaseModel):
    """Settings controlling how groups are compared."""
    mode: Literal["group", "all_vs_all"] = "group"
    reference_group: Optional[str] = None
    residue_key: str = "resi_struct"
    chain: Optional[str] = None  # None = all chains


class MetricsConfig(BaseModel):
    """Which metric columns to include in comparison."""
    include_categories: List[str] = ["structural", "graph", "bonds"]
    exclude_columns: List[str] = []
    custom_columns: Optional[List[str]] = None  # overrides categories if set

    @model_validator(mode="after")
    def validate_categories(self) -> "MetricsConfig":
        valid = set(METRIC_CATEGORY_PATTERNS.keys()) | {"all"}
        for cat in self.include_categories:
            if cat not in valid:
                raise ValueError(
                    f"Unknown metric category '{cat}'. "
                    f"Valid: {sorted(valid)}"
                )
        return self


class GroupedAnalysisConfig(BaseModel):
    """Top-level config for a grouped analysis run."""
    name: str
    output_dir: Path = Path("output/")
    structures: List[StructureEntry]
    comparison: ComparisonConfig = ComparisonConfig()
    metrics: MetricsConfig = MetricsConfig()

    @model_validator(mode="after")
    def validate_groups(self) -> "GroupedAnalysisConfig":
        groups = {e.group for e in self.structures}
        if self.comparison.mode == "group" and self.comparison.reference_group:
            if self.comparison.reference_group not in groups:
                raise ValueError(
                    f"reference_group '{self.comparison.reference_group}' "
                    f"not found in structure groups: {groups}"
                )
        return self

    @classmethod
    def from_toml(cls, path: str | Path) -> "GroupedAnalysisConfig":
        """Load config from a TOML file, resolving relative paths relative to the TOML."""
        path = Path(path).resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Config file not found: {path}")

        with open(path, "rb") as f:
            data = tomli.load(f)

        toml_dir = path.parent

        # Resolve structure paths relative to the TOML directory
        for entry in data.get("structures", []):
            entry_path = Path(entry["path"])
            if not entry_path.is_absolute():
                entry["path"] = str(toml_dir / entry_path)

        # Resolve output_dir relative to TOML directory
        if "output_dir" in data:
            od = Path(data["output_dir"])
            if not od.is_absolute():
                data["output_dir"] = str(toml_dir / od)

        return cls.model_validate(data)
