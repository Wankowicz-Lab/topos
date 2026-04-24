"""
Parse a grouped_structures.toml config into StructureEntry objects.

This module is shared by pairwise_rmsd.py and run_analysis.py. This is to group comparison PDBs together. 

Config format
-------------
See example_config.toml for a fully annotated example.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib  # type: ignore[no-redef]


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class MutationSpec:
    resi: int          # residue number in structure numbering
    wt_aa: str         # single-letter WT amino acid
    mut_aa: str        # single-letter mutant amino acid

    def __str__(self) -> str:
        return f"{self.wt_aa}{self.resi}{self.mut_aa}"


@dataclass
class LigandSpec:
    name: str          # 3-letter HET residue name (e.g. "BZB")
    chain: str = "A"   # chain where the ligand lives in the PDB


@dataclass
class StructureEntry:
    label: str
    pdb_id: str
    state: str                          # "apo" | "bound"
    genotype: str                       # "wt" | "mutant"
    chain: str | list[str]
    pdb_path: Optional[str] = None      # local path; None → fetch from RCSB
    ligand: Optional[LigandSpec] = None
    mutations: list[MutationSpec] = field(default_factory=list)

    @property
    def mutation_summary(self) -> str:
        if not self.mutations:
            return ""
        return ", ".join(str(m) for m in self.mutations)


@dataclass
class PairConfig:
    reference: str      # label of the reference StructureEntry
    comparison: str     # label of the comparison StructureEntry
    description: str = ""


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------

def load_config(
    config_path: Path,
) -> tuple[list[StructureEntry], list[PairConfig], dict]:
    """
    Parse *config_path* and return ``(entries, pairs, settings)``.

    Parameters
    ----------
    config_path:
        Path to the TOML config file.

    Returns
    -------
    entries:
        Ordered list of StructureEntry objects (one per ``[[structures]]``
        block in the config).
    pairs:
        List of PairConfig objects (one per ``[[pairs]]`` block).
        If the config contains no ``[[pairs]]`` blocks, every non-reference
        structure is automatically paired against the first WT/apo structure.
    settings:
        Dict of values from the ``[settings]`` block.
    """
    with open(config_path, "rb") as fh:
        data = tomllib.load(fh)

    settings: dict = data.get("settings", {})

    # -- structures ----------------------------------------------------------
    entries: list[StructureEntry] = []
    for raw in data.get("structures", []):
        ligand = None
        raw_lig = raw.get("ligand")
        if isinstance(raw_lig, dict):
            ligand = LigandSpec(
                name=raw_lig["name"],
                chain=raw_lig.get("chain", "A"),
            )
        elif isinstance(raw_lig, str):
            ligand = LigandSpec(name=raw_lig)

        mutations = [
            MutationSpec(
                resi=int(m["resi"]),
                wt_aa=m["wt_aa"],
                mut_aa=m["mut_aa"],
            )
            for m in raw.get("mutations", [])
        ]

        chain = raw.get("chain", "A")

        entries.append(
            StructureEntry(
                label=raw["label"],
                pdb_id=raw["pdb_id"],
                state=raw.get("state", "apo"),
                genotype=raw.get("genotype", "wt"),
                chain=chain,
                pdb_path=raw.get("pdb_path", None),
                ligand=ligand,
                mutations=mutations,
            )
        )

    if not entries:
        sys.exit("ERROR: No [[structures]] blocks found in config.")

    # -- pairs ---------------------------------------------------------------
    raw_pairs = data.get("pairs", [])
    pairs: list[PairConfig] = []

    if raw_pairs:
        label_set = {e.label for e in entries}
        for raw_p in raw_pairs:
            ref = raw_p["reference"]
            cmp = raw_p["comparison"]
            if ref not in label_set:
                sys.exit(f"ERROR: pair 'reference' label '{ref}' not found in [[structures]].")
            if cmp not in label_set:
                sys.exit(f"ERROR: pair 'comparison' label '{cmp}' not found in [[structures]].")
            pairs.append(
                PairConfig(
                    reference=ref,
                    comparison=cmp,
                    description=raw_p.get("description", f"{ref} vs {cmp}"),
                )
            )
    else:
        # Auto-pair: every structure against the first wt/apo entry.
        ref = next(
            (e for e in entries if e.genotype == "wt" and e.state == "apo"),
            entries[0],
        )
        for e in entries:
            if e.label != ref.label:
                pairs.append(
                    PairConfig(
                        reference=ref.label,
                        comparison=e.label,
                        description=f"{ref.label} vs {e.label}",
                    )
                )

    return entries, pairs, settings
