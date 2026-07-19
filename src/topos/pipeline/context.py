"""
Pipeline context and configuration for protein analysis.

This module provides the Context and Config classes for managing protein
structure data, configuration, and cached computations in the analysis pipeline.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from importlib.resources import files
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

import biotite.structure as struc
import numpy as np
import pandas as pd
from pydantic import BaseModel, Field

from topos.metrics.aaindex_schema import validate_aaindex_columns

# Import structure-loading helpers
from topos.structure.structure_context import ensure_altloc_annotation, residue_table


def _bundled_data_file(name: str) -> Path:
    """Resolve a CSV shipped inside the topos package."""
    return Path(str(files("topos.data").joinpath(name)))


class Config(BaseModel):
    """
    Configuration settings for protein structure analysis pipeline.

    This class manages all configurable parameters for the pipeline including structure data sources,
    membrane protein settings, mutagenesis data, and feature calculation options.

    Attributes
    ----------
    name: Optional[str]
        Name of the protein
    pdb_id : Optional[str]
        PDB identifier for fetching structure from RCSB.
    uniprot_id : Optional[str]
        UniProt accession for fetching an AlphaFold-predicted structure.
    pdb_path : Optional[Path]
        Local path to structure file (PDB or mmCIF format).
    membrane_protein : Optional[bool]
        Whether the protein is a membrane protein (affects analysis methods).
    membrane_thickness : Optional[float]
        Half-thickness of membrane in Angstroms (default: 15).
    remove_hydrogens : bool
        Whether to remove hydrogen atoms from the structure after loading (default: True).
    altloc_policy : Literal["highest", "all"] = "highest"
        Policy for handling alternate locations. 'highest' keeps the
        highest occupancy conformer, 'all' keeps all conformers.
    mutation_data_path : Optional[Path]
        Path to CSV file containing mutagenesis data.
    mutation_data_chain : Optional[str]
        Chain identifier for mutagenesis data alignment.
    mutation_residue_col_name : str
        Column name for wildtype residues in mutation data (default: "wildtype").
    mutation_residue_idx_name : str
        Column name for residue positions in mutation data (default: "position").
    mutation_col_name : str
        Column name for mutant residues in mutation data (default: "mutation").
    mutation_type_col_name : str
        Column name for mutation types in mutation data (default: "type").
    mutation_score_col_name : str
        Column name for mutation effect scores in mutation data (default: "effect").
    aaindex_path : Path
        Path to amino acid index database (default: packaged aaindex CSV).
    kidera_path: Path
        Path to Kidera factors data (default: packaged kidera CSV).
    structural_feature_chains : Optional[List[str]]
        List of chain IDs to include in structural feature calculations.
        If None (default), all chains are included.
    """


    # Allow values to be changed after initialization
    model_config = {"validate_assignment": True}

    # structure data
    name: Optional[str] = None
    pdb_id: Optional[str] = None
    uniprot_id: Optional[str] = None
    pdb_path: Optional[Path] = None
    membrane_protein: Optional[bool] = False

    # structure parameters
    membrane_thickness: Optional[float] = 15
    remove_hydrogens: bool = True
    altloc_policy: Literal["highest", "all"] = "highest"
    structural_feature_chains: Optional[List[str]] = None

    # mutagenesis data
    mutation_data_path: Optional[Path] = None
    mutation_data_chain: Optional[str] = None
    alignment_cutoff: float = 0.95
    mutation_residue_col_name: str = "wildtype"
    mutation_residue_idx_name: str = "position"
    mutation_col_name: str = "mutation"
    mutation_type_col_name: str = "type"
    mutation_score_col_name: str = "effect"

    # sequence features (defaults resolve to package data)
    aaindex_path: Path = Field(
        default_factory=lambda: _bundled_data_file("aaindex_parsed_small.csv")
    )
    kidera_path: Path = Field(
        default_factory=lambda: _bundled_data_file("kidera_factors.csv")
    )

    # pipeline parameters
    output_dir: Optional[Path] = None
    output_prefix: Optional[str] = None

    def model_post_init(self, __context):
        if self.mutation_data_path is not None:
            if not Path(self.mutation_data_path).is_file():
                raise ValueError(f"Mutation data file not found at {self.mutation_data_path}")

            if self.mutation_data_chain is None:
                raise ValueError("If mutation_data_path is provided, "
                                 "mutation_data_chain must also be provided.")

        if not Path(self.aaindex_path).is_file():
            raise ValueError(f"AA index data file not found at {self.aaindex_path}")

        if not Path(self.kidera_path).is_file():
            raise ValueError(f"Kidera factors data file not found at {self.kidera_path}")


@dataclass
class Context:
    """
    Context object for protein structure analysis.

    Holds the protein structure data, configuration, and cached computations.

    Attributes
    ----------
    array : struc.AtomArray or struc.AtomArrayStack
        The protein structure.
    aa : struc.AtomArray, optional
        Amino acid atoms only (filtered from array).
    residue_table : pd.DataFrame, optional
        DataFrame with chain, resi, resn, altloc for each residue. Residues with no altlocs have blanks
    kdtree : Any, optional
        KD-tree for spatial queries (built on demand).
    neighbor_cache : dict
        Cache for neighbor computations keyed by cutoff distance.
    extras : dict
        Additional data storage (e.g., aaindex data).
    config : Config, optional
        Configuration settings for the analysis.
    """
    array: struc.AtomArray | struc.AtomArrayStack
    aa: struc.AtomArray = field(init=False)
    residue_table: pd.DataFrame = field(init=False)
    kdtree: Any = None
    neighbor_cache: Dict[float, list[np.ndarray]] = field(default_factory=dict)
    extras: Dict[str, Any] = field(default_factory=dict)
    config: Config = field(default_factory=Config)

    def __post_init__(self):
        if isinstance(self.array, struc.AtomArray):
            self.array = ensure_altloc_annotation(self.array)
            aa = self.array[struc.filter_amino_acids(self.array)]
        else:
            aa0 = self.array[0]
            aa = aa0[struc.filter_amino_acids(aa0)]
        self.aa = aa
        self.residue_table = residue_table(aa)
        
        if self.config.aaindex_path is not None:
            aa_index = pd.read_csv(self.config.aaindex_path)
            validate_aaindex_columns(aa_index)
            self.extras['aaindex'] = aa_index

        if self.config.kidera_path is not None:
            kidera_factors = pd.read_csv(self.config.kidera_path)
            self.extras['kidera'] = kidera_factors
