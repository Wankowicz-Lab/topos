# structure_context.py
"""
Structure context and metric registry for protein analysis.

This module provides the Context class for managing protein structure data
and a registry for metric functions.
"""
from __future__ import annotations
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Set, Any, Protocol, Literal, Union
import numpy as np
import pandas as pd
import biotite.structure as struc
from biotite.structure.io.pdb import PDBFile
from biotite.structure.io.pdbx import CIFFile, get_structure
from pydantic import BaseModel

logger = logging.getLogger(__name__)


# ---------------- Registry ----------------
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
    tags: Set[str] = frozenset()
    requires: Set[str] = frozenset()


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
    meta = MetricMeta(name=name, provides=list(provides),
                      tags=set(tags), requires=set(requires))
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

# --------------- Context ------------------
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
    pdb_path : Optional[Path]
        Local path to structure file (PDB or mmCIF format).
    pdb_ext : Optional[str]
        File extension of the structure file.
    membrane_protein : Optional[bool]
        Whether the protein is a membrane protein (affects analysis methods).
    membrane_thickness : Optional[float]
        Half-thickness of membrane in Angstroms (default: 15).
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
        Path to amino acid index database (default: 'data/aaindex_parsed_small.csv').
    kidera_path: Path
        Path to Kidera factors data (default: 'data/kidera_factors.csv').
    """


    # Allow values to be changed after initialization
    model_config = {"validate_assignment": True}

    # structure data
    name: Optional[str] = None
    pdb_id: Optional[str] = None
    pdb_path: Optional[Path] = None
    pdb_ext: Optional[str] = None
    membrane_protein: Optional[bool] = False

    # structure parameters
    membrane_thickness: Optional[float] = 15

    # mutagenesis data
    mutation_data_path: Optional[Path] = None
    mutation_data_chain: Optional[str] = None
    alignment_cutoff: float = 0.95
    mutation_residue_col_name: str = "wildtype"
    mutation_residue_idx_name: str = "position"
    mutation_col_name: str = "mutation"
    mutation_type_col_name: str = "type"
    mutation_score_col_name: str = "effect"

    # sequence features
    aaindex_path: Path = 'data/aaindex_parsed_small.csv'
    kidera_path: Path = 'data/kidera_factors.csv'

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
    aa: Optional[struc.AtomArray] = None
    residue_table: Optional[pd.DataFrame] = None
    kdtree: Any = None
    neighbor_cache: Optional[Dict[float, list[np.ndarray]]] = None
    extras: Optional[Dict[str, Any]] = None
    config: Optional[Config] = None

    def __post_init__(self):
        self.neighbor_cache = {}
        self.extras = {} if self.extras is None else self.extras
        
        if self.config is None:
            self.config = Config()

        if isinstance(self.array, struc.AtomArray):
            self.array = _ensure_altloc_annotation(self.array)
            aa = self.array[struc.filter_amino_acids(self.array)]
        else:
            aa0 = self.array[0]
            aa = aa0[struc.filter_amino_acids(aa0)]
        self.aa = aa
        self.residue_table = residue_table(aa)
        
        if self.config.aaindex_path is not None:
            aa_index = pd.read_csv(self.config.aaindex_path)
            self.extras['aaindex'] = aa_index

        if self.config.kidera_path is not None:
            kidera_factors = pd.read_csv(self.config.kidera_path)
            self.extras['kidera'] = kidera_factors

def residue_table(array: struc.AtomArray) -> pd.DataFrame:
    """
    Create a residue table from an AtomArray.

    Parameters
    ----------
    array : struc.AtomArray
        Biotite AtomArray containing protein structure data.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns 'chain', 'resi', 'resn', 'altloc' for each residue.
    """
    res_starts = struc.get_residue_starts(array)
    chains = array.chain_id[res_starts]
    resi   = array.res_id[res_starts]
    resn   = array.res_name[res_starts]
    altloc = array.altloc[res_starts]
    
    return pd.DataFrame({"chain": chains, "resi": resi, "resn": resn, "altloc": altloc})

def load_structure(
    path: Union[str, Path],
    model: Optional[int] = 1,
    altloc_policy: Literal["highest", "all"] = "highest",
    pdb_ext: str = "pdb"
) -> struc.AtomArray:
    """
    Load a protein structure from a PDB or mmCIF file.

    Parameters
    ----------
    path : str or Path
        Path to the structure file (PDB or mmCIF format).
    model : int, optional
        Model number to load. Default is 1. Use None to load all models.
    altloc_policy : {'highest', 'all'}, optional
        Policy for handling alternate locations. 'highest' keeps the
        highest occupancy conformer, 'all' keeps all conformers.
        Default is 'highest'.
    pdb_ext : str, optional
        File extension hint ('pdb', 'cif', or 'mmcif'). Default is 'pdb'.

    Returns
    -------
    struc.AtomArray
        Loaded protein structure. The 'altloc' annotation contains the alternate
        location identifier for each atom (empty string if no alternate location).
    """
    extra_fields = ["b_factor", "occupancy"]
    
    if pdb_ext in ("cif", "mmcif"):
        cif = CIFFile.read(str(path))
        arr = get_structure(cif, model=model or 1, extra_fields=extra_fields)
    else:
        pdb = PDBFile.read(str(path))
        models = pdb.get_model_count()
        if model is None and models > 1:
            arr = pdb.get_structure(model=None, extra_fields=extra_fields)
        else:
            arr = pdb.get_structure(model=model or 1, extra_fields=extra_fields)
    
    # Ensure altloc annotation exists
    if isinstance(arr, struc.AtomArray):
        arr = _ensure_altloc_annotation(arr)
        
        if altloc_policy == "highest":
            keep = _keep_highest_occ_per_atom(arr)
            arr = arr[keep]
    elif isinstance(arr, struc.AtomArrayStack):
        # For multi-model structures, ensure altloc on each model
        for i in range(arr.stack_depth()):
            arr[i] = _ensure_altloc_annotation(arr[i])
    
    return arr


def _ensure_altloc_annotation(array: struc.AtomArray) -> struc.AtomArray:
    """
    Ensure the array has an 'altloc' annotation.
    
    If 'altloc_id' exists (from PDB file), copy it to 'altloc'.
    Otherwise, create an empty 'altloc' annotation.
    
    Parameters
    ----------
    array : struc.AtomArray
        Input atom array.
    
    Returns
    -------
    struc.AtomArray
        Array with 'altloc' annotation guaranteed to exist.
    """
    if "altloc" not in array.get_annotation_categories():
        if "altloc_id" in array.get_annotation_categories():
            # Copy altloc_id to altloc, normalizing empty values
            altloc_vals = np.array([
                str(a).strip() if a is not None else '' 
                for a in array.altloc_id
            ])
            array.set_annotation("altloc", altloc_vals)
        else:
            # No altloc information available, set empty strings
            array.set_annotation("altloc", np.array([''] * array.array_length()))
    return array


def _keep_highest_occ_per_atom(array: struc.AtomArray) -> np.ndarray:
    """Keep atoms with the highest occupancy for each unique atom position."""
    keep = np.zeros(array.array_length(), dtype=bool)
    for idx in struc.group(array, ["chain_id","res_id","atom_name"]):
        occ = array.occupancy[idx]
        keep[idx[int(np.argmax(occ))]] = True
    return keep

