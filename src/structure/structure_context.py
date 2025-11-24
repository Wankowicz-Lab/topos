# metrics_core.py
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Set, Any, Protocol, Literal
import inspect
import numpy as np
import pandas as pd
import biotite.structure as struc
from biotite.structure.io.pdb import PDBFile
from pydantic import BaseModel
from biotite.structure.io.pdbx import CIFFile, get_structure, get_model_count

# ---------------- Registry ----------------
@dataclass(frozen=True)
class MetricMeta:
    name: str
    provides: List[str]             # column names this metric adds
    tags: Set[str] = frozenset()    # e.g., {"geom","packing","water"}
    requires: Set[str] = frozenset()# dependency on other metric outputs (by column name)

class MetricFunc(Protocol):
    def __call__(self, ctx: "Context", **kwargs: Any) -> pd.DataFrame: ...

_REGISTRY: Dict[str, tuple[MetricMeta, MetricFunc]] = {}

def register_metric(*, name: str, provides: Iterable[str],
                    tags: Iterable[str] = (), requires: Iterable[str] = ()):
    meta = MetricMeta(name=name, provides=list(provides),
                      tags=set(tags), requires=set(requires))
    def _wrap(fn: MetricFunc):
        if name in _REGISTRY:
            raise ValueError(f"Metric '{name}' already registered")
        _REGISTRY[name] = (meta, fn)
        return fn
    return _wrap

def metric_names() -> List[str]:
    return sorted(_REGISTRY.keys())

def metrics_with_tag(tag: str) -> List[str]:
    return sorted(m for m,(meta,_) in _REGISTRY.items() if tag in meta.tags)

# --------------- Context ------------------
class Config(BaseModel):
    """
    Configuration settings for protein structure analysis pipeline.

    This class manages all configurable parameters for the pipeline including structure data sources,
    membrane protein settings, mutagenesis data, and feature calculation options.

    Attributes
    ----------
    pdb_id : Optional[str]
        PDB identifier for fetching structure from RCSB.
    pdb_path : Optional[Path]
        Local path to structure file (PDB or mmCIF format).
    pdb_ext : Optional[str]
        File extension of the structure file.
    membrane_protein : Optional[bool]
        Whether the protein is a membrane protein (affects analysis methods).
    vdw_radii : str
        Van der Waals radii set to use for calculations (default: "ProtOr").
    membrane_thickness : Optional[float]
        Half-thickness of membrane in Angstroms (default: 15).
    mutation_data_path : Optional[Path]
        Path to CSV file containing mutagenesis data.
    mutation_data_chain : Optional[str]
        Chain identifier for mutagenesis data alignment.
    aaindex_path : Path
        Path to amino acid index database (default: 'data/aaindex_parsed_small.csv').
    """


    # Allow values to be changed after initialization
    model_config = {"validate_assignment": True}

    # structure data
    pdb_id: Optional[str] = None
    pdb_path: Optional[Path] = None
    pdb_ext: Optional[str] = None
    membrane_protein: Optional[bool] = False

    # structure parameters
    vdw_radii: str = "ProtOr"
    membrane_thickness: Optional[float] = 15

    # mutagenesis data
    mutation_data_path: Optional[Path] = None
    mutation_data_chain: Optional[str] = None

    # sequence features
    aaindex_path: Path = 'data/aaindex_parsed_small.csv'

    def model_post_init(self, __context):
        if self.mutation_data_path is not None:
            if not Path(self.mutation_data_path).is_file():
                raise ValueError(f"Mutation data file not found at {self.mutation_data_path}")

            if self.mutation_data_chain is None:
                raise ValueError("If mutation_data_path is provided, "
                                 "mutation_data_chain must also be provided.")

        if not Path(self.aaindex_path).is_file():
            raise ValueError(f"AA index data file not found at {self.aaindex_path}")


@dataclass
class Context:
    array: struc.AtomArray | struc.AtomArrayStack
    aa: Optional[struc.AtomArray] = None        # amino-acid only
    residue_table: Optional[pd.DataFrame] = None     # (chain, resi, ins, resn)
    kdtree: Any = None                          # built on demand
    neighbor_cache: Dict[float, list[np.ndarray]] = None # cutoff -> neighbor lists
    extras: Dict[str, Any] = None               # room for DSSP, graphs, etc.
    config: Optional[Config] = None

    def __post_init__(self):
        self.neighbor_cache = {}
        self.extras = {} if self.extras is None else self.extras
        if isinstance(self.array, struc.AtomArray):
            aa = self.array[struc.filter_amino_acids(self.array)]
        else:
            aa0 = self.array[0]
            aa = aa0[struc.filter_amino_acids(aa0)]
        self.aa = aa
        self.residue_table = residue_table(aa)

        if self.config is None:
            self.config = Config()

        if self.config.aaindex_path is not None:
            aa_index = pd.read_csv(self.config.aaindex_path)
            self.extras['aaindex'] = aa_index

def residue_table(array: struc.AtomArray) -> pd.DataFrame:
    res_starts = struc.get_residue_starts(array)
    chains = array.chain_id[res_starts]
    resi   = array.res_id[res_starts]
    ins    = getattr(array, "ins_code", None)
    ins    = ins[res_starts] if ins is not None else np.array([None]*len(res_starts), dtype=object)
    resn   = array.res_name[res_starts]
    return pd.DataFrame({"chain": chains, "resi": resi, "ins": ins, "resn": resn})

def load_structure(path: str | Path,
                   model: Optional[int] = 1,
                   altloc_policy: Literal["occupancy","first","all"] = "occupancy",
                   pdb_ext: str = "pdb") -> struc.AtomArray:


    pdb = PDBFile.read(str(path))
    models = pdb.get_model_count()
    arr = pdb.get_structure(model=None) if (model is None and models > 1) else pdb.get_structure(model or 1)
    if isinstance(arr, struc.AtomArray) and altloc_policy != "all":
        if "altloc_id" in arr.get_annotation_categories():
            if altloc_policy == "occupancy" and "occupancy" in arr.get_annotation_categories():
                keep = _keep_highest_occ_per_atom(arr)
            else:
                keep = _keep_first_altloc_per_atom(arr)
            arr = arr[keep]
    return arr

def _keep_highest_occ_per_atom(array: struc.AtomArray) -> np.ndarray:
    keep = np.zeros(array.array_length(), dtype=bool)
    for idx in struc.group(array, ["chain_id","res_id","atom_name"]):
        occ = array.occupancy[idx]
        keep[idx[int(np.argmax(occ))]] = True
    return keep

def _keep_first_altloc_per_atom(array: struc.AtomArray) -> np.ndarray:
    keep = np.zeros(array.array_length(), dtype=bool)
    for idx in struc.group(array, ["chain_id","res_id","atom_name","altloc_id"]):
        keep[idx[0]] = True
    return keep
