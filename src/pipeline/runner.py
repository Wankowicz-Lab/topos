from dataclasses import dataclass
from pathlib import Path
import pandas as pd

from biotite.database import rcsb
from biotite.structure.io.pdb import PDBFile
from biotite.structure.io.pdbx import CIFFile, get_structure, get_model_count

from src.structure import structure_context, metrics
from src.sequence import sequence_context
from src.structure import pdbtm

from typing import List, Optional
from src.structure.structure_context import _REGISTRY
import src.sequence.metrics


@dataclass
class Runner:
    # TODO: read inputs from a single config file
    pdb_id: str
    pdb_path: Optional[Path] = None
    membrane_protein: bool = False
    mutation_data_path: Optional[Path] = None
    mutation_data_chain: Optional[str] = None
    aa_index_path: Path = 'data/aaindex_parsed_small.csv'

    def __post_init__(self):

        # Make sure paths and extensions are set correctly
        if self.pdb_path is None:
            self.pdb_path = rcsb.fetch(self.pdb_id, format="cif")
            self.pdb_ext = "cif"
        else:
            self.pdb_path = Path(self.pdb_path)
            self.pdb_ext = self.pdb_path.suffix.lstrip(".")

        # Load structure using appropriate parser
        # TODO: update this code to use load_structure function in structure_context.py once altloc handling is decided
        if self.pdb_ext in ("cif", "mmcif"):
            mm = CIFFile.read(self.pdb_path)
            arr = get_structure(mm, model=1, extra_fields=["b_factor", "occupancy"])
        else:
            pdb = PDBFile.read(self.pdb_path)
            arr = pdb.get_structure(model=1, extra_fields=["b_factor", "occupancy"])

        self.array = arr

        # TODO: move these to config file
        self.context = structure_context.Context(self.array)
        self.context.membrane_protein = self.membrane_protein
        self.context.membrane_thickness = 15.0
        self.context.vdw_radii = "ProtOr"

        # load amino acid index data
        aa_index = pd.read_csv(self.aa_index_path)
        self.context.aa_index = aa_index

        if self.membrane_protein:
            # TODO: simplify this code to only return pdbtm_df
            pdbtm_df, _ = pdbtm.fetch_pdbtm_annotation(self.pdb_id)
            self.context.residue_table = pdbtm.add_pdbtm_regions(residue_table=self.context.residue_table, pdbtm_regions=pdbtm_df)

        if self.mutation_data_path is not None:
            # TODO: change function names to be more general (not DMS-specific)
            self.mutation_data = sequence_context.load_dms_scores(self.mutation_data_path)
            self.context = sequence_context.merge_dms_scores(
                dms_scores=self.mutation_data,
                ctx=self.context,
                chain=self.mutation_data_chain
            )

    def run(self, metrics: List[str]) -> pd.DataFrame:
        """Compute specified metrics and return as a merged DataFrame.

        Parameters
        ----------
        metrics : List[str]
            List of metric names to compute.
        """

        # filter unknown metrics
        metrics = [m for m in metrics if m in _REGISTRY]
        # TODO: resolve dependencies
        #order = _topological_order(metrics)
        order = metrics.copy()
        result_frames = []
        for m in order:
            meta, func = _REGISTRY[m]

            # TODO: update context to contain all fields needed by metrics, no others
            df = func(self.context)

            # ensure returned DataFrame has index aligned with ctx.res_keys (or positional)
            result_frames.append(df)

            # Optionally store in extras by name
            self.context.extras[m] = df

        # merge all results into one DataFrame
        mutations = self.mutation_data_path is not None
        merged = self._merge_features(result_frames, mutations=mutations)
        return merged

    def _merge_features(self, dfs: List[pd.DataFrame], mutations) -> pd.DataFrame:
        """Merge feature DataFrames on chain, resi, resn, and resm columns.

        Parameters
        ----------
        dfs : List[pd.DataFrame]
            List of DataFrames to merge.

        mutations : bool
            Whether mutation-level data is included (i.e., resm column present).

        Returns
        -------
        pd.DataFrame
            Merged DataFrame.
        """
        # Get all unique rows based on chain, resi, resn, resm to merge on
        keep_cols = ['resi', 'chain', 'resn']
        keep_cols += ['resm'] if mutations else []
        merged_df = self.context.residue_table[keep_cols].drop_duplicates().reset_index(drop=True)

        for df in dfs:
            keep_cols = ['resi', 'chain', 'resn'] + (['resm'] if 'resm' in df.columns else [])
            merged_df = pd.merge(merged_df, df, on=keep_cols, how='outer')

        return merged_df

