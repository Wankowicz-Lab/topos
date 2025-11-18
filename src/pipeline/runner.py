from dataclasses import dataclass
from pathlib import Path
import pandas as pd

import biotite.structure as struc
from biotite.database import rcsb
from biotite.structure.io.pdb import PDBFile
from biotite.structure.io.pdbx import CIFFile, get_structure, get_model_count

from src.structure import structure_context, metrics
from src.structure.pdbtm import fetch_pdbtm_annotation, add_pdbtm_regions
from src.sequence import sequence_context

from src.structure.pdbtm import add_pdbtm_regions, define_secondary_structure

pdb_id = "8smv"
file_path = rcsb.fetch(pdb_id, format="cif")  # or format="mmtf", "cif"

pdb_file = PDBFile.read(file_path)
array = pdb_file.get_structure(model=1)
struc.get_residue_starts(array)


pdb_file = CIFFile.read(file_path)
array = get_structure(pdb_file)
count = get_model_count(pdb_file)
struc.get_residue_starts(array)


myrunner = Runner(pdb_id='8smv', pdb_path=None, membrane_protein=True,
                  mutation_data_path='/Users/ngreenwald/Library/CloudStorage/Box-Box/WCM Lab/Noah/biogenesis/metadata/GPR161_processed_scores.csv',
                  mutation_data_chain='R')
myrunner.define_secondary_structure()

myrunner = Runner(pdb_id='8smv', pdb_path='/Users/ngreenwald/Library/CloudStorage/Box-Box/WCM Lab/Noah/biogenesis/structural/example/data/pdb/GPR161_8SMV.pdb')

merged = myrunner.context.res_keys



@dataclass
class Runner:
    # TODO: read inputs from a single config file
    pdb_id: str
    pdb_path: Path or None = None
    membrane_protein: bool = False
    mutation_data_path: Path or None = None
    mutation_data_chain: str or None = None

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

        # TODO: decide if we need to keep Context object, or if we can just merge attributes into Runner
        self.context = structure_context.Context(self.array)

        if self.membrane_protein:
            # TODO: simplify this code to only return pdbtm_df
            pdbtm_df, _ = fetch_pdbtm_annotation(self.pdb_id)
            self.context.residue_table = add_pdbtm_regions(residue_table=self.context.res_keys, pdbtm_regions=pdbtm_df)

        if self.mutation_data_path is not None:
            # TODO: change function names to be more general (not DMS-specific)
            self.mutation_data = load_dms_scores(self.mutation_data_path)
            self.context = sequence_context.merge_dms_scores(
                dms_scores=self.mutation_data,
                ctx=self.context,
                chain=self.mutation_data_chain
            )

    # TODO: should this be a metric instead?
    def define_secondary_structure(self):
        """Calculate secondary structure and merge adjacent regions based on heuristics or membrane information"""

        ss_vals = metrics.calculate_secondary_structure(self.context.array)
        res_starts = struc.get_residue_starts(self.context.array)
        chains = array.chain_id[res_starts]
        resi = array.res_id[res_starts]

        ss_df = pd.DataFrame({
            "chain": chains,
            "resi": resi,
            "sse": ss_vals
        })

        if self.membrane_protein:
            self.context.res_keys = define_secondary_structure(self.context.res_keys, ss_df)
        else:
            pass
            # TODO: decide if we want to do any merging of secondary structure regions for non-membrane proteins
            # TODO: implement basic sequential numbering + renaming of ss_df objects for non-membrane proteins













