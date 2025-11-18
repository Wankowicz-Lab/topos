from dataclasses import dataclass
from pathlib import Path
import pandas as pd

import biotite.structure as struc
from biotite.database import rcsb
from biotite.structure.io.pdb import PDBFile
from biotite.structure.io.pdbx import CIFFile, get_structure, get_model_count

from src.structure import structure_context
from src.structure.pdbtm_annotation import fetch_pdbtm_annotation, annotate_pdbtm_detailed, merge_pdbtm_regions
from src.sequence import sequence_context

from src.structure.pdbtm import add_pdbtm_regions

pdb_id = "8smv"
file_path = rcsb.fetch(pdb_id, format="cif")  # or format="mmtf", "cif"

pdb_file = PDBFile.read(file_path)
array = pdb_file.get_structure(model=1)
struc.get_residue_starts(array)


pdb_file = CIFFile.read(file_path)
array = get_structure(pdb_file)
count = get_model_count(pdb_file)
struc.get_residue_starts(array)


myrunner = Runner(pdb_id='8smv', pdb_path=None, mutation_data_path=Path('/data/mutation_data'))
myrunner = Runner(pdb_id='8smv', pdb_path='/Users/ngreenwald/Library/CloudStorage/Box-Box/WCM Lab/Noah/biogenesis/structural/example/data/pdb/GPR161_8SMV.pdb')


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
            self.context.residue_table = add_pdbtm_regions(residue_table=self.context.residue_table, pdbtm_regions=pdbtm_df)

        if self.mutation_data_path is not None:
            self.mutation_data = sequence_context.load_dms_scores(self.mutation_data_path)
            self.context = sequence_context.merge_dms_scores(
                dms_scores=self.mutation_data,
                ctx=self.context,
                chain=self.mutation_data_chain
            )

    def define_secondary_structure(self):






