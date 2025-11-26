from dataclasses import dataclass
from tempfile import NamedTemporaryFile

from pathlib import Path
import pandas as pd
import tomli
import warnings

from biotite.database import rcsb
from biotite.structure.io.pdb import PDBFile
from biotite.structure.io.pdbx import CIFFile, get_structure, get_model_count

from src.structure import structure_context, metrics
from src.sequence import sequence_context
from src.structure import pdbtm

from typing import List, Optional, Dict, Any
from src.structure.structure_context import _REGISTRY, Config
import src.sequence.metrics


@dataclass
class Runner:
    pdb_id: Optional[str] = None
    pdb_path: Optional[Path] = None
    membrane_protein: Optional[bool] = None
    mutation_data_path: Optional[Path] = None
    config_path: Path = Path("example/test_config.toml")

    def __post_init__(self):

        # Create override dictionary from input parameters
        overrides = {}
        if self.pdb_id is not None:
            overrides['pdb_id'] = self.pdb_id
        if self.pdb_path is not None:
            overrides['pdb_path'] = self.pdb_path
        if self.membrane_protein is not None:
            overrides['membrane_protein'] = self.membrane_protein
        if self.mutation_data_path is not None:
            overrides['mutation_data_path'] = self.mutation_data_path

        # load and merge config with overrides
        try:
            with self.config_path.open("rb") as f:
                config_dict = tomli.load(f)
                # convert empty strings to None
                config_dict = {k: (None if v == "" else v) for k, v in config_dict.items()}
                config = Config(**config_dict)
        except FileNotFoundError:
            raise FileNotFoundError(f"Configuration file not found at {self.config_path}")
        except tomli.TOMLDecodeError as e:
            raise ValueError(f"Invalid TOML in configuration file {self.config_path}: {e}")
        config = self._merge_config(base=config, overrides=overrides)

        # If the user did not provide a pdb_path, fetch from RCSB and save to a temp file
        if config.pdb_path is None:
            obj = rcsb.fetch(config.pdb_id, format="cif")
            tmp_file = NamedTemporaryFile(delete=False, suffix=".cif")
            tmp_file.write(obj.getvalue().encode("utf-8"))
            tmp_file.close()
            config.pdb_ext = "cif"
            config.pdb_path = Path(tmp_file.name)

        # Otherwise just add parameters directly from config
        else:
            config.pdb_path = Path(config.pdb_path)
            config.pdb_ext = config.pdb_path.suffix.lstrip(".")

        # Load structure using appropriate parser
        # TODO: update this code to use load_structure function in structure_context.py once altloc handling is decided
        if config.pdb_ext in ("cif", "mmcif"):
            mm = CIFFile.read(config.pdb_path)
            arr = get_structure(mm, model=1, extra_fields=["b_factor", "occupancy"])
        else:
            pdb = PDBFile.read(config.pdb_path)
            arr = pdb.get_structure(model=1, extra_fields=["b_factor", "occupancy"])

        # create context object
        self.context = structure_context.Context(arr, config=config)

        if self.membrane_protein:
            pdbtm_df, tmatrix = pdbtm.fetch_pdbtm_annotation(self.pdb_id)
            self.context.residue_table = pdbtm.add_pdbtm_regions(residue_table=self.context.residue_table, pdbtm_regions=pdbtm_df)
            self.context.array.coord = pdbtm.transform_coordinates(self.context.array.coord, tmatrix)

        if self.context.config.mutation_data_path is not None:
            # TODO: change function names to be more general (not DMS-specific)
            # TODO: pass keyword args for column names
            self.context.extras['mutation_data'] = sequence_context.load_dms_scores(self.context.config.mutation_data_path)
            self.context.residue_table = sequence_context.merge_dms_scores(
                dms_scores=self.context.extras['mutation_data'],
                residue_table=self.context.residue_table,
                chain=self.context.config.mutation_data_chain
            )


    def _merge_config(self, base: Config, overrides: Dict[str, Any]) -> Config:
        """Merge configuration overrides with base configuration.

        Parameters
        ----------
        base : Config
            Base configuration loaded from config file.
        overrides : Dict[str, Any]
            Dictionary of override values from Runner initialization.

        Returns
        -------
        Config
            New Config object with overrides applied to base configuration.

        Warnings
        --------
        Issues a warning if unknown configuration keys are provided in overrides. """

        # only keep known fields
        valid = list(Config.model_fields.keys())
        filtered = {k: v for k, v in overrides.items() if k in valid}
        unknown = set(overrides.keys()) - set(filtered.keys())
        if unknown:
            warnings.warn(f"Unknown arguments ignored: {unknown}")
        if not filtered:
            return base

        # construct new Config with overrides
        base_dict = base.model_dump()
        base_dict.update(filtered)

        return Config(**base_dict)


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

            df = func(self.context)

            # ensure returned DataFrame has index aligned with ctx.res_keys (or positional)
            result_frames.append(df)

            # Optionally store in extras by name
            self.context.extras[m] = df

        # merge all results into one DataFrame (outer join by index)
        # TODO: fix this to merge on chain/resi instead of index, handle mutation data (resm) as needed
        merged = pd.concat(result_frames, axis=1)
        return merged
