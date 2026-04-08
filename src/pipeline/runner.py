import json
import logging
import shutil
import warnings
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import biotite.structure as struc
import numpy as np
import pandas as pd
import tomli

# Import metric modules for registration side effects.
import src.metrics.bonds  # noqa: F401
import src.metrics.sequence  # noqa: F401
import src.metrics.structure  # noqa: F401
from src.databases import pdbtm
from src.metrics.averaging_metrics import METRICS_TO_AVERAGE
from src.metrics.graph_metrics import calculate_graph_metrics
from src.metrics.registry import _REGISTRY, metrics_with_tag
from src.pipeline.context import Config, Context
from src.pipeline.ligands import calculate_protein_ligand_interactions
from src.pipeline.neighbors import calculate_neighborhood_features, compute_residue_neighbors
from src.pipeline.secondary_structure_features import calculate_secondary_structure_features
from src.pipeline.sequence_window_features import calculate_sequence_window_features
from src.pipeline.sequence_alignment import load_mutation_scores, merge_mutation_scores
from src.structure.secondary_structure import (
    define_membrane_secondary_structure,
    define_soluble_secondary_structure,
    get_secondary_structure_annotations,
)
from src.structure.structure_context import load_structure

logger = logging.getLogger(__name__)

def _sort_residue_table(residue_table: pd.DataFrame, mutation_chain: Optional[str] = None) -> pd.DataFrame:
    """Sort residue table by chain and alignment position.
    
    If mutation_chain is provided, that chain comes first, followed by other chains alphabetically.
    Within each chain, residues are sorted by align_pos (alignment position).
    
    Parameters
    ----------
    residue_table : pd.DataFrame
        The residue table to sort.
    mutation_chain : Optional[str]
        Chain identifier for mutation data. If provided, this chain will be sorted first.
        
    Returns
    -------
    pd.DataFrame
        Sorted residue table with index reset.
    """
    residue_table = residue_table.copy()
    
    if mutation_chain is not None:
        # Create custom sort key: mutation_data_chain first, then alphabetical
        residue_table['_chain_sort'] = residue_table['chain'].apply(
            lambda c: (0, c) if c == mutation_chain else (1, c)
        )
        residue_table = residue_table.sort_values(
            ['_chain_sort', 'align_pos'], 
            kind='mergesort'
        ).drop(columns=['_chain_sort']).reset_index(drop=True)
    else:
        # No mutation data: sort alphabetically by chain, then by align_pos
        residue_table = residue_table.sort_values(
            ['chain', 'align_pos'], 
            kind='mergesort'
        ).reset_index(drop=True)
    
    return residue_table
@dataclass
class Runner:
    pdb_id: Optional[str] = None
    name: Optional[str] = None
    pdb_path: Optional[Path] = None
    membrane_protein: Optional[bool] = None
    mutation_data_path: Optional[Path] = None
    config_path: Optional[Path|str] = None

    def __post_init__(self):
        logger.info("Initializing pipeline")

        # Ensure that either pdb_id, pdb_path, or config_path is provided
        if self.pdb_id is None and self.pdb_path is None and self.config_path is None:
            raise ValueError("Either pdb_id, pdb_path, or config_path must be provided.")

        # Ensure that either name or config_path is provided
        if self.name is None and self.config_path is None and self.pdb_id is None and self.pdb_path is None:
            raise ValueError("Either name or config_path must be provided.")

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
        if self.name is not None:
            overrides['name'] = self.name

        # Set up config
        if self.config_path is None:
            config = Config()
        else:
            self.config_path = Path(self.config_path)

            # load config from file
            try:
                logger.info("Loading configuration")
                with self.config_path.open("rb") as f:
                    config_dict = tomli.load(f)
                    # convert empty strings to None
                    config_dict = {k: (None if v == "" else v) for k, v in config_dict.items()}
                    config = Config(**config_dict)
            except FileNotFoundError:
                raise FileNotFoundError(f"Configuration file not found at {self.config_path}")
            except tomli.TOMLDecodeError as e:
                raise ValueError(f"Invalid TOML in configuration file {self.config_path}: {e}")

        # merge overrides
        config = self._merge_config(base=config, overrides=overrides)

        # Load structure using load_structure function
        logger.info("Loading structure")
        arr = load_structure(
            path=config.pdb_path,
            pdb_id=config.pdb_id,
            uniprot_id=config.uniprot_id,
            altloc_policy=config.altloc_policy
        )

        # Track hydrogen presence before any removal
        self._had_hydrogens: bool = bool(np.any(arr.element == "H"))

        # Remove hydrogens if configured
        if config.remove_hydrogens:
            if self._had_hydrogens:
                logger.info("Removing hydrogen atoms from structure")
            arr = arr[arr.element != "H"]

        # create context object
        logger.info("Creating context object")
        self.context = Context(arr, config=config)

        # Validate structural_feature_chains if specified
        if self.context.config.structural_feature_chains is not None:
            if len(self.context.config.structural_feature_chains) == 0:
                # Empty list treated as None
                self.context.config.structural_feature_chains = None
            else:
                available_chains = self.context.residue_table['chain'].unique()
                invalid_chains = set(self.context.config.structural_feature_chains) - set(available_chains)
                if invalid_chains:
                    raise ValueError(
                        f"Specified structural_feature_chains {sorted(invalid_chains)} not "
                        f"found in structure chains {sorted(available_chains)}"
                    )

        # Set up df with secondary structure info
        mkdssp_path = shutil.which("mkdssp")
        if mkdssp_path is None:
            warnings.warn(
                "mkdssp was not found in PATH; falling back to pydssp secondary structure assignment.",
                UserWarning,
            )
            self.context.extras["ss_backend"] = "pydssp"
        else:
            self.context.extras["ss_backend"] = "mkdssp"
            self.context.extras["mkdssp_path"] = mkdssp_path

        ss_df = get_secondary_structure_annotations(self.context)

        if self.context.config.membrane_protein:
            if self.context.config.pdb_id is None and self.context.config.uniprot_id is not None:
                warnings.warn(
                    "Skipping PDBTM annotation for AlphaFold-derived structure because "
                    "PDBTM requires a PDB ID. Membrane features will not be generated; "
                    "continuing with soluble secondary-structure assignment.",
                    UserWarning,
                )
                self.context.config.membrane_protein = False
                self.context.residue_table = define_soluble_secondary_structure(self.context.residue_table, ss_df)
            else:
                try:
                    logger.info("Fetching PDBTM annotation")
                    pdbtm_df, tmatrix = pdbtm.fetch_pdbtm_annotation(self.context.config.pdb_id)
                    self.context.residue_table = pdbtm.add_pdbtm_regions(residue_table=self.context.residue_table, pdbtm_regions=pdbtm_df)
                    self.context.array.coord = pdbtm.transform_coordinates(self.context.array.coord, tmatrix)
                    self.context.aa = self.context.array[struc.filter_amino_acids(self.context.array)]
                    self.context.residue_table = define_membrane_secondary_structure(self.context.residue_table, ss_df)
                except RuntimeError as e:
                    raise RuntimeError(f"Failed to fetch PDBTM annotation for {self.context.config.pdb_id}: {e}. Rerun with membrane_protein=False to calculate soluble secondary structure.")
        else:
            self.context.residue_table = define_soluble_secondary_structure(self.context.residue_table, ss_df)

        # Load mutation data if provided
        if self.context.config.mutation_data_path is not None:
            logger.info("Loading mutation data")

            self.context.extras['mutation_data'] = load_mutation_scores(
                path=self.context.config.mutation_data_path,
                residue_col_name=self.context.config.mutation_residue_col_name,
                residue_idx_name=self.context.config.mutation_residue_idx_name,
                mutation_col_name=self.context.config.mutation_col_name,
                mutation_type_col_name=self.context.config.mutation_type_col_name,
                score_col_name=self.context.config.mutation_score_col_name
            )
  
            if self.context.config.mutation_data_chain not in self.context.residue_table['chain'].unique():
                raise ValueError(f"Specified mutation_data_chain '{self.context.config.mutation_data_chain}' not "
                                 f"found in structure chains {self.context.residue_table['chain'].unique()}")

            self.context.residue_table, self.context.extras['sequence_alignment_merged'] = merge_mutation_scores(
                mutation_scores=self.context.extras['mutation_data'],
                residue_table=self.context.residue_table,
                chain=self.context.config.mutation_data_chain,
                alignment_cutoff=self.context.config.alignment_cutoff
            )
            
            # Sort residue table with mutation_data_chain first
            self.context.residue_table = _sort_residue_table(
                self.context.residue_table,
                mutation_chain=self.context.config.mutation_data_chain
            )
        # Otherwise create mutation columns from structure data
        else:
            self.context.extras['sequence_alignment_merged'] = None
            self.context.residue_table.rename(columns={'resn': 'resn_struct', 'resi': 'resi_struct'}, inplace=True)
            self.context.residue_table['resn_mut'] = self.context.residue_table['resn_struct']
            self.context.residue_table['resi_mut'] = self.context.residue_table['resi_struct']
            self.context.residue_table['mut_info'] = True
            self.context.residue_table['struct_info'] = True
            
            # Add align_pos for consistency (sequential across all chains since no alignment is performed)
            # This ensures all code paths have align_pos, even though it doesn't represent an alignment position
            self.context.residue_table['align_pos'] = range(len(self.context.residue_table))
            
            # Sort residue table alphabetically by chain
            self.context.residue_table = _sort_residue_table(
                self.context.residue_table,
                mutation_chain=None
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

        # construct new Config with overrides
        base_dict = base.model_dump()
        base_dict.update(filtered)

        if base_dict.get('name') is None:
            if base_dict.get('pdb_id') is not None:
                # Use pdb_id as a sensible default name when not provided
                base_dict['name'] = base_dict['pdb_id']
            elif base_dict.get('pdb_path') is not None:
                # Fall back to the structure filename stem
                base_dict['name'] = Path(base_dict['pdb_path']).stem
            elif base_dict.get('uniprot_id') is not None:
                # Fall back to the UniProt accession for AlphaFold-derived structures
                base_dict['name'] = base_dict['uniprot_id']
            else:
                raise ValueError(
                    "'name' must be provided either in config file or directly to Runner."
                )
        if (
            base_dict.get('pdb_id') is None
            and base_dict.get('pdb_path') is None
            and base_dict.get('uniprot_id') is None
        ):
            raise ValueError(
                "Either 'pdb_id', 'pdb_path', or 'uniprot_id' must be provided "
                "(in config file or directly to Runner)."
            )

        return Config(**base_dict)


    def _merge_features(self, dfs: List[pd.DataFrame], mutations) -> pd.DataFrame:
        """Merge feature DataFrames on chain and appropriate resi/resn/resm columns.

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
        # Get all unique rows to merge on

        potential_cols = ['chain', 'resi_struct', 'resn_struct', 'resi_mut', 'resn_mut', 'align_pos']
        keep_cols = [col for col in potential_cols if col in self.context.residue_table.columns]

        # Add mutation columns if mutations are present
        keep_cols += ['resm'] if mutations else []

        merged_df = self.context.residue_table[keep_cols].drop_duplicates().reset_index(drop=True)

        # Filter merged_df to only include structural_feature_chains if specified
        # This ensures the final output only contains the specified chains, even though
        # residue_table itself is not filtered (filtering happens in individual metrics)
        if self.context.config.structural_feature_chains is not None:
            merged_df = merged_df[
                merged_df['chain'].isin(self.context.config.structural_feature_chains)
            ].reset_index(drop=True)

        for df in dfs:
            # Determine merge columns based on what's available in the df
            merge_cols = ['chain']

            # Check if this is a sequence-based or structure-based metric
            if 'resi_mut' in df.columns:
                merge_cols.extend(['resi_mut', 'resn_mut'])
                if 'resm' in df.columns:
                    merge_cols.append('resm')
            elif 'resi_struct' in df.columns:
                merge_cols.extend(['resi_struct', 'resn_struct'])

            merged_df = pd.merge(merged_df, df, on=merge_cols, how='left')

        # Sort using the same logic as residue_table
        mutation_chain = self.context.config.mutation_data_chain if mutations else None
        merged_df = _sort_residue_table(merged_df, mutation_chain=mutation_chain)

        # Drop align_pos before returning (it's just for internal sorting)
        merged_df = merged_df.drop(columns=['align_pos'])

        return merged_df


    def run_metrics(self, metrics: List[str], mutations: bool = False) -> pd.DataFrame:
        """Compute specified metrics and return as a merged DataFrame.

        Parameters
        ----------
        metrics : Optional[List[str]] = None
            List of metric names to compute.
        mutations : bool
            Whether to include mutation-level data.
            
        Returns
        -------
        pd.DataFrame
            Merged DataFrame of all metrics.
        """
        
        result_frames = []
        sequence_metric_columns: list[str] = []
        
        for m in metrics:
            meta, func = _REGISTRY[m]

            logger.info(f"Calculating metric: {m}")
            df = func(self.context)
            if "sequence" in meta.tags:
                sequence_metric_columns.extend(
                    column
                    for column in df.columns
                    if column not in {"chain", "resi_struct", "resn_struct", "resi_mut", "resn_mut", "resm"}
                )

            result_frames.append(df)
        
        # merge features
        logger.info("Merging features")
        features = self._merge_features(result_frames, mutations=mutations)
        features['name'] = self.context.config.name
        self._sequence_metric_columns = list(dict.fromkeys(sequence_metric_columns))
        return features
    
    
    def run(self, metrics: Optional[List[str]] = None) -> None:
        """Compute features for the pipeline.

        Parameters
        ----------
        metrics : Optional[List[str]] = None
            List of metric names to compute.
        """

        # Run metrics
        if metrics is None:
            metrics = list(_REGISTRY.keys())
        
        # Configure metrics based on presence of mutation data
        mutations = self.context.config.mutation_data_path is not None
        if not mutations:
            exclude_metrics = metrics_with_tag('sequence')
            metrics = [m for m in metrics if m not in exclude_metrics]
        if self.context.extras.get("ss_backend") != "mkdssp":
            exclude_metrics = metrics_with_tag("dssp")
            metrics = [m for m in metrics if m not in exclude_metrics]
        
        # Track which metrics were run for log output
        self._metrics_run: List[str] = list(metrics)

        # Run metrics
        self.features = self.run_metrics(metrics=metrics, mutations=mutations)
        sequence_window_features = calculate_sequence_window_features(
            self.context,
            self.features,
            seq_metric_columns=getattr(self, "_sequence_metric_columns", []),
            window_size=5,
        )
        if not sequence_window_features.empty:
            self.features = pd.merge(
                self.features,
                sequence_window_features,
                on=["chain", "resi_mut", "resn_mut"],
                how="left",
                validate="many_to_one",
            )

        merge_cols = ['chain', 'resi_struct', 'resn_struct']

        # Run secondary structure metrics
        secondary_structure_features = calculate_secondary_structure_features(
            self.context,
            self.features,
            ss_metrics=METRICS_TO_AVERAGE,
        )
        self.features = pd.merge(
            self.features,
            secondary_structure_features,
            on=merge_cols,
            how='left',
            validate='many_to_one',
        )

        # Run neighborhood metrics using the configured shared averaging list.
        self.run_neighborhood(cutoff=5)

        # Run ligand interactions
        protein_ligand_interactions = calculate_protein_ligand_interactions(self.context, self.context.extras['bonds_df'])
        self.features = pd.merge(
            self.features,
            protein_ligand_interactions,
            on=merge_cols,
            how='left',
            validate='many_to_one',
        )

        # Run graph metrics
        bond_types = ['all', 'vdw_contact', 'hbond']
        for bond_type in bond_types:
            if bond_type == 'all':
                bonds_df = self.context.extras['bonds_df']
            else:
                bonds_df = self.context.extras['bonds_df'][self.context.extras['bonds_df']['bond_type'] == bond_type]
            if len(bonds_df) == 0:
                continue
            bonds_df = bonds_df.loc[bonds_df.protein_protein, :]
            graph_metrics = calculate_graph_metrics(bonds_df, self.context.residue_table)

            graph_metric_columns = [c for c in graph_metrics.columns if c.startswith('graph_')]
            graph_metrics.rename(columns={c: f'graph_{bond_type}_{c}' for c in graph_metric_columns}, inplace=True)
            self.features = pd.merge(
                self.features,
                graph_metrics,
                on=merge_cols,
                how='left',
                validate='many_to_one',
            )

    def run_neighborhood(
        self,
        cutoff: float,
    ) -> None:
        """Compute neighborhood metrics and merge into self.features.

        Requires run() to have been called (self.features exists). Computes
        residue neighbors within cutoff (Angstroms), runs neighborhood metric
        functions, aggregates their outputs, and merges into self.features.

        Parameters
        ----------
        cutoff : float
            Distance cutoff in Angstroms for neighbor definition (heavy atoms).
        """
        if not hasattr(self, "features"):
            raise ValueError("No features to extend. Please call run() first.")
        # TODO: once we have a single source of truth for residue neighbors, this can be calculated earlier, 
        # and this function can instead be a simple calculate_neighborhood_features call that is imported from neighbors.py
        compute_residue_neighbors(self.context, cutoff=cutoff)
        neighborhood_df = calculate_neighborhood_features(self.context, self.features)
        merge_cols = ["chain", "resi_struct", "resn_struct"]
        
        self.features = pd.merge(
            self.features, neighborhood_df, on=merge_cols, how="left", validate="many_to_one"
        )


    def save_results(self, output_dir: Optional[Path] = None, output_prefix: Optional[str] = None) -> None:
        """Save results to CSV files.

        Parameters
        ----------
        output_dir : Optional[Path] = None
            Directory to save output files. If not provided, uses output_dir from config,
            or the directory of config_path if available.
        output_prefix : Optional[str] = None
            Prefix for output file names.
        """
        if not hasattr(self, 'features'):
            raise ValueError("No features to save. Please call run() first.")

        if output_dir is None:
            if self.context.config.output_dir is not None:
                output_dir = self.context.config.output_dir
            elif self.config_path is not None:
                output_dir = Path(self.config_path).parent
            else:
                raise ValueError("If output_dir is not provided, config_path must be provided to determine output location.")

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Generate prefix using pdb_id if available, otherwise fall back to name
        identifier = self.context.config.pdb_id or self.context.config.name
        if output_prefix is not None:
            prefix = output_prefix + "_" + identifier
        else:
            prefix = identifier

        # Save features
        logger.info("Saving results")
        merged_path = output_dir / f"{prefix}_features.csv"
        self.features.to_csv(merged_path, index=False)

        # Save metadata from residue table
        metadata_cols = (['chain', 'resi_struct', 'resn_struct', 'resi_mut', 'resn_mut', 'struct_info', 'mut_info', 'ss_domains', 'ss_group'] +
                         (['resm'] if self.context.config.mutation_data_path is not None else []))

        output_df = self.context.residue_table[metadata_cols].drop_duplicates().reset_index(drop=True)
        metadata_path = output_dir / f"{prefix}_metadata.csv"
        output_df.to_csv(metadata_path, index=False)

        # Save bonds (one row per unique pair)
        if 'bonds_df' in self.context.extras and len(self.context.extras['bonds_df']) > 0:
            bonds = self.context.extras['bonds_df'].copy()
            bonds['category'] = np.where(bonds['bond_type'] == 'hbond', bonds['extras'], '')
            bonds['geometry'] = np.where(bonds['bond_type'] == 'pi_stacking', bonds['extras'], '')
            bonds['role'] = np.where(bonds['bond_type'] == 'cation_pi', bonds['extras'], '')
            bonds = bonds.drop(columns=['extras'])
            bonds = bonds[bonds['residue_key'] <= bonds['partner_residue_key']].reset_index(drop=True)
            bonds_path = output_dir / f"{prefix}_bonds.csv"
            bonds.to_csv(bonds_path, index=False)
            logger.info(f"Saved {len(bonds)} bond rows to {bonds_path}")
        # Save run log
        log_path = output_dir / f"{prefix}_run_log.txt"
        self._save_run_log(log_path, merged_path, metadata_path)


    def _save_run_log(self, log_path: Path, features_path: Path, metadata_path: Path) -> None:
        log = {
            "run_date": datetime.now().isoformat(),
            "config": self.context.config.model_dump(mode="json"),
            "chains": sorted(self.context.residue_table['chain'].unique().tolist()),
            "n_residues": int(self.context.residue_table['resi_struct'].nunique()),
            "had_hydrogens": self._had_hydrogens,
            "metrics_run": getattr(self, '_metrics_run', []),
            "feature_rows": len(self.features) if hasattr(self, 'features') else 0,
            "feature_columns": len(self.features.columns) if hasattr(self, 'features') else 0,
            "output_files": {
                "features": str(features_path),
                "metadata": str(metadata_path),
            },
        }
        log_path.with_suffix(".json").write_text(json.dumps(log, indent=2, default=str))
