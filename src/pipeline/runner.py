from dataclasses import dataclass

from tempfile import NamedTemporaryFile

from pathlib import Path
import numpy as np
import pandas as pd
import tomli
import warnings
import logging

from biotite.database import rcsb
import biotite.structure as struc

from src.structure.structure_context import load_structure
from src.pipeline.context import Context, Config
from src.pipeline.sequence_alignment import load_mutation_scores, merge_mutation_scores
from src.databases import pdbtm

from typing import List, Optional, Dict, Any

# import files containing metrics to register them in _REGISTRY
import src.metrics.sequence
import src.metrics.structure
from src.metrics.registry import _REGISTRY
from src.structure.secondary_structure import get_secondary_structure_annotations, define_membrane_secondary_structure, define_soluble_secondary_structure

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


def _residue_key(chain: str, resi: Any, resn: str) -> tuple:
    """Normalize (chain, resi, resn) for set/map lookups."""
    return (str(chain).strip(), int(resi), str(resn).strip() if resn is not None else "")


def calculate_protein_ligand_interactions(
    context: Context,
    contacting_residues_df: pd.DataFrame,
    ligand_radius: float = 4.5,
    second_shell_cutoff: float = 5.0,
) -> pd.DataFrame:
    """
    Label protein residues as contact, binding site, or second shell per ligand chain.

    For each chain in config.ligand_chains, finds protein residues within ligand_radius
    of any ligand atom (contact if in contacting_residues_df, else binding site), and
    residues within second_shell_cutoff of those (second shell). Adds one column
    ligand_<chain>_interactions to the residue table.

    Parameters
    ----------
    context : Context
        Pipeline context (structure, config, residue_table).
    contacting_residues_df : pd.DataFrame
        DataFrame with columns chain, resi_struct, resn_struct, partner_chain; residues in this set
        that also fall within ligand_radius of a ligand are labeled "contact".
    ligand_radius : float, optional
        Max distance (Å) from any ligand atom to a protein residue for binding site.
        Default is 4.5.
    second_shell_cutoff : float, optional
        Max residue-residue distance (Å) for second shell. Default is 5.0.

    Returns
    -------
    pd.DataFrame
        context.residue_table with same rows plus columns ligand_<chain>_interactions
        (values: "contact", "binding site", "second shell", or NaN). If
        ligand_chains is None or empty, returns residue_table unchanged.
    """
    ligand_chains = context.config.ligand_chains
    if ligand_chains is None or len(ligand_chains) == 0:
        logger.warning(
            "ligand_chains is None or empty; skipping protein-ligand interaction analysis."
        )
        return context.residue_table.copy()

    # Protein atoms (amino acids only)
    protein = context.aa
    if context.config.structural_feature_chains is not None:
        chain_mask = np.isin(protein.chain_id, context.config.structural_feature_chains)
        protein = protein[chain_mask]
    
    # Remove ligand chains from protein array
    protein = protein[~np.isin(protein.chain_id, ligand_chains)]

    # Contacting set from provided df: (chain, resi_struct, resn_struct) normalized
    required = ["chain", "resi_struct", "resn_struct"]

    # Residue metadata for protein array
    res_starts = struc.get_residue_starts(protein)
    protein_chains = protein.chain_id[res_starts]
    protein_res_ids = protein.res_id[res_starts]
    protein_res_names = protein.res_name[res_starts]
    n_protein_res = len(res_starts)
    
    # Map protein atom index -> residue index
    atom_to_res_idx = np.repeat(np.arange(n_protein_res), np.diff(list(res_starts) + [protein.array_length()]))

    # Cell list on protein coords for radius queries
    cell_size = max(ligand_radius, second_shell_cutoff) + 0.01
    protein_cell = struc.CellList(protein, cell_size=cell_size)

    out = context.residue_table.copy()
    arr = context.array
    
    for L in ligand_chains:
        
        # Contacting residues for this ligand
        contact_keys = set()
        ligand_contacting_residues_df = contacting_residues_df[contacting_residues_df["partner_chain"] == L]
        for _, row in ligand_contacting_residues_df[required].drop_duplicates().iterrows():
            contact_keys.add(_residue_key(row["chain"], row["resi_struct"], row["resn_struct"]))
        
        ligand_mask = arr.chain_id == L
        ligand_atoms = arr[ligand_mask]
        ligand_coords = ligand_atoms.coord

        # Protein atoms within radius of any ligand atom -> R_binding residues
        protein_atom_indices = set()
        for i in range(ligand_coords.shape[0]):
            near = protein_cell.get_atoms(ligand_coords[i], radius=ligand_radius)
            protein_atom_indices.update(near.tolist())
        binding_res_indices = set(atom_to_res_idx[list(protein_atom_indices)])

        # (chain, resi, resn) for R_binding
        binding_keys = set()
        for ri in binding_res_indices:
            ch = protein_chains[ri]
            rid = protein_res_ids[ri]
            rn = protein_res_names[ri]
            binding_keys.add(_residue_key(ch, rid, rn))

        # Contact vs binding site
        contact_labels = binding_keys & contact_keys
        binding_only = binding_keys - contact_keys

        # Create binding atom mask that is True for all atoms in the binding residues
        binding_atom_mask = np.zeros(protein.array_length(), dtype=bool)
        for ri in binding_res_indices:
            start = res_starts[ri]
            end = res_starts[ri + 1] if ri + 1 < n_protein_res else protein.array_length()
            binding_atom_mask[start:end] = True
        binding_coords = protein.coord[binding_atom_mask]

        # Second shell: protein residues within second_shell_cutoff of any binding atom
        second_shell_res_indices = set()
        for i in range(binding_coords.shape[0]):
            near = protein_cell.get_atoms(binding_coords[i], radius=second_shell_cutoff)
            for ai in near:
                ri = atom_to_res_idx[ai]
                if ri not in binding_res_indices:
                    second_shell_res_indices.add(ri)
        
        # (chain, resi, resn) for second shell
        second_shell_keys = set()
        for ri in second_shell_res_indices:
            ch = protein_chains[ri]
            rid = protein_res_ids[ri]
            rn = protein_res_names[ri]
            second_shell_keys.add(_residue_key(ch, rid, rn))

        # Map (chain, resi, resn) -> label for this ligand chain
        label_map = {}
        for k in contact_labels:
            label_map[k] = "contact"
        for k in binding_only:
            label_map[k] = "binding site"
        for k in second_shell_keys:
            label_map[k] = "second shell"

        def lookup(row):
            key = _residue_key(row["chain"], row["resi_struct"], row["resn_struct"])
            return label_map.get(key, np.nan)

        out[f"ligand_{L}_interactions"] = out.apply(lookup, axis=1)

    return out


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

        # Ensure that either pdb_id or config_path is provided
        if self.pdb_id is None and self.config_path is None:
            raise ValueError("Either pdb_id or config_path must be provided.")

        # Ensure that either name or config_path is provided
        if self.name is None and self.config_path is None:
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
            altloc_policy=config.altloc_policy
        )

        # Remove hydrogens if configured
        if config.remove_hydrogens:
            logger.info("Removing hydrogen atoms")
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

        # Validate ligand_chains if specified
        if self.context.config.ligand_chains is not None:
            if len(self.context.config.ligand_chains) == 0:
                self.context.config.ligand_chains = None
            else:
                array_chains = np.unique(self.context.array.chain_id)
                invalid_ligand = set(self.context.config.ligand_chains) - set(array_chains)
                if invalid_ligand:
                    raise ValueError(
                        f"Specified ligand_chains {sorted(invalid_ligand)} not "
                        f"found in structure chains {sorted(array_chains)}"
                    )

        # Set up df with secondary structure info
        ss_df = get_secondary_structure_annotations(self.context)

        if self.context.config.membrane_protein:
            try:
                logger.info("Fetching PDBTM annotation")
                pdbtm_df, tmatrix = pdbtm.fetch_pdbtm_annotation(self.context.config.pdb_id)
                self.context.residue_table = pdbtm.add_pdbtm_regions(residue_table=self.context.residue_table, pdbtm_regions=pdbtm_df)
                self.context.array.coord = pdbtm.transform_coordinates(self.context.array.coord, tmatrix)
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

            self.context.residue_table = merge_mutation_scores(
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
            raise ValueError("'name' must be provided either in config file or directly to Runner.")
        if base_dict.get('pdb_id') is None:
            raise ValueError("'pdb_id' must be provided either in config file or directly to Runner.")

        return Config(**base_dict)


    def run(self, metrics: List[str] = None) -> None:
        """Compute specified metrics and return as a merged DataFrame.

        Parameters
        ----------
        metrics : Optional[List[str]] = None
            List of metric names to compute.
        """

        if metrics is None:
            metrics = list(_REGISTRY.keys())

        # filter unknown metrics
        else:
            metrics = [m for m in metrics if m in _REGISTRY]
        # TODO: resolve dependencies
        #order = _topological_order(metrics)
        order = metrics.copy()
        result_frames = []
        for m in order:
            meta, func = _REGISTRY[m]
            
            logger.info(f"Calculating metric: {m}")
            df = func(self.context)

            # ensure returned DataFrame has index aligned with ctx.res_keys (or positional)
            result_frames.append(df)

            # Optionally store in extras by name
            self.context.extras[m] = df

        # merge all results into one DataFrame
        logger.info("Merging features")
        mutations = self.context.config.mutation_data_path is not None
        self.features = self._merge_features(result_frames, mutations=mutations)
        self.features['name'] = self.context.config.name


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


    def save_results(self, output_dir: Path = None, output_prefix: str = None) -> None:
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

        # Generate prefix
        if output_prefix is not None:
            prefix = output_prefix + "_" + self.context.config.pdb_id
        else:
            prefix = self.context.config.pdb_id

        # Save features
        logger.info("Saving results")
        merged_path = output_dir / f"{prefix}_features.csv"
        self.features.to_csv(merged_path, index=False)

        # Save metadata from residue table
        metadata_cols = (['chain', 'resi_struct', 'resn_struct', 'resi_mut', 'resn_mut', 'struct_info', 'mut_info'] +
                         (['resm'] if self.context.config.mutation_data_path is not None else []) +
                         (['pdbtm_region', 'pdbtm_region_detailed'] if self.context.config.membrane_protein else []))

        output_df = self.context.residue_table[metadata_cols].drop_duplicates().reset_index(drop=True)
        metadata_path = output_dir / f"{prefix}_metadata.csv"
        output_df.to_csv(metadata_path, index=False)
