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

from typing import List, Optional, Dict, Any, Tuple

# import files containing metrics to register them in _REGISTRY
import src.metrics.sequence
import src.metrics.structure
import src.metrics.bonds
from src.metrics.registry import _REGISTRY, metrics_with_tag
from src.metrics.secondary_structure import ss_domain_lengths, ss_domain_log2_aa_group_ratios

from src.structure.secondary_structure import get_secondary_structure_annotations, define_membrane_secondary_structure, define_soluble_secondary_structure
from src.structure.utils import res_key, is_heavy
from src.metrics.neighborhood_metrics import NEIGHBORHOOD_METRIC_FUNCTIONS

logger = logging.getLogger(__name__)

# Metric column names eligible to be averaged per ss_domain (only those present in features are used)
SS_METRICS: List[str] = [
    "pos_effect",
    "effect_variance",
    "effect_variance_rank",
    "effect",
    "effect_ranking",
    "sasa",
    "sasa_backbone",
    "sasa_sidechain",
    "sasa_polar",
    "sasa_nonpolar",
    "kyte_doolittle",
    "distance_from_membrane_edge",
    "bb_hbond_count",
    "sc_hbond_count",
    "total_hbond_count",
    "packing_n_atoms",
    "packing_n_neighbor_residues",
    "packing_contact_density",
    "blosum90",
    "phat_score",
]
# Hetero residue sets for find_ligands
PROTEIN_MODS = {
    "MSE", "SEP", "TPO", "PTR", "HYP", "CSO", "MHO", "KCX", "CSD", "CME", "CSX",
    "TRY", "LYS", "ALY", "CMH", "CAF",
}
SOLVENT = {"HOH", "WAT", "H2O", "DOD", "HOD"}
COMMON_BUFFER = {"SO4", "PO4", "GOL", "MPD", "EDO", "PEG"}
# Crystallization ions, metals, and common inorganic ions (3-letter codes)
IONS = frozenset({
    "NA", "CL", "K", "MG", "CA", "ZN", "MN", "FE", "CU", "NI", "CD", "CO",
    "SO4", "PO4", "ACT", "F", "BR", "IOD", "AU", "AG", "BA", "SR", "LI", "RB",
    "CS", "AL", "CR", "MO", "W", "V", "PT", "PD", "IR", "RH", "RU", "OS", "RE",
    "AZI", "IUM", "MMC", "NO3",
})
# Known ligand 3-letter codes; used only for warning when absent (unknown ligands still included)
KNOWN_LIGANDS = frozenset({
    "ATP", "ADP", "AMP", "GTP", "GDP", "GMP", "NAD", "NAH", "NAI", "NAP", "NDP",
    "FAD", "FMN", "HEM", "NAG", "NDG", "MAN", "GAL", "GLC", "BMA", "FUC", "SIA",
    "DMS", "PG4", "LI1", "SQU", "PLP", "TPP", "COA", "ACP", "SAM", "SAH",
    "STU", "BOG", "DDQ", "LDA", "MLA", "OLA", "RET", "CHL", "CLR", "LMT", "LPP",
})


def format_ligand_id(chain: str, res_id: int, res_name: str) -> str:
    """
    Canonical string identifier for a ligand (chain, res_id, res_name).

    Use this format for partner_ligand_id in contacting_residues_df so rows
    match ligands produced by find_ligands. Normalization: strip whitespace,
    int for res_id, strip res_name.

    Parameters
    ----------
    chain : str
        Chain ID.
    res_id : int
        Residue index.
    res_name : str
        Residue name (3-letter code).

    Returns
    -------
    str
        Canonical ID, e.g. "A:1:ATP".
    """
    res_name_str = "" if res_name is None else str(res_name).strip()
    return f"{str(chain).strip()}:{int(res_id)}:{res_name_str}"


def find_ligands(
    array: struc.AtomArray | struc.AtomArrayStack,
    exclude_protein_mods: bool = True,
    exclude_solvent: bool = True,
    exclude_ions: bool = True,
    exclude_common_buffer: bool = True,
    exclude_cholesterol: bool = True,
    warn_unknown_ligands: bool = True,
) -> List[Tuple[str, int, str]]:
    """
    Identify ligand molecules from hetero atoms.

    Each exclusion category is applied only when its corresponding flag is True.
    Residue names are normalized (strip, upper) for set membership checks.
    When warn_unknown_ligands is True, ligands whose res_name is not in KNOWN_LIGANDS
    are reported with a warning but still included in the return value.

    Parameters
    ----------
    array : struc.AtomArray or struc.AtomArrayStack
        Structure; if stack, first model is used.
    exclude_protein_mods : bool, optional
        Exclude residues in PROTEIN_MODS (e.g. MSE, SEP, CSO). Default True.
    exclude_solvent : bool, optional
        Exclude solvent residues (HOH, WAT, H2O, etc.). Default True.
    exclude_ions : bool, optional
        Exclude residues in IONS (crystallization ions, metals). Default True.
    exclude_common_buffer : bool, optional
        Exclude common crystallization additives (SO4, GOL, MPD, etc.). Default False.
    exclude_cholesterol : bool, optional
        Exclude cholesterol residues (CLR). Default False.
    warn_unknown_ligands : bool, optional
        Log a warning for each remaining ligand not in KNOWN_LIGANDS; still include
        it in the return list. Default True.

    Returns
    -------
    list of tuple of (str, int, str)
        List of (chain_id, res_id, res_name) for each distinct ligand molecule.
    """
    if isinstance(array, struc.AtomArrayStack):
        array = array[0]
    if "hetero" not in array.get_annotation_categories():
        return []
    
    hetero_mask = array.hetero
    if not np.any(hetero_mask):
        return []
    hetero_atoms = array[hetero_mask]

    chains = hetero_atoms.chain_id
    res_ids = hetero_atoms.res_id
    res_names = hetero_atoms.res_name
    res_starts = struc.get_residue_starts(hetero_atoms)
    
    unique_tuples = set()
    
    for i in range(len(res_starts)):
        start = res_starts[i]
        ch = chains[start]
        rid = int(res_ids[start])
        rn_raw = str(res_names[start]).strip() if res_names[start] is not None else ""
        rn_upper = rn_raw.upper()
        
        if exclude_protein_mods and rn_upper in PROTEIN_MODS:
            continue
        if exclude_solvent and rn_upper in SOLVENT:
            continue
        if exclude_ions and rn_upper in IONS:
            continue
        if exclude_common_buffer and rn_upper in COMMON_BUFFER:
            continue
        if exclude_cholesterol and rn_upper == "CLR":
            continue
        unique_tuples.add((str(ch).strip(), rid, rn_raw))

    result = sorted(unique_tuples, key=lambda t: (t[0], t[1], t[2]))

    if warn_unknown_ligands:
        for (ch, res_id, rn_raw) in result:
            rn_upper = rn_raw.upper()
            if rn_upper not in KNOWN_LIGANDS:
                logger.warning(
                    "Ligand not in KNOWN_LIGANDS: %s (chain=%s, res_id=%s)",
                    rn_raw or "(empty)",
                    ch,
                    res_id,
                )

    return result


def calculate_neighborhood_features(
    context: Context, features: pd.DataFrame, extras_key: str = "residue_neighbors"
) -> pd.DataFrame:
    """Run neighborhood metric functions and aggregate into one DataFrame.

    Loops over NEIGHBORHOOD_METRIC_FUNCTIONS, calls each with (context, features, extras_key),
    and merges returned DataFrames on chain, resi_struct, resn_struct.

    Parameters
    ----------
    context : Context
        Context with extras[extras_key] neighbor mapping.
    features : pd.DataFrame
        Merged features from Runner.run().
    extras_key : str, optional
        Key in context.extras for the neighbor mapping.

    Returns
    -------
    pd.DataFrame
        One row per (chain, resi_struct, resn_struct) from features, with all
        neighborhood metric columns merged in.
    """
    merge_cols = ["chain", "resi_struct", "resn_struct"]

    base = features[merge_cols].drop_duplicates().reset_index(drop=True)
    for func in NEIGHBORHOOD_METRIC_FUNCTIONS:
        df = func(context, features, extras_key=extras_key)
        base = pd.merge(base, df, on=merge_cols, how="left")
    return base


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


def _sanitize_column_name(s: str) -> str:
    """Replace spaces with underscore for use in pandas column names."""
    return str(s).strip().replace(" ", "_")


def calculate_protein_ligand_interactions(
    context: Context,
    contacting_residues_df: pd.DataFrame,
    ligand_radius: float = 4.5,
    second_shell_cutoff: float = 5.0,
) -> pd.DataFrame:
    """
    Label protein residues as contact, binding site, or second shell per ligand.

    Ligands are detected via find_ligands (hetero flag). For each ligand (chain, res_id, res_name),
    finds protein residues within ligand_radius of any ligand atom (contact if in
    contacting_residues_df for that ligand, else binding site), and residues within
    second_shell_cutoff of those (second shell). Adds one column per ligand, e.g.
    ligand_<chain>_<res_id>_<resn>_interactions.

    contacting_residues_df must include partner_ligand_id in the canonical format
    produced by format_ligand_id (e.g. "A:1:ATP") so rows match ligands.

    Parameters
    ----------
    context : Context
        Pipeline context (structure, config, residue_table).
    contacting_residues_df : pd.DataFrame
        DataFrame with columns chain, resi_struct, resn_struct, and partner_ligand_id.
        partner_ligand_id must be in the canonical format from format_ligand_id for matching.
    ligand_radius : float, optional
        Max distance (Å) from any ligand atom to a protein residue for binding site.
        Default is 4.5.
    second_shell_cutoff : float, optional
        Max residue-residue distance (Å) for second shell. Default is 5.0.

    Returns
    -------
    pd.DataFrame
        context.residue_table with same rows plus one column per ligand
        (values: "contact", "binding site", "second shell", or NaN). If no ligands
        found, returns residue_table unchanged.
    """
    arr = context.array
    if isinstance(arr, struc.AtomArrayStack):
        arr = arr[0]
    ligands = find_ligands(arr)
    if len(ligands) == 0:
        logger.warning(
            "No ligands found (hetero atoms absent or all filtered); skipping protein-ligand interaction analysis."
        )
        return context.residue_table.copy()

    # Protein atoms (amino acids only)
    protein = context.aa
    if context.config.structural_feature_chains is not None:
        chain_mask = np.isin(protein.chain_id, context.config.structural_feature_chains)
        protein = protein[chain_mask]
    
    # get protein chains, res_ids, res_names
    res_starts = struc.get_residue_starts(protein)
    protein_chains = protein.chain_id[res_starts]
    protein_res_ids = protein.res_id[res_starts]
    protein_res_names = protein.res_name[res_starts]
    n_protein_res = len(res_starts)

    # get atom to residue index mapping
    atom_to_res_idx = np.repeat(np.arange(n_protein_res), np.diff(list(res_starts) + [protein.array_length()]))

    # create protein cell list
    cell_size = max(ligand_radius, second_shell_cutoff) + 0.01
    protein_cell = struc.CellList(protein, cell_size=cell_size)
    out = context.residue_table.copy()

    # iterate over ligands
    for (lig_chain, lig_res_id, lig_res_name) in ligands:
        # get ligand id
        ligand_id = format_ligand_id(lig_chain, lig_res_id, lig_res_name)
        
        # get contacting residues
        ligand_contacting_df = contacting_residues_df[contacting_residues_df["partner_ligand_id"] == ligand_id]
        
        # get contact keys
        contact_keys = set()
        for _, row in ligand_contacting_df[["chain", "resi_struct", "resn_struct"]].drop_duplicates().iterrows():
            contact_keys.add(res_key(row["chain"], row["resi_struct"], row["resn_struct"]))

        # create ligand mask
        ligand_mask = (
            (arr.chain_id == lig_chain)
            & (arr.res_id == lig_res_id)
            & (arr.res_name == lig_res_name)
        )
        
        ligand_atoms = arr[ligand_mask]
        ligand_coords = ligand_atoms.coord

        # loop over ligand atoms and get protein atoms within ligand_radius, which are the binding sites
        protein_atom_indices = set()
        for i in range(ligand_coords.shape[0]):
            near = protein_cell.get_atoms(ligand_coords[i], radius=ligand_radius)
            protein_atom_indices.update(near.tolist())
        binding_res_indices = set(atom_to_res_idx[list(protein_atom_indices)])

        # get binding keys
        binding_keys = set()
        for ri in binding_res_indices:
            ch = protein_chains[ri]
            rid = protein_res_ids[ri]
            rn = protein_res_names[ri]
            binding_keys.add(res_key(ch, rid, rn))

        # get contact labels (binding sites that are also contacting residues) and binding only (binding sites that are not contacting residues)
        contact_labels = binding_keys & contact_keys
        binding_only = binding_keys - contact_keys

        # Get all atoms in binding site residues
        binding_atom_mask = np.zeros(protein.array_length(), dtype=bool)
        for ri in binding_res_indices:
            start = res_starts[ri]
            end = res_starts[ri + 1] if ri + 1 < n_protein_res else protein.array_length()
            binding_atom_mask[start:end] = True
        binding_coords = protein.coord[binding_atom_mask]

        # Get all atoms that are within second_shell_cutoff of any binding site atom
        second_shell_res_indices = set()
        for i in range(binding_coords.shape[0]):
            near = protein_cell.get_atoms(binding_coords[i], radius=second_shell_cutoff)
            for ai in near:
                ri = atom_to_res_idx[ai]
                if ri not in binding_res_indices:
                    second_shell_res_indices.add(ri)

        # Get all second-shell residue keys
        second_shell_keys = set()
        for ri in second_shell_res_indices:
            ch = protein_chains[ri]
            rid = protein_res_ids[ri]
            rn = protein_res_names[ri]
            second_shell_keys.add(res_key(ch, rid, rn))

        # Create label map
        label_map = {}
        for k in contact_labels:
            label_map[k] = "contact"
        for k in binding_only:
            label_map[k] = "binding site"
        for k in second_shell_keys:
            label_map[k] = "second shell"

        # lookup label for each residue
        def lookup(row):
            key = res_key(row["chain"], row["resi_struct"], row["resn_struct"])
            return label_map.get(key, np.nan)
 
        # add column to residue table
        col = f"ligand_{_sanitize_column_name(lig_chain)}_{lig_res_id}_{_sanitize_column_name(lig_res_name)}_interactions"
        out[col] = out.apply(lookup, axis=1)

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


    def run_metrics(self, metrics: List[str], mutations: bool = False) -> None:
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
        
        for m in metrics:
            meta, func = _REGISTRY[m]

            logger.info(f"Calculating metric: {m}")
            df = func(self.context)

            result_frames.append(df)
        
        # merge features
        logger.info("Merging features")
        features = self._merge_features(result_frames, mutations=mutations)
        features['name'] = self.context.config.name
        return features
    
    
    def run(self, metrics: List[str] = None) -> None:
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
        
        # Run metrics
        self.features = self.run_metrics(metrics=metrics, mutations=mutations)

        # Run secondary structure metrics
        secondary_structure_features = self.run_secondary_structure(ss_metrics=SS_METRICS)
        self.features = pd.merge(self.features, secondary_structure_features, on=['chain', 'resi_struct', 'resn_struct'], how='left')


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


    def run_secondary_structure(
        self,
        ss_metrics: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """Aggregate features by secondary structure domain and compute domain-level metrics.

        Averages selected residue-level metrics per ss_domain (using ss_domains from
        residue_table), then adds domain-level metrics via :mod:`src.metrics.secondary_structure`:
        ss_length (residue count per domain) and log2 ratio of residue proportion per
        aa group vs protein-wide proportion. AA group is derived from resn_struct in
        that module. Only metric columns that exist in self.features and are listed in
        ss_metrics are averaged; NA values are ignored when computing means. Rows with
        missing ss_domains are excluded from aggregation.

        Parameters
        ----------
        ss_metrics : Optional[List[str]] = None
            Metric column names to average per ss_domain. If None, uses module-level SS_METRICS.

        Returns
        -------
        pd.DataFrame
            One row per ss_domain with columns: ss_domains, ss_length, averaged metric
            columns (only those present), and log2_ratio_<group> for each aa group.
        """
        metrics_to_avg = ss_metrics if ss_metrics is not None else SS_METRICS
        merge_cols = ['chain', 'resi_struct', 'resn_struct']

        # Attach ss_domains to features
        rt_subset = self.context.residue_table[merge_cols + ['ss_domains']].drop_duplicates(merge_cols)
        merged = pd.merge(self.features, rt_subset, on=merge_cols, how='left')
        merged = merged.dropna(subset=['ss_domains'])

        cols_to_avg = [c for c in metrics_to_avg if c in merged.columns]

        # Average metrics per ss_domain (skipna=True so NAs are ignored)
        agg_dict = {c: 'mean' for c in cols_to_avg}
        by_domain = merged.groupby(['chain', 'ss_domains'], as_index=False).agg({**agg_dict})

        # Compute domain-level metrics
        lengths = ss_domain_lengths(merged)
        by_domain = by_domain.merge(lengths, on=['chain', 'ss_domains'], how='left')
        log2_df = ss_domain_log2_aa_group_ratios(merged)
        by_domain = by_domain.merge(log2_df, on=['chain', 'ss_domains'], how='left')

        return by_domain
    
    
    def _compute_residue_neighbors(
        self, cutoff: float, extras_key: str = 'residue_neighbors'
    ) -> Dict[str, List[str]]:
        """Compute residue_key -> [residue_key, ...] for residues within cutoff (Angstroms).

        Uses heavy amino-acid atoms only; two residues are neighbors if any pair of
        heavy atoms is within cutoff. Result is stored in context.extras[extras_key].
        """
        # Generate full list of residue keys
        array = self.context.aa
        res_starts = struc.get_residue_starts(array)
        chains = array.chain_id[res_starts]
        res_ids = array.res_id[res_starts]
        res_names = array.res_name[res_starts]
        full_keys = np.array(
            [res_key(ch, ri, rn) for ch, ri, rn in zip(chains, res_ids, res_names)],
            dtype=object,
        )

        # Filter to only heavy amino-acid atoms
        aa_mask = struc.filter_amino_acids(array)
        heavy_mask = np.array([is_heavy(n) for n in array.atom_name], dtype=bool)
        mask = aa_mask & heavy_mask
        arr = array[mask]

        if arr.array_length() == 0:
            mapping = {k: [] for k in full_keys.tolist()}
            self.context.extras[extras_key] = mapping
            return mapping

        residue_ids = np.array(
            [res_key(c, r, rn) for c, r, rn in zip(arr.chain_id, arr.res_id, arr.res_name)],
            dtype=object,
        )

        # Get unique residue keys
        unique_res = np.unique(residue_ids)
        coords = arr.coord.astype(float)
        cutoff2 = cutoff * cutoff

        mapping: Dict[str, List[str]] = {}
        for res_uid in unique_res:
            # Get indices of all atoms in the residue
            idxs = np.where(residue_ids == res_uid)[0]
            if len(idxs) == 0:
                continue
            res_atoms = arr[idxs]
            
            # Calculate distances between residue atoms and all other atoms
            rcoords = res_atoms.coord
            diff = rcoords[:, None, :] - coords[None, :, :]
            d2 = np.einsum("ijk,ijk->ij", diff, diff)
            
            # Filter to only atoms within cutoff
            within_cutoff = d2 <= cutoff2
            close_atom_idxs = np.where(within_cutoff.any(axis=0))[0]
            neighbor_res_keys = set(residue_ids[close_atom_idxs].tolist())
            
            # Remove self from neighbors
            neighbor_res_keys.discard(res_uid)
            mapping[str(res_uid)] = sorted(neighbor_res_keys)

        # Ensure every residue has an entry (including those with no neighbors)
        for k in full_keys.tolist():
            if k not in mapping:
                mapping[k] = []

        self.context.extras[extras_key] = mapping
        return mapping


    def run_neighborhood(
        self, cutoff: float, extras_key: str = "residue_neighbors"
    ) -> None:
        """Compute neighborhood metrics and merge into self.features.

        Requires run() to have been called (self.features exists). Computes
        residue neighbors within cutoff (Angstroms), runs neighborhood metric
        functions, aggregates their outputs, and merges into self.features.

        Parameters
        ----------
        cutoff : float
            Distance cutoff in Angstroms for neighbor definition (heavy atoms).
        extras_key : str, optional
            Key in context.extras for the neighbor mapping. Default 'residue_neighbors'.
        """
        if not hasattr(self, "features"):
            raise ValueError("No features to extend. Please call run() first.")
        self._compute_residue_neighbors(cutoff=cutoff, extras_key=extras_key)
        neighborhood_df = calculate_neighborhood_features(
            self.context, self.features, extras_key=extras_key
        )
        merge_cols = ["chain", "resi_struct", "resn_struct"]
        
        self.features = pd.merge(
            self.features, neighborhood_df, on=merge_cols, how="left"
        )


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
