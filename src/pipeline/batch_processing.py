from pathlib import Path
import pandas as pd
import itertools
import logging

from typing import List, Dict, Any

from src.pipeline.runner import Runner

logger = logging.getLogger(__name__)



def batch_process(batch_file_path: str) -> pd.DataFrame:
    """Process multiple proteins specified in a batch file.

    Each row in the batch file should contain the necessary parameters for processing a protein,
    including name, PDB ID(s), mutation data path(s), and config file. If multiple PDB IDs or mutation
    data paths are provided in a single row (separated by '|'), separate entries will be created for each.

    Parameters
    ----------
    batch_file_path : str
        Path to a CSV file containing batch specifications. The CSV file must contain the following columns:
            - name: Name of the protein.
            - pdb_id: PDB ID(s) for the protein (multiple IDs can be separated by '|').
            - membrane_protein: Boolean or indicator if the protein is a membrane protein.
            - mutation_data_path: Path(s) to mutation data file(s) (multiple paths can be separated by '|').
            - config_path: Path to the configuration file for the run.

    Returns
    -------
    pd.DataFrame
        Merged DataFrame containing results for all proteins in the batch.
    """

    if not Path(batch_file_path).is_file():
        raise FileNotFoundError(f"Batch file not found at {batch_file_path}")

    logger.info(f"Loading batch file from: {batch_file_path}")
    batch_df = pd.read_csv(batch_file_path)
    logger.info(f"Batch file loaded: {len(batch_df)} protein entries")

    # Expand DFs if multiple entries per protein
    expanded_args = expand_batch_arguments(batch_df)
    num_proteins = len(expanded_args)
    logger.info(f"Starting batch processing for {num_proteins} proteins")

    all_results = []
    for idx, args in enumerate(expanded_args, start=1):
        protein_name = args.get('name', 'Unknown')
        logger.info(f"Processing protein {idx} of {num_proteins}: {protein_name}")
        runner = Runner(
            pdb_id=args.get('pdb_id'),
            name=args.get('name'),
            pdb_path=args.get('pdb_path'),
            membrane_protein=args.get('membrane_protein'),
            mutation_data_path=args.get('mutation_data_path'),
            config_path=args.get('config_path')
        )
        result_df = runner.run()
        all_results.append(result_df)

    logger.info(f"Batch processing completed for {num_proteins} proteins")
    merged_results = pd.concat(all_results, ignore_index=True)
    return merged_results


def expand_batch_arguments(batch_df: pd.DataFrame) -> List[Dict[str, Any]]:
    """Expand batch DataFrame rows into individual argument dictionaries.

    If multiple PDB IDs or mutation data paths are provided in a single row (separated by '|'),
    this function creates separate entries for each combination.

    Parameters
    ----------
    batch_df : pd.DataFrame
        DataFrame containing batch specifications.

    Returns
    -------
    List[Dict[str, Any]]
        List of argument dictionaries for each protein entry.
    """
    # Check inputs
    required_cols = ['name', 'pdb_id', 'membrane_protein', 'mutation_data_path', 'config_path']
    for col in required_cols:
        if col not in batch_df.columns:
            raise ValueError(f"Batch file is missing required column: {col}")

    # Loop through each row
    expanded_args = []
    for _, row in batch_df.iterrows():
        pdb_ids = str(row.get('pdb_id', '')).split('|')
        mutation_paths = str(row.get('mutation_data_path', '')).split('|') if pd.notna(row.get('mutation_data_path')) else [None]

        # Create all combinations of pdb_ids and mutation_paths if multiple are provided
        combos = list(itertools.product(pdb_ids, mutation_paths))

        for pdb_id, mut_path in combos:
            args = {
                'name': row.get('name'),
                'pdb_id': pdb_id,
                'membrane_protein': row.get('membrane_protein'),
                'mutation_data_path': None if pd.isna(mut_path) else mut_path, # handle NaN
                'config_path': row.get('config_path')
            }
            expanded_args.append(args)

    return expanded_args