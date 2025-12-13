from pathlib import Path
import pandas as pd
import itertools

from typing import List, Dict, Any

from src.pipeline.runner import Runner



def batch_process(batch_file_path: str) -> pd.DataFrame:
    """Process multiple PDB entries specified in a batch file.

    Parameters
    ----------
    batch_file_path : str
        Path to a text file containing one protein per line, specifying the name, PDB ID(s),
        mutation_data_path(s), and config file.

    Returns
    -------
    pd.DataFrame
        Merged DataFrame containing results for all proteins in the batch.
    """

    if not Path(batch_file_path).is_file():
        raise FileNotFoundError(f"Batch file not found at {batch_file_path}")

    batch_df = pd.read_csv(batch_file_path)

    # Expand DFs if multiple entries per protein
    expanded_args = expand_batch_arguments(batch_df)

    all_results = []
    for args in expanded_args:
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

    merged_results = pd.concat(all_results, ignore_index=True)
    return merged_results


def expand_batch_arguments(batch_df: pd.DataFrame) -> List[Dict[str, Any]]:
    """Expand batch DataFrame rows into individual argument dictionaries.

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
            print(batch_df.columns)
            raise ValueError(f"Batch file is missing required column: {col}")

    # Loop through each row
    expanded_args = []
    for _, row in batch_df.iterrows():
        pdb_ids = str(row.get('pdb_id', '')).split('|')
        mutation_paths = str(row.get('mutation_data_path', '')).split('|') if pd.notna(row.get('mutation_data_path')) else [None]

        # Create all combinations of pdb_ids and mutation_paths if multiple are provided
        combos = list(itertools.product(pdb_ids, mutation_paths))
        if len(combos) > 1:
            pdb_ids, mutation_paths = zip(*combos)
            pdb_ids = list(pdb_ids)
            mutation_paths = list(mutation_paths)

        for pdb_id, mut_path in zip(pdb_ids, mutation_paths):
            args = {
                'name': row.get('name'),
                'pdb_id': pdb_id,
                'membrane_protein': row.get('membrane_protein'),
                'mutation_data_path': None if pd.isna(mut_path) else mut_path, # handle NaN,
                'config_path': row.get('config_path')
            }
            expanded_args.append(args)

    return expanded_args