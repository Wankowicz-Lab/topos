from string import ascii_uppercase

import numpy as np
import pandas as pd
import random


def _random_AA_seq(length=1):
    amino_acids = ['ALA', 'ARG', 'CYS', 'ASP', 'GLU', 'PHE', 'GLY', 'HIS', 'ILE', 'LYS',
                   'LEU', 'MET', 'ASN', 'PRO', 'GLN', 'SER', 'THR', 'VAL', 'TRP', 'TYR']
    if length == 1:
        return random.choice(amino_acids)
    else:
        # return a list of length=length, each a random amino acid
        return random.choices(amino_acids, k=length)


def _make_residue_table(num_residues=10, num_chains=2, start_resis=1, make_muts=True):
    """Main helper function to create a residue table for testing.

    Parameters:
    -----------
    num_residues : int or list of int
        Number of residues per chain. If a list, must be of length `num_chains`.
    num_chains : int
        Number of chains to include.
    start_resis : int or list of int
        Starting residue index per chain. If a list, must be of length `num_chains`.
    make_muts : bool or list of bool
        Whether to make mutations per chain. If a list, must be of length `num_chains`.

    Returns:
    --------
    pd.DataFrame
        Residue table dataframe
    """

    # validate input
    if isinstance(num_residues, int):
        num_residues = [num_residues] * num_chains
    elif isinstance(num_residues, list):
        if len(num_residues) != num_chains:
            raise ValueError("If num_residues is a list, it must be of length num_chains.")
    else:
        raise TypeError("num_residues must be an int or a list of int.")

    if isinstance(start_resis, int):
        start_resis = [start_resis] * num_chains
    elif isinstance(start_resis, list):
        if len(start_resis) != num_chains:
            raise ValueError("If start_resi is a list, it must be of length num_chains.")
    else:
        raise TypeError("start_resi must be an int or a list of int.")

    if isinstance(make_muts, bool):
        make_muts = [make_muts] * num_chains
    elif isinstance(make_muts, list):
        if len(make_muts) != num_chains:
            raise ValueError("If make_muts is a list, it must be of length num_chains.")
    else:
        raise TypeError("make_muts must be a bool or a list of bool.")

    data = []
    for chain_idx in range(num_chains):
        # get chain-specific parameters
        chain_id = ascii_uppercase[chain_idx]
        num_residue = num_residues[chain_idx]
        start_resi = start_resis[chain_idx]
        make_mut = make_muts[chain_idx]

        if make_mut:
            chain_list = [chain_id] * num_residue * 20
            resi_list = np.repeat(range(start_resi, start_resi + num_residue), 20)
            resn_list = np.repeat([_random_AA_seq(num_residue)], 20)
            resm_list = [x for x in 'ACDEFGHIKLMNPQRSTVWY'] * num_residue
            eff_list = np.random.normal(loc=0.0, scale=1.0, size=num_residue * 20)
            type_list = ['missense'] * num_residue * 20

            chain_df = pd.DataFrame({
                'chain': chain_list,
                'resi': resi_list,
                'resn': resn_list,
                'resm': resm_list,
                'effect': eff_list,
                'type': type_list,
                'struct_info': True,
                'seq_info': True
            })
        else:
            chain_list = [chain_id] * num_residue
            resi_list = range(start_resi, start_resi + num_residue)
            resn_list = _random_AA_seq(num_residue)

            chain_df = pd.DataFrame({
                'chain': chain_list,
                'resi': resi_list,
                'resn': resn_list,
                'struct_info': True,
                'seq_info': False
            })
        data.append(chain_df)

    residue_table = pd.concat(data, ignore_index=True)
    return residue_table
