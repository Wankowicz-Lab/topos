from string import ascii_uppercase

import numpy as np
import pandas as pd
import random

import biotite.structure as struc

AA_LIST = ['ALA', 'ARG', 'CYS', 'ASP', 'GLU', 'PHE', 'GLY', 'HIS', 'ILE', 'LYS',
           'LEU', 'MET', 'ASN', 'PRO', 'GLN', 'SER', 'THR', 'VAL', 'TRP', 'TYR']


def _random_AA_seq(length=1):
    if length == 1:
        return random.choice(AA_LIST)
    else:
        # return a list of length=length, each a random amino acid
        return random.choices(AA_LIST, k=length)


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
            resm_list = AA_LIST * num_residue
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



def _make_atoms(atom_names, coords, res_name="UNK", res_id=1, chain_id="A", element=None):
    """
    Create a small AtomArray with manually specified atom names & coordinates.

    Parameters:
    -----------
    atom_names : list of str
        List of atom names.
    coords : list of list of float
        List of coordinates corresponding to each atom.
    res_name : str
        Residue name to assign to all atoms.
    res_id : int
        Residue ID to assign to all atoms.
    chain_id : str
        Chain ID to assign to all atoms.
    element : list of str, optional
        List of element symbols corresponding to each atom. If None, inferred from atom names.

    Returns:
    --------
    struc.AtomArray
        The constructed AtomArray.
    """
    n = len(atom_names)
    arr = struc.AtomArray(n)

    arr.atom_name = np.array(atom_names)
    arr.coord = np.array(coords, dtype=float)
    arr.res_name = np.array([res_name] * n)
    arr.res_id = np.array([res_id] * n)
    arr.chain_id = np.array([chain_id] * n)

    # Optional: infer element from atom name if not provided
    if element is None:
        arr.element = np.array([name[0] for name in atom_names])
    else:
        arr.element = np.array(element)

    return arr


AA_BACKBONE = ["N", "CA", "C", "O"]

# Sidechain atoms for all 20 AAs (minimal sets)
AA_SIDECHAIN = {
    "GLY": [],
    "ALA": ["CB"],
    "VAL": ["CB", "CG1", "CG2"],
    "LEU": ["CB", "CG", "CD1", "CD2"],
    "ILE": ["CB", "CG1", "CG2", "CD1"],
    "SER": ["CB", "OG"],
    "THR": ["CB", "OG1", "CG2"],
    "CYS": ["CB", "SG"],
    "MET": ["CB", "CG", "SD", "CE"],
    "PRO": ["CB", "CG", "CD"],
    "PHE": ["CB", "CG", "CD1", "CD2", "CE1", "CE2", "CZ"],
    "TYR": ["CB", "CG", "CD1", "CD2", "CE1", "CE2", "CZ", "OH"],
    "TRP": ["CB", "CG", "CD1", "CD2", "NE1", "CE2", "CE3", "CZ2", "CZ3", "CH2"],
    "ASP": ["CB", "CG", "OD1", "OD2"],
    "GLU": ["CB", "CG", "CD", "OE1", "OE2"],
    "ASN": ["CB", "CG", "OD1", "ND2"],
    "GLN": ["CB", "CG", "CD", "OE1", "NE2"],
    "HIS": ["CB", "CG", "ND1", "CD2", "CE1", "NE2"],
    "LYS": ["CB", "CG", "CD", "CE", "NZ"],
    "ARG": ["CB", "CG", "CD", "NE", "CZ", "NH1", "NH2"]
}


def _make_residue(res_name, res_id=1, chain_id="A", coords=None):
    """
    Make a synthetic residue with correct atoms but made-up geometry.

    Parameters:
    -----------
    res_name : str
        Three-letter residue name (e.g., "ALA", "GLY").
    res_id : int
        Residue ID to assign.
    chain_id : str
        Chain ID to assign.
    coords : list of list of float, optional
        List of coordinates for each atom. If None, generates simple linear geometry.

    Returns:
    --------
    struc.AtomArray
        The constructed residue as an AtomArray.
    """
    atom_names = AA_BACKBONE + AA_SIDECHAIN[res_name]

    # Simple fake geometry: place atoms in a line or grid
    if coords is None:
        coords = [[i * 1.5, 0.0, 0.0] for i in range(len(atom_names))]
    else:
        # if list of lists provided, use directly
        if isinstance(coords[0], list):
            if len(coords) != len(atom_names):
                raise ValueError("Length of coords must match number of atoms.")
        # otherwise use increment x coord for atoms while keeping y,z fixed
        else:
            x_coord, y_coord, z_coord = coords
            coords = [[x_coord * i, y_coord, z_coord] for i in range(len(atom_names))]

    return _make_atoms(atom_names, coords, res_name, res_id, chain_id)


def _make_chain(aa_list, chain_id="A", coords=None):
    """
    Create a biotite AtomArray representing a protein chain from a list of amino acids.

    Parameters:
    -----------
    aa_list : list of str
        List of three-letter amino acid codes (e.g., ["ALA", "GLY", "SER"]).
    chain_id : str
        Chain identifier to assign to all residues.
    coords : list of list of float, optional
        List of coordinates for each atom in the chain. If None, generates simple linear geometry.

    Returns:
    --------
    struc.AtomArray
        The constructed protein chain as an AtomArray.

    """

    residues = []
    for i, aa in enumerate(aa_list, start=1):
        res = _make_residue(aa, res_id=i, chain_id=chain_id, coords=coords[i-1] if coords else None)
        residues.append(res)

    return struc.concatenate(residues)

