"""
Test utilities for biogenesis tests.

Provides helper functions for creating test data including residue tables,
atom arrays, and synthetic structure files.
"""
from pathlib import Path
from string import ascii_uppercase

import numpy as np
import pandas as pd
import random
import tomli_w

import biotite.structure as struc

# Seed RNGs for reproducible test data generation
np.random.seed(42)
random.seed(42)

AA_LIST = ['ALA', 'ARG', 'CYS', 'ASP', 'GLU', 'PHE', 'GLY', 'HIS', 'ILE', 'LYS',
           'LEU', 'MET', 'ASN', 'PRO', 'GLN', 'SER', 'THR', 'VAL', 'TRP', 'TYR']


from typing import Optional, List, Union


def _random_AA_seq(length: int = 1, seed: Optional[int] = None) -> Union[str, List[str]]:
    """
    Generate a random amino acid sequence.

    Parameters
    ----------
    length : int, optional
        Length of the sequence. Default is 1.
    seed : int, optional
        Random seed for reproducibility. If provided, reseeds the RNG.

    Returns
    -------
    str or list of str
        Single amino acid code if length is 1, otherwise a list of codes.
    """
    if seed is not None:
        random.seed(seed)
    if length == 1:
        return random.choice(AA_LIST)
    else:
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


def _make_mmcif_file(pdb_id: str, chains: dict) -> str:
    """
    Create a synthetic mmCIF file content for testing.

    Parameters:
    -----------
    pdb_id : str
        PDB ID to use in the mmCIF file.
    chains : dict
        Dictionary where keys are chain IDs and values are lists of amino acid codes.

    Returns:
    --------
    str
        The content of the synthetic mmCIF file as a string.
    """
    mmcif_lines = [
        f"data_{pdb_id}",
        "_entry.id                   " + pdb_id,
        "_struct.title               Synthetic structure for testing",
        "_struct.pdbx_descriptor     Synthetic mmCIF file",
        "",
        "loop_",
        "_atom_site.group_PDB",
        "_atom_site.id",
        "_atom_site.type_symbol",
        "_atom_site.label_atom_id",
        "_atom_site.label_alt_id",
        "_atom_site.label_comp_id",
        "_atom_site.label_asym_id",
        "_atom_site.label_entity_id",
        "_atom_site.label_seq_id",
        "_atom_site.pdbx_PDB_ins_code",
        "_atom_site.Cartn_x",
        "_atom_site.Cartn_y",
        "_atom_site.Cartn_z",
        "_atom_site.occupancy",
        "_atom_site.B_iso_or_equiv",
        "_atom_site.pdbx_formal_charge",
        "_atom_site.auth_seq_id",
        "_atom_site.auth_comp_id",
        "_atom_site.auth_asym_id",
        "_atom_site.auth_atom_id",
        "_atom_site.pdbx_PDB_model_num"
    ]

    atom_id = 1
    for chain_id, aa_list in chains.items():
        for res_idx, res_name in enumerate(aa_list, start=1):
            residue = _make_residue(res_name, res_id=res_idx, chain_id=chain_id)
            for i in range(len(residue)):
                atom_name = residue.atom_name[i]
                x, y, z = residue.coord[i]
                mmcif_lines.append(
                    f"ATOM  {atom_id:>5} {residue.element[i]:<2} {atom_name:<4} . {res_name:<3} "
                    f"{chain_id} 1 {res_idx} . {x:>8.3f} {y:>8.3f} {z:>8.3f} 1.00 20.00 . "
                    f"{res_idx} {res_name} {chain_id} {atom_name} 1"
                )
                atom_id += 1

    mmcif_content = "\n".join(mmcif_lines)
    return mmcif_content


def _write_mmcif_file(file_path: str, chains: dict, pdb_id: str) -> None:
    """
    Write a synthetic mmCIF file to disk for testing.

    Parameters:
    -----------
    file_path : str
        Path to write the mmCIF file.
    chains : dict
        Dictionary where keys are chain IDs and values are lists of amino acid codes.
    pdb_id : str
        PDB ID to use in the mmCIF file.
    """
    mmcif_content = _make_mmcif_file(pdb_id, chains)
    with open(file_path, "w") as f:
        f.write(mmcif_content)


def _make_aaindex_data(accessions):
    """Create a small AAindex DataFrame for testing."""
    aaindex_data = pd.DataFrame({
        'accession': accessions,
        'description': ['Hydrophobicity', 'Volume'],
        'ALA': [0.5, 1.0],
        'CYS': [1.5, 2.0],
        'ASP': [2.5, 3.0],
        'GLU': [3.5, 4.0],
        'PHE': [4.5, 5.0],
        'GLY': [5.5, 6.0],
        'HIS': [6.5, 7.0],
        'ILE': [7.5, 8.0],
        'LYS': [8.5, 9.0],
        'LEU': [9.5, 10.0],
        'MET': [10.5, 11.0],
        'ASN': [11.5, 12.0],
        'PRO': [12.5, 13.0],
        'GLN': [13.5, 14.0],
        'SER': [14.5, 15.0],
        'THR': [15.5, 16.0],
        'VAL': [16.5, 17.0],
        'TRP': [17.5, 18.0],
        'TYR': [18.5, 19.0],
        'ARG': [19.5, 20.0],
    })

    return aaindex_data

def _make_config_file(file_path: Path, pdb_id='8smv', name='test_protein', membrane_protein=False,
                      mutation_data_path=None,
                      mutation_data_chain=None, aaindex_path=None) -> None:
    """Write a configuration file for testing in .toml format with the following defaults"""

    defaults = {"pdb_id": pdb_id,
                'name': name,
                "membrane_protein": membrane_protein,
                "mutation_data_path": str(mutation_data_path) if mutation_data_path is not None else None,
                "mutation_data_chain": mutation_data_chain,
                "aaindex_path": str(aaindex_path) if aaindex_path is not None else None}

    # Remove any keys with None values
    remove_keys = [key for key, value in defaults.items() if value is None]
    for key in remove_keys:
        del defaults[key]

    # write TOML
    with file_path.open("wb") as f:
        tomli_w.dump(defaults, f)