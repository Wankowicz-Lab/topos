import logging

import biotite.structure as struc
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from src.metrics.registry import register_metric
from src.pipeline.context import Context
from src.structure.utils import (
    build_sites_biotite,
    detect_hbonds,
    get_metadata_cols,
    is_heavy,
    res_key,
)

logger = logging.getLogger(__name__)

## INTERACTION DEFINITIONS

ACIDIC_RESIDUES = {'ASP', 'GLU'}
BASIC_RESIDUES = {'LYS', 'ARG', 'HIS'}
PROTONATION_STATE_RESIDUES = {'HIS', 'TYR', 'SER'}
AROMATIC_RESIDUES = {'PHE', 'TYR', 'TRP'}
CATIONIC_RESIDUES = {'LYS', 'ARG'}  

SALT_BRIDGE_ATOMS = {
    'ASP': ['OD1', 'OD2'],
    'GLU': ['OE1', 'OE2'],
    'LYS': ['NZ'],
    'ARG': ['NH1', 'NH2', 'NE'],
}

AROMATIC_RING_ATOMS = {
    'PHE': ['CG', 'CD1', 'CD2', 'CE1', 'CE2', 'CZ'],
    'TYR': ['CG', 'CD1', 'CD2', 'CE1', 'CE2', 'CZ'],
    'TRP': ['CD2', 'CE2', 'CE3', 'CZ2', 'CZ3', 'CH2'],
}

VDW_RADII = {
    'C': 1.70, 'N': 1.55, 'O': 1.52, 'S': 1.80, 'H': 1.20,
}

def get_residue_atoms(array: struc.AtomArray, chain: str, resi: int, atom_names: list) -> np.ndarray:
    """Get coordinates of specific atoms in a residue."""
    mask = (array.chain_id == chain) & (array.res_id == resi) & np.isin(array.atom_name, atom_names)
    return array.coord[mask]


def get_ring_center(array: struc.AtomArray, chain: str, resi: int, resname: str) -> np.ndarray | None:
    """Get the center of an aromatic ring."""
    if resname not in AROMATIC_RING_ATOMS:
        return None
    coords = get_residue_atoms(array, chain, resi, AROMATIC_RING_ATOMS[resname])
    if len(coords) < 3:
        return None
    return coords.mean(axis=0)


def get_ring_normal(array: struc.AtomArray, chain: str, resi: int, resname: str) -> np.ndarray | None:
    """Get the normal vector of an aromatic ring plane."""
    if resname not in AROMATIC_RING_ATOMS:
        return None
    coords = get_residue_atoms(array, chain, resi, AROMATIC_RING_ATOMS[resname])
    if len(coords) < 3:
        return None
    v1 = coords[1] - coords[0]
    v2 = coords[2] - coords[0]
    normal = np.cross(v1, v2)
    norm = np.linalg.norm(normal)
    if norm < 1e-6:
        return None
    return normal / norm


def classify_bond_types(bond_results: pd.DataFrame, array: struc.AtomArray) -> pd.DataFrame:
    """Classify bond types to categorize whether they are protein-protein or not.
    
    Parameters
    ----------
    bond_results: pd.DataFrame
        DataFrame with bond results.
    array: struc.AtomArray
        Biotite AtomArray containing protein structure data.

    Returns
    -------
    pd.DataFrame
        DataFrame with bond types classified as protein-protein or not. 
    """
    filtered_array = array[struc.filter_amino_acids(array)]
    metadata = get_metadata_cols(filtered_array)
    metadata['residue_key'] = metadata.apply(lambda x: res_key(x['chain'], x['resi_struct'], x['resn_struct']), axis=1)

    bond_results['protein_protein'] = bond_results['residue_key'].isin(metadata['residue_key']) & bond_results['partner_residue_key'].isin(metadata['residue_key'])

    return bond_results


def identify_hbonds(array: struc.AtomArray) -> pd.DataFrame:
    """Identify hydrogen bonds between donor and acceptor sites.

    Returns a DataFrame with two rows per hbond (donor as residue, acceptor as partner;
    acceptor as residue, donor as partner). extras['category'] is 'residue_type-partner_type'
    so the first part always describes this row's residue (backbone or sidechain).

    Parameters
    ----------
    array : struc.AtomArray
        Biotite AtomArray containing structure data.

    Returns
    -------
    pd.DataFrame
        DataFrame with standard bond columns and extras['category'].
    """
    donors, acceptors = build_sites_biotite(array)
    hbonds = detect_hbonds(donors, acceptors)
    standard_columns = ['chain', 'resi_struct', 'resn_struct', 'residue_key', 'partner_chain', 'partner_resi', 'partner_resn', 'partner_residue_key', 'bond_type', 'extras']
    results = []
    for h in hbonds:
        cat = h['category']
        # Donor-view row: residue = donor, partner = acceptor; category is already donor-acceptor
        results.append({
            'chain': h['donor_chain'], 'resi_struct': int(h['donor_resi']), 'resn_struct': h['donor_resname'],
            'residue_key': res_key(h['donor_chain'], h['donor_resi'], h['donor_resname']),
            'partner_chain': h['acceptor_chain'], 'partner_resi': int(h['acceptor_resi']), 'partner_resn': h['acceptor_resname'],
            'partner_residue_key': res_key(h['acceptor_chain'], h['acceptor_resi'], h['acceptor_resname']),
            'bond_type': 'hbond', 'extras': {'category': cat}
        })
        # Acceptor-view row: residue = acceptor, partner = donor; category so first part = this row's residue (acceptor)
        donor_part, acceptor_part = cat.split('-')
        results.append({
            'chain': h['acceptor_chain'], 'resi_struct': int(h['acceptor_resi']), 'resn_struct': h['acceptor_resname'],
            'residue_key': res_key(h['acceptor_chain'], h['acceptor_resi'], h['acceptor_resname']),
            'partner_chain': h['donor_chain'], 'partner_resi': int(h['donor_resi']), 'partner_resn': h['donor_resname'],
            'partner_residue_key': res_key(h['donor_chain'], h['donor_resi'], h['donor_resname']),
            'bond_type': 'hbond', 'extras': {'category': f'{acceptor_part}-{donor_part}'}
        })
    if results:
        return pd.DataFrame(results)
    return pd.DataFrame(columns=standard_columns)


def identify_salt_bridges(array: struc.AtomArray, cutoff: float = 4.0) -> pd.DataFrame:
    """
    Identify salt bridge interactions in a protein structure.

    Parameters
    ----------
    array: struc.AtomArray
        Biotite AtomArray containing protein structure data.
    cutoff: float
        Cutoff distance for salt bridge interactions.

    Returns
    -------
    pd.DataFrame
        DataFrame with salt bridge interactions with columns:
        chain, resi_struct, resn_struct, partner_chain, partner_resi, partner_resn, bond_type, extras

    """
    res_starts = struc.get_residue_starts(array)
    chains = array.chain_id[res_starts]
    res_ids = array.res_id[res_starts]
    resnames = array.res_name[res_starts]
    
    results = []
    
    acidic_indices = [i for i, rn in enumerate(resnames) if rn in ACIDIC_RESIDUES]
    basic_indices = [i for i, rn in enumerate(resnames) if rn in BASIC_RESIDUES]
    
    cutoff2 = cutoff * cutoff
    
    # Iterate over acidic residues
    for acid_idx in acidic_indices:
        acid_chain, acid_resi, acid_resn = chains[acid_idx], res_ids[acid_idx], resnames[acid_idx]
        acid_atoms = get_residue_atoms(array, acid_chain, acid_resi, SALT_BRIDGE_ATOMS[acid_resn])
        if len(acid_atoms) == 0:
            continue
            
        # Iterate over basic residues
        for base_idx in basic_indices:
            base_chain, base_resi, base_resn = chains[base_idx], res_ids[base_idx], resnames[base_idx]
            base_atoms = get_residue_atoms(array, base_chain, base_resi, SALT_BRIDGE_ATOMS[base_resn])
            if len(base_atoms) == 0:
                continue
            
            # Calculate distance between acid and base atoms
            diff = acid_atoms[:, None, :] - base_atoms[None, :, :]
            d2 = np.einsum("ijk,ijk->ij", diff, diff)
            
            # If distance is less than cutoff, add to results
            if d2.min() <= cutoff2:
                results.append({
                    'chain': acid_chain, 'resi_struct': int(acid_resi), 'resn_struct': acid_resn, 'residue_key': res_key(acid_chain, acid_resi, acid_resn),
                    'partner_chain': base_chain, 'partner_resi': int(base_resi), 'partner_resn': base_resn, 'partner_residue_key': res_key(base_chain, base_resi, base_resn),
                    'bond_type': 'salt_bridge',
                    'extras': {}
                })
                results.append({
                    'chain': base_chain, 'resi_struct': int(base_resi), 'resn_struct': base_resn, 'residue_key': res_key(base_chain, base_resi, base_resn),
                    'partner_chain': acid_chain, 'partner_resi': int(acid_resi), 'partner_resn': acid_resn, 'partner_residue_key': res_key(acid_chain, acid_resi, acid_resn),
                    'bond_type': 'salt_bridge',
                    'extras': {}
                })
    
    # Define standard columns
    standard_columns = ['chain', 'resi_struct', 'resn_struct', 'residue_key', 'partner_chain', 'partner_resi', 'partner_resn', 'partner_residue_key', 'bond_type', 'extras']
    
    if results:
        return pd.DataFrame(results)
    else:
        return pd.DataFrame(columns=standard_columns)


@register_metric(name='salt_bridge_count', provides=['salt_bridge_count'], tags={'bonds'})
def calculate_salt_bridges(context: Context, cutoff: float = 4.0) -> pd.DataFrame:
    """
    Calculate salt bridge interactions in a protein structure.

    Parameters
    ----------
    context: Context
        Context object containing residue metadata, structural information, and mutation information.
    cutoff: float
        Cutoff distance for salt bridge interactions.

    Returns
    -------
    pd.DataFrame
        DataFrame with salt bridge interactions.        
    """
    array = context.array
    salt_bridges = identify_salt_bridges(array, cutoff)
    salt_bridges = classify_bond_types(salt_bridges, array)
    metadata = get_metadata_cols(array)
    metadata['salt_bridge_count'] = 0
    if len(salt_bridges) > 0:
        counts = salt_bridges[salt_bridges['protein_protein']].groupby(['chain', 'resi_struct']).size()
        for (chain, resi), count in counts.items():
            metadata.loc[(metadata['chain'] == chain) & (metadata['resi_struct'] == resi), 'salt_bridge_count'] = count
    
    # Consolidate into bonds_df
    if 'bonds_df' not in context.extras:
        context.extras['bonds_df'] = pd.DataFrame(columns=['chain', 'resi_struct', 'resn_struct', 'partner_chain', 'partner_resi', 'partner_resn', 'bond_type', 'extras'])
    if len(salt_bridges) > 0:
        context.extras['bonds_df'] = pd.concat([context.extras['bonds_df'], salt_bridges], ignore_index=True)
    
    return metadata


def identify_ionic_bonds(array: struc.AtomArray, cutoff: float = 4.0) -> pd.DataFrame:
    """Identify ionic bonds in a protein structure.

    Parameters
    ----------
    array: struc.AtomArray
        Biotite AtomArray containing protein structure data.
    cutoff: float
        Cutoff distance for ionic bonds.

    Returns
    -------
    pd.DataFrame
        DataFrame with ionic bonds with columns:
        chain, resi_struct, resn_struct, partner_chain, partner_resi, partner_resn, bond_type, extras
    """
    res_starts = struc.get_residue_starts(array)
    chains = array.chain_id[res_starts]
    res_ids = array.res_id[res_starts]
    resnames = array.res_name[res_starts]
    
    results = []
    
    acidic_indices = [i for i, rn in enumerate(resnames) if rn in ACIDIC_RESIDUES]
    ionic_indices = [i for i, rn in enumerate(resnames) if rn in PROTONATION_STATE_RESIDUES]
    
    cutoff2 = cutoff * cutoff
    
    # Iterate over ionic residues
    for ionic_idx in ionic_indices:
        ionic_chain, ionic_resi, ionic_resn = chains[ionic_idx], res_ids[ionic_idx], resnames[ionic_idx]
        ionic_atoms = get_residue_atoms(array, ionic_chain, ionic_resi, SALT_BRIDGE_ATOMS[ionic_resn])
        if len(ionic_atoms) == 0:
            continue
            
        # Iterate over acidic residues
        for acidic_idx in acidic_indices:
            acidic_chain, acidic_resi, acidic_resn = chains[acidic_idx], res_ids[acidic_idx], resnames[acidic_idx]
            acidic_atoms = get_residue_atoms(array, acidic_chain, acidic_resi, SALT_BRIDGE_ATOMS[acidic_resn])
            if len(acidic_atoms) == 0:
                continue
            
            # Exclude adjacent residues if part of the same chain
            if ionic_chain == acidic_chain:
                if abs(ionic_resi - acidic_resi) == 1:
                    continue
            
            # Calculate distance between ionic and acidic atoms
            diff = ionic_atoms[:, None, :] - acidic_atoms[None, :, :]
            d2 = np.einsum("ijk,ijk->ij", diff, diff)
            
            # If distance is less than cutoff, add to results (acidic first, then ionic, matching salt_bridge order)
            if d2.min() <= cutoff2:
                results.append({
                    'chain': acidic_chain, 'resi_struct': int(acidic_resi), 'resn_struct': acidic_resn, 'residue_key': res_key(acidic_chain, acidic_resi, acidic_resn),
                    'partner_chain': ionic_chain, 'partner_resi': int(ionic_resi), 'partner_resn': ionic_resn, 'partner_residue_key': res_key(ionic_chain, ionic_resi, ionic_resn),
                    'bond_type': 'ionic',
                    'extras': {}
                })
                results.append({
                    'chain': ionic_chain, 'resi_struct': int(ionic_resi), 'resn_struct': ionic_resn, 'residue_key': res_key(ionic_chain, ionic_resi, ionic_resn),
                    'partner_chain': acidic_chain, 'partner_resi': int(acidic_resi), 'partner_resn': acidic_resn, 'partner_residue_key': res_key(acidic_chain, acidic_resi, acidic_resn),
                    'bond_type': 'ionic',
                    'extras': {}
                })
    
    # Define standard columns
    standard_columns = ['chain', 'resi_struct', 'resn_struct', 'residue_key', 'partner_chain', 'partner_resi', 'partner_resn', 'partner_residue_key', 'bond_type', 'extras']
    
    if results:
        return pd.DataFrame(results)
    else:
        return pd.DataFrame(columns=standard_columns)


@register_metric(name='ionic_bond_count', provides=['ionic_bond_count'], tags={'bonds'})
def calculate_ionic_bond_count(context: Context, cutoff: float = 4.0) -> pd.DataFrame:
    """Calculate the number of ionic bonds in a protein structure.

    Parameters
    ----------
    context: Context
        Context object containing residue metadata, structural information, and mutation information.
    cutoff: float
        Cutoff distance for ionic bonds.


    Returns
    -------
    pd.DataFrame
        DataFrame with the number of ionic bonds.
    """
    array = context.array
    ionic_bonds = identify_ionic_bonds(array, cutoff)
    ionic_bonds = classify_bond_types(ionic_bonds, array)
    metadata = get_metadata_cols(array)
    metadata['ionic_bond_count'] = 0
    if len(ionic_bonds) > 0:
        counts = ionic_bonds[ionic_bonds['protein_protein']].groupby(['chain', 'resi_struct']).size()
        for (chain, resi), count in counts.items():
            metadata.loc[(metadata['chain'] == chain) & (metadata['resi_struct'] == resi), 'ionic_bond_count'] = count
    
    # Consolidate into bonds_df
    if 'bonds_df' not in context.extras:
        context.extras['bonds_df'] = pd.DataFrame(columns=['chain', 'resi_struct', 'resn_struct', 'partner_chain', 'partner_resi', 'partner_resn', 'bond_type', 'extras'])
    if len(ionic_bonds) > 0:
        context.extras['bonds_df'] = pd.concat([context.extras['bonds_df'], ionic_bonds], ignore_index=True)
    return metadata


def identify_disulfide_bonds(array: struc.AtomArray, cutoff: float = 2.5) -> pd.DataFrame:
    """Identify disulfide bonds in a protein structure.

    Parameters
    ----------
    array: struc.AtomArray
        Biotite AtomArray containing protein structure data.
    cutoff: float
        Cutoff distance for disulfide bonds.

    Returns
    -------
    pd.DataFrame
        DataFrame with disulfide bonds with columns:
        chain, resi_struct, resn_struct, partner_chain, partner_resi, partner_resn, bond_type, extras
    """
    res_starts = struc.get_residue_starts(array)
    chains = array.chain_id[res_starts]
    res_ids = array.res_id[res_starts]
    resnames = array.res_name[res_starts]
    
    results = []
    
    cys_indices = [i for i, rn in enumerate(resnames) if rn == 'CYS']
    
    cutoff2 = cutoff * cutoff
    
    # Iterate over cysteine residues
    for i, cys1_idx in enumerate(cys_indices):
        cys1_chain, cys1_resi = chains[cys1_idx], res_ids[cys1_idx]
        cys1_sg = get_residue_atoms(array, cys1_chain, cys1_resi, ['SG'])
        if len(cys1_sg) == 0:
            continue
            
        # Iterate over cysteine residues
        for cys2_idx in cys_indices[i+1:]:
            cys2_chain, cys2_resi = chains[cys2_idx], res_ids[cys2_idx]
            cys2_sg = get_residue_atoms(array, cys2_chain, cys2_resi, ['SG'])
            if len(cys2_sg) == 0:
                continue
            
            d2 = np.sum((cys1_sg[0] - cys2_sg[0])**2)
            
            # If distance is less than cutoff, add to results
            if d2 <= cutoff2:
                results.append({
                    'chain': cys1_chain, 'resi_struct': int(cys1_resi), 'resn_struct': 'CYS', 'residue_key': res_key(cys1_chain, cys1_resi, 'CYS'),
                    'partner_chain': cys2_chain, 'partner_resi': int(cys2_resi), 'partner_resn': 'CYS', 'partner_residue_key': res_key(cys2_chain, cys2_resi, 'CYS'),
                    'bond_type': 'disulfide',
                    'extras': {}
                })
                results.append({
                    'chain': cys2_chain, 'resi_struct': int(cys2_resi), 'resn_struct': 'CYS', 'residue_key': res_key(cys2_chain, cys2_resi, 'CYS'),
                    'partner_chain': cys1_chain, 'partner_resi': int(cys1_resi), 'partner_resn': 'CYS', 'partner_residue_key': res_key(cys1_chain, cys1_resi, 'CYS'),
                    'bond_type': 'disulfide',
                    'extras': {}
                })
    
    # Define standard columns
    standard_columns = ['chain', 'resi_struct', 'resn_struct', 'residue_key', 'partner_chain', 'partner_resi', 'partner_resn', 'partner_residue_key', 'bond_type', 'extras']
    
    if results:
        return pd.DataFrame(results)
    else:
        return pd.DataFrame(columns=standard_columns)


@register_metric(name='disulfide_bond_count', provides=['disulfide_bond_count'], tags={'bonds'})
def calculate_disulfide_bond_count(context: Context, cutoff: float = 2.5) -> pd.DataFrame:
    """Calculate the number of disulfide bonds in a protein structure.

    Parameters
    ----------
    context: Context
        Context object containing residue metadata, structural information, and mutation information.
    cutoff: float
        Cutoff distance for disulfide bonds.
    
    Returns
    -------
    pd.DataFrame
        DataFrame with the number of disulfide bonds.
    """
    array = context.array
    disulfide_bonds = identify_disulfide_bonds(array, cutoff)
    disulfide_bonds = classify_bond_types(disulfide_bonds, array)
    metadata = get_metadata_cols(array)
    metadata['disulfide_bond_count'] = 0
    if len(disulfide_bonds) > 0:
        counts = disulfide_bonds[disulfide_bonds['protein_protein']].groupby(['chain', 'resi_struct']).size()
        for (chain, resi), count in counts.items():
            metadata.loc[(metadata['chain'] == chain) & (metadata['resi_struct'] == resi), 'disulfide_bond_count'] = count
    
    # Consolidate into bonds_df
    if 'bonds_df' not in context.extras:
        context.extras['bonds_df'] = pd.DataFrame(columns=['chain', 'resi_struct', 'resn_struct', 'partner_chain', 'partner_resi', 'partner_resn', 'bond_type', 'extras'])
    if len(disulfide_bonds) > 0:
        context.extras['bonds_df'] = pd.concat([context.extras['bonds_df'], disulfide_bonds], ignore_index=True)
    return metadata



def identify_pi_stacking(
    array: struc.AtomArray,
    distance_cutoff: float = 5.5,
    parallel_angle_cutoff: float = 20.0,
    perpendicular_angle_cutoff: float = 70.0,
    max_parallel_displacement: float = 2.5,
) -> pd.DataFrame:
    """Identify pi-stacking interactions in a protein structure.

    Parameters
    ----------
    array : struc.AtomArray
        Biotite AtomArray containing protein structure data.
    distance_cutoff : float
        Maximum centroid-centroid distance (Angstroms) to consider.
    parallel_angle_cutoff : float
        Maximum interplanar angle (degrees) to classify as parallel
        (sandwich or displaced). Pairs below this are parallel.
    perpendicular_angle_cutoff : float
        Minimum interplanar angle (degrees) to classify as T-shaped.
        Pairs above this are T-shaped. Angles between
        parallel_angle_cutoff and perpendicular_angle_cutoff are
        rejected as ambiguous.
    max_parallel_displacement : float
        Maximum horizontal displacement (Angstroms) for parallel
        stacking. Pairs with displacement above this are rejected
        even if the angle is parallel.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns: chain, resi_struct, resn_struct,
        residue_key, partner_chain, partner_resi, partner_resn,
        partner_residue_key, bond_type, extras.
        extras contains 'geometry'.
    """
    res_starts = struc.get_residue_starts(array)
    chains = array.chain_id[res_starts]
    res_ids = array.res_id[res_starts]
    resnames = array.res_name[res_starts]

    # Collect aromatic ring data
    aromatic_data = []
    for i, (ch, ri, rn) in enumerate(zip(chains, res_ids, resnames)):
        if rn in AROMATIC_RESIDUES:
            center = get_ring_center(array, ch, ri, rn)
            normal = get_ring_normal(array, ch, ri, rn)
            if center is not None and normal is not None:
                aromatic_data.append((i, ch, ri, rn, center, normal))

    cutoff2 = distance_cutoff * distance_cutoff
    results = []

    for i, (idx1, ch1, ri1, rn1, center1, normal1) in enumerate(aromatic_data):
        for idx2, ch2, ri2, rn2, center2, normal2 in aromatic_data[i + 1:]:
            # Quick centroid distance filter
            connecting_vec = center2 - center1
            d2 = np.dot(connecting_vec, connecting_vec)
            if d2 > cutoff2:
                continue

            # Interplanar angle (folded to 0-90 via abs)
            dot = abs(np.dot(normal1, normal2))
            angle_rad = np.arccos(np.clip(dot, 0.0, 1.0))
            angle_deg = np.degrees(angle_rad)

            # Projection along normal1 (vertical component)
            parallel_comp = abs(np.dot(connecting_vec, normal1))
            # Horizontal displacement in the ring plane
            displacement = np.sqrt(max(d2 - parallel_comp ** 2, 0.0))

            if angle_deg < parallel_angle_cutoff:
                # Parallel — enforce displacement limit
                if displacement > max_parallel_displacement:
                    continue
                geometry = 'sandwich' if displacement < 1.0 else 'parallel_displaced'

            elif angle_deg > perpendicular_angle_cutoff:
                # T-shaped — validate centroid-to-plane distance
                # Use the "top" ring's normal to get plane distance
                geometry = 't_shaped'

            else:
                # Ambiguous interplanar angle — skip
                continue

            extras = {
                'geometry': geometry,
            }

            results.append({
                'chain': ch1, 'resi_struct': int(ri1), 'resn_struct': rn1,
                'residue_key': res_key(ch1, ri1, rn1),
                'partner_chain': ch2, 'partner_resi': int(ri2), 'partner_resn': rn2,
                'partner_residue_key': res_key(ch2, ri2, rn2),
                'bond_type': 'pi_stacking',
                'extras': extras,
            })
            results.append({
                'chain': ch2, 'resi_struct': int(ri2), 'resn_struct': rn2,
                'residue_key': res_key(ch2, ri2, rn2),
                'partner_chain': ch1, 'partner_resi': int(ri1), 'partner_resn': rn1,
                'partner_residue_key': res_key(ch1, ri1, rn1),
                'bond_type': 'pi_stacking',
                'extras': extras,
            })

    standard_columns = [
        'chain', 'resi_struct', 'resn_struct', 'residue_key',
        'partner_chain', 'partner_resi', 'partner_resn', 'partner_residue_key',
        'bond_type', 'extras',
    ]

    if results:
        return pd.DataFrame(results)
    else:
        return pd.DataFrame(columns=standard_columns)


@register_metric(name='pi_stacking_count', provides=['pi_stacking_count'], tags={'bonds'})
def calculate_pi_stacking_count(context: Context, distance_cutoff: float = 5.5) -> pd.DataFrame:
    """Calculate the number of pi-stacking interactions in a protein structure.

    Parameters
    ----------
    context: Context
        Context object containing residue metadata, structural information, and mutation information.
    distance_cutoff: float
        Cutoff distance for pi-stacking interactions.

    Returns
    -------
    pd.DataFrame
        DataFrame with the number of pi-stacking interactions.
    """
    array = context.array
    pi_stacking = identify_pi_stacking(array, distance_cutoff)
    pi_stacking = classify_bond_types(pi_stacking, array)
    metadata = get_metadata_cols(array)
    metadata['pi_stacking_count'] = 0
    if len(pi_stacking) > 0:
        counts = pi_stacking.groupby(['chain', 'resi_struct']).size()
        for (chain, resi), count in counts.items():
            metadata.loc[(metadata['chain'] == chain) & (metadata['resi_struct'] == resi), 'pi_stacking_count'] = count

    # Consolidate into bonds_df
    if 'bonds_df' not in context.extras:
        context.extras['bonds_df'] = pd.DataFrame(columns=['chain', 'resi_struct', 'resn_struct', 'partner_chain', 'partner_resi', 'partner_resn', 'bond_type', 'extras'])
    if len(pi_stacking) > 0:
        context.extras['bonds_df'] = pd.concat([context.extras['bonds_df'], pi_stacking], ignore_index=True)
    return metadata


def identify_cation_pi(array: struc.AtomArray, cutoff: float = 6.0) -> pd.DataFrame:
    """Identify cation-pi interactions in a protein structure.

    Parameters
    ----------
    array: struc.AtomArray
        Biotite AtomArray containing protein structure data.
    cutoff: float
        Cutoff distance for cation-pi interactions.

    Returns
    -------
    pd.DataFrame
        DataFrame with cation-pi interactions with columns:
        chain, resi_struct, resn_struct, partner_chain, partner_resi, partner_resn, bond_type, extras
        (extras contains 'role' key)
    """

    # Get residue starts and metadata
    res_starts = struc.get_residue_starts(array)
    chains = array.chain_id[res_starts]
    res_ids = array.res_id[res_starts]
    resnames = array.res_name[res_starts]
    
    results = []
    
    # Get cation atoms
    cation_atoms = {'LYS': ['NZ'], 'ARG': ['CZ']}
    
    # Iterate over cationic residues to get cation coordinates
    cation_data = []
    for i, (ch, ri, rn) in enumerate(zip(chains, res_ids, resnames)):
        if rn in CATIONIC_RESIDUES:
            atoms = get_residue_atoms(array, ch, ri, cation_atoms.get(rn, []))
            if len(atoms) > 0:
                cation_data.append((i, ch, ri, rn, atoms[0]))
    
    # Iterate over aromatic residues to get ring center coordinates
    aromatic_data = []
    for i, (ch, ri, rn) in enumerate(zip(chains, res_ids, resnames)):
        if rn in AROMATIC_RESIDUES:
            center = get_ring_center(array, ch, ri, rn)
            if center is not None:
                aromatic_data.append((i, ch, ri, rn, center))
    
    # Calculate cutoff distance
    cutoff2 = cutoff * cutoff
    
    # Iterate over cationic residues to identify cation-pi interactions
    for cat_idx, cat_ch, cat_ri, cat_rn, cat_coord in cation_data:
        for aro_idx, aro_ch, aro_ri, aro_rn, aro_center in aromatic_data:
            d2 = np.sum((cat_coord - aro_center)**2)
            
            if d2 <= cutoff2:
                results.append({
                    'chain': cat_ch, 'resi_struct': int(cat_ri), 'resn_struct': cat_rn, 'residue_key': res_key(cat_ch, cat_ri, cat_rn),
                    'partner_chain': aro_ch, 'partner_resi': int(aro_ri), 'partner_resn': aro_rn, 'partner_residue_key': res_key(aro_ch, aro_ri, aro_rn),
                    'bond_type': 'cation_pi',
                    'extras': {'role': 'cation'}
                })
                results.append({
                    'chain': aro_ch, 'resi_struct': int(aro_ri), 'resn_struct': aro_rn, 'residue_key': res_key(aro_ch, aro_ri, aro_rn),
                    'partner_chain': cat_ch, 'partner_resi': int(cat_ri), 'partner_resn': cat_rn, 'partner_residue_key': res_key(cat_ch, cat_ri, cat_rn),
                    'bond_type': 'cation_pi',
                    'extras': {'role': 'aromatic'}
                })
    
    # Define standard columns
    standard_columns = ['chain', 'resi_struct', 'resn_struct', 'residue_key', 'partner_chain', 'partner_resi', 'partner_resn', 'partner_residue_key', 'bond_type', 'extras']
    
    if results:
        return pd.DataFrame(results)
    else:
        return pd.DataFrame(columns=standard_columns)


@register_metric(name='cation_pi_count', provides=['cation_pi_count'], tags={'bonds'})
def calculate_cation_pi_count(context: Context, cutoff: float = 6.0) -> pd.DataFrame:
    """Calculate the number of cation-pi interactions in a protein structure.

    Parameters
    ----------
    context: Context
        Context object containing residue metadata, structural information, and mutation information.
    cutoff: float
        Cutoff distance for cation-pi interactions.

    Returns
    -------
    pd.DataFrame
        DataFrame with the number of cation-pi interactions.
    """
    array = context.array
    cation_pi = identify_cation_pi(array, cutoff)
    cation_pi = classify_bond_types(cation_pi, array)
    metadata = get_metadata_cols(array)
    metadata['cation_pi_count'] = 0
    if len(cation_pi) > 0:
        counts = cation_pi[cation_pi['protein_protein']].groupby(['chain', 'resi_struct']).size()
        for (chain, resi), count in counts.items():
            metadata.loc[(metadata['chain'] == chain) & (metadata['resi_struct'] == resi), 'cation_pi_count'] = count
    
    # Consolidate into bonds_df
    if 'bonds_df' not in context.extras:
        context.extras['bonds_df'] = pd.DataFrame(columns=['chain', 'resi_struct', 'resn_struct', 'partner_chain', 'partner_resi', 'partner_resn', 'bond_type', 'extras'])
    if len(cation_pi) > 0:
        context.extras['bonds_df'] = pd.concat([context.extras['bonds_df'], cation_pi], ignore_index=True)
    return metadata


def identify_vdw_contacts(array: struc.AtomArray, cutoff_factor: float = 1.0) -> pd.DataFrame:
    """Identify van der Waals contacts in a protein structure.

    Parameters
    ----------
    array: struc.AtomArray
        Biotite AtomArray containing protein structure data.
    cutoff_factor: float
        Cutoff factor for van der Waals contacts.

    Returns
    -------
    pd.DataFrame
        DataFrame with van der Waals contacts with columns:
        chain, resi_struct, resn_struct, partner_chain, partner_resi, partner_resn, bond_type, extras
    """
    
    # Get heavy atoms
    heavy_mask = np.array([is_heavy(n) for n in array.atom_name], dtype=bool)
    heavy_array = array[heavy_mask]
    
    # Get atom chains, residue IDs, residue names, elements, and coordinates
    atom_chains = heavy_array.chain_id
    atom_res_ids = heavy_array.res_id
    atom_res_names = heavy_array.res_name
    atom_elements = heavy_array.element
    coords = heavy_array.coord
    
    radii = np.array([VDW_RADII.get(e, 1.70) for e in atom_elements])
    
    tree = cKDTree(coords)
    max_radius = max(VDW_RADII.values())
    max_cutoff = 2 * max_radius + cutoff_factor
    
    pairs = tree.query_pairs(max_cutoff)
    
    seen_pairs = set()
    results = []
    
    for i, j in pairs:
        key_i = f"{atom_chains[i]}:{atom_res_ids[i]}"
        key_j = f"{atom_chains[j]}:{atom_res_ids[j]}"
        if key_i == key_j:
            continue
        
        pair_key = tuple(sorted([key_i, key_j]))
        if pair_key in seen_pairs:
            continue
        
        dist = np.linalg.norm(coords[i] - coords[j])
        vdw_sum = radii[i] + radii[j] + cutoff_factor
        
        if dist <= vdw_sum:
            seen_pairs.add(pair_key)
            results.append({
                'chain': atom_chains[i], 'resi_struct': int(atom_res_ids[i]), 'resn_struct': atom_res_names[i], 'residue_key': res_key(atom_chains[i], atom_res_ids[i], atom_res_names[i]),
                'partner_chain': atom_chains[j], 'partner_resi': int(atom_res_ids[j]), 'partner_resn': atom_res_names[j], 'partner_residue_key': res_key(atom_chains[j], atom_res_ids[j], atom_res_names[j]),
                'bond_type': 'vdw_contact',
                'extras': {}
            })
            results.append({
                'chain': atom_chains[j], 'resi_struct': int(atom_res_ids[j]), 'resn_struct': atom_res_names[j], 'residue_key': res_key(atom_chains[j], atom_res_ids[j], atom_res_names[j]),
                'partner_chain': atom_chains[i], 'partner_resi': int(atom_res_ids[i]), 'partner_resn': atom_res_names[i], 'partner_residue_key': res_key(atom_chains[i], atom_res_ids[i], atom_res_names[i]),
                'bond_type': 'vdw_contact',
                'extras': {}
            })
    
    # Define standard columns
    standard_columns = ['chain', 'resi_struct', 'resn_struct', 'residue_key', 'partner_chain', 'partner_resi', 'partner_resn', 'partner_residue_key', 'bond_type', 'extras']
    
    if results:
        return pd.DataFrame(results)
    else:
        return pd.DataFrame(columns=standard_columns)


@register_metric(name='vdw_contact_count', provides=['vdw_contact_count'], tags={'bonds'})
def calculate_vdw_contact_count(context: Context, cutoff_factor: float = 1.0) -> pd.DataFrame:
    """Calculate the number of van der Waals contacts in a protein structure.

    Parameters
    ----------
    context: Context
        Context object containing residue metadata, structural information, and mutation information.
    cutoff_factor: float
        Cutoff factor for van der Waals contacts.

    Returns
    -------
    pd.DataFrame
        DataFrame with the number of van der Waals contacts.
    """
    array = context.array
    vdw_contacts = identify_vdw_contacts(array, cutoff_factor)
    vdw_contacts = classify_bond_types(vdw_contacts, array)
    metadata = get_metadata_cols(array)
    metadata['vdw_contact_count'] = 0
    if len(vdw_contacts) > 0:
        counts = vdw_contacts[vdw_contacts['protein_protein']].groupby(['chain', 'resi_struct']).size()
        for (chain, resi), count in counts.items():
            metadata.loc[(metadata['chain'] == chain) & (metadata['resi_struct'] == resi), 'vdw_contact_count'] = count
    
    # Consolidate into bonds_df
    if 'bonds_df' not in context.extras:
        context.extras['bonds_df'] = pd.DataFrame(columns=['chain', 'resi_struct', 'resn_struct', 'partner_chain', 'partner_resi', 'partner_resn', 'bond_type', 'extras'])
    if len(vdw_contacts) > 0:
        context.extras['bonds_df'] = pd.concat([context.extras['bonds_df'], vdw_contacts], ignore_index=True)
    return metadata


@register_metric(name='calculate_hbond_metrics', provides=['bb_hbond_count', 'sc_hbond_count', 'total_hbond_count'], tags={'structure', 'interaction'})
def calculate_hbond_metrics(context: Context) -> pd.DataFrame:
    """
    Compute per-residue hydrogen bond metrics.

    Uses an altloc-aware donor/acceptor model to detect hydrogen bonds and count them per residue.

    Parameters
    ----------
    context : Context
        Context object containing residue metadata, structural information, and mutation information.

    Returns
    -------
    pd.DataFrame
        DataFrame with 'bb_hbond_count', 'sc_hbond_count', 'total_hbond_count' along with residue metadata.
    """
    array = context.array
    if context.config.structural_feature_chains is not None:
        chain_mask = np.isin(array.chain_id, context.config.structural_feature_chains)
        array = array[chain_mask]

    hbonds_df = identify_hbonds(array)
    hbonds_df = classify_bond_types(hbonds_df, array)

    metadata_df = get_metadata_cols(array)
    n_res = len(metadata_df)
    bb_counts = np.zeros(n_res, dtype=float)
    sc_counts = np.zeros(n_res, dtype=float)
    total_counts = np.zeros(n_res, dtype=float)

    res_starts = struc.get_residue_starts(array)
    chains = array.chain_id[res_starts]
    res_ids = array.res_id[res_starts]
    resnames = array.res_name[res_starts]
    residue_to_idx = {(ch, int(ri), rn): i for i, (ch, ri, rn) in enumerate(zip(chains, res_ids, resnames))}

    for _, row in hbonds_df[hbonds_df['protein_protein']].iterrows():
        idx = residue_to_idx.get((row['chain'], int(row['resi_struct']), row['resn_struct']))
        if idx is None:
            continue
        total_counts[idx] += 1
        parts = row['extras']['category'].split('-')
        if parts[0] == 'backbone':
            bb_counts[idx] += 1
        else:
            sc_counts[idx] += 1

    metadata_df['bb_hbond_count'] = bb_counts
    metadata_df['sc_hbond_count'] = sc_counts
    metadata_df['total_hbond_count'] = total_counts

    standard_columns = ['chain', 'resi_struct', 'resn_struct', 'residue_key', 'partner_chain', 'partner_resi', 'partner_resn', 'partner_residue_key', 'bond_type', 'extras']
    if 'bonds_df' not in context.extras:
        context.extras['bonds_df'] = pd.DataFrame(columns=standard_columns)
    if len(hbonds_df) > 0:
        context.extras['bonds_df'] = pd.concat([context.extras['bonds_df'], hbonds_df], ignore_index=True)

    return metadata_df


@register_metric(
    name='total_bond_count',
    provides=['total_bond_count', 'total_within_chain_bonds', 'total_between_chain_bonds'],
    tags={'bonds'},
)
def calculate_total_bond_count(context: Context) -> pd.DataFrame:
    """Aggregate row-level bond participation counts per residue.

    Aggregates counts from ``context.extras['bonds_df']`` after filtering to
    protein-protein interactions. Produces total counts, plus within-chain and
    between-chain splits based on ``chain`` vs ``partner_chain``.
    """
    if 'bonds_df' not in context.extras:
        raise ValueError(
            "context.extras['bonds_df'] is missing. Run one or more bond metrics "
            "before computing total_bond_count."
        )

    # Get metadata columns
    metadata = get_metadata_cols(context.array)
    metadata['total_bond_count'] = 0
    metadata['total_within_chain_bonds'] = 0
    metadata['total_between_chain_bonds'] = 0

    bonds_df = context.extras['bonds_df']
    protein_bonds = bonds_df[bonds_df['protein_protein']]

    # Group by chain and residue to get total counts, within-chain counts, and between-chain counts
    total_counts = protein_bonds.groupby(['chain', 'resi_struct']).size()
    within_counts = protein_bonds[
        protein_bonds['chain'] == protein_bonds['partner_chain']
    ].groupby(['chain', 'resi_struct']).size()
    between_counts = protein_bonds[
        protein_bonds['chain'] != protein_bonds['partner_chain']
    ].groupby(['chain', 'resi_struct']).size()

    # Add counts to metadata
    for (chain, resi), count in total_counts.items():
        metadata.loc[
            (metadata['chain'] == chain) & (metadata['resi_struct'] == resi),
            'total_bond_count',
        ] = count
    for (chain, resi), count in within_counts.items():
        metadata.loc[
            (metadata['chain'] == chain) & (metadata['resi_struct'] == resi),
            'total_within_chain_bonds',
        ] = count
    for (chain, resi), count in between_counts.items():
        metadata.loc[
            (metadata['chain'] == chain) & (metadata['resi_struct'] == resi),
            'total_between_chain_bonds',
        ] = count

    return metadata
