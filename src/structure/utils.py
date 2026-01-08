# utils.py
"""
Structure utility functions for hydrogen bond detection and residue packing.

This module provides utilities for analyzing protein structures, including
hydrogen bond detection, residue metadata extraction, and packing calculations.
"""
from __future__ import annotations
import math
from collections import defaultdict, namedtuple
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import biotite.structure as struc


def get_metadata_cols(array: struc.AtomArray) -> pd.DataFrame:
    """
    Extract metadata columns (chain, resi_struct, resn_struct, altloc) from an AtomArray.

    Parameters
    ----------
    array : struc.AtomArray
        Biotite AtomArray containing protein structure data.

    Returns
    -------
    pd.DataFrame

    """
    res_starts = struc.get_residue_starts(array)
    chains = array.chain_id[res_starts]
    resi = array.res_id[res_starts]
    resn = array.res_name[res_starts]
    
    try:
        annot_categories = array.get_annotation_categories()
        if 'altloc' in annot_categories:
            altloc = array.altloc[res_starts]
        elif 'altloc_id' in annot_categories:
            altloc = array.altloc_id[res_starts]
        else:
            altloc = np.array([''] * len(res_starts))
    except (AttributeError, TypeError):
        altloc = np.array([''] * len(res_starts))
        
    return pd.DataFrame({"chain": chains, "resi_struct": resi, "resn_struct": resn, "altloc": altloc})


#________________HYDROGEN BONDS__________________________

DA_MAX = 3.5           # Å donor–acceptor distance
H_A_MAX = 2.6          # Å H–acceptor distance
ANGLE_MIN = 120.0      # degrees minimum angle

INCLUDE_WATER = False
INCLUDE_LIGANDS = False

DonorSite = namedtuple(
    "DonorSite",
    (
        "res_key resname chain resi atom_name altloc "
        "base_atom base_altloc coord_base coord_donor "
        "has_explicit_H H_name H_altloc H_coord "
        "is_backbone"
    ),
)

AcceptorSite = namedtuple(
    "AcceptorSite",
    "res_key resname chain resi atom_name altloc coord is_backbone",
)


def _norm_vec(v: np.ndarray) -> np.ndarray:
    x = np.linalg.norm(v)
    return v / x if x > 1e-8 else v


def angle_deg(u: np.ndarray, v: np.ndarray) -> float:
    """
    Calculate the angle between two vectors in degrees.

    Parameters
    ----------
    u : np.ndarray
        First vector.
    v : np.ndarray
        Second vector.

    Returns
    -------
    float
        Angle between the vectors in degrees.
    """
    un = _norm_vec(u)
    vn = _norm_vec(v)
    dot = np.clip(np.dot(un, vn), -1.0, 1.0)
    return float(math.degrees(math.acos(dot)))


def _res_key(chain, resi, resname) -> str:
    return f"{chain}:{int(resi)}:{resname}"


def is_backbone_atom(atom_name: str) -> bool:
    """
    Check if an atom is part of the protein backbone.

    Parameters
    ----------
    atom_name : str
        Name of the atom (e.g., 'N', 'CA', 'C', 'O').

    Returns
    -------
    bool
        True if the atom is a backbone atom, False otherwise.
    """
    return atom_name in ("N", "CA", "C", "O", "OXT", "H", "H1", "H2", "H3")


def norm_alt(a: Any) -> str:
    """
    Normalize an alternate location identifier.

    Parameters
    ----------
    a : Any
        Alternate location identifier (may be None or string).

    Returns
    -------
    str
        Normalized uppercase string, or empty string if input is None or empty.
    """
    if a is None:
        return ""
    s = str(a).strip()
    return s.upper() if s else ""


def altloc_compatible(d_alt: Any, a_alt: Any) -> bool:
    """
    Check if two alternate location identifiers are compatible.

    Two altloc identifiers are compatible if either is empty/None, or if they
    are identical (case-insensitive).

    Parameters
    ----------
    d_alt : Any
        First alternate location identifier.
    a_alt : Any
        Second alternate location identifier.

    Returns
    -------
    bool
        True if the identifiers are compatible, False otherwise.
    """
    d = norm_alt(d_alt)
    a = norm_alt(a_alt)
    if d == "" or a == "":
        return True
    return d == a

def _split_by_residue(arr: struc.AtomArray):
    """
    Split an AtomArray into residues.

    Yields
    ------
    tuple
        A tuple of (resname, chain_id, res_id, idxs, base_arr) for each residue.
    """
    if arr.array_length() == 0:
        return

    starts = struc.get_residue_starts(arr)
    
    starts = np.append(starts, arr.array_length())

    for s, e in zip(starts[:-1], starts[1:]):
        # idxs are indices into the current arr slice
        idxs = np.arange(s, e, dtype=int)
        
        yield (
            arr.res_name[s],
            arr.chain_id[s],
            int(arr.res_id[s]),
            idxs,
            arr,
        )

def _residue_ok(resname: str,
                is_protein_flag: bool,
                include_water: bool = True,
                include_ligands: bool = False) -> bool:
    """
    Check if a residue should be included in hydrogen bond analysis.

    Parameters
    ----------
    resname : str
        Three-letter residue name.
    is_protein_flag : bool
        Whether the residue is a protein amino acid.
    include_water : bool, optional
        Whether to include water molecules. Default is True.
    include_ligands : bool, optional
        Whether to include ligand molecules. Default is False.

    Returns
    -------
    bool
        True if the residue should be included, False otherwise.
    """
    if is_protein_flag:
        return True
    if include_water and resname.upper() in ("HOH", "WAT", "H2O"):
        return True
    return include_ligands

def _donor_acceptor_templates(resname: str, names_set: set[str]):
    donors = []
    acceptors = []
    resname3 = resname.strip().upper()

    # water
    if resname3 in ("HOH", "WAT", "H2O"):
        if "O" in names_set:
            donors.append(("O", None, "H*"))
            acceptors.append("O")
        return donors, acceptors

    # mainchain N/O
    if "N" in names_set:
        base = "CA" if "CA" in names_set else ("C" if "C" in names_set else None)
        donors.append(("N", base, "H*"))
    if "O" in names_set:
        acceptors.append("O")
    if "OXT" in names_set:
        acceptors.append("OXT")

    # Ser / Thr / Tyr sidechain OH
    if resname3 in ("SER", "THR", "TYR"):
        oname = "OG" if resname3 == "SER" else ("OG1" if resname3 == "THR" else "OH")
        if oname in names_set:
            donors.append((oname, "CB" if "CB" in names_set else None, "H*"))
            acceptors.append(oname)

    # Lys
    if resname3 == "LYS" and "NZ" in names_set:
        donors.append(("NZ", "CE" if "CE" in names_set else None, "H*"))

    # Arg
    if resname3 == "ARG":
        for d, base in (("NE", "CD"), ("NH1", "CZ"), ("NH2", "CZ")):
            if d in names_set:
                donors.append((d, base if base in names_set else None, "H*"))

    # His
    if resname3 == "HIS":
        for d, base in (("ND1", "CG"), ("NE2", "CD2")):
            if d in names_set:
                donors.append((d, base if base in names_set else None, None))
        for a in ("ND1", "NE2"):
            if a in names_set:
                acceptors.append(a)

    # Asp / Glu
    if resname3 == "ASP":
        for a in ("OD1", "OD2"):
            if a in names_set:
                acceptors.append(a)
    if resname3 == "GLU":
        for a in ("OE1", "OE2"):
            if a in names_set:
                acceptors.append(a)

    # Asn / Gln
    if resname3 == "ASN":
        if "OD1" in names_set:
            acceptors.append("OD1")
        if "ND2" in names_set:
            donors.append(("ND2", "CG" if "CG" in names_set else None, None))
    if resname3 == "GLN":
        if "OE1" in names_set:
            acceptors.append("OE1")
        if "NE2" in names_set:
            donors.append(("NE2", "CD" if "CD" in names_set else None, None))

    return donors, acceptors


def _index_by_name_alt(arr: struc.AtomArray, idxs: np.ndarray):
    d: Dict[str, Dict[str, int]] = defaultdict(dict)
    # Check for altloc annotation - could be named 'altloc' or 'altloc_id'
    altloc_attr = None
    try:
        annot_categories = arr.get_annotation_categories()
        if "altloc" in annot_categories:
            altloc_attr = "altloc"
        elif "altloc_id" in annot_categories:
            altloc_attr = "altloc_id"
    except (AttributeError, TypeError):
        pass
    
    for i in idxs:
        name = arr.atom_name[i].strip()
        if altloc_attr:
            try:
                alt = norm_alt(getattr(arr, altloc_attr)[i])
            except (AttributeError, KeyError, IndexError):
                alt = ""
        else:
            alt = ""
        d[name][alt] = i
    return d


def _find_h_for_name_alt(name_to_alt_to_idx, donor_alt):
    candidates = (
        "H", "H1", "H2", "H3", "HG", "HG1", "HH",
        "HZ", "HZ1", "HZ2", "HZ3",
    )
    for nm in candidates:
        if nm in name_to_alt_to_idx:
            if donor_alt in name_to_alt_to_idx[nm]:
                return nm, donor_alt, name_to_alt_to_idx[nm][donor_alt]
            if "" in name_to_alt_to_idx[nm]:
                return nm, "", name_to_alt_to_idx[nm][""]
    return None, None, None


def _pick_base(name_to_alt_to_idx, base_name, donor_alt):
    if base_name is None or base_name not in name_to_alt_to_idx:
        return None, None, None
    d = name_to_alt_to_idx[base_name]
    if donor_alt in d:
        return base_name, donor_alt, d[donor_alt]
    if "" in d:
        return base_name, "", d[""]
    return None, None, None


def build_sites_biotite(
    arr: struc.AtomArray,
    include_water: bool = INCLUDE_WATER,
    include_ligands: bool = INCLUDE_LIGANDS,
) -> Tuple[List[DonorSite], List[AcceptorSite]]:
    """
    Build DonorSite and AcceptorSite lists from a Biotite AtomArray.

    Parameters
    ----------
    arr : struc.AtomArray
        Biotite AtomArray containing protein structure data.
    include_water : bool, optional
        Whether to include water molecules. Default is False.
    include_ligands : bool, optional
        Whether to include ligand molecules. Default is False.

    Returns
    -------
    tuple
        A tuple of (donors, acceptors) where each is a list of
        DonorSite or AcceptorSite namedtuples.
    """
    prot_mask = struc.filter_amino_acids(arr)
    donors: List[DonorSite] = []
    acceptors: List[AcceptorSite] = []

    for resname, chain_id, resi, idxs, base_arr in _split_by_residue(arr):
        is_protein = bool(prot_mask[idxs].any())
        if not _residue_ok(resname, is_protein, include_water, include_ligands):
            continue

        res_key = _res_key(chain_id, resi, resname)
        name_to_alt = _index_by_name_alt(base_arr, idxs)
        names_set = set(name_to_alt.keys())
        dtempl, atempl = _donor_acceptor_templates(resname, names_set)

        # Acceptors
        for aname in atempl:
            if aname not in name_to_alt:
                continue
            for a_alt, ai in name_to_alt[aname].items():
                coord = base_arr.coord[ai].astype(float)
                acceptors.append(
                    AcceptorSite(
                        res_key,
                        resname,
                        chain_id,
                        resi,
                        aname,
                        a_alt,
                        coord,
                        is_backbone_atom(aname),
                    )
                )

        # Donors
        for dname, base_name, hsentinel in dtempl:
            if dname not in name_to_alt:
                continue
            for d_alt, di in name_to_alt[dname].items():
                D_coord = base_arr.coord[di].astype(float)
                base_nm, base_alt, bi = _pick_base(name_to_alt, base_name, d_alt)
                B_coord = base_arr.coord[bi].astype(float) if bi is not None else None

                if hsentinel:
                    H_nm, H_alt, hi = _find_h_for_name_alt(name_to_alt, d_alt)
                    H_coord = base_arr.coord[hi].astype(float) if hi is not None else None
                    has_H = hi is not None
                else:
                    H_nm = H_alt = H_coord = None
                    has_H = False

                donors.append(
                    DonorSite(
                        res_key,
                        resname,
                        chain_id,
                        resi,
                        dname,
                        d_alt,
                        base_nm,
                        base_alt,
                        B_coord,
                        D_coord,
                        has_H,
                        H_nm,
                        H_alt,
                        H_coord,
                        is_backbone_atom(dname),
                    )
                )

    return donors, acceptors


def detect_hbonds(
    donors: List[DonorSite],
    acceptors: List[AcceptorSite],
    da_max: float = DA_MAX,
    h_a_max: float = H_A_MAX,
    angle_min: float = ANGLE_MIN,
) -> List[Dict[str, Any]]:
    """
    Detect hydrogen bonds between donor and acceptor sites.

    Parameters
    ----------
    donors : list of DonorSite
        List of hydrogen bond donor sites.
    acceptors : list of AcceptorSite
        List of hydrogen bond acceptor sites.
    da_max : float, optional
        Maximum donor-acceptor distance in Angstroms. Default is 3.5.
    h_a_max : float, optional
        Maximum hydrogen-acceptor distance in Angstroms. Default is 2.6.
    angle_min : float, optional
        Minimum bond angle in degrees. Default is 120.0.

    Returns
    -------
    list of dict
        List of dictionaries containing hydrogen bond information with keys:
        donor_chain, donor_resi, donor_resname, donor_atom, donor_altloc,
        acceptor_chain, acceptor_resi, acceptor_resname, acceptor_atom,
        acceptor_altloc, DA_dist, angle, category. The category indicates
        the type of H-bond (e.g., 'backbone-backbone', 'backbone-sidechain').
    """
    hbonds = []
    for d in donors:
        for a in acceptors:
            if not altloc_compatible(d.altloc, a.altloc):
                continue
            if d.res_key == a.res_key and d.atom_name == a.atom_name and d.altloc == a.altloc:
                continue

            DA = float(np.linalg.norm(d.coord_donor - a.coord))
            if DA > da_max:
                continue

            ok = False
            used_angle = None

            # Prefer explicit H if available
            if d.has_explicit_H and d.H_coord is not None:
                HA = float(np.linalg.norm(d.H_coord - a.coord))
                if HA <= h_a_max:
                    ang = angle_deg(d.coord_donor - d.H_coord, a.coord - d.H_coord)
                    used_angle = ang
                    if ang >= angle_min:
                        ok = True

            # Fallback to base-atom angle
            if not ok and d.coord_base is not None:
                ang = angle_deg(d.coord_base - d.coord_donor,
                                a.coord - d.coord_donor)
                used_angle = ang
                if ang >= angle_min:
                    ok = True

            if not ok:
                continue

            cat = (
                ("backbone" if d.is_backbone else "sidechain")
                + "-"
                + ("backbone" if a.is_backbone else "sidechain")
            )

            hbonds.append(
                {
                    "donor_chain": d.chain,
                    "donor_resi": d.resi,
                    "donor_resname": d.resname,
                    "donor_atom": d.atom_name,
                    "donor_altloc": norm_alt(d.altloc),
                    "acceptor_chain": a.chain,
                    "acceptor_resi": a.resi,
                    "acceptor_resname": a.resname,
                    "acceptor_atom": a.atom_name,
                    "acceptor_altloc": norm_alt(a.altloc),
                    "DA_dist": round(DA, 3),
                    "angle": round(used_angle if used_angle is not None else -1.0, 1),
                    "category": cat,
                }
            )
    return hbonds

#_________________________________PACKING_________________________
def is_heavy(atom_name: str) -> bool:
    """
    Check if an atom is a heavy atom (non-hydrogen).

    Parameters
    ----------
    atom_name : str
        Name of the atom.

    Returns
    -------
    bool
        True if the atom is heavy (does not start with 'H' or 'D'),
        False otherwise.
    """
    n = atom_name.strip()
    return not (n.startswith("H") or n.startswith("D"))


def residue_key(chain_id: str, res_id: int) -> str:
    """
    Build a unique residue identifier string.

    Parameters
    ----------
    chain_id : str
        Chain identifier.
    res_id : int
        Residue number.

    Returns
    -------
    str
        A unique identifier in the format 'chain:resi'.
    """
    return f"{chain_id}:{int(res_id)}"
