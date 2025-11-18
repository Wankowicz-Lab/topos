# utils.py
from __future__ import annotations
import math
from collections import defaultdict, namedtuple
from typing import Dict, List, Tuple

import numpy as np
import biotite.structure as struc

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
    un = _norm_vec(u)
    vn = _norm_vec(v)
    dot = np.clip(np.dot(un, vn), -1.0, 1.0)
    return float(math.degrees(math.acos(dot)))


def _res_key(chain, resi, resname) -> str:
    return f"{chain}:{int(resi)}:{resname}"


def is_backbone_atom(atom_name: str) -> bool:
    return atom_name in ("N", "CA", "C", "O", "OXT", "H", "H1", "H2", "H3")


def norm_alt(a) -> str:
    if a is None:
        return ""
    s = str(a).strip()
    return s.upper() if s else ""


def altloc_compatible(d_alt, a_alt) -> bool:
    d = norm_alt(d_alt)
    a = norm_alt(a_alt)
    if d == "" or a == "":
        return True
    return d == a

def _split_by_residue(arr: struc.AtomArray):
    """
    Yield per residue:
      (resname, chain_id, res_id, idxs, base_arr)

    Residues are sorted by (chain_id, res_id).
    """
    if arr.array_length() == 0:
        return

    order = np.lexsort((arr.res_id, arr.chain_id))
    arr = arr[order]

    starts = [0]
    for i in range(1, arr.array_length()):
        if (arr.chain_id[i] != arr.chain_id[i - 1] or
                arr.res_id[i] != arr.res_id[i - 1]):
            starts.append(i)
    starts.append(arr.array_length())

    for s, e in zip(starts[:-1], starts[1:]):
        yield (
            arr.res_name[s],
            arr.chain_id[s],
            int(arr.res_id[s]),
            np.arange(s, e, dtype=int),
            arr,
        )

def _residue_ok(resname: str,
                is_protein_flag: bool,
                include_water: bool = True,
                include_ligands: bool = False) -> bool:
    '''
    If considering protein/ligand/solvent
    '''
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
    for i in idxs:
        name = arr.atom_name[i].strip()
        alt = norm_alt(arr.altloc_id[i])
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
):
    """
    Build DonorSite and AcceptorSite lists from a Biotite AtomArray.
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
):
    """
    Return a list of H-bond dicts with keys:
      donor_chain, donor_resi, donor_resname, donor_atom, donor_altloc
      acceptor_chain, acceptor_resi, acceptor_resname, acceptor_atom, acceptor_altloc
      DA_dist, angle, category
    where category is one of: backbone-backbone, backbone-sidechain, etc.
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
    Return True if the atom is considered heavy (non-hydrogen).
    """
    n = atom_name.strip()
    return not (n.startswith("H") or n.startswith("D"))


def residue_key(chain_id, res_id) -> str:
    """
    Build a unique residue identifier (chain:resi).
    """
    return f"{chain_id}:{int(res_id)}"
