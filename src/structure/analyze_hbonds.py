#!/usr/bin/env python3
import argparse
import math
import os
from collections import defaultdict, Counter, namedtuple

import numpy as np
import pandas as pd
import networkx as nx

from biotite.structure import AtomArray, filter_amino_acids
from biotite.structure.io.pdb import PDBFile
# from biotite.structure.io.mmcif import MMCIFFile


# ----------------------------- Tunables --------------------------------------

DA_MAX = 3.5           # Å donor–acceptor distance
H_A_MAX = 2.6          # Å H–acceptor distance
ANGLE_MIN = 120.0      # degrees minimum angle

INCLUDE_WATER = True
INCLUDE_LIGANDS = False

# -----------------------------------------------------------------------------


DonorSite = namedtuple("DonorSite",
    "res_uid resname chain resi atom_name altloc base_atom base_altloc coord_base coord_donor "
    "has_explicit_H H_name H_altloc H_coord is_backbone"
)
AcceptorSite = namedtuple("AcceptorSite",
    "res_uid resname chain resi atom_name altloc coord is_backbone"
)


# --------------------------- Helpers ---------------------------

def norm(v):
    x = np.linalg.norm(v)
    return v / x if x > 1e-8 else v

def angle_deg(u, v):
    un = norm(u); vn = norm(v)
    dot = np.clip(np.dot(un, vn), -1.0, 1.0)
    return math.degrees(math.acos(dot))

def distance(a, b):
    return float(np.linalg.norm(a - b))

def residue_uid(chain_id, resi):
    return f"{chain_id}:{resi}"

def is_backbone_atom(atom_name):
    return atom_name in ("N", "CA", "C", "O", "OXT", "H", "H1", "H2", "H3")

def classify_backbone_sidechain(atom_name):
    return "backbone" if is_backbone_atom(atom_name) else "sidechain"

def norm_alt(a):
    if a is None: return ''
    s = str(a).strip()
    return s.upper() if s else ''

def altloc_compatible(d_alt, a_alt):
    d = norm_alt(d_alt)
    a = norm_alt(a_alt)
    if d == '' or a == '':
        return True
    return d == a


# ------------------------- Biotite utilities ---------------------------

def load_structure_any(path) -> AtomArray:
    ext = os.path.splitext(path)[1].lower()
    if ext in (".cif", ".mmcif"):
        mm = MMCIFFile.read(path)
        arr = mm.get_structure(model=1, extra_fields=["b_factor", "occupancy", "altloc_id"])
    else:
        pdb = PDBFile.read(path)
        arr = pdb.get_structure(model=1, extra_fields=["b_factor", "occupancy", "altloc_id"])
    return arr

def split_by_residue(arr: AtomArray):
    if arr.array_length() == 0:
        return
    order = np.lexsort((arr.res_id, arr.chain_id))
    arr = arr[order]

    starts = [0]
    for i in range(1, arr.array_length()):
        if arr.chain_id[i] != arr.chain_id[i-1] or arr.res_id[i] != arr.res_id[i-1]:
            starts.append(i)
    starts.append(arr.array_length())

    for s, e in zip(starts[:-1], starts[1:]):
        yield arr.res_name[s], arr.chain_id[s], int(arr.res_id[s]), np.arange(s, e, dtype=int), arr

def residue_ok(resname, is_protein_flag):
    if is_protein_flag:
        return True
    if INCLUDE_WATER and resname.upper() in ("HOH", "WAT", "H2O"):
        return True
    return INCLUDE_LIGANDS


# ---------------------- Chemistry templates ----------------------

def donor_acceptor_templates(resname, names_set):
    donors = []
    acceptors = []
    resname3 = resname.strip().upper()

    if resname3 in ("HOH", "WAT", "H2O"):
        if "O" in names_set:
            donors.append(("O", None, "H*"))
            acceptors.append("O")
        return donors, acceptors

    if "N" in names_set:
        base = "CA" if "CA" in names_set else ("C" if "C" in names_set else None)
        donors.append(("N", base, "H*"))
    if "O" in names_set:
        acceptors.append("O")
    if "OXT" in names_set:
        acceptors.append("OXT")

    if resname3 in ("SER", "THR", "TYR"):
        oname = "OG" if resname3 == "SER" else ("OG1" if resname3 == "THR" else "OH")
        if oname in names_set:
            donors.append((oname, "CB" if "CB" in names_set else None, "H*"))
            acceptors.append(oname)

    if resname3 == "LYS" and "NZ" in names_set:
        donors.append(("NZ", "CE" if "CE" in names_set else None, "H*"))

    if resname3 == "ARG":
        for d, base in (("NE","CD"),("NH1","CZ"),("NH2","CZ")):
            if d in names_set:
                donors.append((d, base if base in names_set else None, "H*"))

    if resname3 == "HIS":
        for d, base in (("ND1","CG"),("NE2","CD2")):
            if d in names_set:
                donors.append((d, base if base in names_set else None, None))
        for a in ("ND1","NE2"):
            if a in names_set:
                acceptors.append(a)

    if resname3 == "ASP":
        for a in ("OD1","OD2"):
            if a in names_set:
                acceptors.append(a)
    if resname3 == "GLU":
        for a in ("OE1","OE2"):
            if a in names_set:
                acceptors.append(a)

    if resname3 == "ASN":
        if "OD1" in names_set: acceptors.append("OD1")
        if "ND2" in names_set: donors.append(("ND2", "CG" if "CG" in names_set else None, None))
    if resname3 == "GLN":
        if "OE1" in names_set: acceptors.append("OE1")
        if "NE2" in names_set: donors.append(("NE2", "CD" if "CD" in names_set else None, None))

    return donors, acceptors


# ----------------------- Site building (keep altlocs) -----------------------

def _index_by_name_alt(arr: AtomArray, idxs):
    d = defaultdict(dict)
    for i in idxs:
        name = arr.atom_name[i].strip()
        alt = norm_alt(arr.altloc_id[i])
        d[name][alt] = i
    return d

def _find_h_for_name_alt(name_to_alt_to_idx, donor_alt):
    candidates = ("H", "H1", "H2", "H3", "HG", "HG1", "HH", "HZ", "HZ1", "HZ2", "HZ3")
    for nm in candidates:
        if nm in name_to_alt_to_idx:
            if donor_alt in name_to_alt_to_idx[nm]:
                return nm, donor_alt, name_to_alt_to_idx[nm][donor_alt]
            if '' in name_to_alt_to_idx[nm]:
                return nm, '', name_to_alt_to_idx[nm]['']
    return None, None, None

def _pick_base(name_to_alt_to_idx, base_name, donor_alt):
    if base_name is None or base_name not in name_to_alt_to_idx:
        return None, None, None
    d = name_to_alt_to_idx[base_name]
    if donor_alt in d:
        return base_name, donor_alt, d[donor_alt]
    if '' in d:
        return base_name, '', d['']
    return None, None, None

def build_sites_biotite(arr: AtomArray):
    prot_mask = filter_amino_acids(arr)
    donors, acceptors = [], []

    for resname, chain_id, resi, idxs, base_arr in split_by_residue(arr):
        is_protein = bool(prot_mask[idxs].any())
        if not residue_ok(resname, is_protein):
            continue

        name_to_alt = _index_by_name_alt(base_arr, idxs)
        names_set = set(name_to_alt.keys())
        dtempl, atempl = donor_acceptor_templates(resname, names_set)

        # Acceptors
        for aname in atempl:
            if aname not in name_to_alt:
                continue
            for a_alt, ai in name_to_alt[aname].items():
                A_coord = base_arr.coord[ai].astype(float)
                acceptors.append(
                    AcceptorSite(
                        residue_uid(chain_id, resi), resname, chain_id, resi, aname, a_alt,
                        A_coord, classify_backbone_sidechain(aname) == "backbone"
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
                H_nm, H_alt, hi = _find_h_for_name_alt(name_to_alt, d_alt) if hsentinel else (None,None,None)
                H_coord = base_arr.coord[hi].astype(float) if hi is not None else None

                donors.append(
                    DonorSite(
                        residue_uid(chain_id, resi), resname, chain_id, resi, dname, d_alt,
                        base_nm, base_alt, B_coord, D_coord,
                        hi is not None, H_nm, H_alt, H_coord,
                        classify_backbone_sidechain(dname) == "backbone"
                    )
                )
    return donors, acceptors


# ----------------------- H-bond detection -----------------------

def detect_hbonds(donors, acceptors):
    hbonds = []
    for d in donors:
        for a in acceptors:
            if not altloc_compatible(d.altloc, a.altloc):
                continue
            if d.res_uid == a.res_uid and d.atom_name == a.atom_name and d.altloc == a.altloc:
                continue
            DA = distance(d.coord_donor, a.coord)
            if DA > DA_MAX:
                continue
            ok = False
            used_angle = None
            if d.has_explicit_H and d.H_coord is not None:
                HA = distance(d.H_coord, a.coord)
                if HA <= H_A_MAX:
                    ang = angle_deg(d.coord_donor - d.H_coord, a.coord - d.H_coord)
                    used_angle = ang
                    if ang >= ANGLE_MIN:
                        ok = True
            if not ok and d.coord_base is not None:
                ang = angle_deg(d.coord_base - d.coord_donor, a.coord - d.coord_donor)
                used_angle = ang
                if ang >= ANGLE_MIN:
                    ok = True
            if not ok:
                continue
            cat = (
                ("backbone" if d.is_backbone else "sidechain")
                + "-"
                + ("backbone" if a.is_backbone else "sidechain")
            )
            hbonds.append({
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
                "DA_dist": round(DA,3),
                "angle": round(used_angle if used_angle is not None else -1.0,1),
                "category": cat
            })
    return hbonds


# ------------------------- Graph -------------------------

def build_graph(hbonds):
    G = nx.Graph()
    for h in hbonds:
        u = f"{h['donor_chain']}:{h['donor_resi']}:{h['donor_resname']}"
        v = f"{h['acceptor_chain']}:{h['acceptor_resi']}:{h['acceptor_resname']}"
        for node, chain, resi, resn in ((u,h['donor_chain'],h['donor_resi'],h['donor_resname']),
                                        (v,h['acceptor_chain'],h['acceptor_resi'],h['acceptor_resname'])):
            if not G.has_node(node):
                G.add_node(node, chain=chain, resi=int(resi), resname=resn)
        if not G.has_edge(u,v):
            G.add_edge(u,v,count=0,categories=Counter(),distances=[],angles=[],altloc_pairs=Counter())
        G[u][v]['count'] += 1
        G[u][v]['categories'][h['category']] += 1
        G[u][v]['distances'].append(h['DA_dist'])
        G[u][v]['angles'].append(h['angle'])
        G[u][v]['altloc_pairs'][(h["donor_altloc"],h["acceptor_altloc"])] += 1
    return G


# ------------------------- Pandas I/O -------------------------

def write_hbond_csv_pandas(hbonds, outpath):
    cols = ["donor_chain","donor_resi","donor_resname","donor_atom","donor_altloc",
            "acceptor_chain","acceptor_resi","acceptor_resname","acceptor_atom","acceptor_altloc",
            "DA_dist","angle","category"]
    df = pd.DataFrame(hbonds)
    # ensure all expected columns exist even if list is empty
    for c in cols:
        if c not in df.columns:
            df[c] = []
    df = df[cols]
    df.to_csv(outpath, index=False)

def write_residue_summary_pandas(G, outpath):
    if G.number_of_nodes() == 0:
        pd.DataFrame(columns=["chain","resi","resname","degree","betweenness","donor_count_est","acceptor_count_est"]).to_csv(outpath, index=False)
        return
    deg = dict(G.degree())
    bc = nx.betweenness_centrality(G)
    rows = []
    for n, data in G.nodes(data=True):
        dcount = acount = 0
        for nbr in G.neighbors(n):
            e = G[nbr][n]
            dcount += e.get("count", 0)//2
            acount += e.get("count", 0)//2
        rows.append({
            "chain": data["chain"],
            "resi": int(data["resi"]),
            "resname": data["resname"],
            "degree": deg.get(n, 0),
            "betweenness": round(bc.get(n, 0.0), 4),
            "donor_count_est": dcount,
            "acceptor_count_est": acount
        })
    pd.DataFrame(rows).sort_values(["chain","resi"]).to_csv(outpath, index=False)


# -------------------------- Main --------------------------

def analyze_single_pdb(pdb_path, outdir):
    arr = load_structure_any(pdb_path)
    donors, acceptors = build_sites_biotite(arr)
    hb = detect_hbonds(donors, acceptors)
    G = build_graph(hb)
    os.makedirs(outdir, exist_ok=True)
    base = os.path.basename(pdb_path)
    write_hbond_csv_pandas(hb, os.path.join(outdir, f"{base}_hbonds.csv"))
    write_residue_summary_pandas(G, os.path.join(outdir, f"{base}_residue_summary.csv"))
    print(f"[OK] {pdb_path}: {len(hb)} H-bonds; nodes={G.number_of_nodes()} edges={G.number_of_edges()}")

def main():
    ap = argparse.ArgumentParser(description="Altloc-aware hydrogen-bond network analysis for a single PDB/mmCIF (Biotite).")
    ap.add_argument("--pdb", required=True, help="Path to PDB/mmCIF file.")
    ap.add_argument("--outdir", default="hbond_out", help="Output directory.")
    args = ap.parse_args()
    analyze_single_pdb(args.pdb, args.outdir)

if __name__ == "__main__":
    main()
