"""
Re-number biogenesis output CSVs so that resi_struct matches the user-specified reference
PDB via pairwise sequence alignment.

Logic per structure
-------------------
1. Reference PDB and list of PDBs to renumber are chosen by the user, either
   by listing PDB IDs with --pdbs, or by auto-detecting from --input-dir
2. For every chain in the query CSV, align its sequence to the reference
   sequence (chain can be specified if needed, default A).
3. Build a mapping  query_resi -> ref_resi  from the alignment.
4. Count aligned positions where resn differs (true sequence mismatches).
5. If total mismatches across ALL chains exceed --max-mismatches, discard the
   structure and report it.
6. Otherwise renumber resi_struct and resi_mut in the features, metadata, and
   bonds CSVs and save to <output_dir>/renumbered/.
"""

import sys
from pathlib import Path

import biotite.sequence.align as align
import pandas as pd
from biotite.sequence import ProteinSequence

DEFAULT_INPUT_DIR = "."
RENUMBERED_DIR = "renumbered"

# Non-standard amino acids not covered by biotite's standard protein alphabet.
_NONSTANDARD_AA = {"SEC": "U", "PYL": "O"}

def to1(resn: str) -> str:
    """Convert a 3-letter residue name to a 1-letter code (returns 'X' for unknowns)."""
    if resn in _NONSTANDARD_AA:
        return _NONSTANDARD_AA[resn]
    try:
        return ProteinSequence.convert_letter_3to1(resn)
    except KeyError:
        return "X"


def build_alignment_params() -> tuple[align.SubstitutionMatrix, tuple[int, float]]:
    """
    Return the BLOSUM62 substitution matrix and affine gap penalties for protein global alignments.
    This function encapsulates the alignment scoring choices.
    """
    matrix = align.SubstitutionMatrix.std_protein_matrix()
    gap_penalty = (-4, -0.5)  # open, extend — similar to prior BioPython settings
    return matrix, gap_penalty


def get_chain_sequence(df: pd.DataFrame, chain: str) -> list[tuple[int, str]]:
    """
    Extract the sequence for a given chain in (resi, 1-letter-code) tuples, sorted by residue number.
    Used for both reference and query PDBs.
    """
    sub = df[df["chain"] == chain].sort_values("resi_struct")
    return [(int(row.resi_struct), to1(row.resn_struct)) for row in sub.itertuples()]


def align_and_map(
    ref_seq: list[tuple[int, str]],
    qry_seq: list[tuple[int, str]],
    matrix: align.SubstitutionMatrix,
    gap_penalty: tuple[int, float],
) -> tuple[dict[int, int | None], int]:
    """
    Perform global sequence alignment (query to reference) and map query residue numbers to reference.
    
    Returns
    -------
    mapping : dict  Maps query residue numbers to reference residue numbers (None if in a gap).
    mismatches : int  Number of aligned positions where residues differ.
    """
    ref_prot = ProteinSequence("".join(r for _, r in ref_seq))
    qry_prot = ProteinSequence("".join(r for _, r in qry_seq))

    # Perform global alignment of query to reference sequence
    alignment = align.align_optimal(
        ref_prot, qry_prot, matrix,
        gap_penalty=gap_penalty,
        local=False,
    )[0]

    mismatches = 0
    mapping: dict[int, int | None] = {}

    # Iterate through alignment trace to build mapping and count mismatches
    for r_idx, q_idx in alignment.trace:
        if r_idx == -1 or q_idx == -1:
            continue
        if ref_prot[r_idx] != qry_prot[q_idx]:
            mismatches += 1
        mapping[qry_seq[q_idx][0]] = ref_seq[r_idx][0]

    # Any residues in the query chain not aligned are mapped to None
    aligned_q = {qry_seq[q][0] for _, q in alignment.trace if q != -1}
    for resi, _ in qry_seq:
        if resi not in aligned_q:
            mapping[resi] = None

    return mapping, mismatches


def renumber_df(df: pd.DataFrame, mapping: dict[int, int | None]) -> pd.DataFrame:
    """
    Apply a residue number mapping to the DataFrame's resi_struct column (and resi_mut if present).
    Returns a copy of the DataFrame with renumbered residues.
    """
    df = df.copy()
    df["resi_struct"] = df["resi_struct"].map(mapping)
    if "resi_mut" in df.columns:
        df["resi_mut"] = df["resi_mut"].map(mapping)
    return df


def detect_pdbs(input_dir: Path, ref_pdb: str = None) -> list[str]:
    """
    Detect all unique PDB IDs in the input directory based on *_features.csv files.
    Optionally prioritize the reference PDB as first in the list.
    """
    pdbs = set()
    for f in input_dir.glob("*_features.csv"):
        pdb = f.name.split("_features.csv")[0]
        if pdb:
            pdbs.add(pdb)
    if ref_pdb and ref_pdb in pdbs:
        return [ref_pdb] + sorted(p for p in pdbs if p != ref_pdb)
    return sorted(pdbs)


def renumber_structures(
    ref_pdb: str,
    max_mismatches: int,
    input_dir: str = DEFAULT_INPUT_DIR,
    pdb_list: list[str] = None,
    ref_chain: str = "A",
) -> None:
    """
    Driver: Renumber all structures in input_dir to match the reference PDB's residue numbering.

    For each query PDB:
        - Align its sequence(s) to the reference chain.
        - Build mapping from query residue numbers to reference.
        - Discard if total mismatches exceed threshold.
        - Otherwise, renumber all related CSVs and save to output directory.
    """
    input_dir = Path(input_dir)
    renum_dir = input_dir / RENUMBERED_DIR
    renum_dir.mkdir(parents=True, exist_ok=True)
    matrix, gap_penalty = build_alignment_params()

    # Determine PDB IDs to process (either user-supplied or auto-detect)
    if pdb_list is None or len(pdb_list) == 0:
        pdb_ids = detect_pdbs(input_dir, ref_pdb=ref_pdb)
    else:
        pdb_ids = pdb_list

    # Load reference features and extract the chain sequence used for alignment
    ref_path = input_dir / f"{ref_pdb}_features.csv"
    if not ref_path.exists():
        sys.exit(f"Reference CSV not found: {ref_path}")

    ref_df = pd.read_csv(ref_path)
    ref_chain_seq = get_chain_sequence(ref_df, ref_chain)
    print(f"Reference: {ref_pdb}  chain {ref_chain}  ({len(ref_chain_seq)} residues)")
    print(f"Max mismatches allowed: {max_mismatches}\n")

    kept, removed = [], []

    # Process each PDB
    for pdb_id in pdb_ids:
        feat_path = input_dir / f"{pdb_id}_features.csv"
        if not feat_path.exists():
            continue

        # Special case: reference PDB, copy all CSVs through unchanged
        if pdb_id == ref_pdb:
            for suffix in ("features", "metadata", "bonds"):
                src = input_dir / f"{pdb_id}_{suffix}.csv"
                if src.exists():
                    pd.read_csv(src).to_csv(renum_dir / src.name, index=False)
            kept.append(pdb_id)
            continue

        feat_df = pd.read_csv(feat_path)
        chains = sorted(feat_df["chain"].unique())

        # Align each chain found in the features file; accumulate mapping and mismatch info
        chain_maps: dict[str, dict[int, int | None]] = {}
        total_mismatches = 0
        chain_info = []

        for chain in chains:
            qry_seq = get_chain_sequence(feat_df, chain)
            chain_map, mm = align_and_map(ref_chain_seq, qry_seq, matrix, gap_penalty)
            chain_maps[chain] = chain_map
            total_mismatches += mm
            chain_info.append(f"{chain}:{mm}")

        mismatch_str = "  mismatches: " + ", ".join(chain_info) + f"  (total={total_mismatches})"

        # Remove query structures with excessive mismatches to the reference
        if total_mismatches > max_mismatches:
            print(f"[REMOVED] {pdb_id}{mismatch_str}")
            removed.append(pdb_id)
            continue
        kept.append(pdb_id)

        # Apply mapping to all relevant CSVs (features, metadata, bonds)
        for suffix in ("features", "metadata", "bonds"):
            src = input_dir / f"{pdb_id}_{suffix}.csv"
            if not src.exists():
                continue
            df = pd.read_csv(src)

            # Remap resi_struct and optionally resi_mut using chain-specific map
            def remap_resi(row, col):
                # Look up the query residue number in the per-chain mapping
                m = chain_maps.get(row["chain"], {})
                return m.get(row[col], row[col])

            df["resi_struct"] = df.apply(lambda r: remap_resi(r, "resi_struct"), axis=1)
            if "resi_mut" in df.columns:
                df["resi_mut"] = df.apply(lambda r: remap_resi(r, "resi_mut"), axis=1)

            df.to_csv(renum_dir / src.name, index=False)
