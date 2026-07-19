"""
Tests for renumber_to_referencePDB.py
"""
from pathlib import Path

import biotite.sequence.align as align
import pandas as pd
from src.grouped_analysis.renumber_to_referencePDB import (
    align_and_map,
    build_alignment_params,
    get_chain_sequence,
    renumber_structures,
    to1,
)

def test_to1_standard_residues():
    assert to1("ALA") == "A"
    assert to1("GLY") == "G"
    assert to1("TRP") == "W"
    assert to1("MET") == "M"

def test_to1_unknown_returns_X():
    assert to1("UNK") == "X"
    assert to1("HOH") == "X"
    assert to1("XYZ") == "X"

def test_to1_nonstandard_known():
    assert to1("SEC") == "U"
    assert to1("PYL") == "O"


def test_build_alignment_params_returns_matrix_and_gap_penalty():
    matrix, gap_penalty = build_alignment_params()
    assert isinstance(matrix, align.SubstitutionMatrix)
    assert isinstance(gap_penalty, tuple)
    assert len(gap_penalty) == 2

def test_build_alignment_params_gap_penalty_values():
    _, gap_penalty = build_alignment_params()
    open_penalty, extend_penalty = gap_penalty
    assert open_penalty == -4
    assert extend_penalty == -0.5

def _make_chainseq_df():
    return pd.DataFrame({
        "chain": ["A", "A", "A", "B"],
        "resi_struct": [1, 2, 3, 1],
        "resn_struct": ["ALA", "GLY", "TRP", "LEU"],
    })

def test_get_chain_sequence_returns_correct_chain():
    df = _make_chainseq_df()
    seq = get_chain_sequence(df, "A")
    assert seq == [(1, "A"), (2, "G"), (3, "W")]

def test_get_chain_sequence_other_chain():
    df = _make_chainseq_df()
    seq = get_chain_sequence(df, "B")
    assert seq == [(1, "L")]

def test_get_chain_sequence_sorted_by_resi():
    df = pd.DataFrame({
        "chain": ["A", "A", "A"],
        "resi_struct": [10, 1, 5],
        "resn_struct": ["ALA", "GLY", "TRP"],
    })
    seq = get_chain_sequence(df, "A")
    assert [r for r, _ in seq] == [1, 5, 10]

def test_get_chain_sequence_unknown_residue_becomes_X():
    df = pd.DataFrame({
        "chain": ["A"],
        "resi_struct": [1],
        "resn_struct": ["UNK"],
    })
    seq = get_chain_sequence(df, "A")
    assert seq == [(1, "X")]

def aligner_params():
    matrix, gap_penalty = build_alignment_params()
    return matrix, gap_penalty

def test_align_and_map_identical_sequences_no_mismatches(aligner_params):
    matrix, gap_penalty = aligner_params
    ref = [(1, "A"), (2, "G"), (3, "V")]
    qry = [(10, "A"), (20, "G"), (30, "V")]
    mapping, mm = align_and_map(ref, qry, matrix, gap_penalty)
    assert mm == 0
    assert mapping[10] == 1
    assert mapping[20] == 2
    assert mapping[30] == 3

def test_align_and_map_one_mismatch(aligner_params):
    matrix, gap_penalty = aligner_params
    ref = [(1, "A"), (2, "G"), (3, "V")]
    qry = [(1, "A"), (2, "W"), (3, "V")]  # W != G
    _, mm = align_and_map(ref, qry, matrix, gap_penalty)
    assert mm == 1

def test_align_and_map_all_mismatches(aligner_params):
    matrix, gap_penalty = aligner_params
    ref = [(1, "A"), (2, "G"), (3, "V")]
    qry = [(1, "W"), (2, "W"), (3, "W")]
    _, mm = align_and_map(ref, qry, matrix, gap_penalty)
    assert mm == 3

def test_align_and_map_query_subset_maps_to_reference(aligner_params):
    """Query shorter than reference: aligned portion must map correctly."""
    matrix, gap_penalty = aligner_params
    ref = [(1, "A"), (2, "G"), (3, "V"), (4, "L")]
    qry = [(1, "A"), (2, "G")]
    mapping, mm = align_and_map(ref, qry, matrix, gap_penalty)
    assert mapping[1] == 1
    assert mapping[2] == 2

def test_align_and_map_mapping_values_are_reference_resis(aligner_params):
    matrix, gap_penalty = aligner_params
    ref = [(5, "A"), (6, "G"), (7, "L")]
    qry = [(1, "A"), (2, "G"), (3, "L")]
    mapping, _ = align_and_map(ref, qry, matrix, gap_penalty)
    assert set(mapping.values()).issubset({5, 6, 7, None})

def _write_features(directory: Path, pdb_id: str, resi_range, resnames):
    rows = [{
        "chain": "A",
        "resi_struct": r,
        "resi_mut": r,
        "resn_struct": resnames[i % len(resnames)],
    } for i, r in enumerate(resi_range)]
    pd.DataFrame(rows).to_csv(
        directory / f"{pdb_id}_features.csv", index=False
    )

def test_main_reference_is_copied_unchanged(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    resnames = ["ALA", "GLY", "VAL", "LEU", "SER"]
    _write_features(tmp_path, "REFA", range(1, 6), resnames)
    _write_features(tmp_path, "BBBB", range(1, 6), resnames)
    renumber_structures(ref_pdb="REFA", max_mismatches=5,
         pdb_list=["REFA", "BBBB"], input_dir=str(tmp_path))
    out = pd.read_csv(tmp_path / "renumbered" / "REFA_features.csv")
    assert list(out["resi_struct"]) == list(range(1, 6))

def test_main_identical_seqs_kept_and_renumbered(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    resnames = ["ALA", "GLY", "VAL", "LEU", "SER"]
    _write_features(tmp_path, "REFA", range(1, 6), resnames)
    _write_features(tmp_path, "BBBB", range(10, 15), resnames)
    renumber_structures(ref_pdb="REFA", max_mismatches=0,
         pdb_list=["REFA", "BBBB"], input_dir=str(tmp_path))
    out = pd.read_csv(tmp_path / "renumbered" / "BBBB_features.csv")
    assert set(out["resi_struct"]) == set(range(1, 6))

def test_main_too_many_mismatches_removes_structure(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    ref_resnames = ["ALA", "GLY", "VAL"]
    qry_resnames = ["TRP", "TRP", "TRP"]  # 3 mismatches
    _write_features(tmp_path, "REFA", range(1, 4), ref_resnames)
    _write_features(tmp_path, "BBBB", range(1, 4), qry_resnames)
    renumber_structures(ref_pdb="REFA", max_mismatches=0,
         pdb_list=["REFA", "BBBB"], input_dir=str(tmp_path))
    assert not (tmp_path / "renumbered" / "BBBB_features.csv").exists()

def test_main_missing_features_csv_skipped(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    resnames = ["ALA", "GLY", "VAL"]
    _write_features(tmp_path, "REFA", range(1, 4), resnames)
    # BBBB has no features CSV
    renumber_structures(ref_pdb="REFA", max_mismatches=5,
         pdb_list=["REFA", "BBBB"], input_dir=str(tmp_path))
    assert not (tmp_path / "renumbered" / "BBBB_features.csv").exists()

def test_main_creates_renumbered_dir(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    resnames = ["ALA"]
    _write_features(tmp_path, "REFA", range(1, 2), resnames)
    renumber_structures(ref_pdb="REFA", max_mismatches=5,
         pdb_list=["REFA"], input_dir=str(tmp_path))
    assert (tmp_path / "renumbered").is_dir()
