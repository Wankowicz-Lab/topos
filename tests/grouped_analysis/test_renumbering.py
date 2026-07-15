"""
Tests for renumber_to_reference.py
"""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


from renumber_to_reference import (
    align_and_map,
    build_aligner,
    get_chain_sequence,
    main,
    to1,
)


class TestTo1:
    def test_standard_residues(self):
        assert to1("ALA") == "A"
        assert to1("GLY") == "G"
        assert to1("TRP") == "W"
        assert to1("MET") == "M"

    def test_case_insensitive(self):
        assert to1("ala") == "A"
        assert to1("Gly") == "G"

    def test_unknown_returns_X(self):
        assert to1("UNK") == "X"
        assert to1("HOH") == "X"
        assert to1("XYZ") == "X"

    def test_nonstandard_known(self):
        assert to1("SEC") == "U"
        assert to1("PYL") == "O"


class TestBuildAligner:
    def test_returns_aligner(self):
        from Bio.Align import PairwiseAligner
        aligner = build_aligner()
        assert isinstance(aligner, PairwiseAligner)

    def test_mode_is_global(self):
        aligner = build_aligner()
        assert aligner.mode == "global"

    def test_scores(self):
        aligner = build_aligner()
        assert aligner.match_score == 2
        assert aligner.mismatch_score == -1
        assert aligner.open_gap_score == -4
        assert aligner.extend_gap_score == -0.5



class TestGetChainSequence:
    def _make_df(self):
        return pd.DataFrame({
            "chain": ["A", "A", "A", "B"],
            "resi_struct": [1, 2, 3, 1],
            "resn_struct": ["ALA", "GLY", "TRP", "LEU"],
        })

    def test_returns_correct_chain(self):
        df = self._make_df()
        seq = get_chain_sequence(df, "A")
        assert seq == [(1, "A"), (2, "G"), (3, "W")]

    def test_other_chain(self):
        df = self._make_df()
        seq = get_chain_sequence(df, "B")
        assert seq == [(1, "L")]

    def test_sorted_by_resi(self):
        df = pd.DataFrame({
            "chain": ["A", "A", "A"],
            "resi_struct": [10, 1, 5],
            "resn_struct": ["ALA", "GLY", "TRP"],
        })
        seq = get_chain_sequence(df, "A")
        assert [r for r, _ in seq] == [1, 5, 10]

    def test_unknown_residue_becomes_X(self):
        df = pd.DataFrame({
            "chain": ["A"],
            "resi_struct": [1],
            "resn_struct": ["UNK"],
        })
        seq = get_chain_sequence(df, "A")
        assert seq == [(1, "X")]



class TestAlignAndMap:
    @pytest.fixture(autouse=True)
    def _aligner(self):
        self.aligner = build_aligner()

    def test_identical_sequences_no_mismatches(self):
        ref = [(1, "A"), (2, "G"), (3, "V")]
        qry = [(10, "A"), (20, "G"), (30, "V")]
        mapping, mm = align_and_map(ref, qry, self.aligner)
        assert mm == 0
        assert mapping[10] == 1
        assert mapping[20] == 2
        assert mapping[30] == 3

    def test_one_mismatch(self):
        ref = [(1, "A"), (2, "G"), (3, "V")]
        qry = [(1, "A"), (2, "W"), (3, "V")]  # W != G
        _, mm = align_and_map(ref, qry, self.aligner)
        assert mm == 1

    def test_all_mismatches(self):
        ref = [(1, "A"), (2, "G"), (3, "V")]
        qry = [(1, "W"), (2, "W"), (3, "W")]
        _, mm = align_and_map(ref, qry, self.aligner)
        assert mm == 3

    def test_query_subset_maps_to_reference(self):
        """Query shorter than reference: no assertion on gap residues."""
        ref = [(1, "A"), (2, "G"), (3, "V"), (4, "L")]
        qry = [(1, "A"), (2, "G")]
        mapping, mm = align_and_map(ref, qry, self.aligner)
        # The aligned portion must map correctly
        assert mapping[1] == 1
        assert mapping[2] == 2

    def test_mapping_values_are_reference_resis(self):
        ref = [(5, "A"), (6, "G"), (7, "L")]
        qry = [(1, "A"), (2, "G"), (3, "L")]
        mapping, _ = align_and_map(ref, qry, self.aligner)
        assert set(mapping.values()).issubset({5, 6, 7, None})


# ── main (integration) ────────────────────────────────────────────────────────

class TestMainIntegration:
    def _write_features(self, directory: Path, pdb_id: str,
                        resi_range, resnames):
        rows = [{
            "chain": "A",
            "resi_struct": r,
            "resi_mut": r,
            "resn_struct": resnames[i % len(resnames)],
        } for i, r in enumerate(resi_range)]
        pd.DataFrame(rows).to_csv(
            directory / f"{pdb_id}_features.csv", index=False
        )

    def test_reference_is_copied_unchanged(self, tmp_path):
        resnames = ["ALA", "GLY", "VAL", "LEU", "SER"]
        self._write_features(tmp_path, "REFA", range(1, 6), resnames)
        self._write_features(tmp_path, "BBBB", range(1, 6), resnames)
        main(ref_pdb="REFA", max_mismatches=5,
             pdb_ids=["REFA", "BBBB"], output_dir=tmp_path)
        out = pd.read_csv(tmp_path / "renumbered" / "REFA_features.csv")
        assert list(out["resi_struct"]) == list(range(1, 6))

    def test_identical_seqs_kept_and_renumbered(self, tmp_path):
        resnames = ["ALA", "GLY", "VAL", "LEU", "SER"]
        self._write_features(tmp_path, "REFA", range(1, 6), resnames)
        self._write_features(tmp_path, "BBBB", range(10, 15), resnames)
        main(ref_pdb="REFA", max_mismatches=0,
             pdb_ids=["REFA", "BBBB"], output_dir=tmp_path)
        out = pd.read_csv(tmp_path / "renumbered" / "BBBB_features.csv")
        # Renumbered to reference resi (1-5), not original (10-14)
        assert set(out["resi_struct"]) == set(range(1, 6))

    def test_too_many_mismatches_removes_structure(self, tmp_path):
        ref_resnames = ["ALA", "GLY", "VAL"]
        qry_resnames = ["TRP", "TRP", "TRP"]  # 3 mismatches
        self._write_features(tmp_path, "REFA", range(1, 4), ref_resnames)
        self._write_features(tmp_path, "BBBB", range(1, 4), qry_resnames)
        main(ref_pdb="REFA", max_mismatches=0,
             pdb_ids=["REFA", "BBBB"], output_dir=tmp_path)
        assert not (tmp_path / "renumbered" / "BBBB_features.csv").exists()

    def test_missing_features_csv_skipped(self, tmp_path, capsys):
        resnames = ["ALA", "GLY", "VAL"]
        self._write_features(tmp_path, "REFA", range(1, 4), resnames)
        # BBBB has no features CSV
        main(ref_pdb="REFA", max_mismatches=5,
             pdb_ids=["REFA", "BBBB"], output_dir=tmp_path)
        assert not (tmp_path / "renumbered" / "BBBB_features.csv").exists()

    def test_creates_renumbered_dir(self, tmp_path):
        resnames = ["ALA"]
        self._write_features(tmp_path, "REFA", range(1, 2), resnames)
        main(ref_pdb="REFA", max_mismatches=5,
             pdb_ids=["REFA"], output_dir=tmp_path)
        assert (tmp_path / "renumbered").is_dir()
