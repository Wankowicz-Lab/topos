"""Tests for sequence utility functions."""
import warnings

import pytest

from topos.sequence.utils import (
    AA_1_TO_3,
    AA_3_TO_1,
    convert_amino_acid_1to3,
    convert_amino_acid_3to1,
)


def test_convert_amino_acid_3_to_1():
    """Test conversion from 3-letter to 1-letter codes."""
    assert convert_amino_acid_3to1("ALA") == "A"
    assert convert_amino_acid_3to1("GLY") == "G"
    assert convert_amino_acid_3to1("TRP") == "W"
    assert convert_amino_acid_3to1("ala") == "A"  # case insensitive


def test_convert_amino_acid_1_to_3():
    """Test conversion from 1-letter to 3-letter codes."""
    assert convert_amino_acid_1to3("A") == "ALA"
    assert convert_amino_acid_1to3("G") == "GLY"
    assert convert_amino_acid_1to3("W") == "TRP"
    assert convert_amino_acid_1to3("a") == "ALA"  # case insensitive


def test_convert_amino_acid_unknown_1letter_code():
    """Test that unknown codes return the input and issue a warning."""
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        result = convert_amino_acid_1to3("Z")
        assert result == "Z"
        assert len(w) == 1
        assert "Unknown 1 letter code" in str(w[0].message)

    # test with force_convert
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        result = convert_amino_acid_1to3("Z", force_convert=True)
        assert result == "ZZZ"
        assert len(w) == 1
        assert "Unknown 1 letter code" in str(w[0].message)


def test_convert_amino_acid_unknown_3letter_code():
    """Test that unknown 3-letter codes return the input and issue a warning."""
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        result = convert_amino_acid_3to1("XYZ")
        assert result == "XYZ"
        assert len(w) == 1
        assert "Unknown 3 letter code" in str(w[0].message)

    # test with force_convert
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        result = convert_amino_acid_3to1("XYZ", force_convert=True)
        assert result == "X"
        assert len(w) == 1
        assert "Unknown 3 letter code" in str(w[0].message)


def test_convert_amino_acid_unexpected_length():
    """Test that unexpected lengths return the input and issue a warning."""
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        result = convert_amino_acid_3to1("ALAA")
        assert result == "ALAA"
        assert len(w) == 1
        assert "Unexpected amino acid code length" in str(w[0].message)

    # test with force_convert raises ValueError
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        assert convert_amino_acid_3to1("ALAA", force_convert=True) == "X"
        assert len(w) == 1
        assert "Unexpected amino acid code length" in str(w[0].message)

    with pytest.raises(ValueError):
        convert_amino_acid_1to3("ALAA", force_convert=True)


def test_convert_amino_acid_whitespace():
    """Test that whitespace is handled correctly."""
    assert convert_amino_acid_3to1("  ALA  ") == "A"
    assert convert_amino_acid_1to3("  A  ") == "ALA"


def test_convert_amino_acid_special_mutation_tokens():
    """Special mutant tokens should collapse to X in 3-to-1 conversion."""
    for token in ["DEL", "DEL1", "DEL2", "DEL3", "INS1", "INS2", "INS3"]:
        assert convert_amino_acid_3to1(token) == "X"


def test_amino_acid_mappings_bidirectional():
    """Test that canonical mappings stay consistent across both tables."""
    assert len(AA_3_TO_1) == 28
    assert len(AA_1_TO_3) == 21

    for one, three in AA_1_TO_3.items():
        assert AA_3_TO_1[three] == one
