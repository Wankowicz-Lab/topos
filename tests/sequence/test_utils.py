"""Tests for sequence utility functions."""
import pytest
import warnings

from src.sequence.utils import convert_amino_acid, AA_3_TO_1, AA_1_TO_3


def test_convert_amino_acid_3_to_1():
    """Test conversion from 3-letter to 1-letter codes."""
    assert convert_amino_acid("ALA") == "A"
    assert convert_amino_acid("GLY") == "G"
    assert convert_amino_acid("TRP") == "W"
    assert convert_amino_acid("ala") == "A"  # case insensitive


def test_convert_amino_acid_1_to_3():
    """Test conversion from 1-letter to 3-letter codes."""
    assert convert_amino_acid("A") == "ALA"
    assert convert_amino_acid("G") == "GLY"
    assert convert_amino_acid("W") == "TRP"
    assert convert_amino_acid("a") == "ALA"  # case insensitive


def test_convert_amino_acid_unknown_code():
    """Test that unknown codes return the input and issue a warning."""
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        result = convert_amino_acid("Z")
        assert result == "Z"
        assert len(w) == 1
        assert "Unknown 1 letter code" in str(w[0].message)

    # test with force_convert
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        result = convert_amino_acid("Z", force_convert=True)
        assert result == "ZZZ"
        assert len(w) == 1
        assert "Unknown 1 letter code" in str(w[0].message)


def test_convert_amino_acid_unknown_3letter_code():
    """Test that unknown 3-letter codes return the input and issue a warning."""
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        result = convert_amino_acid("XYZ")
        assert result == "XYZ"
        assert len(w) == 1
        assert "Unknown 3 letter code" in str(w[0].message)

    # test with force_convert
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        result = convert_amino_acid("XYZ", force_convert=True)
        assert result == "X"
        assert len(w) == 1
        assert "Unknown 3 letter code" in str(w[0].message)


def test_convert_amino_acid_unexpected_length():
    """Test that unexpected lengths return the input and issue a warning."""
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        result = convert_amino_acid("ALAA")
        assert result == "ALAA"
        assert len(w) == 1
        assert "Unexpected amino acid code length" in str(w[0].message)

    # test with force_convert raises ValueError
    with pytest.raises(ValueError):
        convert_amino_acid("ALAA", force_convert=True)


def test_convert_amino_acid_whitespace():
    """Test that whitespace is handled correctly."""
    assert convert_amino_acid("  ALA  ") == "A"
    assert convert_amino_acid("  A  ") == "ALA"


def test_amino_acid_mappings_bidirectional():
    """Test that the AA mappings are complete and bidirectional."""
    # All 20 standard amino acids should be present
    assert len(AA_3_TO_1) == 23
    assert len(AA_1_TO_3) == 23
    
    # Bidirectionality
    for three, one in AA_3_TO_1.items():
        assert AA_1_TO_3[one] == three
