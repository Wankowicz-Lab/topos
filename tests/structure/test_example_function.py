"""Unit tests for example_function.py"""

import pytest
from src.structure.example_function import process_integers


def test_process_integers_basic():
    """Test that odd integers are removed and average is appended."""
    result = process_integers([1, 2, 3, 4, 5, 6])
    expected = [2, 4, 6, 4.0]  # evens: [2, 4, 6], average: 4.0
    assert result == expected


def test_process_integers_all_even():
    """Test with all even integers."""
    result = process_integers([2, 4, 6, 8])
    expected = [2, 4, 6, 8, 5.0]  # evens: [2, 4, 6, 8], average: 5.0
    assert result == expected


def test_process_integers_all_odd():
    """Test with all odd integers returns empty list."""
    result = process_integers([1, 3, 5, 7])
    expected = []
    assert result == expected


def test_process_integers_empty_list():
    """Test with empty list returns empty list."""
    result = process_integers([])
    expected = []
    assert result == expected


def test_process_integers_single_even():
    """Test with single even number."""
    result = process_integers([4])
    expected = [4, 4.0]
    assert result == expected


def test_process_integers_single_odd():
    """Test with single odd number."""
    result = process_integers([3])
    expected = []
    assert result == expected
