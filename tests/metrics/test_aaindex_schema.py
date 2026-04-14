"""Tests for AAindex CSV schema validation."""
import pandas as pd
import pytest

from src.metrics.aaindex_schema import (
    AAINDEX_REQUIRED_COLUMNS,
    validate_aaindex_columns,
)


def test_validate_aaindex_columns_accepts_expected_layout():
    df = pd.DataFrame(columns=list(AAINDEX_REQUIRED_COLUMNS))
    validate_aaindex_columns(df)


def test_validate_aaindex_columns_rejects_wrong_order():
    cols = list(AAINDEX_REQUIRED_COLUMNS)
    cols[0], cols[1] = cols[1], cols[0]
    df = pd.DataFrame(columns=cols)
    with pytest.raises(ValueError, match="AAindex CSV columns must match"):
        validate_aaindex_columns(df)


def test_validate_aaindex_columns_rejects_missing_column():
    df = pd.DataFrame(columns=[c for c in AAINDEX_REQUIRED_COLUMNS if c != "VAL"])
    with pytest.raises(ValueError, match="AAindex CSV columns must match"):
        validate_aaindex_columns(df)
