"""Expected columns and validation for AAindex CSV files."""

from __future__ import annotations

import pandas as pd

# Residue columns in fixed order (matches bundled data/aaindex_parsed_small.csv).
AAINDEX_AA_COLUMNS = (
    'ALA', 'ARG', 'ASN', 'ASP', 'CYS', 'GLN', 'GLU', 'GLY', 'HIS', 'ILE',
    'LEU', 'LYS', 'MET', 'PHE', 'PRO', 'SER', 'THR', 'TRP', 'TYR', 'VAL',
)

AAINDEX_REQUIRED_COLUMNS = ('accession', 'description', 'category') + AAINDEX_AA_COLUMNS


def validate_aaindex_columns(df: pd.DataFrame) -> None:
    """
    Ensure ``df`` has exactly the expected AAindex columns in order.

    Raises
    ------
    ValueError
        If column names or order do not match the pipeline contract.
    """
    expected = list(AAINDEX_REQUIRED_COLUMNS)
    actual = list(df.columns)
    if actual != expected:
        raise ValueError(
            f"AAindex CSV columns must match {expected!r} in order; got {actual!r}"
        )
