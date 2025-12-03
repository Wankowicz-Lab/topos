# Contributing to biogenesis

Thank you for your interest in contributing to biogenesis! This document provides guidelines and conventions for contributing to this project.

## Development Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/Wankowicz-Lab/biogenesis.git
   cd biogenesis
   ```

2. Install the package in development mode with test dependencies:
   ```bash
   pip install -e ".[test]"
   ```

3. Run tests to verify your setup:
   ```bash
   pytest tests/
   ```

## Coding Conventions

### Docstrings

We use **NumPy-style docstrings** for all public functions, classes, and methods. Here's an example:

```python
def calculate_metric(context: Context, threshold: float = 1.0) -> pd.DataFrame:
    """
    Calculate a per-residue metric from protein structure.

    This function computes the metric value for each residue in the
    structure based on the provided threshold.

    Parameters
    ----------
    context : Context
        Context object containing residue metadata and structural information.
    threshold : float, optional
        Cutoff value for the calculation. Default is 1.0.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns 'chain', 'resi', 'resn', and 'metric_value'
        for each residue.

    Raises
    ------
    ValueError
        If the context does not contain required structural data.

    Examples
    --------
    >>> from src.structure.structure_context import Context
    >>> context = Context(array=my_structure)
    >>> result = calculate_metric(context, threshold=2.0)
    >>> print(result.head())
    """
    pass
```

### Path Handling

- Use `pathlib.Path` instead of string paths for file operations
- Accept both `str` and `Path` types in function signatures using `Union[str, Path]`
- Use context managers (`with` statements) when opening files

```python
from pathlib import Path
from typing import Union

def load_data(path: Union[str, Path]) -> pd.DataFrame:
    """Load data from a CSV file."""
    path = Path(path)
    with path.open('r') as f:
        return pd.read_csv(f)
```

### Mutable Default Arguments

Never use mutable objects (lists, dicts, sets) as default argument values. Use `None` and initialize inside the function:

```python
# Good
def process_data(items: list = None):
    if items is None:
        items = []
    # ...

# Bad - don't do this!
def process_data(items: list = []):
    # ...
```

### Type Hints

Use type hints for function parameters and return values:

```python
from typing import List, Optional, Dict

def analyze_residues(
    residue_ids: List[int],
    chain: str,
    options: Optional[Dict[str, Any]] = None
) -> pd.DataFrame:
    ...
```

## Testing

### Running Tests

```bash
# Run all tests
pytest tests/

# Run tests with coverage
pytest tests/ --cov=src --cov-report=html

# Run a specific test file
pytest tests/sequence/test_metrics.py

# Run tests matching a pattern
pytest tests/ -k "test_calculate"
```

### Writing Tests

- Place tests in the `tests/` directory, mirroring the `src/` structure
- Use pytest fixtures for common setup
- **Seed random number generators** for deterministic tests
- Test edge cases (empty inputs, boundary values, etc.)

Example test structure:

```python
import numpy as np
import pandas as pd
import pytest

# Seed RNG at module level for deterministic tests
np.random.seed(42)

def test_calculate_metric_basic():
    """Test basic metric calculation."""
    # Arrange
    residue_table = create_test_residue_table()
    context = MockContext(residue_table)
    
    # Act
    result = calculate_metric(context)
    
    # Assert
    assert 'metric_value' in result.columns
    assert len(result) == len(residue_table)


def test_calculate_metric_empty_input():
    """Test metric calculation with empty input."""
    context = MockContext(pd.DataFrame())
    result = calculate_metric(context)
    assert len(result) == 0


def test_calculate_metric_invalid_input():
    """Test that invalid input raises appropriate error."""
    with pytest.raises(ValueError, match="expected pattern"):
        calculate_metric(None)
```

### Random Number Generation

For reproducibility, always seed RNGs in tests:

```python
import numpy as np
import random

# At module level
np.random.seed(42)
random.seed(42)

# Or in fixtures
@pytest.fixture
def seeded_rng():
    np.random.seed(42)
    random.seed(42)
    yield
```

## Adding New Features

### Adding a New Metric

1. Create the metric function in the appropriate module (`src/sequence/metrics.py` or `src/structure/metrics.py`)
2. Use the `@register_metric` decorator to register it
3. Follow the NumPy docstring convention
4. Add tests in the corresponding test file

```python
from src.structure.structure_context import Context, register_metric

@register_metric(
    name='my_new_metric',
    provides=['metric_column'],
    tags={'structure'},
    requires=set()
)
def calculate_my_new_metric(context: Context) -> pd.DataFrame:
    """
    Calculate my new metric per residue.

    Parameters
    ----------
    context : Context
        Context object containing structural information.

    Returns
    -------
    pd.DataFrame
        DataFrame with 'metric_column' for each residue.
    """
    # Implementation
    pass
```

## Code Review

Before submitting a pull request:

1. Run the test suite and ensure all tests pass
2. Add tests for any new functionality
3. Update docstrings if you change function signatures
4. Run linting: `ruff check src/ tests/`
5. Ensure your changes don't break existing functionality

## Questions?

If you have questions about contributing, please open an issue on GitHub.
