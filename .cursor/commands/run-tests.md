# Run Tests

## Overview
Run the test suite for the biogenesis project using pytest with an incremental workflow: first test only changed functions, iterate on failures until they pass, then run the full suite.

## Required Workflow

You MUST follow these steps in order:

### Step 1: Identify Changed Source Files

Run this command to get all changed Python source files:

```bash
git diff --name-only HEAD | grep "^src/" | grep "\.py$"
```

If the output is empty (no changed files), skip to Step 5 and run the full test suite.

### Step 2: Map Changed Files to Test Files

For each changed source file from Step 1, identify the corresponding test file using this mapping pattern:

- `src/sequence/utils.py` → `tests/sequence/test_utils.py`
- `src/metrics/structure.py` → `tests/metrics/test_structure.py`
- `src/pipeline/runner.py` → `tests/pipeline/test_runner.py`
- `src/structure/utils.py` → `tests/structure/test_utils.py`
- `src/databases/pdbtm.py` → `tests/databases/test_pdbtm.py`

**Mapping rule**: `src/{module}/{file}.py` → `tests/{module}/test_{file}.py`

Check if each test file exists. If a test file doesn't exist for a changed source file, note it but continue with the test files that do exist.

### Step 3: Run Tests for Changed Files Only

Run pytest on ONLY the test files identified in Step 2. Collect all test file paths and run them in a single command:

```bash
pytest tests/sequence/test_utils.py tests/metrics/test_structure.py -v
```

(Replace with the actual test files identified in Step 2)

### Step 4: Iterate on Failures (REPEAT UNTIL PASSING)

If any tests fail in Step 3:

1. **Analyze the failure**: Carefully read error messages, stack traces, and test output
2. **Fix the code**: Make necessary changes to source code or tests to address the failures
3. **Re-run the same tests**: Use the exact same pytest command from Step 3
4. **Check exit code**: If pytest exit code is non-zero, repeat steps 1-3

Continue this iteration loop until all tests for changed files pass (pytest exit code = 0).

During iteration, you may use the `-x` flag to stop on first failure:
```bash
pytest tests/sequence/test_utils.py -v -x
```

### Step 5: Run Full Test Suite

**ONLY AFTER** all tests from Step 3 are passing, run the complete test suite.

The full suite includes tests that fetch structures from RCSB (files.rcsb.org) and PDBTM (pdbtm.unitmp.org). **Request network permissions** when running the full suite so these tests do not fail with `ConnectionError` or `NameResolutionError`:

```bash
pytest tests/ -v
```

Run this command with **network access enabled** (e.g. `required_permissions: ["network"]` or `full_network` in Cursor) so that tests in `tests/pipeline/test_runner.py`, `tests/pipeline/test_examples.py`, `tests/databases/test_pdbtm.py`, and `tests/structure/test_structure_context.py` that use `pdb_id` to download from RCSB/PDBTM can succeed.

If the full suite reveals failures in unrelated tests, investigate and fix them as needed.

## Prerequisites
Ensure you have installed the test dependencies:

```bash
pip install -e ".[test]"
```

## Test Structure
- Tests are located in the `tests/` directory
- Tests mirror the `src/` directory structure
- Each test file follows the naming convention `test_*.py`
- Individual test functions can be run by specifying the path: `tests/path/to/test_file.py::test_function_name`

## Notes
- Use `-v` flag for verbose output showing individual test results
- Use `-x` flag to stop on first failure during iteration
- Use `-k` flag to run tests matching a pattern: `pytest -k "test_convert" -v`
- If no files have changed (clean git state), skip to running the full test suite
