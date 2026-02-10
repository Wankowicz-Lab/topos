# Coding and Testing Guidelines

These guidelines apply to all code in this repository.

## Testing

- Verify output creation with a single assertion; avoid checking every field/attribute/column
- Focus on high-value, broad functionality rather than exhaustive edge cases
- Test correctness with known expected results, not just successful completion
- Use toy data with known outcomes to validate correctness
- Where possible, use small inputs where you can compute the expected output by hand or from a reference

**Anti-patterns:**
- Don't systematically check every column: `assert 'col1' in df.columns; assert 'col2' in df.columns; ...` — Prefer: a single assertion that the output has the expected shape or a key value.
- Don't test unlikely edge cases: `test_empty_input()`, `test_none_input()`, `test_malformed_data()`
- Don't only assert completion: `result = func(); assert result is not None` — Prefer: assert against a known expected value or shape.

## Code Style

- Avoid defensive coding when data is expected to be present
- Don't catch exceptions when there's no reason to expect them
- Don't use `.get(key, None)` or return `None` when data should exist
- Non-user-facing functions should expect correctly formatted inputs
- Let errors surface naturally rather than masking them

**Anti-patterns:**
- Don't use: `value = data.get('key', None)` when `data['key']` is expected — Prefer: `value = data['key']` and let KeyError surface.
- Don't write: `try: result = func(); except: return None` without reason
- Don't return: `return None` when an exception would be more informative

## Documentation

- Add explanatory comments in function bodies to explain logic
- In dense or non-obvious logic, consider a comment every ~10 lines; avoid long stretches without explanation
- Comments should explain "why" and "how", not restate what the code does

## When to break the rules

- User-facing or I/O boundaries (e.g. API handlers, file parsing) may use defensive checks and clear error messages; internal helpers can assume valid inputs.
