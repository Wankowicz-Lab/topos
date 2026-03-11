# Coding and Testing Guidelines

These guidelines apply to all code in this repository.

## Code Style

- Assume valid inputs in internal/downstream pipeline code; avoid explicit input verification there.
- Validate inputs robustly at user-facing or I/O boundaries and return clear errors.
- Let failures surface naturally in internal code; avoid broad exception handling that hides issues.
- Prefer direct access over defensive defaults when required data is expected (e.g., `data['key']` over `data.get(...)`).

**Anti-patterns:**
- `value = data.get('key', None)` when `data['key']` is required.
- `try: result = func(); except: return None` in internal pipeline code.
- Guard clauses that validate required inputs in downstream/internal transforms.

## Testing

- Prefer high-value tests of core behavior over exhaustive edge-case enumeration.
- Use small, deterministic toy inputs where expected outputs can be computed by hand/reference.
- Assert correctness (key value/shape/invariant), not just successful execution.
- Keep assertions focused; avoid long checklists for every output field/column.

**Anti-patterns:**
- Column-by-column assertion spam when one correctness assertion would cover behavior.
- Tests that only assert completion (`assert result is not None`).
- Low-value edge cases that do not reflect real usage.

## Documentation

- Add concise comments for key logic and assumptions, especially around transformations, lookup/indexing strategy, and merge behavior.
- Comment dense/non-obvious sections frequently enough that intent is clear without reverse-engineering.
- Explain "why/how", not line-by-line "what".

## When to break the rules

- User-facing or I/O boundaries (e.g., API handlers, CLI entrypoints, file parsing) should use defensive checks and clear error messages; internal pipeline helpers should assume valid upstream inputs.
