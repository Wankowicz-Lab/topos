---
name: run-tests
description: Runs Ruff (`ruff check src tests`) per CI, then pytest using a fast, incremental workflow (scoped tests first with fail-fast, then full suite). Covers install per project docs, CI parity, and network for remote-dependent tests. Use when running tests, fixing failures, or preparing a PR; use when the user mentions pytest, ruff, test suite, CI tests, or local verification.
compatibility: Outbound network is often required for integration tests (HTTP, registries, external services). Use terminal network/full_network permissions in sandboxed agent runs when tests fail with connection or DNS errors. Prefer a **project-local `.venv`** for `pip`/`pytest` so installs do not write outside the workspace; if `pip install` hits permission errors on user site-packages, create `.venv` in the repo or use full permissions once.
---

# Run tests (pytest)

## Prerequisites

- **Python version and install**: follow the project **README** (or `pyproject.toml` / `requirements.txt`). Typical editable install with test extras:

```bash
pip install -e ".[test]"
```

- **Optional system or native dependencies** (compilers, CLIs, databases): only what the README or CI workflow documents—install when tests or docs indicate they are required.

- **Virtual environment (agent/sandbox)**: If `.venv/` exists at the repo root, **activate it** before `pip`/`pytest` (e.g. `. .venv/bin/activate` on Unix). That avoids permission failures when the sandbox blocks writes to user-level `site-packages`. If there is no venv and `pip install -e ".[test]"` fails with `Operation not permitted` outside the workspace, create one in-repo (`python3 -m venv .venv`) and reinstall, or rerun install with full permissions once.

## Philosophy

- **Before opening a PR**, run tests locally—do not rely only on CI.
- **Tight loop first**: scope to tests tied to **files you changed** in the working tree, fix failures, repeat.
- **Fail fast**: if a test fails, fix it before running more tests in that phase (`pytest -x` or `--maxfail=1`). No need to burn time on the rest until the current failure is resolved.
- **Full suite last**: once scoped tests pass, run the **entire** test tree (same expectation as CI).
- **Testing style**: if the repo has [AGENTS.md](../../../AGENTS.md), follow it; otherwise prefer high-value assertions and small deterministic inputs.

## Ruff (lint, CI parity)

CI runs **Ruff** before pytest on Python 3.11 (see [`.github/workflows/test.yml`](../../../.github/workflows/test.yml) “Run Ruff”). After `pip install -e ".[test]"` (which includes `ruff`), run the same check locally:

```bash
ruff check src tests
```

Fix any reported issues before merging. Running Ruff early (e.g. right after install, or after small edits) catches style issues before you spend time on pytest. CI only executes this step on one matrix version; locally you can run it with any Python that has `ruff` installed.

## Network / sandbox (default)

Many projects have tests that call **remote services** (HTTP APIs, package indexes, cloud resources). **Assume pytest may need outbound network** unless you know the suite is offline-safe.

When using the agent terminal in a **sandboxed** environment, **request network permission by default** for pytest in this repo (e.g. `required_permissions: ["network"]`, or `full_network` if needed). Without it, you may see `ConnectionError`, timeouts, or resolution failures. If only a subset needs network, narrow scope after reading tracebacks or project docs.

## Step 1 — Changed `src/` files

List Python files under `src/` that differ from `HEAD` (working tree + index):

```bash
git diff --name-only HEAD | grep '^src/' | grep '\.py$'
```

If this is **empty** (no local `src/` changes, or clean tree), **skip to Step 4** and run the full suite—unless you need to validate a branch against mainline before a PR:

```bash
git fetch origin main 2>/dev/null || true
git diff --name-only origin/main...HEAD | grep '^src/' | grep '\.py$'
```

If the project does not use a `src/` layout, adapt: diff the project’s main package directory (e.g. `lib/`, `mypackage/`) and map to tests the same way the repo mirrors code vs `tests/`.

## Step 2 — Map to test files

Common mirror layout:

- `src/<pkg>/<module>.py` → `tests/<pkg>/test_<module>.py`

Example: `src/foo/bar.py` → `tests/foo/test_bar.py`.

If a test file does not exist for a changed source file, skip that path and continue with the rest.

## Step 3 — Scoped pytest (fail fast)

Run **only** the mapped test files in **one** command; use `-x` to stop on the first failure:

```bash
pytest <path/to/test_a.py> <path/to/test_b.py> -v -x
```

**If anything fails:** read the traceback, fix code or tests, then **re-run the same pytest command** until exit code is 0. Do not move to the full suite until scoped tests pass.

For a single test: `pytest tests/path/test_file.py::test_function_name -v`. For a name filter: `pytest -k "pattern" -v` (use sparingly).

## Step 4 — Full suite (after scoped tests pass)

Run **Ruff** (`ruff check src tests`) if you have not already, so local results match CI lint + test order.

Align with CI if the repo has a workflow (e.g. [`.github/workflows/test.yml`](../../../.github/workflows/test.yml))—match its pytest arguments when practical:

```bash
pytest tests/ -v --cov --cov-report=term --cov-report=xml
```

Faster without coverage:

```bash
pytest tests/ -v
```

Always run the full suite with **network enabled** in the sandbox when the project’s tests are known to use the network.

If the full suite fails in areas you did not touch, still investigate—regressions should be fixed or understood before merge.

## If stuck

- Re-read the failure traceback; confirm install per README and network permissions.
- Re-check Steps 1–4 above: scoped tests must pass before the full suite.
