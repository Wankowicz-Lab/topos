---
name: create-worktree
description: Creates a git worktree for isolated agent work in this repository. Use when the user wants a worktree, isolated branch, or asks an agent to do implementation work in a separate checkout.
compatibility: Linked worktrees may require full terminal permissions because git stores refs in the main repository metadata. On this machine, non-interactive shells often expose `python3` but not `python`, and `gh` may need `/opt/homebrew/bin` on PATH.
---

# Create a linked worktree

## Goal

Use a separate checkout and branch for isolated agent work. Prefer a sibling directory named after the change, such as `../topos-my-feature`.

## 1. Create the worktree safely

- If the target directory is outside the current workspace, sandboxed terminals may block directory creation or git ref updates. If `git worktree add` fails with `Operation not permitted`, `Permission denied`, or ref-locking errors, rerun the same command with full permissions.
- When creating a new branch, pass an explicit base such as `origin/main`. Do not rely on the current checkout `HEAD`, which may be on a feature branch or detached.
- Create a new branch with the worktree when starting fresh:

```bash
git worktree add -b <branch-name> ../topos-<slug> origin/main
```

- If the branch already exists, attach it explicitly:

```bash
git worktree add ../topos-<slug> <branch-name>
```

- If the intended base is not `origin/main`, still pass it explicitly:

```bash
git worktree add -b <branch-name> ../topos-<slug> <base-commit-ish>
```

## 2. Verify the checkout immediately

Before editing, staging, or committing, confirm the terminal is operating in the intended worktree:

```bash
pwd
git rev-parse --show-toplevel
git branch --show-current
git status --short
```

Expected state:

- `git rev-parse --show-toplevel` points at the new worktree path.
- `git branch --show-current` is the feature branch, not `main`.
- `git status` only shows files relevant to this task.

If you see the primary checkout, `main`, or unrelated untracked files, stop immediately. Do not commit. Re-run commands with the worktree as the explicit working directory and, if needed, with full permissions.

## 3. Bootstrap a local Python environment

Do not assume the worktree has a usable Python environment or inherited tools on `PATH`.

- Prefer a worktree-local `.venv`.
- Use `python3`, not `python`.
- After creating the venv, call tools from `.venv/bin/...` instead of relying on shell activation.

```bash
python3 -m venv .venv
.venv/bin/pip install -e ".[test]"
```

Common commands:

```bash
.venv/bin/python -m pytest tests/path/test_file.py -v
.venv/bin/ruff check src tests
```

If imports fail in a fresh worktree, install dependencies into the local `.venv` before treating the failure as a code regression.

## 4. Stage and commit from the worktree root

- Run git commands from the worktree root, not the primary checkout.
- If `git add <path>` reports `pathspec ... did not match any files`, first re-check `git rev-parse --show-toplevel` and make sure the path is relative to that root.
- If the path is correct but explicit path staging is still flaky, use `git add -A` from the worktree root when you intend to stage the full worktree diff.
- If a commit appears to land on `main` or includes unrelated files, stop immediately and re-verify the checkout before continuing.

## 5. Validate inside the worktree

Follow [run-tests](../run-tests/SKILL.md) from the worktree-local environment.

- Run Ruff from the worktree venv: `.venv/bin/ruff check src tests`
- Run focused tests first, then broader regression coverage.
- If Ruff is missing, install test extras instead of skipping lint.
- If an untouched test fails because a local dependency is missing, report it as an environment blocker unless the current task is to fix environment setup.

## 6. Push and open a PR

- Push the branch before creating a PR if it has no upstream yet:

```bash
git push -u origin HEAD
```

- If `gh` is not on `PATH`, try:

```bash
PATH="/opt/homebrew/bin:/usr/local/bin:$PATH" gh pr create ...
```

- If needed, call the binary directly:

```bash
/opt/homebrew/bin/gh pr create ...
```

Then follow [open-github-pr](../open-github-pr/SKILL.md).

## Checklist

- [ ] Worktree created on a feature branch
- [ ] Verified `pwd`, repo root, current branch, and status before editing
- [ ] Created `.venv` in the worktree and used `python3`
- [ ] Ran Ruff and pytest from `.venv/bin`
- [ ] Staged and committed from the worktree root
- [ ] Pushed with `-u` before `gh pr create` when needed
