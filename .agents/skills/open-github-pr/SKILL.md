---
name: open-github-pr
description: Prepares and opens a GitHub pull request for this repository. Use when opening a PR, merging via GitHub, pushing a feature branch, or when the user says pull request, gh pr, or ready to merge.
compatibility: Requires git; `git push` and GitHub API need network. Install GitHub CLI (`gh`) for `gh pr create`, or open the compare URL in a browser.
---

# Open a GitHub pull request

## Before a PR

1. Follow **[run-tests](../run-tests/SKILL.md)**: scoped tests, then full `pytest` over the project’s test tree with network when needed, until green. Do not open a PR while tests fail locally unless the user explicitly overrides.

2. **Commit** all intended changes (`git status` clean for the work you are including).

## Worktree vs branch

The same repo may be edited in a **linked worktree** (`git worktree list` shows this path) or in the primary clone. Behavior differs only when you need a **branch name** for the PR.

| Situation | Action |
|-----------|--------|
| **Detached HEAD** (`git branch --show-current` empty) | Create a branch before push: `git checkout -b <short-descriptive-name>` |
| **On `main` / default branch with local-only commits** | Create a feature branch: `git checkout -b <short-descriptive-name>` — do not PR directly from default unless the user asks |
| **Already on a feature branch** | No extra branch step; ensure it is pushed |

Use a branch name that reflects the change. If unsure, derive it from the main commit theme.

## Push

From the repository root (worktree or main clone):

```bash
git push -u origin HEAD
```

If the branch already tracks a remote, `git push` may suffice.

## PR title and body (no prompts)

Draft the **title** and **body yourself** from `git log`, `git diff origin/main...HEAD` (or `main...HEAD` if no `origin/main`), and changed file paths. **Do not ask the user** to supply the description unless they volunteer edits.

**Title:** One line, imperative mood, ≤ ~72 characters when possible. Summarizes the whole change set (not only the last commit).

**Body:** Use short, complete sentences (see repository [AGENTS.md](../../../AGENTS.md) if present). Structure:

1. **Goal** — What problem or outcome this PR addresses (why merge it).
2. **Implementation** — How the changes achieve that (modules, key mechanisms, tests added/updated).
3. **Other areas** — Call out incidental edits (docs, skills, CI, refactors, formatting) or explicitly state *None* if the diff is tightly scoped.

If the PR is large, prefer accurate summary over listing every file.

## Open the PR

**Preferred (GitHub CLI):**

```bash
gh pr create --title "..." --body "..."
```

Use `--draft` if the user wants a draft. If `gh` is missing, say so and give the remote compare URL pattern: `https://github.com/<org>/<repo>/compare/<branch>?expand=1` (derive org/repo from `git remote -v`).

## Checklist

- [ ] Tests passed per **run-tests** workflow
- [ ] On a **non-default** branch (unless explicitly PR’ing to default)
- [ ] Pushed to `origin`
- [ ] Title + body drafted from the diff; user not blocked on writing copy
