# MOADyna Contributor Workflow

This document explains how students and collaborators should work with the
MOADyna repository. It assumes no prior Git experience.

The short version:

- `main` is the stable branch.
- Students and collaborators do not push directly to `main`.
- Every change goes through a short-lived branch and a merge request.
- Yi Lu reviews and merges changes into `main`.
- If you are unsure, open an issue before writing code.

## Roles

### Maintainer

The maintainer is responsible for:

- merging changes into `main`;
- deciding release timing;
- resolving API and physics-convention questions;
- reviewing correctness-sensitive code;
- protecting the long-term design of the package.

At the moment, this role is Yi Lu.

### Contributors

Students and collaborators can:

- clone or pull the repository;
- run tests and examples;
- open issues;
- create short-lived branches;
- push those branches;
- open merge requests;
- respond to review comments.

Contributors should not:

- push directly to `main`;
- force-push over other people's work;
- rewrite shared history;
- merge their own merge requests;
- commit large generated files or private data unless explicitly discussed.

## Basic Concepts

### Repository

The repository is the project folder tracked by Git.

### Remote / origin

Your local clone is a working copy. The public copy on GitHub is called the
**remote**. Git refers to the default remote as **`origin`**. So
`git push -u origin my-branch` means "send `my-branch` from my local clone
up to the remote."

If you don't have push access to `CorrelatedSpectra/MOADyna.jl` (most
contributors won't), first **fork** the repository on GitHub and clone your
fork — then `origin` is your fork, and pull requests are opened against
`CorrelatedSpectra/MOADyna.jl`.

### Hosting

`github.com/CorrelatedSpectra/MOADyna.jl` is the canonical repository —
issues, pull requests, releases, and CI all live there. (A read-only
backup copy is kept on an institutional GitLab server; it is not part of
the contribution workflow.)

You can see the remote URL with:

```bash
git remote -v
```

### Branch

A branch is a separate line of work.

`main` is the stable branch. A feature or fix branch is a temporary branch used
for one specific task.

Examples:

```text
main
feat/dipole-builder
fix/kanamori-sign
docs/nio-xas-example
test/quanty-c4v-fixture
```

### Short-Lived Branch

A short-lived branch is a temporary branch for one focused change. It should
usually live for hours or a few days, not weeks.

Good branch names:

```text
feat/shellmodel-parser
feat/dipole-builder
fix/d2h-label-order
docs/nio-example
test/quanty-c4v-fixture
```

Poor branch names:

```text
my-work
big-refactor
new-version
try
yilu-dev
```

### Commit

A commit is a saved snapshot of your changes.

A good commit message uses the form `<area>: <description>`, where `<area>` is
the module or doc category being changed:

```text
shells: add shell-tag parser
multiplets: fix kanamori spin-flip sign
docs: add NiO XAS example
algebra: rename herm → add_hc
chore: ignore editor backup files
```

Common areas: `algebra`, `bases`, `ed`, `spectroscopy`, `pointgroups`,
`shells`, `multiplets`, `quantyio`, `docs`, `chore`, `test`, `refactor`.

### Merge Request

A merge request asks the maintainer to merge your branch into `main`.

Use a merge request even for small changes. It gives the maintainer a place to
review the change, ask questions, and check tests.

Some platforms call this a pull request. In this document, "merge request" and
"pull request" mean the same thing.

## One-Time Setup

First, clone the repository:

```bash
git clone https://github.com/CorrelatedSpectra/MOADyna.jl.git
cd MOADyna.jl
```

Check that Git knows where the remote repository is:

```bash
git remote -v
```

Tell Git your name and email if you have not done this before:

```bash
git config --global user.name "Your Name"
git config --global user.email "your.email@example.com"
```

Install Julia dependencies:

```bash
julia --project=. -e 'using Pkg; Pkg.instantiate()'
```

Run the test suite once:

```bash
julia --project=. -e 'using Pkg; Pkg.test()'
```

This currently takes about 90 seconds on a recent laptop. The first run
may take longer because Julia precompiles dependencies.

If this fails on a fresh clone, report it before starting new work.

## Daily Workflow

### 1. Start From Current `main`

Before starting work, update your local `main`:

```bash
git switch main
git pull --ff-only
```

The `--ff-only` flag tells Git to update only if the local `main` can be
fast-forwarded — refusing to make weird merge commits if upstream has
unexpectedly diverged. If this fails, something unusual happened (e.g.,
the remote was rewritten); ask the maintainer rather than guessing.

This makes sure your new branch starts from the latest shared code.

### 2. Create a Short-Lived Branch

Create a branch for one task:

```bash
git switch -c fix/kanamori-sign
```

Use a name that describes the task.

Common prefixes:

```text
feat/   new feature
fix/    bug fix
docs/   documentation change
test/   tests or validation fixtures
refactor/ internal cleanup without intended behavior change
```

Examples:

```bash
git switch -c feat/dipole-builder
git switch -c docs/nio-xas-example
git switch -c test/oh-akm-fixture
```

### 3. Make a Small Focused Change

Keep one branch focused on one issue.

Good:

- add `dipole`;
- add tests for `dipole`;
- document `dipole` conventions.

Too broad:

- add `dipole`, rewrite `coulomb`, change `ShellModel`, and reformat all docs.

If your branch grows too large, split it into multiple branches or ask for help.

### 4. Check What Changed

Before committing, inspect your changes:

```bash
git status
git diff
```

Only commit files related to your task.

If you see unrelated files, do not add them.

### 5. Run Relevant Tests

For small documentation-only changes, tests may not be necessary.

For code changes, run at least the relevant tests. If you are unsure, run the
full suite:

```bash
julia --project=. -e 'using Pkg; Pkg.test()'
```

For physics changes, also run the relevant validation script or comparison
against a known reference.

Examples of acceptable validation:

- exact analytic result;
- symmetry or degeneracy check;
- Quanty comparison;
- PyQuanty comparison;
- ED comparison;
- existing regression fixture.

### 6. Commit Your Work

Stage the files you intend to include:

```bash
git add src/multiplets/dipole.jl test/multiplets/test_dipole.jl
```

Commit:

```bash
git commit -m "multiplets: add electric dipole builder"
```

For documentation:

```bash
git add docs/src/man/multiplets.md
git commit -m "docs: clarify dipole normalization"
```

### 7. Push Your Branch

Push the branch to your fork (or to the shared repository, if you have
push access):

```bash
git push -u origin fix/kanamori-sign
```

Use your actual branch name.

### 8. Open a Merge Request

After `git push -u origin your-branch`, GitHub will print a URL like

```text
https://github.com/<owner>/MOADyna.jl/pull/new/your-branch
```

Open that URL, fill in the description, and click "Create pull request".

If you have the `gh` CLI installed, you can do this from the terminal:

```bash
gh pr create
```

This will prompt you for the title and description.

The merge request should include:

- what changed;
- why it changed;
- tests or validation you ran;
- any physics convention involved;
- any open question.

Example:

```text
Summary
- Adds electric-dipole builder for p -> d transitions.
- Uses Racah-normalized C^1_q with radial integral set to 1.
- Adds 2p -> 3d reference fixture.

Validation
- Pkg.test() passed locally.
- Compared 2p -> 3d matrix elements against Quanty fixture.

Open questions
- None.
```

## Issue Workflow

Open an issue when:

- something fails;
- documentation is confusing;
- a validation does not match;
- you want to propose a new feature;
- you are unsure what branch to create.

Good issue title examples:

```text
NiO XAS example fails on Julia 1.11
D4h Akm convention unclear for f shell
Need example for RIXS polarization channels
Kanamori pair-hop sign check
```

Good issue body:

```text
What I tried:
I ran test/shells/validation/test_nio_xas_native.jl.

What happened:
The L3 peak is shifted by about 0.2 eV.

What I expected:
Agreement with docs/dev/validation/.../XAS_lanczos_cont_frac.txt.

Environment:
Julia 1.11.5, macOS/Linux, commit abc123.

Files/scripts:
...
```

For bugs, include:

- exact command;
- full error message;
- commit hash if possible:

```bash
git rev-parse --short HEAD
```

## Recovery — What To Do When You Broke Something

These are the most common beginner panic moments. They are all fixable. Read
this section once now, before you need it.

### "I changed files I did not mean to change."

If the change is not committed yet:

```bash
git status                  # see what changed
git restore path/to/file    # undo changes to that file
git restore .               # undo ALL unstaged changes (use with care)
```

### "I committed to main by accident."

If you have not pushed yet:

```bash
git log --oneline -3              # find your accidental commit's hash
git reset --soft HEAD~1           # un-commit, keep the changes staged
git switch -c fix/proper-branch   # move to a real branch
git commit                        # commit on the branch instead
```

### "I want to throw away my branch and start over."

```bash
git switch main
git branch -D bad-branch          # delete the local bad branch
```

(Capital `-D` forces the delete even if the branch has unmerged work, which
is what you want when discarding.)

### "git switch refuses — `your local changes would be overwritten`"

You have uncommitted changes that conflict with the target branch. Either
commit them on the current branch, or temporarily stash:

```bash
git stash                # set aside your changes
git switch main
# ...do your update...
git switch your-branch
git stash pop            # bring your changes back
```

### "My branch is in a tangle and I do not know what state it's in."

Stop. Do **NOT** run `git reset --hard`, `git push --force`, or
`git clean -fdx`. Open an issue or message the maintainer with the output of:

```bash
git status
git log --oneline -10
```

It is almost always recoverable, but the fix depends on details. Your local
work is safer if you do not improvise.

## Code Review Expectations

The maintainer may ask for changes. This is normal.

Common review requests:

- add a test;
- simplify an API;
- clarify a convention;
- compare against Quanty or another reference;
- split a large merge request into smaller ones;
- remove unrelated changes.

When responding to review:

- make the requested change on the same branch;
- commit it;
- push again;
- leave a short comment explaining what changed.

Example:

```bash
git add test/multiplets/test_dipole.jl
git commit -m "test dipole parity rejection"
git push
```

The merge request updates automatically.

## Rules for Scientific Correctness

MOADyna is scientific software. A change is not complete just because the code
runs.

For physics-facing changes, separate these clearly:

- derived formula;
- assumed convention;
- phenomenological parameter;
- numerical validation.

Every physics-facing merge request should include at least one validation:

- analytic limit;
- symmetry check;
- degeneracy check;
- Quanty comparison;
- PyQuanty comparison;
- ED comparison;
- published benchmark;
- existing fixture.

Examples:

```text
Good:
The dipole operator uses Racah-normalized C^1_q. The 2p -> 3d matrix
elements match the Quanty fixture to 1e-12.

Not enough:
The spectrum looks reasonable.
```

## What Not To Do

Do not commit directly to `main`.

Do not run:

```bash
git push --force origin main
git reset --hard origin/main
git rm -rf .
git clean -fdx
```

unless the maintainer explicitly tells you to. These commands can destroy
work irrecoverably.

Do not mix unrelated work in one branch.

Do not commit:

- large generated files;
- private unpublished data;
- temporary scratch outputs;
- local machine paths;
- credentials or tokens;
- `.DS_Store`;
- editor backup files.

The repository's `.gitignore` already excludes editor backup files, OS-level
files (`.DS_Store`, `*.swp`), and Julia build artifacts (`Manifest.toml`,
`docs/build/`). You generally do not need to do anything — just don't override
it. If `git status` ever shows a file you did not expect to add, do NOT
`git add` it; ask first.

Do not silently change conventions. If a convention changes, document it and
add a validation test.

## Keeping Your Branch Up To Date

If your branch is open for more than a few days, `main` may change.

To update your branch:

```bash
git switch main
git pull --ff-only
git switch your-branch-name
git merge main
```

If Git reports conflicts, stop and ask for help if you are unsure.

After resolving conflicts:

```bash
git add <resolved-files>
git commit
git push
```

## Releases

Students should use tagged releases for published calculations whenever
possible.

Use `main` for active testing and development. Use release tags for stable
reproducible calculations.

In Julia, a tagged version can be installed with:

```julia
pkg> add https://github.com/CorrelatedSpectra/MOADyna.jl#v0.3.0
```

## Quick Command Reference

Update local `main`:

```bash
git switch main
git pull --ff-only
```

Create a branch:

```bash
git switch -c feat/my-change
```

See changed files:

```bash
git status
```

See detailed changes:

```bash
git diff
```

Stage files:

```bash
git add path/to/file.jl
```

Commit:

```bash
git commit -m "short description"
```

Push:

```bash
git push -u origin feat/my-change
```

Run tests:

```bash
julia --project=. -e 'using Pkg; Pkg.test()'
```

Find current commit:

```bash
git rev-parse --short HEAD
```

See recent history:

```bash
git log --oneline -10
```

List local branches:

```bash
git branch
```

Set aside uncommitted changes temporarily:

```bash
git stash
git stash pop      # bring them back later
```

## When To Ask For Help

Ask before continuing if:

- Git reports a conflict and you do not know how to resolve it;
- your branch has grown large;
- tests fail and you do not understand why;
- you need to change a public API;
- you need to change a physics convention;
- you are about to commit generated data;
- you are unsure whether something belongs in MOADyna.

Opening an issue early is better than fixing a large incorrect branch later.
