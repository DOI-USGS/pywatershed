# Developing `pywatershed`

This document describes how to set up a development environment, install
dependencies, generate example data, and run the tests
and example notebooks.

<!-- START doctoc generated TOC please keep comment here to allow auto update -->
<!-- DON'T EDIT THIS SECTION, INSTEAD RE-RUN doctoc TO UPDATE -->

- [Requirements](#requirements)
  - [Git](#git)
  - [Compilers](#compilers)
  - [Python](#python)
    - [Creating a virtual environment](#creating-a-virtual-environment)
      - [Mamba](#mamba)
      - [Pip](#pip)
    - [Installing `pywatershed` in development mode](#installing-pywatershed-in-development-mode)
- [Branching model](#branching-model)
- [Maintenance ledger](#maintenance-ledger)
- [CI](#ci)
- [Testing](#testing)
  - [Step 1: get the answers](#step-1-get-the-answers)
  - [Step 2: run the tests](#step-2-run-the-tests)
  - [Running pieces by hand](#running-pieces-by-hand)
- [Linting](#linting)
- [Committing Jupyter Notebooks](#committing-jupyter-notebooks)
- [pre-commit hooks](#pre-commit-hooks)
- [Documentation](#documentation)
- [Miscellaneous](#miscellaneous)
  - [Locating the root](#locating-the-root)

<!-- END doctoc generated TOC please keep comment here to allow auto update -->

## Requirements

### Git

[Git](https://git-scm.com/) is required to contribute code back to the
repository. GitHub's
[Guide to Installing Git](https://help.github.com/articles/set-up-git)
is a good source of information.

### Compilers

Compilers are optional. `pywatershed` itself is pure Python.

- The test answers are PRMS output. They are produced in a separate
  repository, `pws-prms-oracle`, and published as release assets.
  Producing them needs a Fortran compiler; downloading a released set
  does not. See [Testing](#testing).

### Python

Python >= 3.9 is required.

#### Creating a virtual environment

We suggest Mamba over a Python virtual environment (venv), but a venv is better
than nothing and can be created with Python's builtin `venv`.

##### Mamba

We highly recommend using Mamba. To create a mamba/conda environment with all
core, testing and optional dependencies, run from the project root:

```
mamba env create -f environment.yml
```

The conda `environment.yml` contains all dependencies needed to develop, test,
and run example notebooks. More detailed Conda environment installation
instructions can be found in `examples/00_python_virtual_env.ipynb`.

##### Pip

To install all dependencies with `pip`:

```
pip install ".[all]"
```

Several dependency groups are defined in `pyproject.toml` and can be selected
instead of `all` for a more lightweight environment:

- `lint`
- `test`
- `optional`

#### Installing `pywatershed` in development mode

Once the python environment and dependencies are established and activated
(e.g. `conda activate pywatershed` or, assuming a virtual environment in `venv`
in the root, `source venv/bin/activate`), `pywatershed` can be installed in
[editable development
mode](https://setuptools.pypa.io/en/latest/userguide/development_mode.html)
with:

`pip install -e . `

## Branching model

This project uses the [git
flow](https://nvie.com/posts/a-successful-git-branching-model/): development
occurs on the `develop` branch, while `main` is reserved for the state of the
latest release. Development PRs are typically squashed to `develop`, to avoid
merge commits. At release time, release branches are merged to `main`, and then
`main` is merged back into `develop`.

## Maintenance ledger

Maintenance todos that are blocked on external events (dependency
releases, cross-repo work like conda-forge feedstocks) are tracked in
[`MAINTENANCE.md`](MAINTENANCE.md) at the repo root, with mechanically
checkable unblock conditions. The `/maintenance` Claude skill checks
them live and reports what is actionable.

## CI

The automated practices of installing, linting, and testing described below are
all formally encoded in `.github/workflows/ci.yaml` and
`.github/workflows/ci_examples.yaml` files.

CI does not compile PRMS. Each domain test job downloads its domain's pinned
answer set from the pws-prms-oracle releases with
`test_data/generate/oracle_fetch.py` and populates `test_data/` from it with
`oracle_populate.py` (see Testing below). The oracle repository is internal,
so the fetch step passes the `PWS_ORACLE_TOKEN` repository secret -- a
classic personal access token with the `repo` scope, SSO-authorized for
DOI-USGS -- as `GITHUB_TOKEN`. Two consequences:

- `pull_request` runs from a fork do not receive secrets, so the domain jobs
  of a fork PR fail at the fetch. Push feature branches to this repository.
- When the token expires or is revoked, every domain job fails at the fetch.
  Replace the secret with a new token (or drop it, once the oracle is public).

The downloaded assets are cached per domain and platform (`actions/cache`,
keyed on `autotest/prms_oracle.toml`): a pin bump re-downloads, nothing else
does.

CI runs at two levels. Pushes to any branch — in this repository or on a fork —
run a skeleton: installs, linting, domainless tests, the documentation build,
and the example notebooks on ubuntu only. The full suite (the domain test jobs
on all three platforms) runs for pull requests (including drafts), pushes to
`develop`/`main`, and `workflow_dispatch`.

To also run domain test jobs on branch pushes, put a `ci-<token>` anywhere in
the branch name or in the pushed (head) commit message. The tokens are
`ci-all`, `ci-sagehen`, `ci-hru1`, `ci-drb`, `ci-ucb`, and `ci-fgr`.

The two variants complement each other:

- **Branch name — sticky.** A branch named `my_feature_ci-fgr` runs both fgr
  domain jobs on every push, with nothing to clean up before merging since
  the opt-in lives in the branch name.
- **Commit message — one shot.** The gate is evaluated fresh on each push
  with no memory of earlier pushes: a push whose head commit message contains
  `ci-drb` runs the drb job for that push only, even on a branch previously
  pushed without any token, and later pushes without a token drop back to the
  skeleton. Only the _head_ (most recent) commit of a push is checked — when
  pushing several commits at once, the token must be in the last one. To
  trigger a domain run on the current state without changing any code, push
  an empty commit:

  ```shell
  git commit --allow-empty -m "trigger ci-fgr"
  git push <fork-remote> my_branch
  ```

Alternatively, open a draft PR or use the workflow dispatch button (on your
fork) to get the full suite on any branch.

Token matching is a plain substring test — there is no parsing of token
lists, separators, or suffixes. Consequences:

- Tokens can only add jobs, never subtract them: appending to a token
  (`ci-sagehen_gridded`) does not narrow the selection, it matches the
  `ci-sagehen` token and triggers every job in that family.
- A commit message that merely _mentions_ a token triggers it — easy to do
  accidentally in a commit message about the CI configuration itself. Write
  `ci-<token>` (as in this file) rather than a literal token when referring
  to the mechanism.
- Tokens are matched anywhere in the branch name, so avoid naming a branch
  with a literal token unless the sticky behavior is wanted.

Notes for maintaining the gates in `ci.yaml`:

- New domain test jobs must copy the `if:` gate from an existing domain job
  (with an appropriate token) — an ungated job runs on every push to every
  branch, silently defeating the skeleton/full split.
- When adding finer-grained tokens, no token may be a substring of another:
  a bare `ci-sagehen` cannot coexist with a distinct `ci-sagehen-gridded`
  switch, because writing the longer token always matches the shorter one.
  To split a family, replace the family token with mutually non-substring
  tokens, e.g. `ci-sagehen-5yr`, `ci-sagehen-gridded`, and `ci-sagehen-all`
  for the whole family.

## Testing

The tests compare `pywatershed` to reference answers (PRMS output). The
answers come from a separate repository,
[pws-prms-oracle](https://github.com/DOI-USGS/pws-prms-oracle) ("the
oracle"). Its `domains/<domain>/` directories hold each test watershed's
inputs and, under `output*/`, the answers.

### Step 1: get the answers

Choose one.

**A. Use a published release** (no compiler needed)

The oracle repository is internal to DOI-USGS, so you need a GitHub
token that can read it. Create a classic personal access token with the
`repo` scope, authorize it for the DOI-USGS organization (SSO), and:

```shell
export GITHUB_TOKEN=<token>
```

`autotest/ci_local.sh` then downloads the pinned release assets into
`test_data/.oracle/` (gitignored) the first time it runs. The pins are
in `autotest/prms_oracle.toml`.

**B. Generate locally** (Fortran compiler needed)

1. Clone the oracle:

   ```shell
   git clone https://github.com/DOI-USGS/pws-prms-oracle /path/to/pws-prms-oracle
   ```

2. Follow the oracle's `README.md` to build PRMS and run every domain.
   Answers (`output*/`) are never committed; this step creates them.

3. Tell `pywatershed` where they are:

   ```shell
   export PWS_ORACLE_DIR=/path/to/pws-prms-oracle/domains
   ```

   `pywatershed` only reads this directory. Several clones can share it.
   Unset, it means `test_data/.oracle/` (option A).

### Step 2: run the tests

From `autotest/`:

```shell
./ci_local.sh
```

For every domain it:

1. Populates `test_data/<domain>/` from the answers: links to the inputs
   and answers (copies on Windows), plus `pywatershed`'s derived answer
   variables written alongside.
2. Runs `pytest` for that domain with the same arguments CI uses.

It also lints, builds the documentation, and runs the example notebooks.
Flags turn parts off; `./ci_local.sh -h` lists them. `-g` skips
populating and tests what is already in `test_data/`.

### Running pieces by hand

Populate only, from the repository root:

```shell
python test_data/generate/oracle_populate.py                    # every domain
python test_data/generate/oracle_populate.py --domain hru_1     # one domain
```

Test one domain and control file, from `autotest/`:

```shell
pytest -n=auto --domain=hru_1 --control_pattern=nhm.control test_prms_atmosphere.py
```

Not every test applies to every domain and control file. `pytest` cannot
tell which do; `ci_local.sh` (and `ci.yaml`) encode that selection with
`--ignore` lists. Copy the `pytest` block for your domain from
`ci_local.sh` rather than running the whole suite.

All tests should pass, XPASS, or XFAIL (an expected failure). See
[`test_data/README.md`](test_data/README.md) and
[`autotest/README.md`](autotest/README.md) for more.

## Linting

Automated linting procedures are performed in CI and enforced, these are

```shell
ruff check .
ruff format .
```

And you'll need to run these locally to pass CI checks.

## Committing Jupyter Notebooks

All outputs are required to be stripped from jupyter notebooks prior to
committing. To facilitate this we have
[pre-commit hooks](https://pre-commit.com/) which will strip
outputs and metadata from jupyter notebooks. When a `git commit` is attempted,
the hook will check all staged `*.ipynb` files. If the file is modified after
running the hook (which runs
[nbstripout](https://github.com/kynan/nbstripout)), then the
commit is abandoned and the changes resulting from the hook need added/staged
before the commit can be attempted again.

The pre-commit hook is highly recommended because it acts at the appropritae
time to keep very large diffs out of the repository history. If you are using
`environment.yml` this will be installed. Othewise, your commits will be bloating the repository and versions of noteboks may need removed from the history.

The maximal amount of metadata can be stripped from Jupyter notebooks by following the example configuration found in the [nbstripout section on stripping metadata](https://github.com/kynan/nbstripout#stripping-metadata).

## pre-commit hooks

Pre-commit hooks apply actionas at commit-time. These are available when
`pre-commit` is installed in the environment, as in the `environment.yml`
supplied. To install yourself

```
pre-commit install
```

As specified in `.pre-commit-config.yaml`, we adopt the following pre-commit
hooks

- [nbstripout](https://github.com/kynan/nbstripout):
  strip outputs from jupyter notebooks
- [blackdoc](https://github.com/keewis/blackdoc):
  apply black within documentation
- [doctoc](https://github.com/thlorenz/doctoc): auto generate tables of
  contents in markdown docs

## Documentation

[Google-style docstrings](https://google.github.io/styleguide/pyguide.html#38-comments-and-docstrings)
are used for documenting source code. (Though numpy style is supposedly handled
as well by Napolean, preference is for google-style.)

## Miscellaneous

### Locating the root

Python scripts often need to reference files elsewhere in the project. To allow
scripts to be run from anywhere in the project hierarchy, scripts should locate
the project root relative to themselves, then use paths relative to the root
for file access, rather than using relative paths (e.g., `../some/path`).

For a script in a subdirectory of the root, for instance, the conventional
approach would be:

`Python project_root_path = Path(__file__).parent.parent `
