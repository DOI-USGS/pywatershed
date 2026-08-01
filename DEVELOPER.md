# Developing `pywatershed`

This document describes how to set up a development environment, install
dependencies, compile C/Fortran code, generate example data, and run the tests
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
    - [F2PY](#f2py)
- [Branching model](#branching-model)
- [CI](#ci)
- [Testing](#testing)
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

C and Fortran compilers are required. We are currently using gnu (gcc,
gfortran) 11 and 12 as well as intel (icc, ifort) 2021 on Windows, Linux, and
MacOS (including Apple Silicon). Both of these are freely obtainable but the
installation process varies widely. We are looking for a conda-based approach
to obtaining compilers, but currently do not have a solution. Compilers are
needed for two applications:

1. Compiling and running C/Fortran PRMS code to generate testing/verification
   data 2. Compiling (installing) and running fortran backends/kernels for some
   hydrological process representations in pywatershed

On Apple Silicon, the PRMS source code is only currently known to compile with
intel while the fortran kernels in pywatershed only compile with gnu.

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

#### F2PY

The numpy extension F2PY is used to provide fortran compiled kernels of core
calculations to boost performance. F2PY is documented [within
numpy](https://numpy.org/doc/stable/f2py/index.html). This repository is
configured NOT to compile on install by default. Currently, we have not
established this compilation procedure for Windows. On Linux and MacOS,
compilation of fortran kernels on package installation is achieved by setting
several environent variables before installing the `pywatershed` module. For
instance, from the project root:

```
export SETUPTOOLS_ENABLE_FEATURES="legacy-editable"
export CC=path/to/gcc  # for example
export FC=path/to/gfortran  # for example
export PYWS_FORTRAN=true
pip install -e .
```

Note that an editable (`-e` above) is required to compile the fotran code.

## Branching model

This project uses the [git
flow](https://nvie.com/posts/a-successful-git-branching-model/): development
occurs on the `develop` branch, while `main` is reserved for the state of the
latest release. Development PRs are typically squashed to `develop`, to avoid
merge commits. At release time, release branches are merged to `main`, and then
`main` is merged back into `develop`.

## CI

The automated practices of installing, linting, and testing described below are
all formally encoded in `.github/workflows/ci.yaml` and
`.github/workflows/ci_examples.yaml` files.

## Testing

Once the dependencies are available, we want to verify the software by running
its test suite. However, we first need to generate the test data. This consists
of running binaries (PRMS) and then converting the output to netcdf files used
by autotest as the answers or reference results. In the `autotest/` directory,
test data is generated using the following command.

```shell
python generate_test_data.py -n=auto
```

Additional options may be supplied to `generate_test_data.py`. For more details
on generating the test data, see [`test_data/README.md`](test_data/README.md).

Finally, the tests can be run from the `autotest` directory:

```shell
pytest -v -n=auto
```

All tests should pass, XPASS, or XFAIL. XFAIL is an expected
failure. The flag `-n auto` to automatically use all available cores on your
machine.

For more details on the autotests, see [`autotest/README.md`](autotest/README.md).

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
