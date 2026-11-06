<!-- START doctoc generated TOC please keep comment here to allow auto update -->
<!-- DON'T EDIT THIS SECTION, INSTEAD RE-RUN doctoc TO UPDATE -->
**Table of Contents**

- [Autotest](#autotest)
  - [Test data](#test-data)
  - [Usage](#usage)
  - [The simulation fixture](#the-simulation-fixture)
  - [Answers in test files](#answers-in-test-files)

<!-- END doctoc generated TOC please keep comment here to allow auto update -->

# Autotest

The test suite for pywatershed. Run everything from this directory.

Two kinds of tests:

- **Domainless tests** need no data files. Marked `domainless`; run with
  `pytest -m domainless`.
- **Domain tests** run pywatershed on a test watershed (a "domain") and
  compare it to PRMS answers. They use the `simulation` fixture and need
  `--domain`.

## Test data

Domain tests read inputs and answers from `../test_data/<domain>/`. That
directory is populated from the pws-prms-oracle repository; see
[`DEVELOPER.md`](../DEVELOPER.md), "Testing", for setup and
[`test_data/README.md`](../test_data/README.md) for what is in it.

Before collecting a domain's tests, `conftest.py` checks two markers in
each answer directory: `.oracle_okey` (the answers) and `.derived_okey`
(the answers pywatershed's derived variables were computed from). If they
differ or are missing, collection stops with an error that names the
`oracle_populate.py` command to run.

## Usage

The recommended way:

```shell
./ci_local.sh
```

It populates every domain and runs the same `pytest` commands CI runs.
`./ci_local.sh -h` lists flags for running only parts of it.

By hand, one domain and one control file:

```shell
pytest -n=auto --domain=hru_1 --control_pattern=nhm test_prms_atmosphere.py
```

Options added by `conftest.py` (`pytest --help`, "Custom options"):

- `--domain=<name>` — a directory name under `test_data/`, not a path.
  Repeatable. Required for domain tests.
- `--control_pattern=<glob>` — substring of the control file name
  (`.control` may be dropped). Repeatable. Default: every `*.control` in
  the domain.
- `--all_domains` — shorthand for `--domain=hru_1 --domain=drb_2yr
  --domain=ucb_2yr`.
- `--print_ans` — print each test's results instead of asserting (see
  "Answers in test files").

Not every test applies to every domain and control file; `pytest` cannot
tell which do. `ci_local.sh` and `ci.yaml` encode the selection with
`--ignore` lists. Copy the block for your domain and control from
`ci_local.sh` rather than running the whole suite.

## The simulation fixture

Domain tests take a `simulation` argument. One test runs once per
domain x control file selected; its id is `<domain>:<control stem>`, for
example `drb_2yr:nhm`. The fixture is a dict:

- `name` — the id above
- `dir` — `test_data/<domain>/`
- `control_file` — path to the `.control` file
- `output_dir` — the answer directory the control writes to
  (`netcdf_output_dir` in the control)
- `print_ans` — the `--print_ans` flag

## Answers in test files

A few tests compare summary statistics (not PRMS output) to values kept
in a dict in the test file, keyed by domain (for example `test_ans` in
`test_prms_parameters.py`). `utils.assert_or_print` does the comparison.
To refresh those values after an intended change:

```shell
pytest -s --print_ans --domain=drb_2yr test_prms_parameters.py
```

`-s` shows the printed values; paste them into the dict.
