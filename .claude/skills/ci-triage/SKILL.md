---
name: ci-triage
description: Triage failing GitHub Actions runs on pywatershed PRs or branches using the public API - identify failing jobs, read annotations when logs are locked, distinguish stale runs from real failures, and set up local reproduction. Use when the user reports CI failures or invokes /ci-triage, optionally with a PR number or run URL.
---

# Triaging pywatershed CI failures

You diagnose; the human clicks. You run read-only API queries (curl,
no auth); the human re-runs workflows, pushes fixes, and merges.
Local reproduction (section 5) runs code and modifies the local
environment (deletes/reinstalls binaries, regenerates test data) —
propose those steps and get explicit permission before running any of
them. Unauthenticated API calls are limited to 60/hour per IP — batch
queries and cache responses to /tmp files.

## 1. Identify the target

- PR number → head sha: `curl -s "https://api.github.com/repos/DOI-USGS/pywatershed/pulls?state=open"`
  (confirm which PR the user means — numbers are easy to confuse).
- All checks for a sha: `curl -s ".../commits/<sha>/check-runs?per_page=100"`.
  Group by `conclusion`; a full run is ~30+ checks across four workflows
  (CI, CI example notebooks, Documentation Build, Safety Check — the
  notebook check names are bare OS names like `ubuntu-latest py3.13`).

## 2. First question: is the failure current?

Before reading any error text, compare timestamps. Check `started_at`
on the check runs and `created_at`/`run_attempt` on
`.../actions/runs/<run_id>`. Passing and failing checks on the same sha
with different start times mean someone re-ran one workflow but not the
others — **re-runs are per workflow**; re-running "CI" does not refresh
docs/notebooks/safety.

Correlate failure times with external events before debugging code:
dependency releases on PyPI
(`curl -s https://pypi.org/pypi/<pkg>/json` → `releases[v][0]['upload_time']`),
mf6 nightly builds, flopy releases. A failure that predates the fix for
its own cause needs a re-run, not a diagnosis. The human has re-run
permission on run pages ("Re-run all jobs") even where the
workflow-dispatch button is missing upstream.

## 3. Getting failure details

Job logs (`.../actions/jobs/<job_id>/logs`) return 403 "Must have admin
rights" without auth. Use the public annotations instead:
`curl -s ".../check-runs/<check_run_id>/annotations"` (check-run id ==
job id). Annotations usually contain the terminal error; step-level
detail beyond that requires the human to open the job page.

## 4. Known failure signatures

- **`micromamba ... exit code 1` + ENOENT `micromamba-shell`**: env
  creation failed, almost always the pip section of the env file being
  unresolvable (a version floor not yet published, a new release of a
  transitive dep). Note which env file the workflow uses: CI and docs
  use `environment.yml`; notebooks and safety use
  `environment_w_jupyter.yml`.
- **flopy rejects valid-looking mf6 input, or mf6 rejects data CI used
  to accept** (e.g. "Error converting to an integer"): mf6 develop
  changed a definition and flopy's bundled dfns lag. Both
  `generate_classes` calls in `ci.yaml` need `--ref develop`, and
  `autotest/ci_local.sh` must stay in sync with `ci.yaml`.
- **A test fails only in CI via `--error-for-skips`**: a conditional
  skip; either `--ignore` it in the broad steps or give it a dedicated
  step whose `--control_pattern` avoids the skip.
- **`check_version.yaml` exit codes**: version files must match the
  release branch name; exit 7 = missing entry in `doc/index.rst`
  (major releases).
- **Domain jobs absent from a branch-push run**: not a failure — the
  skeleton/full split gates them (DEVELOPER.md "CI"); tokens or a draft
  PR bring them back.

## 5. Local reproduction

Reproduce with the job's exact pytest line (visible in `ci.yaml`),
from `autotest/`, in the project conda env. Environment parity
checklist, in order:

1. mf6 binary: delete the stale one so the nightly build is
   reinstalled to match CI.
2. `python -m flopy.mf6.utils.generate_classes --ref develop`
   (run from a modflow6 clone's autotest dir, as CI does).
3. Test data: re-populate if stale (`python
   test_data/generate/oracle_populate.py --domain <d>` from the repo
   root, after `oracle_fetch.py` with `GITHUB_TOKEN` set when
   `PWS_ORACLE_DIR` is unset; conftest errors when `.derived_okey` and
   `.oracle_okey` differ).
4. Run the job's pytest command with its `--domain` and
   `--control_pattern`. Domainless quirk: the conftest exemption is an
   exact string match on `-m domainless` — pass exactly that markexpr,
   no `--domain` needed.

## 6. Report format

Lead with the verdict (real failure vs stale run vs infra). When
staleness is involved, show a short timeline table (run times vs the
external event). List the re-run links per failed workflow — the human
clicks them. If code changes are needed, hand over a diagnosis and the
failing test invocation before drafting fixes.
