<!-- START doctoc generated TOC please keep comment here to allow auto update -->
<!-- DON'T EDIT THIS SECTION, INSTEAD RE-RUN doctoc TO UPDATE -->

- [Maintenance ledger](#maintenance-ledger)
  - [Open](#open)
    - [3.0.1 hotfix release (flopy pin unwind + pyPRMS floor)](#301-hotfix-release-flopy-pin-unwind--pyprms-floor)
    - [pyPRMS upstream: close PR #64, port its regression test](#pyprms-upstream-close-pr-64-port-its-regression-test)
    - [Reconcile check_version.yaml with release_preflight.sh](#reconcile-check_versionyaml-with-release_preflightsh)
    - [Deliver CI-usage findings to org admins](#deliver-ci-usage-findings-to-org-admins)
    - [PRMSSnow does not reproduce PRMS at threshold magnitudes](#prmssnow-does-not-reproduce-prms-at-threshold-magnitudes)
    - [Drop the netCDF4 ndarray.shape warning filter](#drop-the-netcdf4-ndarrayshape-warning-filter)
    - [Drop the gfortran <16 ceiling (conda-forge win-64 link failure)](#drop-the-gfortran-16-ceiling-conda-forge-win-64-link-failure)
    - [PR #412 follow-ups: pre-commit notebook coverage, holoviews floor](#pr-412-follow-ups-pre-commit-notebook-coverage-holoviews-floor)
    - [Decide the fate of preprocess_gridded_params before 4.0](#decide-the-fate-of-preprocess_gridded_params-before-40)
  - [Done](#done)

<!-- END doctoc generated TOC please keep comment here to allow auto update -->

# Maintenance ledger

Maintenance todos that are blocked on external events or span
repositories, so they survive between sessions and maintainers. Each
item states what unblocks it as a condition that can be checked
mechanically (PyPI, conda-forge, or GitHub APIs). The `/maintenance`
Claude skill (`.claude/skills/maintenance/`) walks this file, checks
each condition live, and reports what is actionable; humans edit this
file like any other (PRs to `develop`).

Item format: a heading, then **Blocked on** (the condition and how to
check it), **Action** (what to do once unblocked), and optional
**Notes**. Move finished items to the Done section with a date.

## Open

### 3.0.1 hotfix release (flopy pin unwind + pyPRMS floor)

- **Blocked on:** a flopy release containing modflowpy/flopy PR #2730
  (wipe the dfn/toml dir in `generate_classes`). Check: latest version at
  `https://pypi.org/pypi/flopy/json` is newer than 3.10.0 (Feb 2026) and
  its changelog/commits include #2730.
- **Action:** patch release `3.0.1` from `main` per `.github/RELEASE.md`
  (the `/release` skill assists):
  - Replace `"flopy[codegen] @ git+https://github.com/modflowpy/flopy.git"`
    in `environment.yml` and `environment_w_jupyter.yml` with the released
    floor (e.g. `flopy[codegen]>=X`). Leave the `mpsplines` git pin alone —
    it is deliberate.
  - Add a `pyPRMS >=0.10.0` floor to `pyproject.toml` on the release
    branch: 3.0.0's published metadata has no floor, so installing an old
    pyPRMS (<=0.9.10) with packaging >=26.3 still breaks; develop already
    has the floor (PR #406).
  - Afterwards: flopy floor + build-number bump on the conda-forge
    pywatershed feedstock.
- **Notes:** conda-forge flopy 3.10.0 predates the fix, so pywatershed
  codegen paths (e.g. `MmrToMf6Dfw` class generation) are broken with
  released flopy until then; flopy imports lazily so this is not an
  import-time problem.

### pyPRMS upstream: close PR #64, port its regression test

- **Blocked on:** nothing — #64 was closed unmerged 2026-08-17; the
  regression test is not on pyPRMS main (verified 2026-08-19).
- **Action:** port #64's regression test to pyPRMS via a fresh PR
  (`tests/func/test_Parameters.py::TestParametersSharedMetadata` — two
  `Parameters` instances from one shared `MetaData` dict; the fix
  itself shipped in 0.10.0 via commit 3650b285).

### Reconcile check_version.yaml with release_preflight.sh

- **Blocked on:** nothing; low priority.
- **Action:** fold the legacy workflow's `doc/index.rst` major-release
  check (exit code 7) into `.github/scripts/release_preflight.sh`, or
  retire `check_version.yaml` in favor of the preflight + release.yaml
  checks.

### Deliver CI-usage findings to org admins

- **Blocked on:** nothing; conversational.
- **Action:** pywatershed is public and uses only standard runners, so
  its minutes are free and cannot consume the org quota (proof: when the
  org exhausted its minutes, private-repo Actions halted while
  pywatershed kept running). The billing usage CSV attributes metered
  minutes to the private repos that spent them. Org-shared runner
  concurrency was addressed by PR #408 (skeleton/full split +
  cancellation). Also raise the missing upstream push access /
  "Run workflow" button.

### PRMSSnow does not reproduce PRMS at threshold magnitudes

- **Blocked on:** nothing; a large lift rather than an external event.
  Split out of the PRMSCanopy `pkwater_ante` item (Done, 2026-08-25).
- **Action:** bring `PRMSSnow` to the standard the other processes meet,
  i.e. reproduce PRMS's snowpack state to within the 1e-12 tolerance the
  other process tests use, rather than the 1e-3 that
  `test_prms_snow.py` checks.
- **Notes:** `snowcomp` is full of `dnearzero` (2.23e-16) and `nearzero`
  thresholds tested against an accumulated `Pkwater_equiv`, and
  pywatershed's residual tail on a melting pack differs from PRMS's by
  orders of magnitude at those magnitudes. Consequence: single-process
  comparisons of `PRMSCanopy` are faithful, because they read PRMS's own
  `pkwater_ante`, but a fully coupled pywatershed run can still land on
  the other side of a gate from PRMS wherever a vanishing pack is
  involved. Until this is fixed, coupled-run differences of this kind
  are expected and are not canopy bugs.

### Drop the netCDF4 ndarray.shape warning filter

- **Blocked on:** a netCDF4 (netcdf4-python) release after 1.7.4, which
  is the first to contain PR #1469 (merged 2026-02-17). Check:
  `https://api.github.com/repos/Unidata/netcdf4-python/releases?per_page=5`
  for a tag above `v1.7.4rel`, then confirm the fix is in it, e.g. the
  released sdist no longer has `data.shape = tuple(datashape)` in
  `src/netCDF4/_netCDF4.pyx`.
- **Action:** remove `suppress_netcdf4_shape_warning` from
  `pywatershed/utils/netcdf_utils.py` and its three uses (two there,
  one in `pywatershed/base/budget.py`), and raise the netCDF4 floor in
  `environment.yml` to that release. (Until 2026-09-04 this was a
  `filterwarnings` line in `autotest/pytest.ini`, which did not reach
  the notebook subprocesses or users' own scripts.)
- **Notes:** netCDF4 assigns to `ndarray.shape` on every variable write
  (`_netCDF4.pyx:5616`), which NumPy >= 2.5 deprecates. Nothing on the
  pywatershed side avoids it: every assignment form was tried
  (`v[0,:] = a`, `v[0:1,:] = a[np.newaxis, ...]`, `v[0] = a`) and all
  warn, so the two sites the warning is attributed to,
  `pywatershed/utils/netcdf_utils.py:660` and
  `pywatershed/base/budget.py:944`, are correct as written -- the
  warning is raised in Cython and attributed to the nearest Python
  frame, which is ours. Observed with netCDF4 1.7.3 and NumPy 2.5.2:
  ~44,000 warnings in a single `test_prms_canopy.py` run. xarray took
  the same temporary measure (its PR #11146).

### Drop the gfortran <16 ceiling (conda-forge win-64 link failure)

- **Blocked on:** conda-forge restoring `crt2.o` and `default-manifest.o`
  where the win-64 gfortran driver looks for them. Check: a build newer
  than `*_11` at
  `https://api.anaconda.org/package/conda-forge/m2w64-sysroot_win-64`,
  or the upstream issue below being closed.
- **Action:** remove `,<16` from the `gfortran` line in
  `environment.yml` and `environment_w_jupyter.yml`, along with the
  three comment lines above it.
- **Notes:** on 2026-08-26 all five Windows domain jobs that compile
  PRMS failed at the link step with
  `ld.exe: cannot find crt2.o` and `cannot find default-manifest.o`
  (PR #414, run 32922976968). Those two files come from
  `mingw-w64-ucrt-x86_64-crt-git` and
  `mingw-w64-ucrt-x86_64-windows-default-manifest`, both reached through
  `m2w64-sysroot_win-64`, which `gcc_impl_win-64` depends on unpinned.
  Two files from two packages going missing at once points at the
  sysroot layout, not at either package.

  Timeline (UTC, anaconda.org upload times): the last green Windows
  compile was develop's CI at 2026-08-20 22:15; `m2w64-sysroot_win-64`
  build 11 landed at 2026-08-20 23:44 and win-64 `gfortran` 16.2.0
  build 4 at 2026-08-25 02:50.

  develop stayed green only because the micromamba cache key hashes the
  environment file: unchanged file, restored env, no re-solve. Any
  branch that edits `environment.yml` re-solves and hits this.

  **The `<16` ceiling is a probe, not a diagnosis.** Both the sysroot
  and gcc 16.2.0 landed after the last green run. If Windows still
  fails with the ceiling in place, the sysroot is the culprit; pin it
  instead (`m2w64-sysroot_win-64=*=*_10`), checking that a win-64-only
  package in a shared environment file does not break the macOS and
  Linux solves.

  Draft report for `conda-forge/mingw-w64-sysroot-feedstock` (the
  compilers live in `conda-forge/ctng-compilers-feedstock`), not yet
  filed:

  > **Title:** win-64: `crt2.o` and `default-manifest.o` not found at
  > link time with `m2w64-sysroot_win-64` build 11
  >
  > Linking any Fortran or C program on `windows-latest` fails after
  > build 11 of `m2w64-sysroot_win-64` (2026-08-20 23:44 UTC):
  >
  > ```
  > gfortran -O -static -o prms *.o libmmf.a -lgfortran -lgcc -lm
  > .../x86_64-w64-mingw32/bin/ld.exe: cannot find crt2.o: No such file or directory
  > .../x86_64-w64-mingw32/bin/ld.exe: cannot find default-manifest.o: No such file or directory
  > collect2.exe: error: ld returned 1 exit status
  > ```
  >
  > Environment: `gfortran` 16.2.0 `hb5e953d_4` (win-64),
  > `gcc_impl_win-64` 16.2.0, `m2w64-sysroot_win-64` `*_11`, installed
  > with micromamba on GitHub Actions `windows-latest`. The same link
  > line succeeded with `gfortran` 15.3.0 build 2 and the build-10
  > sysroot on 2026-08-20. Both missing files ship in packages the
  > sysroot depends on
  > (`mingw-w64-ucrt-x86_64-crt-git`,
  > `mingw-w64-ucrt-x86_64-windows-default-manifest`), so they appear to
  > be installed somewhere the driver's default search path no longer
  > covers.

### PR #412 follow-ups: pre-commit notebook coverage, holoviews floor

- **Blocked on:** nothing; both were deferred when #412 (lint notebooks)
  was merged on 2026-08-26.
- **Action:**
  - `.pre-commit-config.yaml`'s `ruff-check` hook has `types: [python]`.
    pre-commit's `identify` tags `.ipynb` as `jupyter`, not `python`, so
    a commit staging only notebooks likely skips the hook -- which
    defeats #412's purpose. Check with
    `identify-cli examples/snow_errors.ipynb`; if `python` is absent,
    change to `types_or: [python, jupyter]`.
  - `pyproject.toml`'s `optional` extra lists `hvplot` but never
    `holoviews`, so a pip install resolves to whatever hvplot allows
    (`>=1.19`) and keeps the NumPy 2.5 `nat_as_integer` warning that
    `holoviews>=1.23.0` in the environment files avoids. Add the floor
    for parity.
- **Notes:** the pre-commit hook only ever sees *staged* files while CI
  runs `ruff check .` over the whole tree. That gap is what let #412's
  notebook errors reach CI in the first place.

### Decide the fate of preprocess_gridded_params before 4.0

- **Blocked on:** having reviewed several gridded/inactive-HRU example
  models after the cascades port (PR #407) lands; must be settled before
  the first 4.0 release (check: no `4.0.0` tag at
  `https://api.github.com/repos/DOI-USGS/pywatershed/releases`).
- **Action:** keep or delete
  `pywatershed/utils/preprocess_gridded.py::preprocess_gridded_params`.
  - If kept: state in its docstring that processes never read the
    variables it writes (`active_hru_mask`, `wh_active_hrus`,
    `nactive_hrus` are always derived from `hru_type` by
    `base.HruMixin._set_active_hrus`), and add the `nactive_hru`
    dimension it introduces to `pywatershed/static/metadata/dimensions.yaml`.
  - If deleted: remove it from `doc/api/utils.rst`, delete
    `autotest/test_preprocess_gridded.py`'s tests of it (keep those of
    `get_active_hru_params`, which the mixin uses), and note the removal
    in whats-new.
- **Notes:** the PR #407 review (B3/B7, 2026-09-02) fixed the function's
  crash and removed the mixin's dead "use supplied mask" path, making
  `hru_type` the single source of truth. That left the function public
  with zero callers, writing three variables nothing consumes. Deleting
  was recommended; James kept it pending experience with real gridded
  setups.

## Done

- 2026-08-31: the GSFLOW ifort binary is retired (branch
  `feat_gs_flow_bin_gnu_only`). All three platforms now ship gfortran
  double-precision binaries built from one GSFLOW commit
  (`7f61c53c0278`) by one CI run — the first time provenance is uniform
  and recorded (`bin/README.md`). The arm64 macOS binary (no Rosetta)
  passed the fgr suite locally (`ci_local.sh -ilmosrdu`, 61 + 2 passed);
  the byte-identical `_og` duplicate and a `.bak_x86_64` copy were
  deleted (~36 MB). The oracle-repo end state remains the intended
  destination for GSFLOW/PRMS sources and binaries.
- 2026-08-25: PRMSCanopy's grass rain interception gate now uses
  `pkwater_ante` (an input taken in place of `pk_ice_prev` and
  `freeh2o_prev`), and PRMSSnow declares `pkwater_ante` to supply it,
  matching `intcp.f90:416` and `snowcomp.f90:346-349,946`. Fixed on
  branch `feat_canopy_pkwater_ante`; `test_prms_canopy.py`,
  `test_prms_above_snow.py`, `test_prms_et_canopy.py`, and
  `test_prms_et_can_runoff.py` pass on macOS for `ucb_2yr:nhm`, where
  they had been failing. The caveat in the original item stands and is
  now its own open item: this restores fidelity for the single-process
  comparisons, which read PRMS's own `pkwater_ante`; fully coupled runs
  still depend on PRMSSnow reproducing PRMS's residual tail to ~1e-16.
- 2026-08-19: conda-forge pywatershed 3.0.0 build 1 published (feedstock
  PR #14: `pyprms >=0.10.0` replaces `packaging <26.3`; verified on
  anaconda.org, uploaded 19:35 UTC). The last `packaging <26.3` pin
  anywhere is gone. Also: PR #407's sagehen ubuntu OOM resolved by
  serial pytest (`-n=1`) on Linux.
- 2026-08-19: gh-pages v3.0.0 extended release notes are live upstream
  (`doi-usgs.github.io/pywatershed`, 2026-07-13 post) and all 3.0.0
  release-body links resolve — step 9 done, the 3.0.0 release is
  complete. (Push-access / "Run workflow" org questions moved into the
  CI-usage item.)
- 2026-08-19: PR #407's new sagehen CI jobs carry the skeleton/full
  `if:` gates (token `ci-sagehen`); its remaining ubuntu CI failure is
  being handled on the branch.
- 2026-08-18: PR #406 (pyPRMS>=0.10.0 floor replaces packaging<26.3 in
  pyproject + both env files) and PR #408 (CI skeleton/full split,
  ci-tokens, concurrency cancellation) merged to develop.
- 2026-08-18: pyPRMS 0.10.0 on PyPI and conda-forge; no pywatershed hot
  release needed (3.0.0's published metadata never pinned packaging, so
  fresh installs self-healed).
