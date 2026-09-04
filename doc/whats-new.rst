.. currentmodule:: pywatershed

What's New
==========

.. ipython:: python
    :suppress:

    import numpy as np
    import pywatershed as pws

    np.random.seed(123456)

.. _whats-new.3.1.0:

v3.1.0 (Unreleased)
---------------------

New Features
~~~~~~~~~~~~~~~~
- A migration guide for updating projects from pywatershed 2.x to 3.x,
  ``version_migration_guides/v2_to_v3.md``: the verified, complete list of
  breaking changes with scans, fixes, and symptom lookup, written to be
  followed by a person or handed whole to an AI assistant. Retroactive
  additions to the v3.0.0 Breaking Changes section below record five changes
  it documents that were found undocumented.
  (:pull:`416`) By `James McCreight <https://github.com/jmccreight>`_.
- HRU cascading flow (Hortonian surface runoff and soilzone interflow /
  Dunnian flow) following PRMS ``cascade_flag=1`` with
  ``cascadegw_flag=0``: new process classes
  :class:`PRMSRunoffCascadesNoDprst` and
  :class:`PRMSSoilzoneCascadesNoDprst`, cascade parameter preprocessing
  from PRMS parameter files
  (:func:`~utils.preprocess_cascades.preprocess_cascade_params`), and
  support for inactive HRUs via :class:`base.HruMixin` (which HRUs are
  active is always derived from the ``hru_type`` parameter; results at
  inactive HRUs are masked to ``nan`` and excluded from mass-balance
  checks by the new ``active_mask`` capability of :class:`base.Budget`).
  Verified against PRMS 5.2.1 on the ``sagehen_5yr``
  (``sagehen_no_gw_cascades``) and new gridded ``sagehen_gridded_5yr``
  (5609 cells with inactive cells) test domains, both tested in CI on
  all platforms. (:pull:`407`) By `James McCreight <https://github.com/jmccreight>`_.
- The reference PRMS 5.2.1 binary (with cascades and full-precision CBH
  output patches) is now compiled on demand from ``prms_src`` by the
  test-data generation machinery
  (:func:`~utils.prms_exe_utils.compile_prms`); the gridded sagehen
  domain generates its own CBH forcing files with PRMS, making it fully
  reproducible from a clean clone. (:pull:`407`) By `James McCreight <https://github.com/jmccreight>`_.

Breaking Changes
~~~~~~~~~~~~~~~~
- :class:`PRMSCanopy` takes the input ``pkwater_ante`` in place of the inputs
  ``pk_ice_prev`` and ``freeh2o_prev``, and :class:`PRMSSnow` declares the
  variable ``pkwater_ante`` to supply it. Code which supplies
  :class:`PRMSCanopy` its inputs individually must be updated; models
  assembled from process lists or model dictionaries are unaffected.
  (:pull:`414`) By `James McCreight <https://github.com/jmccreight>`_.
- ``hru_type`` is now a declared parameter of :class:`PRMSAtmosphere`,
  :class:`PRMSAtmosphereTranspFrost`,
  :class:`PRMSAtmosphereTranspFrostDynamic`, :class:`PRMSSolarGeometry`,
  :class:`PRMSCanopy`, :class:`PRMSGroundwater` and
  :class:`PRMSGroundwaterNoDprst`, which use it (through
  :class:`base.HruMixin`) to identify inactive HRUs. A :class:`Parameters`
  object built by hand for one of these processes must now include
  ``hru_type``; PRMS parameter files and the ``parameters_dis_hru.nc``
  discretization file written by
  :func:`~utils.separate_nhm_params.separate_domain_params_dis_to_ncdf`
  already carry it. As a consequence, :class:`PRMSCanopy` now applies
  PRMS's lake treatment to HRUs with ``hru_type`` lake (no interception),
  where it previously treated every HRU as land; results change on domains
  with lake HRUs. Note that lake HRUs are currently untested in pywatershed:
  no test domain contains one.
  (:pull:`407`) By `James McCreight <https://github.com/jmccreight>`_.
- Keyword arguments were inserted mid-signature: ``stream_seg_in=None``
  now precedes ``dprst_flag`` in :class:`PRMSSoilzone` and
  :class:`PRMSSoilzoneNoDprst`, and ``active_mask=False`` precedes
  ``unit_desc`` in :class:`base.Budget`. Code passing those or any later
  arguments positionally must switch to keywords.
  (:pull:`407`) By `James McCreight <https://github.com/jmccreight>`_.

Bug fixes
~~~~~~~~~
- :class:`PRMSCanopy` gates rain interception by grasses on the antecedent
  snowpack, ``pkwater_ante``, as PRMS does (``intcp.f90:416``, which tests
  ``Pkwater_equiv`` before ``snowcomp`` updates it for the timestep;
  ``snowcomp.f90:946`` publishes that value as ``Pkwater_ante``). pywatershed
  instead reconstructed the pack as ``pk_ice_prev + freeh2o_prev``. PRMS
  carries ``Pkwater_equiv`` as its own accumulated double rather than
  recomputing it as that sum, and in the vanishing tail of a melting pack the
  two differ by orders of magnitude and fall on opposite sides of the
  ``dnearzero`` (2.23e-16) threshold, flipping whether a day's rain is
  intercepted. :class:`PRMSSnow` now declares ``pkwater_ante``, as
  ``snowcomp`` does (``snowcomp.f90:346-349``), so coupled models supply it.
  (:pull:`414`) By `James McCreight <https://github.com/jmccreight>`_.
- :class:`Model` reports both candidate filenames when an input file is
  missing. Input discovery looks for ``<variable>.nc`` in the input directory
  and, failing that, retries the same name as ``<variable>.param``; only the
  second name reached the error, so a missing netCDF input was reported as a
  not-found dynamic parameter file. Both lookups are now checked up front and
  a missing input raises naming both candidates.
  (:pull:`407`) By `James McCreight <https://github.com/jmccreight>`_.
- Integer variables written by :func:`dd_to_nc4_ds` (and so by
  :class:`DatasetDict` and :func:`separate_nhm_params`) no longer carry a
  default ``_FillValue`` of -9999, which made xarray promote them to float
  on read and turn legitimate -9999 values into NaN. The in-memory fill used
  to mask inactive HRUs is now the separate ``mask_fill_values_dict``.
  (:pull:`407`) By `James McCreight <https://github.com/jmccreight>`_.
- Reading a PRMS parameter file no longer raises for a parameter with an
  expandable scalar form that is supplied at some other, unhandled shape;
  such parameters pass through unchanged as before. A monthly parameter is
  now recognized by its declared ``nmonth`` dimension rather than by having
  12 values, so a per-HRU array on a 12-HRU domain is no longer misread as
  monthly.
  (:pull:`407`) By `James McCreight <https://github.com/jmccreight>`_.

Internal changes
~~~~~~~~~~~~~~~~
- The public API surface (exports, ``__init__`` signatures, declared
  input/variable/parameter name sets, control options, and variable/parameter
  metadata) is snapshotted in ``autotest/api_surface.txt``,
  generated by ``.github/scripts/api_surface.py`` and checked by
  ``autotest/test_api_surface.py`` (domainless). A surface change fails CI
  until the baseline is regenerated with the script's ``--write`` option —
  and, if any existing line changed or was removed, documented under
  Breaking Changes.
  (:pull:`416`) By `James McCreight <https://github.com/jmccreight>`_.
- Retire the last intel-built artifact, the GSFLOW binary for MacOS, which
  was x86_64 and required Rosetta 2 on Apple Silicon. All three checked-in
  GSFLOW 2.4.0 binaries (macOS arm64, Linux, Windows) are now gfortran
  double-precision builds from a single GSFLOW source commit and CI run,
  with provenance recorded in ``bin/README.md`` for the first time.
  (:pull:`415`) By `James McCreight <https://github.com/jmccreight>`_.
- Quiet the test suite's NumPy >= 2.5 deprecation warnings, which numbered
  tens of thousands per run. ``constants.nat`` is now
  ``np.datetime64("NaT", "ns")``, the generic form being deprecated and ``ns``
  being what its only use already cast to; ``environment.yml`` requires
  ``holoviews >=1.23.0``, the release which dropped the ``nat_as_integer``
  the warning came from; and ``autotest/pytest.ini`` ignores the
  ``ndarray.shape`` assignment that netCDF4 <= 1.7.4 makes on every variable
  write, which is fixed upstream by netcdf4-python PR #1469 but unreleased
  and so is tracked in ``MAINTENANCE.md``.
  (:pull:`414`) By `James McCreight <https://github.com/jmccreight>`_.
- Retire ifort and Intel MacOS. The PRMS binaries are no longer checked in to
  ``bin/``; they are compiled from ``prms_src/`` with gfortran (supplied by
  ``environment.yml``) the first time they are needed, by
  ``pywatershed.utils.compile_prms()``, which ``ci.yaml``,
  ``autotest/ci_local.sh``, and the test-data fixtures all call. The
  ``prms_src`` makelists gained ``-std=gnu17`` for gcc, without which PRMS
  5.2.1's C sources fail to build under the C23 default of gcc 15. Apple
  Silicon binaries are tagged ``mac_arm`` rather than ``m1``, and Intel MacOS
  is no longer detected. GSFLOW binaries remain checked in because their
  source is not part of this repository.
  (:pull:`414`) By `James McCreight <https://github.com/jmccreight>`_.
- Lint Jupyter notebooks. ``[tool.ruff] include`` covered only ``*.py``, so
  ``ruff check .`` and CI never saw notebooks, while the pre-commit hook
  passed ``*.ipynb`` paths explicitly (which overrides ``include``) and did.
  Notebook problems were therefore invisible until someone edited a notebook
  and was met with errors they had not caused. Notebooks are now in
  ``include``, with ``E501`` and ``I001`` exempted: they are narrative, and
  several import a module purely for its side effect (``hvplot.xarray``
  registers a ``.hvplot`` accessor) and must do so after the module they
  extend, which sorting undoes. Notebooks are linted but not auto-formatted,
  for the same line-length reason. This surfaced four broken cells that no
  test covered: two stray ``)``, a string opened with ``"`` and closed with
  ``'``, and a use of ``pl.Path`` with no ``import pathlib``. Three notebooks
  needed further cleanup once the syntax errors stopped masking it: unused
  imports, unused assignments, a semicolon-joined statement, and bare
  ``except`` clauses narrowed to ``except AssertionError``. Also repairs
  four malformed ``# noqa`` directives (including a ``# noaq`` typo) that
  ruff silently ignored, which meant the unused-import fixer would have
  deleted the ``hvplot`` imports they were meant to protect.
  (:pull:`412`) By `James McCreight <https://github.com/jmccreight>`_.
- Add ``MAINTENANCE.md``, a ledger of maintenance todos blocked on external
  events (dependency releases, cross-repo work), each with a mechanically
  checkable unblock condition; the ``/maintenance`` Claude skill checks them
  live and reports what is actionable.
  (:pull:`409`) By `James McCreight <https://github.com/jmccreight>`_.
- Reduce CI footprint with a skeleton/full split: pushes to any branch (in
  this repository or on forks) run a skeleton — installs, linting, domainless
  tests, the docs build, and example notebooks on ubuntu only — while the
  full suite (domain test jobs, all platforms) runs for pull requests
  (including drafts), pushes to ``develop``/``main``, and
  ``workflow_dispatch``. Domain jobs can be opted in on branch pushes with a
  ``ci-<token>`` (e.g. ``ci-fgr``, ``ci-all``) in the branch name or head
  commit message — see DEVELOPER.md. ``concurrency`` groups cancel in-flight
  runs superseded by a newer push on the same non-mainline ref.
  (:pull:`408`) By `James McCreight <https://github.com/jmccreight>`_.
- ``PRMSAtmosphere``, ``PRMSSolarGeometry``, ``PRMSCanopy``,
  ``PRMSSnow``, ``PRMSRunoff*``, ``PRMSSoilzone*`` and
  ``PRMSGroundwater*`` compute over active HRUs (in routing order where
  applicable) rather than all HRUs. All-active domains are unaffected.
  (:pull:`407`) By `James McCreight <https://github.com/jmccreight>`_.
- Require pyPRMS >=0.10.0 and remove the temporary ``packaging <26.3`` pin it
  supersedes (pyPRMS 0.9.10 crashed on import of metadata with packaging >=26.3).
  Also remove calls to pyPRMS methods deprecated in 0.10.0:
  ``Parameters.adjust_bounded_parameters()`` in domain subsetting and
  ``DataFile.data_by_variable()`` in the obsin flow node test.
  (:pull:`406`) By `James McCreight <https://github.com/jmccreight>`_.

.. _whats-new.3.0.0:

v3.0.0 (13 July 2026)
-------------------------

A migration guide for updating a project from v2.x, written to be
followed by a person or handed to an AI assistant, is at
`version_migration_guides/v2_to_v3.md
<https://github.com/DOI-USGS/pywatershed/blob/develop/version_migration_guides/v2_to_v3.md>`_
in the repository.

New Features
~~~~~~~~~~~~~~~~
- The new staticmethod :meth:`Model.solve_inputs` determines where each process
  input comes from — another process or a file — from a process list or model
  dictionary, without instantiating a Model or requiring any files to exist.
  Useful for determining the file inputs a model configuration requires, e.g.
  when forcing a sub-model from another model's outputs. Model construction
  uses the same implementation internally.
  (:pull:`396`) By `James McCreight <https://github.com/jmccreight>`_.
- The :class:`base.ConservativeProcess` class now supports both mass and energy budgets.
  Processes can specify which quantity to budget using the ``quantity`` parameter in
  ``_set_budget()``. The new ``mass_budget`` and ``energy_budget`` properties provide
  explicit access to each budget type. The legacy ``budget`` property is deprecated
  and will be removed in the next major release - use ``mass_budget`` instead.
  (:pull:`343`) By `James McCreight <https://github.com/jmccreight>`_.
- The new :class:`PRMSStreamTemp` and :class:`PRMSStreamTempHumidityCBH` classes provide
  stream temperature simulation using the PRMS stream temperature methodology, computing
  water temperatures based on energy balance in stream segments. The latter class accepts
  time-varying humidity inputs on the HRUs while the former accepts a mean monthly
  humidity for each segment.
  The classes support optional energy flux tracking and budgeting via the
  ``track_energy_fluxes`` parameter (default: True). When enabled, it computes and
  tracks 11 energy flux components including advective heat transport (upstream,
  lateral, outflow), surface energy exchange (solar radiation, longwave
  emission/absorption, evaporative cooling, convective exchange), and internal sources
  (friction heating, groundwater conduction). These fluxes are available as output
  variables and included in the energy budget. When disabled
  (``track_energy_fluxes=False``), energy flux variables are set to None and excluded
  from NetCDF output, with ``imbalance_behavior`` required to be None.
  The classes :class:`PRMSStreamTemp` and :class:`PRMSStreamTempHumidityCBH` take a stream shade
  class as input on initialization. Two stream shade classes have been implemented,
  :class:`PRMSStreamShadeConstant` and :class:`PRMSStreamShadeDynamic`. The former
  works based on 3 parameters: summer shade fraction, winter shade fraction, and segment
  latitude. The latter class computes shade dynamically based on topographic and
  vegetation parameters using solar geometry calculations. This is the default PRMS
  behavior when ``stream_temp_shade_flag = 0`` and requires 13 parameters describing topography
  and vegetation characteristics for each stream segment.
  The classes :class:`PRMSStreamTemp` and :class:`PRMSStreamTempHumidityCBH` also require one of
  :class:`PRMSHydraulicGeometryFull` or :class:`PRMSHydraulicGeometryWidthOnly` as an upstream process to provide
  hydraulic geometry variables needed for energy balance calculations. :class:`PRMSHydraulicGeometryFull`
  computes flow-dependent hydraulic geometry (width, depth, area, velocity) using power-law
  relationships when all parameters are provided, while :class:`PRMSHydraulicGeometryWidthOnly`
  uses PRMS default values for depth parameters (depth_alpha=0.27, depth_m=0.39) when they are
  missing from the parameter file, matching PRMS 5.2.1 behavior. These capabilities are
  demonstrated in notebooks ``examples/01_multi-process_models.ipynb`` and ``examples/02_prms_legacy_models.ipynb``
  as part of the NHM configuration in pywatershed.
  (:pull:`343`) By `James McCreight <https://github.com/jmccreight>`_.
- Option for :class:`Model` class to read from a single netcdf file or (not and,
  the existing option,) from a directory containing multiple netcdf files.
  (:pull:`333`) By `James McCreight <https://github.com/jmccreight>`_.
- A new :class:`SourceSinkFlowNode` class adds or removes flow above some minimum
  flow parameter as specified by an input data file.
  (:pull:`327`) By `James McCreight <https://github.com/jmccreight>`_.
- The :class:`Control` class has new method `edit_init_start_times` to manage changing these times.
  (:pull:`335`) By `James McCreight <https://github.com/jmccreight>`_.
- The :class:`FlowGraph` class has new method `plot` to show an abstract plot of the FlowGraph.
  (:pull:`351`) By `James McCreight <https://github.com/jmccreight>`_.
- The :class:`base.Process` class and subclasses have a new restart capability.
  See notebook ``examples/08_restart_streamflow.ipynb`` for examples.
  (:pull:`349`, :pull:`362`) By `James McCreight <https://github.com/jmccreight>`_.
- The :class:`PRMSAtmosphereTranspFrost` implements the transp_frost module of PRMS.
  (:pull:`354`) By `James McCreight <https://github.com/jmccreight>`_.
- The :class:`PRMSAtmosphereTranspFrostDynamic` extends :class:`PRMSAtmosphereTranspFrost`
  to accept dynamic (time-varying) fall_frost and spring_frost dates from PRMS dynamic
  parameter files, reproducing PRMS/GSFLOW runs with ``dyn_fallfrost_flag`` and/or
  ``dyn_springfrost_flag`` set.
  (:pull:`392`) By `James McCreight <https://github.com/jmccreight>`_.
- The `load()` method of :class:`parameters.PrmsParameters` now supports reading multiple parameter
  files which are treated as addenda to the first parameter file in the list which
  contains the dimension information.
  (:pull:`354`) By `James McCreight <https://github.com/jmccreight>`_.
- The :class:`StarfitSourceSinkFlowNode` allows sources and sinks to interact
  with storage of a Starfit reservoir/FlowNode.
  (:pull:`348`) By `James McCreight <https://github.com/jmccreight>`_.
- The new :class:`~base.output.Output` class provides flexible output collection and statistical
  analysis for models, supporting HRUs of interest (HOI), segments/nodes of interest (NOI),
  and monthly accumulations. Includes Zarr chunked output capability for efficient large-scale
  data writing (~6x faster than NetCDF). See notebook ``examples/09_model_output.ipynb`` for examples.
  (:pull:`363`) By `James McCreight <https://github.com/jmccreight>`_.
- New agricultural water use classes enable simulation of irrigated agriculture based on GSFLOW.
  :class:`PRMSRunoffAg` extends :class:`PRMSRunoff` to calculate infiltration separately for pervious
  and agricultural areas. :class:`PRMSSoilzoneAgObsET` provides dual-area soil moisture accounting
  with iterative adjustment of irrigation to match observed actual ET. :class:`PRMSSoilzoneAg`
  is a simplified version without the observed ET iteration, suitable when ET observations are
  not available. See notebook ``examples/10_ag_irrigation_use.ipynb`` for examples.
  (:pull:`362`) By `James McCreight <https://github.com/jmccreight>`_.
- Add pre-commit hook to run security review on staged files or on entire repository, checks for:
  1. Absolute paths, 2. IP addresses, 3. Internal server hostnames, and 4. Usernames/passwords or
  credentials. See .github/scripts/check_security.py.
  (:pull:`384`) By `James McCreight <https://github.com/jmccreight>`_.
- Bug fixes for PRMS 5.2.1.1: 1) errant code skipped humidity CBH files entirely when they were
  selected to be used, 2) code deletion resulted in ``seg_humid`` not being zeroed each timestep and
  erroneously accumulating. Both fixes are extensively documented. The PRMS code was modified
  (compared to the released 5.2.1.1) and the pywatershed code was made to match PRMS stream
  temperature when using humidity inputs from 1. CBH, 2. scalar parameter, and 3. monthly spatially
  distributed parameters.
  (:pull:`386`) By `James McCreight <https://github.com/jmccreight>`_.
- Add a weekly security scan using Safety CLI in a GitHub Actions workflow
  (``.github/workflows/security_check.yaml``): checks conda-installed packages,
  pip-installed packages, and ``pyproject.toml`` dependencies separately.
  (:pull:`387`) By `James McCreight <https://github.com/jmccreight>`_.
Breaking Changes
~~~~~~~~~~~~~~~~
- The ``budget_type`` parameter has been renamed to ``imbalance_behavior`` in
  :class:`base.ConservativeProcess` and all its subclasses, in :class:`base.FlowGraph`, and in
  control options. Update all ``budget_type`` references to ``imbalance_behavior`` in
  your code and configuration files. This breaking change clarifies what the parameter does
  and is intentionally distinct from the budget quantity parameter.
  (:pull:`343`) By `James McCreight <https://github.com/jmccreight>`_.
- Budget netcdf output filenames have changed to include the quantity type.
  Mass budgets are now named ``ProcessName_mass_budget.nc`` instead of
  ``ProcessName_budget.nc``. Energy budgets use ``ProcessName_energy_budget.nc``.
  (:pull:`343`) By `James McCreight <https://github.com/jmccreight>`_.
- The variable ``seg_width`` was renamed ``seg_flow_width``. It is the only
  variable removed from the metadata in this release; code selecting it for
  output, or reading it from output files, must use the new name.
  (:pull:`343`) By `James McCreight <https://github.com/jmccreight>`_.
- :class:`PRMSAtmosphere` no longer overrides ``get_variables()`` and so
  inherits :class:`base.Process`'s implementation, which returns the keys of
  ``get_init_values()``. The result is a ``list`` of 15 names where it was a
  ``tuple`` of 14; ``tmax_sum`` is now included. Code which iterates the
  result, to select output variables for example, gets a variable it did not
  get before and raises no error.
  (:pull:`349`) By `James McCreight <https://github.com/jmccreight>`_.
- The method ``PRMSAtmosphere.calculate_transp_tindex`` was renamed
  ``calc_transp_tindex``.
  (:pull:`354`) By `James McCreight <https://github.com/jmccreight>`_.
- ``FlowNodeMaker.get_node`` takes ``self`` as its first argument; the base
  class previously declared ``get_node(control, index)`` without it. The
  :class:`base.FlowNodeMaker` subclasses in pywatershed already supplied
  ``self``, but custom subclasses which copied the base class signature must
  add it.
  (:pull:`348`) By `James McCreight <https://github.com/jmccreight>`_.
- :meth:`Control.load_prms` changed in two ways. The options
  ``parameter_file``, ``netcdf_output_dir`` and ``streamflow_module`` are
  unwrapped from their single-element lists only when exactly one value is
  present; two or more values are now retained as a list. And
  ``keep_unused_options=True`` no longer forces ``warn_unused_options`` to
  ``True``, so the two are set independently.
  (:pull:`354`) By `James McCreight <https://github.com/jmccreight>`_.

Bug fixes
~~~~~~~~~
- The :class:`PRMSRunoffAg` mass budget did not balance on days with canopy
  changeover (``intcp_changeover > 0``, at canopy density transitions), for two
  reasons: the budget input term ``intcp_changeover_budget`` was assigned only
  in dead code and remained zero, and depression storage inflow was missing
  changeover water when changeover is carried separately from net_rain
  (``intcp_changeover_in_net_rain=False``). GSFLOW carries changeover inside
  net_rain, whereby its depression storage receives this water. With the fix
  the budget closes to machine precision.
  (:pull:`401`) By `James McCreight <https://github.com/jmccreight>`_.
- :class:`utils.PrmsDynamicParameter` ``daily_data_array`` left fill values for days
  between ``daily_start_date`` and the first dynamic parameter date inside that window,
  instead of forward-filling from the most recent date at or before ``daily_start_date``
  as PRMS applies dynamic updates. Only runs starting between dynamic parameter dates
  were affected.
  (:pull:`393`) By `James McCreight <https://github.com/jmccreight>`_.
- PRMS 5.2.1.1 had a bug in stream temperature where division by HRU area was repeated
  multiple times. In the old code this occurred in routing.f90 on lines 764 and
  765 and then again on 789 and 790, where ``seginc_swrad`` and ``seginc_potet`` were
  divided despite this having already occurred on lines 744 and 745. Comments
  regarding the fix are found on lines 764 and 793 in the fixed code.
  (:pull:`383`) By `James McCreight <https://github.com/jmccreight>`_.

Internal changes
~~~~~~~~~~~~~~~~
- The :class:`base.ConservativeProcess` class now uses ``_mass_budget`` and
  ``_energy_budget`` attributes internally instead of ``budget``. The ``budget``
  property remains as a deprecated alias for ``_mass_budget`` for backward compatibility.
- Release procedures were revamped: ``.github/RELEASE.md`` rewritten as a
  concrete step-by-step guide with a running example, guarded release
  automation jobs (checks, package build/publish, frozen conda environment
  exports per platform), a preflight script shared between local use and CI,
  and a version-consistency test.
  (:pull:`395`) By `James McCreight <https://github.com/jmccreight>`_.
- CI no longer uses ``fortran-lang/setup-fortran``; gfortran is provided by
  the conda environment.
  (:pull:`394`) By `James McCreight <https://github.com/jmccreight>`_.
- CI and the conda environments temporarily install flopy from its develop
  branch (with codegen options) to accommodate a flopy API change, until the
  next flopy release.
  (:pull:`397`) By `James McCreight <https://github.com/jmccreight>`_.
- The autotests now require an explicit ``--domain`` option; the silent
  ``drb_2yr`` default was removed.
  (:pull:`393`) By `James McCreight <https://github.com/jmccreight>`_.
- ``autotest/ci_local.sh`` keeps previously downloaded mf6 binaries on PATH
  so mf6-dependent tests run even when the modflow section is skipped.
  (:pull:`398`) By `James McCreight <https://github.com/jmccreight>`_.
- Refactor of ``test_data/generate/convert_prms_output_to_nc.py`` to put final variables into
  a separate file to run by pytests both after all other variables are generated and
  so the final variables are run serially.
  (:pull:`331`) By `James McCreight <https://github.com/jmccreight>`_.

.. _whats-new.2.0.4:

v2.0.4 (23 February 2026)
--------------------------

New Features
~~~~~~~~~~~~~~~~
Fixes to release workflow, pypi publishing.

.. _whats-new.2.0.3:

v2.0.3 (22 February 2026)
--------------------------

New Features
~~~~~~~~~~~~~~~~
Some minor fixes. This is a data release for the upcoming major release, new data will be an asset on this
release.

.. _whats-new.2.0.2:

v2.0.2 (14 March 2025)
----------------------

Bug fixes
~~~~~~~~~
- Fixed setup.py to allow editable installs, keeping up with changes in
  the pythonverse. Deprecated all fortran code built and interfaced using
  f2py as it was not popular and had only maybe very slight speed advantages
  compared to numba. This was not considered a breaking change because there
  are redundant alternatives to the fortran.
  (:pull:`331`) By `James McCreight <https://github.com/jmccreight>`_.

.. _whats-new.2.0.1:

v2.0.1 (19 December 2024)
-----------------------------

New Features
~~~~~~~~~~~~~~~~
- Corrected disclaimer on top-level README.md. Other minor fixes not to code base (CI, envs, etc).


.. _whats-new.2.0.0:

v2.0.0 (16 December 2024)
-----------------------------

New Features
~~~~~~~~~~~~~~~~
- The :class:`FlowGraph` capabilities are introduced. These allow users to
  combine different kinds flow solutions in arbitrary order on a "flow graph".
  The accompanying base classes :class:`FlowNode` and :class:`FlowNodeMaker`
  are introduced along with their subclasses for modeling
  :class:`PassThroughFlowNode`\ s, :class:`ObsInFlowNode`\ s (flow replacement by
  observations with sink and source tracking in mass balance),
  :class:`PRMSChannelFlowNode`\ s, and :class:`StarfitFlowNode`\ s. A new
  example notebook,
  `examples/06_flow_graph_starfit.ipynb <https://github.com/DOI-USGS/pywatershed/blob/develop/examples/06_flow_graph_starfit.ipynb>`__
  demonstrates adding STARFIT reservoir nodes into a FlowGraph otherwise
  simulating `PRMSChannel` and highlights helper functions for this use case.
  (:pull:`233`) By `James McCreight <https://github.com/jmccreight>`_.
- The :class:`MmrToMf6Dfw` class builds a MF6 simulation with Diffusive Wave
  (DFW) routing from PRMS NHM input files and a few simple assumptions. The
  lateral (to-channel) fluxes from a PRMS are used as time varying boundary
  conditions. A new notebook runs the Delaware River Basin using MF6 DFW:
  `examples/07_mmr_to_mf6_chf_dfw.ipynb <https://github.com/DOI-USGS/pywatershed/blob/develop/examples/07_mmr_to_mf6_chf_dfw.ipynb>`__.
  (:pull:`290`) By `James McCreight <https://github.com/jmccreight>`_.
- No depression storage subclasses are available for PRMSRunoff, PRMSSoilzone,
  and PRMSGroundwater by adding "NoDprst" to the end of the names. Depression
  storage is switched off in sagehen_5yr and in new nhm_no_dprst
  configurations.
  (:pull:`288`) By `James McCreight <https://github.com/jmccreight>`_.
- Dunnian flow is implemented (in PRMSSoilzone) and tested for sagehen_5yr.
  (:pull:`288`) By `James McCreight <https://github.com/jmccreight>`_.
- Preferential flow is implemented (in PRMSSoilzone) and tested for sagehen_5yr.
  (:pull:`288`) By `James McCreight <https://github.com/jmccreight>`_.
- Control instances have a diff method to compare with other instances.
  (:pull:`288`) By `James McCreight <https://github.com/jmccreight>`_.
- Feature to standardize subsetting input data (parameters and forcings) in
  space and time either from file (:func:`utils.netcdf_utils.subset_netcdf_file`) or
  in memory (:func:`utils.netcdf_utils.subset_xr`).
  (:pull:`304`) By `James McCreight <https://github.com/jmccreight>`_.

Breaking Changes
~~~~~~~~~~~~~~~~
- pref_flow_infil_frac now a required parameter input for PRMSSoilzone. The NHM
  values assumed previously are zeros on all HRUs.
  (:pull:`288`) By `James McCreight <https://github.com/jmccreight>`_.

Bug fixes
~~~~~~~~~
- Fixed calculation of the variable transp_on was incorrectly calculated in certain
  situations not covered by NHM configuratons but covered by sagehen_5yr.
  (:pull:`288`) By `James McCreight <https://github.com/jmccreight>`_.
- Fixed calculation of variable dprst_area_open which was not being checked but
  was affecting no other variables.
  (:pull:`288`) By `James McCreight <https://github.com/jmccreight>`_.
- The variable pptmix was incorrectly calculated in certain situations not covered
  by the NHM configurations.
  (:pull:`288`) By `James McCreight <https://github.com/jmccreight>`_.

Internal changes
~~~~~~~~~~~~~~~~
- Testing system refactor to handle pairs of domains and control files
  allowing much more flexibility in configuration/control testing.
  (:pull:`278`) By `James McCreight <https://github.com/jmccreight>`_.
- New testing domain "sagehen_5yr" is added to test_data directory
  with configuration sagehen_no_cascades. This domain introduces multiple
  PRMS capabilities (noted indvidually in this PR) not used in the NHM
  configuration and provides a test for these.
  (:pull:`288`) By `James McCreight <https://github.com/jmccreight>`_.
- Tests are now marked as "domain" or "domainless" to avoid redundant
  runs of domainless tests across test domains.
  (:pull:`288`) By `James McCreight <https://github.com/jmccreight>`_.
- New tests test_prms_above_snow and test_prms_below_snow replace
  test_model and are extremely close to PRMS (PRMSSolarGeometry: 1.0e-8,
  PRMSAtmosphere: 1.0e-5, PRMSCanopy: 1.0e-6, PRMSRunoff: 1.0e-8,
  PRMSRunoffNoDprst: 1.0e-8, PRMSSoilzone: 1.0e-8, PRMSSoilzoneNoDprst: 1.0e-8,
  PRMSGroundwater: 1.0e-8, PRMSGroundwaterNoDprst: 1.0e-8, PRMSChannel: 5.0e-7)
  for all test domains.
  (:pull:`288`) By `James McCreight <https://github.com/jmccreight>`_.
- Migration to Numpy 2.0+.
  (:pull:`310`) By `James McCreight <https://github.com/jmccreight>`_.


.. _whats-new.1.1.0:

v1.1.0 (25 June 2024)
---------------------

New features
~~~~~~~~~~~~
- Minor enhancement to ensure PRMSSnow hru_deplcrv parameter is integer or coercable.
  (:pull:`296`) By `James McCreight <https://github.com/jmccreight>`_.
- Release assests to include new GIS files and an additional domain to support the upcoming
  major release. By `James McCreight <https://github.com/jmccreight>`_.


.. _whats-new.1.0.0:

v1.0.0 (18 December 2023)
-------------------------

New features
~~~~~~~~~~~~
- Control object features including (optional) warnings for unused legacy options, and
  defined and enforced options. Also to_yaml() and __str__ implementations.
  (:pull:`240`) By `James McCreight <https://github.com/jmccreight>`_.
- Example notebook of how to edit Parameters with associated bug fixes to do so.
  (:pull:`232`) By `James McCreight <https://github.com/jmccreight>`_.
- Conda feedstock for pywatershed `<https://github.com/conda-forge/staged-recipes/pull/23428>`_.
  By `Joseph Hughes <https://github.com/jdhughes-usgs>`_.


Breaking changes
~~~~~~~~~~~~~~~~
- The `control.options` "netcdf_output_dir", "netcdf_output_var_names", and
  "netcdf_output_separate_files" match the keyword arguments "output_dir",
  "output_vars", and "separate_files" for both `process.intitalize_netcdf()`
  and `model.initialize_netcdf()`. None of these arguments can be supplied in
  both places (control and method call). It used to be that calling
  `initialize_netcdf()` would override what is supplied in `control.options`
  but this will now throw an error. The suggestion is to use `control.options` and
  not pass arguments to `intialize_netcdf()`. When using
  `Control.load()` (deprecated) or `Control.load_prms()` from a PRMS control
  file, note that the "control.options" of "netcdf_output_dir" and
  "netcdf_output_var_names" are set by values in the PRMS control file. You can
  edit these, but be aware that they are now set in that load.
  (:pull:`257`) By `James McCreight <https://github.com/jmccreight>`_.

Deprecations
~~~~~~~~~~~~
- Deprecation of Control.load() for Control.load_prms().
  (:pull:`240`) By `James McCreight <https://github.com/jmccreight>`_.

Performance
~~~~~~~~~~~


Bug fixes
~~~~~~~~~
- Mass balance fix in PRMS snow for rain on snow followed by evaporation
  consuming the entire snow pack.
  (:pull:`248`) By `James McCreight <https://github.com/jmccreight>`_.
- Fix mass balance issue in PRMSSnow is also present in PRMS,
  snow evap is not taken from freeh2o when there is no pk_ice.
  (:pull:`236`) By `James McCreight <https://github.com/jmccreight>`_.
- Resolve issues with different ways of specifying necdf output options.
  (:pull:`230`) By `James McCreight <https://github.com/jmccreight>`_.
- Resolve issues with different ways of specifiying netcdf output options.
  (:pull:`230`) By `James McCreight <https://github.com/jmccreight>`_.
- PRMSSoilzone remove soil_moist_prev because soil_moist is not prognotic and
  PRMSRunoff was needing it in the advance and not getting the correct value.
  PRMSRunoff now depends on soil_lower_prev and soil_rechr_prev instead.
  (:pull:`244`) By `James McCreight <https://github.com/jmccreight>`_.

Documentation
~~~~~~~~~~~~~
- Implement sphinx_autodoc_typehints.
  (:pull:`257`) By `James McCreight <https://github.com/jmccreight>`_.
- New gh-pages branch (without history) to publish
  `"pywatershed notes" <https://doi-usgs.github.io/pywatershed/>`_ including the
  `extended release notes for v1.0.0 <https://doi-usgs.github.io/pywatershed/2023/12/18/v1-0-0-overview>`_.
  This branch publishes analysis supporting the version 1.0.0 release.
- Add about section for version 1.0 to describe how pywatershed matches PRMS'
  NHM configuration and how to perform the comparison.
  (:pull:`244`) By `James McCreight <https://github.com/jmccreight>`_.

Internal changes
~~~~~~~~~~~~~~~~
- New system for generating test_data, by calling generate_test_data.py from
  `autotest/`. The system helps autotest know if test data were generated
  and if they are up to date.
  (:pull:`253`) By `James McCreight <https://github.com/jmccreight>`_.
- Apply pylint and flake8 everywhere as much as possible.
  (:pull:`251`) By `James McCreight <https://github.com/jmccreight>`_.
- Remove diagnostic variables pkwater_equiv_change, pkwater_ante
  (:pull:`248`) By `James McCreight <https://github.com/jmccreight>`_.
- Use v1 instead of main for fortran-lang/setup-fortran.
  (:pull:`242`, :pull:`243`) By `Wes Bonelli <https://github.com/w-bonelli>`_.
- Refactor test data generation to solve race condition for dependent tests.
  (:pull:`237`) By `Wes Bonelli <https://github.com/w-bonelli>`_.
- Refactor tests against PRMS for consistency, flexibility, and thoroughness.
  (:pull:`244`) By `James McCreight <https://github.com/jmccreight>`_.


.. _whats-new.0.2.1:

v0.2.1 (19 July 2023)
---------------------

Bug fixes
~~~~~~~~~
- Package data was not properly installed.
  (:pull:`219`) By `James McCreight <https://github.com/jmccreight>`_.
- Small addition to notebook 02
  (:pull:`219`) By `James McCreight <https://github.com/jmccreight>`_.


.. _whats-new.0.2.0:

v0.2.0 (18 July 2023)
---------------------

New features
~~~~~~~~~~~~
- New example notebooks. Moved old notebooks to `examples/developer`.
  (:pull:`204`)
  By `James McCreight <https://github.com/jmccreight>`_.
- New way to specify `Model` instantiation either in-memory or from yaml files
  using a model dictionary. The approach is loosely based on MODFLOW 6 input
  organization. See `Model` documentation. Introduced the concept of
  discretizations for PRMS, defining "dis_hru" and "dis_seg". These are
  components of how model dictionaries are specified.
  (:pull:`188`) By `James McCreight <https://github.com/jmccreight>`_.
- New `Control.from_yaml()` method. (:pull:`188`)
  By `James McCreight <https://github.com/jmccreight>`_.
- What's new workflow (behold!) per :issue:`180` and :pull:`181`
  By `James McCreight <https://github.com/jmccreight>`_.
- Add automatic release workflow to PyPi as per :issue:`178`. Associated
  implementation of gitflow and semver conventions. Overhauled
  `CONTRIBUTING.md`, `DEVELOPER.md`, `README.md`, and `.github/RELEASE.md`
  to document adopted practices. Adoption of `git-cliff` to generate change
  logs by filtering comitt messages, see `cliff.toml`. Clean up of environment
  files and streamlining against `pyproject.toml`. Symlink gfortran dylibs to
  `/usr/local/lib` on macOS CI so PRMS binaries included in this repo can find
  them. (:pull:`179`)
  By `Wes Bonelli <https://github.com/w-bonelli>`_.


Breaking changes
~~~~~~~~~~~~~~~~
- Move Control attribute "config" to "options" for handling global options.
  (:pull:`188`)
  By `James McCreight <https://github.com/jmccreight>`_.
- Remove arguments from `Model` initialization. Options pass via control, new
  `set_options()` method on Process and ConservativeProcess
  (:pull:`188`)
  By `James McCreight <https://github.com/jmccreight>`_.
- `Control` no longer takes a `Parameter` object as an initialization argument.
  `Process` subclasses now require arguments `discretization` and `parameters`.
  The firstargument of `Model` not a indefinite number of processes, it is now
  either a list of `Process` subclasses or a model dictionary (see `Model`
  documentation. (:pull:`188`)
  By `James McCreight <https://github.com/jmccreight>`_.


Deprecations
~~~~~~~~~~~~


Performance
~~~~~~~~~~~
- Introduce ASV performance benchmarks for import and various NHM configurations
  in pywatershed. (:issue:`170` and :pull:`184`)
  By `James McCreight <https://github.com/jmccreight>`_.


Bug fixes
~~~~~~~~~
- Remove non-pep-compliant post-release reset PR steps. (:pull:`203`)
  By `Wes Bonelli <https://github.com/w-bonelli>`_.
- Add doc building requirements to environment.yml (:pull:`188`)
  By `James McCreight <https://github.com/jmccreight>`_.
- Revive fortran compiling for editable installs (:pull:`188`)
  By `James McCreight <https://github.com/jmccreight>`_.
- Made the Parameter class data completely private by converting dicts to
  MappingProxyTypes and setting numpy.ndarrays to read-only. (:issue:`177`
  and :pull:`183`)
  By `James McCreight <https://github.com/jmccreight>`_.
- ModelGraph improvements and fixes (however result are platform dependent)
  (:pull:`162`) By `James McCreight <https://github.com/jmccreight>`_.


Documentation
~~~~~~~~~~~~~
- Model class, DatasetDict and general documentation overhaul (:pull:`188`)
  By `James McCreight <https://github.com/jmccreight>`_.


Internal changes
~~~~~~~~~~~~~~~~
- Introduce precommit hooks: `nbstripout`, `blackdoc`, and `doctoc`.
  (:pull:`197`)
  By `James McCreight <https://github.com/jmccreight>`_.
- Rename StorageUnit to ConservativeProcess that subclasses from a new Process
  class that contains most of the StorageUnit functionality. (:pull:`188`)
  By `James McCreight <https://github.com/jmccreight>`_.
- New set_options() method on Process and ConservativeProcess to set their
  initialization options as '_` atrributes. (:pull:`188`)
  By `James McCreight <https://github.com/jmccreight>`_.
- Clean up of how the `calc_method` option assigns function names to reduce
  the total amount of code and do it upon initialization. (:pull:`188`)
  By `James McCreight <https://github.com/jmccreight>`_.
- Rename many modules to use lower-snake-case names including those in base/,
  atmoshpere/, and hydrology/ (:pull:`188`)
  By `James McCreight <https://github.com/jmccreight>`_.
- NHM "self-driving" tests
  (:pull:`160`)
  By `James McCreight <https://github.com/jmccreight>`_.
- Refactor dependencies for standard pypi installation. (:pull:`164`,
  :issue:`178`)
  By `Joseph Hughes <https://github.com/jdhughes-usgs>`_.


.. _whats-new.0.1.1:

v0.1.1 (27 April 2023)
----------------------

Initial release.
