# Network hydraulics export and at-a-station hydraulic geometry

Date: 2026-09-11
Status: draft for review
Branch: `feat_network_hydraulics_export` (off `develop`)

## Purpose

Provide, from a pywatershed PRMS run, everything a 1D river-network
particle tracker needs: reach topology, planform geometry, and daily
per-reach flow, velocity, depth, width and shear velocity. The consumer
is a new network solver in the sibling `fluvial-particle` repository,
whose first target is a passive-tracer proof-of-concept demo on the
Delaware River Basin (DRB) example of `examples/02_prms_legacy_models.ipynb`.

Two pieces of work on the pywatershed side:

1. A more realistic per-segment hydraulic geometry for the example,
   derived from the bankfull width and depth already in the PRMS
   parameter file.
2. A model-agnostic "network hydraulics" NetCDF export that other
   routing models (first candidate: NOAA National Water Model, NWM) can
   also be transformed into.

## Non-goals

- The particle solver itself (fluvial-particle repo, separate spec).
- An NWM transformer. The schema admits it; the code is follow-on work
  recorded in `MAINTENANCE.md`.
- A BMI for either package. The design keeps the door open (see
  "Coupling readiness") without building it.
- Sub-daily routing output from pywatershed. The export is at the
  model's output time step (daily for PRMS); temporal interpolation is
  the consumer's option.
- Settling or retention fields. In 1D a first-order loss rate is
  settling velocity over depth; depth is exported, the settling
  velocity is a solver-side user input. No per-reach retention field
  is fabricated.

## Screening facts that shaped the design (DRB, 2026-09-11)

- 456 segments, 6 ocean outlets, 116 headwaters, tree topology
  (`tosegment` single-valued, 0 at outlets), no lakes.
- `width_alpha` equals `seg_width` and `width_m` is 0.015 everywhere,
  so width is effectively constant at bankfull; `depth_alpha` and
  `depth_m` are absent, so every segment used the PRMS defaults
  (0.27, 0.39). Velocity estimates from hydraulic geometry,
  `seg_length / K_coef`, and Manning disagree by about a factor of two.
- The segment shapefile (`Segments_subset.shp`, Albers, meters) matches
  parameter order when sorted by `model_idx`; 446 of 450 downstream
  line ends meet the next line's start within 1 m; polyline length
  agrees with `seg_length` (median ratio 1.03); vertex counts run
  2 to 1034 per line.
- Median residence time is 3.9 h; in 98 percent of segment-days a
  particle traverses a segment within a day.
- The width-only hydraulic geometry process runs in the notebook but
  its variables were not in the output list; `seg_res_time` was, and
  reproduces `area * length / flow` exactly.

## Section 1: at-a-station hydraulic geometry from bankfull parameters

New function `pywatershed.utils.at_a_station_hydraulic_geometry` in
`pywatershed/utils/hydraulic_geometry.py`.

```python
def at_a_station_hydraulic_geometry(
    parameters: Parameters,
    width_exp: float = 0.26,
    depth_exp: float = 0.40,
    return_bankfull: bool = False,
) -> Parameters | tuple[Parameters, dict[str, np.ndarray]]:
```

Inputs (per segment, from `parameters`): `seg_width` (bankfull width,
m), `seg_depth` (bankfull depth, m), `seg_slope` (m/m), `mann_n`
(s m^-1/3). The DRB values were originally set from Bieger et al.
(2015) regional curves, so anchoring on them is consistent with their
provenance.

Computation, per segment, SI units:

- `A_bf = w_bf * d_bf`; `R_bf = A_bf / (w_bf + 2 d_bf)` (rectangular
  section); `S = max(seg_slope, 1e-7)` (same floor as the stream
  temperature process).
- `Q_bf = A_bf * R_bf**(2/3) * sqrt(S) / n` (m^3/s).
- `width_alpha = w_bf / Q_bf**width_exp`; `width_m = width_exp`.
- `depth_alpha = d_bf / Q_bf**depth_exp`; `depth_m = depth_exp`.
- `v_bf = Q_bf / A_bf` (diagnostic).

Because `PRMSHydraulicGeometryFull` evaluates `alpha * Q_cms**m`, the
curves pass through the bankfull point exactly, and `Q / (w d)` equals
the Manning bankfull velocity at `Q_bf`. Default exponents are the
Leopold and Maddock (1953) at-a-station averages; the velocity
exponent, `1 - width_exp - depth_exp` (0.34 by default), is reported in
the diagnostics, not stored.

Validation: the four inputs must be present with dimension
`nsegment`. `seg_width`, `seg_depth` and `mann_n` must be strictly
positive; `seg_slope` is checked *before* the floor is applied and must
be finite and non-negative, so that a negative slope is not silently
turned into 1e-7. Failure raises `ValueError` naming the parameter and
the count of offending segments.

Return: a new `Parameters` object that is a copy of the input with
`width_alpha`, `width_m`, `depth_alpha`, `depth_m` set or overwritten.
(A fragment to `Parameters.merge` is not viable: merge raises on
duplicate keys with different data, and the DRB file already carries
`width_alpha`/`width_m`.) With `return_bankfull=True` a second value is
returned, a dict with `bankfull_flow`, `bankfull_velocity`,
`velocity_exp`.

## Section 2: network hydraulics export

New function `pywatershed.utils.export_network_hydraulics` in
`pywatershed/utils/network_hydraulics.py`.

```python
def export_network_hydraulics(
    parameters: Parameters,
    run_dir: pathlib.Path,
    out_file: pathlib.Path,
    segment_shp_file: pathlib.Path | None = None,
    shp_id_col: str = "nsegment_v",
    connect_tol: float = 1.0,
    start_time: np.datetime64 | None = None,
    end_time: np.datetime64 | None = None,
) -> pathlib.Path:
```

`run_dir` is a pywatershed NetCDF output directory. Required files:
`seg_outflow`, `seg_inflow`, `seg_flow_width`, `seg_flow_depth`,
`seg_flow_velocity`, `seg_res_time`. Optional: `seg_tave_water`. A
missing required file raises `FileNotFoundError` listing all missing
names. Flows are converted from cfs using the process's `CFS_TO_CMS`.

Inputs are validated before anything is written. The parameters must
carry `nhm_seg`, `tosegment`, `seg_length`, `seg_slope`, `mann_n`,
`seg_width`, `seg_depth`, `hru_segment` and `hru_elev`; all missing
names are listed in one `ValueError`. `tosegment` must be 0 (outlet) or
in `1..nsegment` and must not contain a cycle; the offending segments,
or the cycle's chain, are named in the error. Each run file must carry
an `nhm_seg` coordinate in the parameters' order, must share
`seg_outflow`'s time coordinate exactly, and must carry the `units`
string that `pywatershed.meta` gives for that variable: a differing
`units` raises, an absent one warns and the expected units are assumed.
A `start_time` after `end_time`, or a time selection that leaves no
time steps, raises.

The exporter does not recompute geometry; it consumes the process
outputs (see "Coupling readiness"). It adds shear velocity as a pure
per-step helper `shear_velocity(depth, slope)` = `sqrt(g * depth *
max(slope, 1e-7))`, `g = 9.80665`, exposed in the same module and used
by the exporter; it raises `ValueError` on a negative or non-finite
slope rather than flooring it.

### File schema

One NetCDF4 file, CF-style attributes, SI units, dimensions `reach`,
`time`, and optional `vertex`.

Static, dimension `reach`:

| variable | dtype | units | source (PRMS) |
|---|---|---|---|
| `reach_id` | int64 | - | `nhm_seg` |
| `to_id` | int64 | - | `tosegment_nhm` when present, else derived from `tosegment` and `nhm_seg` (0 = outlet) |
| `to_index` | int32 | - | `tosegment - 1`; -1 at outlets |
| `length` | float64 | m | `seg_length` |
| `slope` | float64 | m/m | `seg_slope` |
| `mann_n` | float64 | s m^-1/3 | `mann_n` |
| `elevation_mid` | float64 | m | walked from outlets, see below |
| `bankfull_width` | float64 | m | `seg_width` |
| `bankfull_depth` | float64 | m | `seg_depth` |
| `is_outlet` | int8 | - | `tosegment == 0` |
| `x_mid`, `y_mid` | float64 | m (projected CRS required) | polyline mid arc-length (only with shapefile) |

`elevation_mid` comes from `calculate_seg_mid_elevations(parameters)`
in `pywatershed/utils/network_hydraulics.py`, the outlet-upward walk
(needs `hru_elev`, `hru_segment`) that formerly lived in
`MmrToMf6Dfw._calculate_seg_mid_elevations`; `MmrToMf6Dfw` now calls
the shared function, behavior unchanged, covered by the existing
`test_mmr_to_mf6_dfw.py`.

Polyline block, present only when `segment_shp_file` is given,
dimension `vertex`:

| variable | dtype | units |
|---|---|---|
| `vertex_x`, `vertex_y` | float64 | m (projected CRS required) |
| `vertex_dist` | float64 | CRS units, cumulative arc length from the reach's upstream end, 0 at the first vertex |
| `reach_vertex_start` (dim `reach`) | int64 | index of the reach's first vertex |
| `reach_vertex_count` (dim `reach`) | int32 | number of vertices |

The segment shapefile must use a projected CRS in meters: a geographic
CRS or a projected CRS not in meters raises `ValueError`; a missing
CRS warns and the vertex, arc-length and midpoint variables are
labeled with units "unknown" (`crs_wkt` is then empty). Global
attribute `crs_wkt` holds the shapefile CRS otherwise. Lines are
matched to reaches by `shp_id_col` against `reach_id`; a mismatch in
set or count raises, as does a geometry that is not a `LineString`
(multipart segments must be exploded or merged first).

Each line is oriented so its downstream end is last: a line is
reversed whenever its first vertex is nearer its downstream reach's
nearest end (the closer of that reach's first or last vertex) than its
last vertex is. For an outlet reach (no downstream reach), the
reference point is instead the last vertices of the upstream reaches
that drain to it (already oriented by the pass above); the outlet's
line is reversed when its last vertex is nearer that reference than
its first vertex is. `connect_tol` plays no part in this
reversal; it only bounds a separate check: after orientation, the
number of reaches whose last vertex is still farther than
`connect_tol` from its downstream reach's first vertex is stored as
global attribute `n_unconnected` and returned in a warning — it is not
an error (the DRB has 4) unless more than half of the reaches that have
a downstream reach fail it, which means an id, CRS or units mismatch
and raises. `n_unconnected` is -1 when no `segment_shp_file` is
supplied, meaning no polyline block was written; `connect_tol` is
written as a global attribute either way. An outlet with no upstream
reach cannot be oriented by either pass, and warns.

Time-varying, dimensions `(time, reach)`, `time` as datetime64:

| variable | units | source | method attribute |
|---|---|---|---|
| `flow_out` | m^3/s | `seg_outflow` | routed |
| `flow_in` | m^3/s | `seg_inflow` | routed |
| `velocity` | m/s | `seg_flow_velocity` | `power_law_at_a_station` |
| `depth` | m | `seg_flow_depth` | same |
| `width` | m | `seg_flow_width` | same |
| `ustar` | m/s | computed | `sqrt(g*depth*slope)` |
| `residence_time` | s | `seg_res_time` | `area*length/flow_out` |
| `water_temperature` | degC | `seg_tave_water` | optional |

Every variable carries `long_name`, `units`, `source_name` (the source
model's variable name) and, where derived, `method`.

Global attributes: `source_model` ("pywatershed PRMS"),
`source_model_version`, `geometry_method`, `pywatershed_version`,
`created`, `title`, `crs_wkt`, `n_unconnected`, `connect_tol`, and
`conventions_note` describing the particle convention below and the
zero-flow caveat. Where `flow_out` is 0 the `velocity`, `depth`,
`width` and `residence_time` are 0, not infinite; each of those four
carries a `note` attribute saying to mask on `flow_out > 0`.

### Particle-position convention (for the consumer)

A particle's state is `(reach index, s)` with `0 <= s <= length[reach]`
measured from the reach's upstream end, in the hydraulic `length`, not
the polyline length. Each step the solver advances `s` by velocity
times dt plus a dispersion increment from `ustar`, `depth` and `width`.
If `s` exceeds `length`, the particle moves to `to_index` carrying the
unused fraction of dt and continues at that reach's velocity; at an
outlet (`to_index == -1`) it exits and its exit time is recorded.
Map position is found by scaling `s / length` onto the polyline's total
`vertex_dist` and interpolating between vertices. Time-varying fields
are held for each day or interpolated between days at the consumer's
option.

### Model-agnostic contract and NWM applicability

The contract is the derived, time-varying hydraulics plus generic
topology, never source parameters. Inspected 2026-09-11:

- NWM v3.0 retrospective channel output (hourly, 2,776,734 reaches):
  `streamflow`, `velocity`, `q_lateral`; static `elevation`, `latitude`,
  `longitude`, `order`. No depth or width.
- NWM v3.1.6 `RouteLink_CONUS.nc`: `to` (0 = outlet), `Length`, `So`,
  `n`, trapezoid `BtmWdth`/`TopWdth`/`ChSlp`, compound `nCC`/`TopWdthCC`,
  `MusK`, `MusX`, `alt`, `order`, midpoint `lat`/`lon`. Planform lines
  come from NHDPlus v2 flowlines by COMID.

An NWM transformer would add `stream_order` (int32, from `order`), a
field PRMS has no equivalent for and which the PRMS export omits.

Mapping: depth by inverting Manning for the trapezoid at each reach's
streamflow (the geometry WRF-Hydro uses internally); width = bottom
width + 2 * side slope * depth; `ustar` from hydraulic radius and slope;
`flow_in = streamflow - q_lateral`; outlets where `to == 0`; subsetting
by upstream trace from an outlet. `velocity` is provided directly. All
of this lands in the same variables with different `source_name` and
`method` attributes.

### Coupling readiness (BMI, later)

pywatershed has no BMI today; its `Model`/`Process` expose
initialize, advance, calculate, finalize and named variables, which a
BMI facade would wrap. For a future step-by-step coupling (for
example water-quality particles):

- The export's variable names are the coupling vocabulary. A stepwise
  consumer reads the same named process variables
  (`seg_flow_velocity`, `seg_flow_depth`, `seg_flow_width`,
  `seg_outflow`) each step; the file and the live path carry identical
  quantities because the exporter never recomputes them.
- `shear_velocity(depth, slope)` is a pure per-step function usable in
  both paths.
- On the fluvial-particle side (its own spec), the solver core takes
  per-step arrays from a provider interface with two implementations,
  file-backed (post-processing) and in-memory (set by a BMI
  `set_value`), and its loop is initialize/update(dt)/finalize.

## Section 3: notebook and packaging changes

New notebook `examples/02a_network_hydraulics_export.ipynb`, an add-on
to `02_prms_legacy_models.ipynb`, which is left unchanged. It is
self-contained (it does not depend on `02` having been run) and is
picked up automatically by `autotest_exs/test_notebooks.py`, which
globs every notebook whose name starts with a digit; `02a_` sorts
directly after `02_`. It writes to `examples/02a_network_hydraulics_export/`.

Contents, in order:

- Introduction: why hydraulic geometry matters for particle tracking,
  the thin default geometry in the DRB parameter file, and the plan
  (bankfull-anchored at-a-station relations, then export).
- Setup mirroring `02`: preprocess the CBH files to NetCDF into the
  notebook's own output directory, load the PRMS parameters and the
  `nhm_stream_temp.control` control file, truncate to the same six
  months.
- Derive geometry:
  `params, bankfull = pws.utils.at_a_station_hydraulic_geometry(params, return_bankfull=True)`,
  with a markdown cell on bankfull anchoring and the exponents, and a
  plot of bankfull discharge and velocity on the network map.
- Run the NHM process list from `02` with
  `PRMSHydraulicGeometryFull` in place of
  `PRMSHydraulicGeometryWidthOnly`, and with `seg_flow_width`,
  `seg_flow_depth`, `seg_flow_area`, `seg_flow_velocity`,
  `seg_res_time` added to `control.options["netcdf_output_var_names"]`.
- Compare the derived geometry with the PRMS-default geometry that `02`
  used: velocity and depth for the last day side by side on the map,
  and a scatter of the two velocity estimates, so the reader sees what
  the new parameterization changed.
- Export: call `export_network_hydraulics` on the run directory with
  the segment shapefile, open the result, print its structure, and
  plot last-day velocity and depth from the file. State that this file
  is the interface to fluvial-particle and summarize the
  particle-position convention.

Packaging:

- Export both functions (and `shear_velocity`,
  `calculate_seg_mid_elevations`) from `pywatershed/utils/__init__.py`
  and list them in `doc/api/utils.rst`.
- `doc/whats-new.rst`: one entry under new features with
  `(:pull:`XXX`)`.
- Regenerate `autotest/api_surface.txt` with
  `python .github/scripts/api_surface.py --write` (additive change).
- `MAINTENANCE.md`: two follow-on items, an NWM transformer against
  this schema, and the fluvial-particle network solver (cross-repo).

## Testing

All new tests are `@pytest.mark.domainless` so they run in the broad
CI step without `--domain`.

`autotest/test_hydraulic_geometry_utils.py`:
- Two-segment hand-computed case: alphas, exponents, bankfull round
  trip (`alpha * Q_bf**m` returns `w_bf`, `d_bf`), bankfull velocity.
- Exponent overrides honored; velocity exponent reported.
- Validation errors: missing parameter, non-positive width, depth, n.
- Slope floor applied.
- Input `Parameters` not mutated.
- DRB parameter file (shipped in `pywatershed/data/drb_2yr`): every
  segment reproduces bankfull width and depth at `Q_bf` to 1e-10
  relative.

`autotest/test_network_hydraulics.py`:
- Synthetic three-reach network (two headwaters into one outlet) with a
  `Parameters` object built in the test and a temporary run directory
  of small NetCDF files: checks every static and time-varying variable,
  units, `to_index`, `is_outlet`, `ustar` values, optional temperature
  presence and absence, time subsetting.
- Polyline handling with a small GeoDataFrame written to a temporary
  shapefile: reversal of a backwards line, `vertex_dist` monotone with
  correct totals, `reach_vertex_start`/`count`, `x_mid`/`y_mid`,
  `n_unconnected` when one line is displaced, id mismatch raises.
- Missing required file raises listing all missing names.
- `calculate_seg_mid_elevations` on the synthetic network matches a
  hand-walked result; `MmrToMf6Dfw` tests continue to pass unchanged.

The executed `02a` notebook in CI exercises the DRB end to end.

## Follow-on work (not in this PR)

- fluvial-particle: network solver spec and implementation (passive
  first), consuming this file; demo of headwater releases arriving at
  Trenton with arrival-time distributions and a map animation.
- NWM transformer producing the same schema.
- Optional BMI facades on both packages.

## References

- Bieger, K., Rathjens, H., Allen, P.M., Arnold, J.G. (2015). Development
  and evaluation of bankfull hydraulic geometry relationships for the
  physiographic regions of the United States. JAWRA 51(3), 842-858.
- Leopold, L.B., Maddock, T. (1953). The hydraulic geometry of stream
  channels and some physiographic implications. USGS Professional
  Paper 252.
- PRMS 5.2.1 `strmflow_character.f90` (source of the power-law form and
  default depth coefficients).
