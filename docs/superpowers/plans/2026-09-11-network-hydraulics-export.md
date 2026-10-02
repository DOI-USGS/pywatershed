# Network Hydraulics Export Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a bankfull-anchored at-a-station hydraulic geometry helper and a model-agnostic network-hydraulics NetCDF exporter to pywatershed, demonstrated in a new example notebook `02a`, so a 1D network particle tracker in fluvial-particle can consume DRB segment flow.

**Architecture:** Two new modules in `pywatershed/utils/`: `hydraulic_geometry.py` (derive `width_alpha`, `width_m`, `depth_alpha`, `depth_m` from bankfull parameters via Manning) and `network_hydraulics.py` (shear velocity helper, the mid-segment elevation walk refactored out of `MmrToMf6Dfw`, and `export_network_hydraulics` which reads a pywatershed run directory plus parameters and an optional segment shapefile and writes one CF-style NetCDF). The exporter never recomputes hydraulics; it consumes the `PRMSHydraulicGeometryFull` process outputs. A new notebook `examples/02a_network_hydraulics_export.ipynb` runs the DRB with the derived geometry and exports.

**Tech Stack:** Python 3.13, numpy, xarray, netCDF4, geopandas/shapely (already dependencies), pytest with the `domainless` marker, nbformat for building the notebook, ruff (line length 79).

**Spec:** `docs/superpowers/specs/2026-09-11-network-hydraulics-export-design.md`

## Global Constraints

- Branch `feat_network_hydraulics_export` off `develop`; never commit the three example notebooks that carry executed outputs (`00_`, `01_`, `02_`) — they are unrelated working-tree noise.
- Python interpreter for all commands: `/home/rmcd/miniforge3/envs/pws/bin/python` (conda env `pws`); run pytest from `autotest/` as `/home/rmcd/miniforge3/envs/pws/bin/python -m pytest ...`.
- All new tests carry `@pytest.mark.domainless` and take no `simulation` fixture.
- `ruff check .` and `ruff format .` clean (line length 79). Run `ruff format` before each commit.
- SI units in the export: m, m/s, m^3/s, s, degC. Flow conversion constant `CFS_TO_CMS = 0.028316847` (import from `pywatershed.hydrology.prms_hydraulic_geometry`).
- Slope floor `1.0e-7`; gravity `9.80665`.
- Default exponents: `width_exp=0.26`, `depth_exp=0.40`.
- Commit messages: conventional prefix (`feat:`, `test:`, `docs:`, `refactor:`), ending with `Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>`.
- Public additions must be exported from `pywatershed/utils/__init__.py`, listed in `doc/api/utils.rst`, get a `doc/whats-new.rst` entry with `(:pull:`XXX`)`, and the API-surface baseline must be regenerated (`python .github/scripts/api_surface.py --write`).

---

## File structure

- Create `pywatershed/utils/hydraulic_geometry.py` — `at_a_station_hydraulic_geometry`.
- Create `pywatershed/utils/network_hydraulics.py` — `G`, `SLOPE_FLOOR`, `shear_velocity`, `calculate_seg_mid_elevations`, `export_network_hydraulics` (+ private helpers `_read_run_vars`, `_polyline_block`).
- Modify `pywatershed/utils/mmr_to_mf6_dfw.py:407-421,786-790,807-914` — delegate to `calculate_seg_mid_elevations`.
- Modify `pywatershed/utils/__init__.py`, `doc/api/utils.rst`, `doc/whats-new.rst`, `autotest/api_surface.txt`, `MAINTENANCE.md`.
- Create `autotest/test_hydraulic_geometry_utils.py`, `autotest/test_network_hydraulics.py`.
- Create `examples/02a_network_hydraulics_export.ipynb` (built by a throwaway script in the scratchpad, not committed).

Shared test fixture (defined in `autotest/test_network_hydraulics.py`, and duplicated verbatim in the geometry test file so each file is self-contained): a three-reach synthetic network.

```
reach index:   0        1        2
nhm_seg:     101      102      103
tosegment:     3        3        0     (1-based; 0 = outlet)
length (m): 1000     2000     1500
slope:      0.01    0.005    0.002
mann_n:     0.04    0.035     0.03
seg_width:     5        8       12     (bankfull, m)
seg_depth:   0.5      0.8      1.2     (bankfull, m)
hru_segment: [1, 2, 3]; hru_elev: [120, 110, 100]
```

Mid elevations by the outlet-upward walk: reach 2 rise 3 m on outlet HRU elevation 100 → upstream end 103, mid 101.5; reaches 0 and 1 each rise 10 m above 103 → mid 108.

---

### Task 1: `network_hydraulics.py` foundations — shear velocity and the elevation walk

**Files:**
- Create: `pywatershed/utils/network_hydraulics.py`
- Modify: `pywatershed/utils/mmr_to_mf6_dfw.py:407-421`, `:786-790`, `:807-914`
- Test: `autotest/test_network_hydraulics.py`

**Interfaces:**
- Produces: `shear_velocity(depth: np.ndarray, slope: np.ndarray) -> np.ndarray`; `calculate_seg_mid_elevations(parameters: Parameters) -> tuple[np.ndarray, dict[int, float]]` returning per-segment midpoint elevation and `{outlet_index: outlet_mid_elevation}`; module constants `G = 9.80665`, `SLOPE_FLOOR = 1.0e-7`.

- [ ] **Step 1: Write the failing tests**

Create `autotest/test_network_hydraulics.py`:

```python
import numpy as np
import pytest

from pywatershed.base.parameters import Parameters

# Three-reach synthetic network: reaches 0 and 1 are headwaters that
# flow into reach 2, which is the outlet.
NSEG = 3
NHRU = 3


def _meta(dims: tuple, units: str) -> dict:
    return {"dims": dims, "attrs": {"units": units}}


@pytest.fixture
def synthetic_params() -> Parameters:
    dims = {"nsegment": NSEG, "nhru": NHRU}
    coords = {
        "nhm_seg": np.array([101, 102, 103], dtype=np.int64),
        "nhm_id": np.array([1, 2, 3], dtype=np.int64),
    }
    data_vars = {
        "tosegment": np.array([3, 3, 0], dtype=np.int64),
        "tosegment_nhm": np.array([103, 103, 0], dtype=np.int64),
        "seg_length": np.array([1000.0, 2000.0, 1500.0]),
        "seg_slope": np.array([0.01, 0.005, 0.002]),
        "mann_n": np.array([0.04, 0.035, 0.03]),
        "seg_width": np.array([5.0, 8.0, 12.0]),
        "seg_depth": np.array([0.5, 0.8, 1.2]),
        "hru_segment": np.array([1, 2, 3], dtype=np.int64),
        "hru_elev": np.array([120.0, 110.0, 100.0]),
    }
    metadata = {
        "global": {},
        "nhm_seg": _meta(("nsegment",), "none"),
        "nhm_id": _meta(("nhru",), "none"),
        "tosegment": _meta(("nsegment",), "none"),
        "tosegment_nhm": _meta(("nsegment",), "none"),
        "seg_length": _meta(("nsegment",), "meters"),
        "seg_slope": _meta(("nsegment",), "decimal fraction"),
        "mann_n": _meta(("nsegment",), "seconds / meter ** (1/3)"),
        "seg_width": _meta(("nsegment",), "meter"),
        "seg_depth": _meta(("nsegment",), "meter"),
        "hru_segment": _meta(("nhru",), "none"),
        "hru_elev": _meta(("nhru",), "meters"),
    }
    return Parameters(
        dims=dims, coords=coords, data_vars=data_vars, metadata=metadata
    )


@pytest.mark.domainless
def test_shear_velocity_values():
    from pywatershed.utils.network_hydraulics import G, shear_velocity

    depth = np.array([1.0, 2.0, 0.0])
    slope = np.array([0.001, 0.0, 0.01])
    result = shear_velocity(depth, slope)
    expected = np.array(
        [np.sqrt(G * 1.0 * 0.001), np.sqrt(G * 2.0 * 1.0e-7), 0.0]
    )
    np.testing.assert_allclose(result, expected, rtol=1e-12)


@pytest.mark.domainless
def test_calculate_seg_mid_elevations(synthetic_params):
    from pywatershed.utils.network_hydraulics import (
        calculate_seg_mid_elevations,
    )

    mid, outlet_mid = calculate_seg_mid_elevations(synthetic_params)
    np.testing.assert_allclose(mid, np.array([108.0, 108.0, 101.5]))
    assert outlet_mid == {2: 101.5}
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `cd autotest && /home/rmcd/miniforge3/envs/pws/bin/python -m pytest test_network_hydraulics.py -m domainless -v`
Expected: 2 FAIL with `ModuleNotFoundError: No module named 'pywatershed.utils.network_hydraulics'`

- [ ] **Step 3: Create the module with the two helpers**

Create `pywatershed/utils/network_hydraulics.py`:

```python
"""Network hydraulics helpers and a model-agnostic NetCDF export.

The export carries reach topology, planform geometry, and per-reach,
per-time-step hydraulics (flow, velocity, depth, width, shear velocity)
in SI units for consumers such as 1D network particle trackers. See
``docs/superpowers/specs/2026-09-11-network-hydraulics-export-design.md``.
"""

import numpy as np

from ..base.parameters import Parameters

G = 9.80665
"""Gravitational acceleration (m/s^2)."""

SLOPE_FLOOR = 1.0e-7
"""Minimum slope (m/m), the floor used by PRMS stream temperature."""


def shear_velocity(depth: np.ndarray, slope: np.ndarray) -> np.ndarray:
    """Shear velocity sqrt(g * depth * slope) with the slope floor.

    Args:
        depth: flow depth (m), any shape.
        slope: channel slope (m/m), broadcastable to ``depth``.

    Returns:
        Shear velocity (m/s), same shape as the broadcast of the inputs.
    """
    slope_floored = np.maximum(np.asarray(slope, dtype=float), SLOPE_FLOOR)
    return np.sqrt(G * np.asarray(depth, dtype=float) * slope_floored)


def calculate_seg_mid_elevations(
    parameters: Parameters,
) -> tuple[np.ndarray, dict[int, float]]:
    """Elevation at the midpoint of each segment, walked up from outlets.

    Each outlet's downstream end takes the lowest elevation of the HRUs
    that drain to it; every segment's upstream end is its downstream
    end plus ``seg_slope * seg_length``; the midpoint is the mean of the
    two. Requires ``tosegment``, ``seg_slope``, ``seg_length``,
    ``hru_segment`` and ``hru_elev``.

    Args:
        parameters: a Parameters object with the parameters above.

    Returns:
        ``(seg_mid_elevation, outlet_mid_elevation)`` where the first is
        an array over segments (m) and the second maps each outlet's
        zero-based segment index to its midpoint elevation (m).
    """
    params = parameters.parameters
    seg_dy = params["seg_slope"] * params["seg_length"]
    nseg = len(seg_dy)
    seg_y = np.full(nseg, np.nan)  # elevation at the upstream end
    tosegment0 = params["tosegment"] - 1
    is_outflow = -1
    hru_seg = params["hru_segment"] - 1
    hru_elev = params["hru_elev"]
    outlet_mid = {}

    for ss in range(nseg):
        if not np.isnan(seg_y[ss]):
            continue
        # walk downstream until a solved segment or an outlet
        chain = []
        ind = ss
        while ind != is_outflow and np.isnan(seg_y[ind]):
            chain.append(ind)
            ind = tosegment0[ind]
        # solve from the most downstream unsolved segment upward
        for seg in reversed(chain):
            down = tosegment0[seg]
            if down == is_outflow:
                outlet_hrus = np.where(hru_seg == seg)
                outlet_elev = hru_elev[outlet_hrus].min()
                seg_y[seg] = seg_dy[seg] + outlet_elev
                outlet_mid[int(seg)] = float(seg_y[seg] - seg_dy[seg] / 2)
            else:
                seg_y[seg] = seg_dy[seg] + seg_y[down]

    return seg_y - seg_dy / 2, outlet_mid
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `cd autotest && /home/rmcd/miniforge3/envs/pws/bin/python -m pytest test_network_hydraulics.py -m domainless -v`
Expected: 2 PASS

- [ ] **Step 5: Refactor `MmrToMf6Dfw` to delegate**

In `pywatershed/utils/mmr_to_mf6_dfw.py` add the import near the other relative imports (after `from .optional_import import import_optional_dependency`):

```python
from .network_hydraulics import calculate_seg_mid_elevations
```

Replace the whole method `_calculate_seg_mid_elevations(self, check=False)` (currently lines 807-914, from `def _calculate_seg_mid_elevations` through its `return`) with:

```python
    def _calculate_seg_mid_elevations(self, check=False):
        mid, outlet_mid = calculate_seg_mid_elevations(self.parameters)
        # constant head at each outlet: 1 m above the midpoint elevation
        self._outlet_chds = {kk: 1.0 + vv for kk, vv in outlet_mid.items()}
        if check:
            params = self.parameters.parameters
            seg_dy = params["seg_slope"] * params["seg_length"]
            tosegment0 = params["tosegment"] - 1
            for ss in range(len(seg_dy)):
                down = tosegment0[ss]
                if down == -1:
                    continue
                # upstream end of ss equals upstream end of down + rise
                up_ss = mid[ss] + seg_dy[ss] / 2
                up_down = mid[down] + seg_dy[down] / 2
                assert abs((up_ss - up_down) - seg_dy[ss]) < 1.0e-7
        self._seg_mid_elevation = mid
        return
```

Leave the two call sites (`self._calculate_seg_mid_elevations(check=False)` near line 419 and the `_outlet_chds` use near line 786) unchanged.

- [ ] **Step 6: Run the existing MF6 converter tests and the new tests**

Run: `cd autotest && /home/rmcd/miniforge3/envs/pws/bin/python -m pytest test_mmr_to_mf6_dfw.py test_network_hydraulics.py -m domainless -v`
Expected: the new 2 PASS; `test_mmr_to_mf6_dfw.py` domainless tests PASS or SKIP with "mf6 binary not available" (skips are acceptable locally; the refactor is also covered by Task 1's elevation test).

Also confirm the DRB walk is unchanged against the pre-refactor code by running this once in the scratchpad (compares against `git stash`-free logic: the old method is reproduced inline from the git history):

```bash
cd /home/rmcd/projects/pywatershed && git show develop:pywatershed/utils/mmr_to_mf6_dfw.py > /tmp/claude-1000/-home-rmcd-projects-pywatershed/fc5eaa5f-83c2-41d5-bdd0-c94cf01de40f/scratchpad/old_mmr.py
/home/rmcd/miniforge3/envs/pws/bin/python - <<'EOF'
import sys, types, numpy as np, pywatershed as pws
sys.path.insert(0, "/tmp/claude-1000/-home-rmcd-projects-pywatershed/fc5eaa5f-83c2-41d5-bdd0-c94cf01de40f/scratchpad")
import old_mmr
from pywatershed.utils.network_hydraulics import calculate_seg_mid_elevations
p = pws.parameters.PrmsParameters.load(pws.constants.__pywatershed_root__ / "data/drb_2yr/myparam.param")
shim = types.SimpleNamespace(parameters=p, _nsegment=p.dims["nsegment"])
old_mmr.MmrToMf6Dfw._calculate_seg_mid_elevations(shim, check=False)
new_mid, new_out = calculate_seg_mid_elevations(p)
np.testing.assert_allclose(shim._seg_mid_elevation, new_mid)
assert {k: 1.0 + v for k, v in new_out.items()} == shim._outlet_chds
print("DRB mid elevations identical:", len(new_out), "outlets")
EOF
```
Expected: prints `DRB mid elevations identical: 6 outlets`.

- [ ] **Step 7: Lint, format, commit**

```bash
cd /home/rmcd/projects/pywatershed && ruff format pywatershed/utils/network_hydraulics.py pywatershed/utils/mmr_to_mf6_dfw.py autotest/test_network_hydraulics.py && ruff check pywatershed autotest
git add pywatershed/utils/network_hydraulics.py pywatershed/utils/mmr_to_mf6_dfw.py autotest/test_network_hydraulics.py
git commit -m "refactor: shared shear velocity and segment mid-elevation helpers

Move the outlet-upward elevation walk out of MmrToMf6Dfw into
pywatershed.utils.network_hydraulics and add shear_velocity.

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 2: `at_a_station_hydraulic_geometry`

**Files:**
- Create: `pywatershed/utils/hydraulic_geometry.py`
- Test: `autotest/test_hydraulic_geometry_utils.py`

**Interfaces:**
- Consumes: `SLOPE_FLOOR` from `pywatershed.utils.network_hydraulics`; `CFS_TO_CMS` is not needed here (bankfull flow is computed in m^3/s directly).
- Produces: `at_a_station_hydraulic_geometry(parameters: Parameters, width_exp: float = 0.26, depth_exp: float = 0.40, return_bankfull: bool = False) -> Parameters | tuple[Parameters, dict]`. The dict has keys `bankfull_flow` (m^3/s, array), `bankfull_velocity` (m/s, array), `velocity_exp` (float).

- [ ] **Step 1: Write the failing tests**

Create `autotest/test_hydraulic_geometry_utils.py`:

```python
import numpy as np
import pytest

import pywatershed as pws
from pywatershed.base.parameters import Parameters
from pywatershed.hydrology.prms_hydraulic_geometry import CFS_TO_CMS

NSEG = 3
NHRU = 3


def _meta(dims: tuple, units: str) -> dict:
    return {"dims": dims, "attrs": {"units": units}}


@pytest.fixture
def synthetic_params() -> Parameters:
    dims = {"nsegment": NSEG, "nhru": NHRU}
    coords = {
        "nhm_seg": np.array([101, 102, 103], dtype=np.int64),
        "nhm_id": np.array([1, 2, 3], dtype=np.int64),
    }
    data_vars = {
        "tosegment": np.array([3, 3, 0], dtype=np.int64),
        "tosegment_nhm": np.array([103, 103, 0], dtype=np.int64),
        "seg_length": np.array([1000.0, 2000.0, 1500.0]),
        "seg_slope": np.array([0.01, 0.005, 0.002]),
        "mann_n": np.array([0.04, 0.035, 0.03]),
        "seg_width": np.array([5.0, 8.0, 12.0]),
        "seg_depth": np.array([0.5, 0.8, 1.2]),
        "hru_segment": np.array([1, 2, 3], dtype=np.int64),
        "hru_elev": np.array([120.0, 110.0, 100.0]),
    }
    metadata = {
        "global": {},
        "nhm_seg": _meta(("nsegment",), "none"),
        "nhm_id": _meta(("nhru",), "none"),
        "tosegment": _meta(("nsegment",), "none"),
        "tosegment_nhm": _meta(("nsegment",), "none"),
        "seg_length": _meta(("nsegment",), "meters"),
        "seg_slope": _meta(("nsegment",), "decimal fraction"),
        "mann_n": _meta(("nsegment",), "seconds / meter ** (1/3)"),
        "seg_width": _meta(("nsegment",), "meter"),
        "seg_depth": _meta(("nsegment",), "meter"),
        "hru_segment": _meta(("nhru",), "none"),
        "hru_elev": _meta(("nhru",), "meters"),
    }
    return Parameters(
        dims=dims, coords=coords, data_vars=data_vars, metadata=metadata
    )


def _manning_bankfull(w, d, s, n):
    area = w * d
    radius = area / (w + 2 * d)
    return area * radius ** (2.0 / 3.0) * np.sqrt(s) / n


@pytest.mark.domainless
def test_at_a_station_hand_computed(synthetic_params):
    from pywatershed.utils.hydraulic_geometry import (
        at_a_station_hydraulic_geometry,
    )

    new, bankfull = at_a_station_hydraulic_geometry(
        synthetic_params, return_bankfull=True
    )
    p = synthetic_params.parameters
    q_bf = _manning_bankfull(
        p["seg_width"], p["seg_depth"], p["seg_slope"], p["mann_n"]
    )
    np.testing.assert_allclose(bankfull["bankfull_flow"], q_bf, rtol=1e-12)
    np.testing.assert_allclose(
        bankfull["bankfull_velocity"],
        q_bf / (p["seg_width"] * p["seg_depth"]),
        rtol=1e-12,
    )
    assert bankfull["velocity_exp"] == pytest.approx(0.34)

    newp = new.parameters
    np.testing.assert_allclose(newp["width_m"], 0.26)
    np.testing.assert_allclose(newp["depth_m"], 0.40)
    np.testing.assert_allclose(
        newp["width_alpha"], p["seg_width"] / q_bf**0.26, rtol=1e-12
    )
    np.testing.assert_allclose(
        newp["depth_alpha"], p["seg_depth"] / q_bf**0.40, rtol=1e-12
    )
    # the process formula (alpha * Q_cms ** m) returns bankfull geometry
    np.testing.assert_allclose(
        newp["width_alpha"] * q_bf ** newp["width_m"],
        p["seg_width"],
        rtol=1e-12,
    )
    np.testing.assert_allclose(
        newp["depth_alpha"] * q_bf ** newp["depth_m"],
        p["seg_depth"],
        rtol=1e-12,
    )
    # the other parameters are carried over untouched
    np.testing.assert_array_equal(newp["seg_length"], p["seg_length"])
    assert new.dims["nsegment"] == NSEG


@pytest.mark.domainless
def test_at_a_station_exponent_override(synthetic_params):
    from pywatershed.utils.hydraulic_geometry import (
        at_a_station_hydraulic_geometry,
    )

    new, bankfull = at_a_station_hydraulic_geometry(
        synthetic_params, width_exp=0.1, depth_exp=0.5, return_bankfull=True
    )
    np.testing.assert_allclose(new.parameters["width_m"], 0.1)
    np.testing.assert_allclose(new.parameters["depth_m"], 0.5)
    assert bankfull["velocity_exp"] == pytest.approx(0.4)


@pytest.mark.domainless
def test_at_a_station_returns_parameters_only_by_default(synthetic_params):
    from pywatershed.utils.hydraulic_geometry import (
        at_a_station_hydraulic_geometry,
    )

    new = at_a_station_hydraulic_geometry(synthetic_params)
    assert isinstance(new, Parameters)
    assert "depth_alpha" in new.parameters


@pytest.mark.domainless
def test_at_a_station_does_not_mutate_input(synthetic_params):
    from pywatershed.utils.hydraulic_geometry import (
        at_a_station_hydraulic_geometry,
    )

    _ = at_a_station_hydraulic_geometry(synthetic_params)
    assert "depth_alpha" not in synthetic_params.parameters
    assert "width_alpha" not in synthetic_params.parameters


@pytest.mark.domainless
def test_at_a_station_overwrites_existing_geometry(synthetic_params):
    from pywatershed.utils.hydraulic_geometry import (
        at_a_station_hydraulic_geometry,
    )

    dd = synthetic_params.to_dd()
    dd.data_vars["width_alpha"] = np.full(NSEG, 99.0)
    dd.metadata["width_alpha"] = _meta(("nsegment",), "unknown")
    dd.data_vars["width_m"] = np.full(NSEG, 0.015)
    dd.metadata["width_m"] = _meta(("nsegment",), "none")
    with_old = Parameters(**dd.data)

    new = at_a_station_hydraulic_geometry(with_old)
    assert not np.any(new.parameters["width_alpha"] == 99.0)
    np.testing.assert_allclose(new.parameters["width_m"], 0.26)


@pytest.mark.domainless
def test_at_a_station_slope_floor(synthetic_params):
    from pywatershed.utils.hydraulic_geometry import (
        at_a_station_hydraulic_geometry,
    )
    from pywatershed.utils.network_hydraulics import SLOPE_FLOOR

    dd = synthetic_params.to_dd()
    dd.data_vars["seg_slope"][:] = 0.0
    flat = Parameters(**dd.data)
    _, bankfull = at_a_station_hydraulic_geometry(flat, return_bankfull=True)
    p = flat.parameters
    expected = _manning_bankfull(
        p["seg_width"], p["seg_depth"], SLOPE_FLOOR, p["mann_n"]
    )
    np.testing.assert_allclose(bankfull["bankfull_flow"], expected)


@pytest.mark.domainless
@pytest.mark.parametrize("name", ["seg_width", "seg_depth", "mann_n"])
def test_at_a_station_nonpositive_raises(synthetic_params, name):
    from pywatershed.utils.hydraulic_geometry import (
        at_a_station_hydraulic_geometry,
    )

    dd = synthetic_params.to_dd()
    dd.data_vars[name][1] = 0.0
    bad = Parameters(**dd.data)
    with pytest.raises(ValueError, match=f"{name}.*1 segment"):
        at_a_station_hydraulic_geometry(bad)


@pytest.mark.domainless
def test_at_a_station_missing_raises(synthetic_params):
    from pywatershed.utils.hydraulic_geometry import (
        at_a_station_hydraulic_geometry,
    )

    dd = synthetic_params.to_dd()
    del dd.data_vars["seg_depth"]
    del dd.metadata["seg_depth"]
    missing = Parameters(**dd.data)
    with pytest.raises(ValueError, match="seg_depth"):
        at_a_station_hydraulic_geometry(missing)


@pytest.mark.domainless
def test_at_a_station_drb_bankfull_round_trip():
    from pywatershed.utils.hydraulic_geometry import (
        at_a_station_hydraulic_geometry,
    )

    param_file = (
        pws.constants.__pywatershed_root__ / "data/drb_2yr/myparam.param"
    )
    params = pws.parameters.PrmsParameters.load(param_file)
    new, bankfull = at_a_station_hydraulic_geometry(
        params, return_bankfull=True
    )
    p = params.parameters
    q_cms = bankfull["bankfull_flow"]
    # mirror PRMSHydraulicGeometryFull: flow_cms = seg_outflow_cfs * CFS_TO_CMS
    q_cfs = q_cms / CFS_TO_CMS
    flow_cms = q_cfs * CFS_TO_CMS
    width = new.parameters["width_alpha"] * flow_cms ** new.parameters["width_m"]
    depth = new.parameters["depth_alpha"] * flow_cms ** new.parameters["depth_m"]
    np.testing.assert_allclose(width, p["seg_width"], rtol=1e-10)
    np.testing.assert_allclose(depth, p["seg_depth"], rtol=1e-10)
    assert np.all(q_cms > 0)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `cd autotest && /home/rmcd/miniforge3/envs/pws/bin/python -m pytest test_hydraulic_geometry_utils.py -m domainless -v`
Expected: all FAIL with `ModuleNotFoundError: No module named 'pywatershed.utils.hydraulic_geometry'`

- [ ] **Step 3: Implement the function**

Create `pywatershed/utils/hydraulic_geometry.py`:

```python
"""At-a-station hydraulic geometry anchored on bankfull parameters."""

import numpy as np

from ..base.parameters import Parameters
from .network_hydraulics import SLOPE_FLOOR

_REQUIRED = ("seg_width", "seg_depth", "seg_slope", "mann_n")

_NEW_META = {
    "width_alpha": {
        "desc": "Alpha coefficient in power function for width calculation",
        "units": "unknown",
    },
    "width_m": {
        "desc": "M value in power function for width calculation",
        "units": "none",
    },
    "depth_alpha": {
        "desc": "Alpha coefficient in power function for depth calculation",
        "units": "meters",
    },
    "depth_m": {
        "desc": "M value in power function for depth calculation",
        "units": "none",
    },
}


def at_a_station_hydraulic_geometry(
    parameters: Parameters,
    width_exp: float = 0.26,
    depth_exp: float = 0.40,
    return_bankfull: bool = False,
) -> Parameters | tuple[Parameters, dict]:
    """Derive power-law hydraulic geometry from bankfull channel parameters.

    Bankfull discharge per segment is computed with Manning's equation
    for a rectangular section of bankfull width ``seg_width`` and depth
    ``seg_depth`` using ``seg_slope`` (floored at 1e-7) and ``mann_n``.
    Width and depth power laws of the form ``alpha * Q**m`` (``Q`` in
    m^3/s, as evaluated by :class:`PRMSHydraulicGeometryFull`) are then
    anchored through that bankfull point with at-a-station exponents.
    The defaults are the Leopold and Maddock (1953) averages; the implied
    velocity exponent is ``1 - width_exp - depth_exp``.

    Args:
        parameters: Parameters containing ``seg_width``, ``seg_depth``,
            ``seg_slope`` and ``mann_n`` on the ``nsegment`` dimension.
        width_exp: at-a-station width exponent.
        depth_exp: at-a-station depth exponent.
        return_bankfull: also return diagnostics.

    Returns:
        A new Parameters object, a copy of the input with ``width_alpha``,
        ``width_m``, ``depth_alpha`` and ``depth_m`` set (overwriting any
        existing values). With ``return_bankfull=True`` a tuple of that
        object and a dict with ``bankfull_flow`` (m^3/s),
        ``bankfull_velocity`` (m/s) and ``velocity_exp``.

    Raises:
        ValueError: a required parameter is missing or not positive.
    """
    params = parameters.parameters
    missing = [kk for kk in _REQUIRED if kk not in params]
    if missing:
        raise ValueError(
            "at_a_station_hydraulic_geometry requires parameters "
            f"{list(_REQUIRED)}; missing {missing}"
        )

    width = np.asarray(params["seg_width"], dtype=float)
    depth = np.asarray(params["seg_depth"], dtype=float)
    slope = np.maximum(np.asarray(params["seg_slope"], dtype=float), SLOPE_FLOOR)
    mann_n = np.asarray(params["mann_n"], dtype=float)

    for name, arr in (
        ("seg_width", width),
        ("seg_depth", depth),
        ("mann_n", mann_n),
    ):
        n_bad = int(np.sum(~(arr > 0.0)))
        if n_bad:
            raise ValueError(
                f"Parameter {name} must be positive; "
                f"{n_bad} segment(s) are not"
            )

    area = width * depth
    radius = area / (width + 2.0 * depth)
    q_bf = area * radius ** (2.0 / 3.0) * np.sqrt(slope) / mann_n

    dd = parameters.to_dd()
    nseg = dd.dims["nsegment"]
    new_vars = {
        "width_alpha": width / q_bf**width_exp,
        "width_m": np.full(nseg, width_exp),
        "depth_alpha": depth / q_bf**depth_exp,
        "depth_m": np.full(nseg, depth_exp),
    }
    for name, values in new_vars.items():
        dd.data_vars[name] = values
        dd.metadata[name] = {
            "dims": ("nsegment",),
            "attrs": dict(_NEW_META[name]),
        }
    new_params = Parameters(**dd.data)

    if not return_bankfull:
        return new_params
    bankfull = {
        "bankfull_flow": q_bf,
        "bankfull_velocity": q_bf / area,
        "velocity_exp": 1.0 - width_exp - depth_exp,
    }
    return new_params, bankfull
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `cd autotest && /home/rmcd/miniforge3/envs/pws/bin/python -m pytest test_hydraulic_geometry_utils.py -m domainless -v`
Expected: 9 PASS (the parametrized test counts as 3)

If `Parameters(**dd.data)` fails validation because `dd.data` lacks an `encoding` entry for the new variables, that is fine: the validator only requires encoding keys to be a subset of variable keys. If `dd.dims` is not subscriptable, use `dd.data["dims"]["nsegment"]`.

- [ ] **Step 5: Lint, format, commit**

```bash
cd /home/rmcd/projects/pywatershed && ruff format pywatershed/utils/hydraulic_geometry.py autotest/test_hydraulic_geometry_utils.py && ruff check pywatershed autotest
git add pywatershed/utils/hydraulic_geometry.py autotest/test_hydraulic_geometry_utils.py
git commit -m "feat: at-a-station hydraulic geometry from bankfull parameters

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 3: `export_network_hydraulics` without planform geometry

**Files:**
- Modify: `pywatershed/utils/network_hydraulics.py`
- Test: `autotest/test_network_hydraulics.py`

**Interfaces:**
- Consumes: `shear_velocity`, `calculate_seg_mid_elevations` (Task 1); `CFS_TO_CMS` from `pywatershed.hydrology.prms_hydraulic_geometry`.
- Produces: `export_network_hydraulics(parameters, run_dir, out_file, segment_shp_file=None, shp_id_col="nsegment_v", connect_tol=1.0, start_time=None, end_time=None) -> pathlib.Path`; module constants `REQUIRED_RUN_VARS`, `OPTIONAL_RUN_VARS`.

- [ ] **Step 1: Write the failing tests**

Append to `autotest/test_network_hydraulics.py` (imports at top of file must gain `import pathlib as pl`, `import xarray as xr`):

```python
NTIME = 4
TIMES = np.arange(
    np.datetime64("1979-01-01"), np.datetime64("1979-01-05")
).astype("datetime64[ns]")


def _write_run_var(run_dir, name, values, units, nhm_seg):
    da = xr.DataArray(
        values,
        dims=("time", "nhm_seg"),
        coords={"time": TIMES, "nhm_seg": nhm_seg},
        name=name,
        attrs={"units": units},
    )
    da.to_netcdf(run_dir / f"{name}.nc")


@pytest.fixture
def synthetic_run_dir(tmp_path, synthetic_params) -> pl.Path:
    """A fake pywatershed output directory for the three-reach network."""
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    nhm_seg = synthetic_params.parameters["nhm_seg"]
    base = np.array([[10.0, 20.0, 35.0]])  # cfs, outlet sums the two
    ramp = np.arange(1, NTIME + 1)[:, None]  # 1..4
    outflow = base * ramp
    inflow = outflow * 0.9
    width = np.array([[4.0, 6.0, 10.0]]) * np.ones((NTIME, 1))
    depth = np.array([[0.3, 0.5, 0.9]]) * ramp * 0.5
    velocity = outflow * 0.028316847 / (width * depth)
    res_time = width * depth * synthetic_params.parameters[
        "seg_length"
    ] / (outflow * 0.028316847)
    _write_run_var(run_dir, "seg_outflow", outflow, "cfs", nhm_seg)
    _write_run_var(run_dir, "seg_inflow", inflow, "cfs", nhm_seg)
    _write_run_var(run_dir, "seg_flow_width", width, "meters", nhm_seg)
    _write_run_var(run_dir, "seg_flow_depth", depth, "meters", nhm_seg)
    _write_run_var(
        run_dir, "seg_flow_velocity", velocity, "meters per second", nhm_seg
    )
    _write_run_var(run_dir, "seg_res_time", res_time, "seconds", nhm_seg)
    return run_dir


@pytest.mark.domainless
def test_export_static_fields(synthetic_params, synthetic_run_dir, tmp_path):
    from pywatershed.utils.network_hydraulics import (
        export_network_hydraulics,
    )

    out = export_network_hydraulics(
        synthetic_params, synthetic_run_dir, tmp_path / "net.nc"
    )
    assert out == tmp_path / "net.nc"
    ds = xr.open_dataset(out)
    p = synthetic_params.parameters
    assert ds.sizes["reach"] == NSEG
    assert ds.sizes["time"] == NTIME
    np.testing.assert_array_equal(ds["reach_id"], p["nhm_seg"])
    np.testing.assert_array_equal(ds["to_id"], p["tosegment_nhm"])
    np.testing.assert_array_equal(ds["to_index"], np.array([2, 2, -1]))
    np.testing.assert_array_equal(ds["is_outlet"], np.array([0, 0, 1]))
    np.testing.assert_array_equal(ds["length"], p["seg_length"])
    np.testing.assert_array_equal(ds["slope"], p["seg_slope"])
    np.testing.assert_array_equal(ds["mann_n"], p["mann_n"])
    np.testing.assert_array_equal(ds["bankfull_width"], p["seg_width"])
    np.testing.assert_array_equal(ds["bankfull_depth"], p["seg_depth"])
    np.testing.assert_allclose(
        ds["elevation_mid"], np.array([108.0, 108.0, 101.5])
    )
    assert "vertex" not in ds.dims
    assert "x_mid" not in ds
    assert ds["length"].attrs["units"] == "m"
    assert ds["to_index"].attrs["source_name"] == "tosegment"
    assert ds.attrs["source_model"] == "pywatershed PRMS"
    assert "pywatershed_version" in ds.attrs
    assert ds.attrs["n_unconnected"] == -1  # no polyline supplied
    ds.close()


@pytest.mark.domainless
def test_export_time_varying_fields(
    synthetic_params, synthetic_run_dir, tmp_path
):
    from pywatershed.hydrology.prms_hydraulic_geometry import CFS_TO_CMS
    from pywatershed.utils.network_hydraulics import (
        export_network_hydraulics,
        shear_velocity,
    )

    out = export_network_hydraulics(
        synthetic_params, synthetic_run_dir, tmp_path / "net.nc"
    )
    ds = xr.open_dataset(out)
    src = {
        nm: xr.open_dataarray(synthetic_run_dir / f"{nm}.nc").load()
        for nm in [
            "seg_outflow",
            "seg_inflow",
            "seg_flow_width",
            "seg_flow_depth",
            "seg_flow_velocity",
            "seg_res_time",
        ]
    }
    np.testing.assert_array_equal(ds["time"], TIMES)
    np.testing.assert_allclose(
        ds["flow_out"], src["seg_outflow"].values * CFS_TO_CMS
    )
    np.testing.assert_allclose(
        ds["flow_in"], src["seg_inflow"].values * CFS_TO_CMS
    )
    np.testing.assert_allclose(ds["width"], src["seg_flow_width"].values)
    np.testing.assert_allclose(ds["depth"], src["seg_flow_depth"].values)
    np.testing.assert_allclose(
        ds["velocity"], src["seg_flow_velocity"].values
    )
    np.testing.assert_allclose(
        ds["residence_time"], src["seg_res_time"].values
    )
    expected_ustar = shear_velocity(
        src["seg_flow_depth"].values,
        synthetic_params.parameters["seg_slope"][None, :],
    )
    np.testing.assert_allclose(ds["ustar"], expected_ustar)
    assert ds["flow_out"].attrs["units"] == "m3 s-1"
    assert ds["flow_out"].attrs["source_name"] == "seg_outflow"
    assert ds["ustar"].attrs["method"] == "sqrt(g*depth*slope)"
    assert ds["velocity"].attrs["method"] == "power_law_at_a_station"
    assert ds["flow_out"].dims == ("time", "reach")
    assert "water_temperature" not in ds
    ds.close()


@pytest.mark.domainless
def test_export_optional_temperature(
    synthetic_params, synthetic_run_dir, tmp_path
):
    from pywatershed.utils.network_hydraulics import (
        export_network_hydraulics,
    )

    temp = np.full((NTIME, NSEG), 12.5)
    _write_run_var(
        synthetic_run_dir,
        "seg_tave_water",
        temp,
        "degrees Celsius",
        synthetic_params.parameters["nhm_seg"],
    )
    ds = xr.open_dataset(
        export_network_hydraulics(
            synthetic_params, synthetic_run_dir, tmp_path / "net.nc"
        )
    )
    np.testing.assert_allclose(ds["water_temperature"], temp)
    assert ds["water_temperature"].attrs["units"] == "degC"
    ds.close()


@pytest.mark.domainless
def test_export_time_subset(synthetic_params, synthetic_run_dir, tmp_path):
    from pywatershed.utils.network_hydraulics import (
        export_network_hydraulics,
    )

    ds = xr.open_dataset(
        export_network_hydraulics(
            synthetic_params,
            synthetic_run_dir,
            tmp_path / "net.nc",
            start_time=np.datetime64("1979-01-02"),
            end_time=np.datetime64("1979-01-03"),
        )
    )
    assert ds.sizes["time"] == 2
    np.testing.assert_array_equal(ds["time"], TIMES[1:3])
    ds.close()


@pytest.mark.domainless
def test_export_missing_files_raise(synthetic_params, synthetic_run_dir, tmp_path):
    from pywatershed.utils.network_hydraulics import (
        export_network_hydraulics,
    )

    (synthetic_run_dir / "seg_flow_depth.nc").unlink()
    (synthetic_run_dir / "seg_res_time.nc").unlink()
    with pytest.raises(FileNotFoundError) as excinfo:
        export_network_hydraulics(
            synthetic_params, synthetic_run_dir, tmp_path / "net.nc"
        )
    assert "seg_flow_depth" in str(excinfo.value)
    assert "seg_res_time" in str(excinfo.value)


@pytest.mark.domainless
def test_export_reach_order_mismatch_raises(
    synthetic_params, synthetic_run_dir, tmp_path
):
    from pywatershed.utils.network_hydraulics import (
        export_network_hydraulics,
    )

    path = synthetic_run_dir / "seg_outflow.nc"
    with xr.open_dataarray(path) as opened:
        da = opened.load()  # close the file before rewriting it (Windows)
    da = da.assign_coords(nhm_seg=np.array([103, 102, 101]))
    da.to_netcdf(path)
    with pytest.raises(ValueError, match="nhm_seg"):
        export_network_hydraulics(
            synthetic_params, synthetic_run_dir, tmp_path / "net.nc"
        )
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `cd autotest && /home/rmcd/miniforge3/envs/pws/bin/python -m pytest test_network_hydraulics.py -m domainless -v`
Expected: the 6 new tests FAIL with `ImportError: cannot import name 'export_network_hydraulics'`; the 2 Task-1 tests still PASS.

- [ ] **Step 3: Implement the exporter (no polyline block yet)**

Append to `pywatershed/utils/network_hydraulics.py`. Add these imports at the top of the module (keep them sorted; `ruff` enforces isort):

```python
import datetime
import pathlib as pl
from warnings import warn

import numpy as np
import xarray as xr

from ..base.parameters import Parameters
from ..hydrology.prms_hydraulic_geometry import CFS_TO_CMS
from ..version import __version__
```

Check that `pywatershed/version.py` defines `__version__` (run `grep -n __version__ pywatershed/version.py`); if the attribute lives elsewhere, use `from .. import __version__` inside the function body to avoid a circular import.

Then append:

```python
REQUIRED_RUN_VARS = (
    "seg_outflow",
    "seg_inflow",
    "seg_flow_width",
    "seg_flow_depth",
    "seg_flow_velocity",
    "seg_res_time",
)
"""Output variables the exporter requires in ``run_dir``."""

OPTIONAL_RUN_VARS = ("seg_tave_water",)
"""Output variables the exporter includes when present in ``run_dir``."""

_GEOMETRY_METHOD = "power_law_at_a_station"

# export name: (source name, units, long_name, method, scale factor)
_TIME_VARS = {
    "flow_out": ("seg_outflow", "m3 s-1", "flow leaving the reach", "routed", CFS_TO_CMS),
    "flow_in": ("seg_inflow", "m3 s-1", "flow entering the reach", "routed", CFS_TO_CMS),
    "velocity": ("seg_flow_velocity", "m s-1", "mean flow velocity", _GEOMETRY_METHOD, 1.0),
    "depth": ("seg_flow_depth", "m", "mean flow depth", _GEOMETRY_METHOD, 1.0),
    "width": ("seg_flow_width", "m", "flow width", _GEOMETRY_METHOD, 1.0),
    "residence_time": ("seg_res_time", "s", "mean residence time of water in the reach", "area*length/flow_out", 1.0),
    "water_temperature": ("seg_tave_water", "degC", "mean water temperature", "PRMS stream temperature", 1.0),
}


def _read_run_vars(
    run_dir: pl.Path,
    reach_id: np.ndarray,
    start_time,
    end_time,
) -> dict[str, xr.DataArray]:
    run_dir = pl.Path(run_dir)
    missing = [
        nm for nm in REQUIRED_RUN_VARS if not (run_dir / f"{nm}.nc").exists()
    ]
    if missing:
        raise FileNotFoundError(
            f"Required output files missing from {run_dir}: {missing}"
        )
    names = list(REQUIRED_RUN_VARS) + [
        nm for nm in OPTIONAL_RUN_VARS if (run_dir / f"{nm}.nc").exists()
    ]
    result = {}
    for nm in names:
        da = xr.open_dataarray(run_dir / f"{nm}.nc").load()
        if "nhm_seg" in da.coords and not np.array_equal(
            da["nhm_seg"].values, reach_id
        ):
            raise ValueError(
                f"{nm}.nc coordinate nhm_seg does not match the "
                "parameters' nhm_seg order"
            )
        if start_time is not None or end_time is not None:
            da = da.sel(time=slice(start_time, end_time))
        result[nm] = da
    return result


def export_network_hydraulics(
    parameters: Parameters,
    run_dir: pl.Path,
    out_file: pl.Path,
    segment_shp_file: pl.Path | None = None,
    shp_id_col: str = "nsegment_v",
    connect_tol: float = 1.0,
    start_time: np.datetime64 | None = None,
    end_time: np.datetime64 | None = None,
) -> pl.Path:
    """Write a model-agnostic network hydraulics NetCDF from a PRMS run.

    The file carries reach topology, optional planform polylines, and
    per-reach time series of flow, velocity, depth, width, shear
    velocity and residence time in SI units, for consumers such as 1D
    network particle trackers. Hydraulics are taken from the
    :class:`PRMSHydraulicGeometryFull` outputs in ``run_dir``; only shear
    velocity is computed here.

    Args:
        parameters: the run's parameters (needs ``nhm_seg``,
            ``tosegment``, ``tosegment_nhm``, ``seg_length``,
            ``seg_slope``, ``mann_n``, ``seg_width``, ``seg_depth``,
            ``hru_segment``, ``hru_elev``).
        run_dir: pywatershed NetCDF output directory containing
            ``seg_outflow``, ``seg_inflow``, ``seg_flow_width``,
            ``seg_flow_depth``, ``seg_flow_velocity`` and ``seg_res_time``
            (``seg_tave_water`` is included when present).
        out_file: path of the NetCDF file to write.
        segment_shp_file: optional shapefile of segment LineStrings; adds
            the ``vertex`` block and reach midpoints.
        shp_id_col: shapefile column holding ``nhm_seg`` identifiers.
        connect_tol: distance (CRS units) within which a reach's last
            vertex must meet its downstream reach's first vertex.
        start_time: optional first time to include.
        end_time: optional last time to include.

    Returns:
        ``out_file`` as a Path.
    """
    params = parameters.parameters
    reach_id = np.asarray(params["nhm_seg"], dtype=np.int64)
    nreach = len(reach_id)
    to_index = (np.asarray(params["tosegment"], dtype=np.int64) - 1).astype(
        np.int32
    )
    is_outlet = (to_index < 0).astype(np.int8)
    elevation_mid, _ = calculate_seg_mid_elevations(parameters)

    run_vars = _read_run_vars(run_dir, reach_id, start_time, end_time)
    time = run_vars["seg_outflow"]["time"].values

    def static(values, dtype, units, long_name, source_name):
        return xr.DataArray(
            np.asarray(values, dtype=dtype),
            dims=("reach",),
            attrs={
                "units": units,
                "long_name": long_name,
                "source_name": source_name,
            },
        )

    data_vars = {
        "reach_id": static(reach_id, np.int64, "-", "reach identifier", "nhm_seg"),
        "to_id": static(params["tosegment_nhm"], np.int64, "-", "downstream reach identifier (0 = outlet)", "tosegment_nhm"),
        "to_index": static(to_index, np.int32, "-", "zero-based index of the downstream reach (-1 = outlet)", "tosegment"),
        "is_outlet": static(is_outlet, np.int8, "-", "1 where the reach drains out of the network", "tosegment"),
        "length": static(params["seg_length"], np.float64, "m", "reach hydraulic length", "seg_length"),
        "slope": static(params["seg_slope"], np.float64, "m m-1", "reach slope", "seg_slope"),
        "mann_n": static(params["mann_n"], np.float64, "s m-1/3", "Manning roughness", "mann_n"),
        "elevation_mid": static(elevation_mid, np.float64, "m", "elevation at the reach midpoint, walked up from outlets", "seg_slope*seg_length, hru_elev"),
        "bankfull_width": static(params["seg_width"], np.float64, "m", "bankfull width", "seg_width"),
        "bankfull_depth": static(params["seg_depth"], np.float64, "m", "bankfull depth", "seg_depth"),
    }

    slope = np.asarray(params["seg_slope"], dtype=float)
    for name, (src, units, long_name, method, scale) in _TIME_VARS.items():
        if src not in run_vars:
            continue
        values = run_vars[src].values * scale
        data_vars[name] = xr.DataArray(
            values,
            dims=("time", "reach"),
            attrs={
                "units": units,
                "long_name": long_name,
                "source_name": src,
                "method": method,
            },
        )
    data_vars["ustar"] = xr.DataArray(
        shear_velocity(data_vars["depth"].values, slope[None, :]),
        dims=("time", "reach"),
        attrs={
            "units": "m s-1",
            "long_name": "shear velocity",
            "source_name": "seg_flow_depth, seg_slope",
            "method": "sqrt(g*depth*slope)",
        },
    )

    n_unconnected = -1
    crs_wkt = ""
    if segment_shp_file is not None:
        poly_vars, n_unconnected, crs_wkt = _polyline_block(
            segment_shp_file, shp_id_col, reach_id, to_index, connect_tol
        )
        data_vars.update(poly_vars)

    ds = xr.Dataset(
        data_vars=data_vars,
        coords={"time": ("time", time)},
        attrs={
            "title": "Network hydraulics for 1D river-network transport",
            "source_model": "pywatershed PRMS",
            "source_model_version": __version__,
            "pywatershed_version": __version__,
            "geometry_method": _GEOMETRY_METHOD,
            "created": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "n_unconnected": n_unconnected,
            "crs_wkt": crs_wkt,
            "conventions_note": (
                "Particle state is (reach index, s) with 0 <= s <= length "
                "from the reach's upstream end. When s exceeds length the "
                "particle moves to to_index carrying the unused fraction of "
                "the time step; to_index == -1 is an outlet. Map position "
                "scales s/length onto the polyline's vertex_dist."
            ),
        },
    )
    out_file = pl.Path(out_file)
    ds.to_netcdf(out_file)
    if n_unconnected > 0:
        warn(
            f"{n_unconnected} reach polyline(s) do not meet their "
            f"downstream reach within {connect_tol}; see the "
            "n_unconnected global attribute"
        )
    return out_file


def _polyline_block(
    segment_shp_file, shp_id_col, reach_id, to_index, connect_tol
):
    raise NotImplementedError("added in the next task")
```

Note for the implementer: the long dictionary/`static(...)` lines above exceed 79 characters; let `ruff format` wrap them. `_polyline_block` is filled in by Task 4; keep the stub so the module imports.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `cd autotest && /home/rmcd/miniforge3/envs/pws/bin/python -m pytest test_network_hydraulics.py -m domainless -v`
Expected: 8 PASS

- [ ] **Step 5: Lint, format, commit**

```bash
cd /home/rmcd/projects/pywatershed && ruff format pywatershed/utils/network_hydraulics.py autotest/test_network_hydraulics.py && ruff check pywatershed autotest
git add pywatershed/utils/network_hydraulics.py autotest/test_network_hydraulics.py
git commit -m "feat: export_network_hydraulics writes topology and hydraulics

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 4: Polyline block from the segment shapefile

**Files:**
- Modify: `pywatershed/utils/network_hydraulics.py` (replace the `_polyline_block` stub)
- Test: `autotest/test_network_hydraulics.py`

**Interfaces:**
- Consumes: `export_network_hydraulics` (Task 3).
- Produces: `_polyline_block(segment_shp_file, shp_id_col, reach_id, to_index, connect_tol) -> tuple[dict[str, xr.DataArray], int, str]` returning the vertex variables plus `reach_vertex_start`, `reach_vertex_count`, `x_mid`, `y_mid`; the count of unconnected reaches; and the CRS WKT.

- [ ] **Step 1: Write the failing tests**

Append to `autotest/test_network_hydraulics.py` (add `import geopandas as gpd` and `from shapely.geometry import LineString` at the top):

```python
def _write_segments_shp(path, lines, ids, crs="EPSG:5070"):
    gdf = gpd.GeoDataFrame(
        {"nsegment_v": ids, "model_idx": np.arange(1, len(ids) + 1)},
        geometry=[LineString(ll) for ll in lines],
        crs=crs,
    )
    gdf.to_file(path)


@pytest.fixture
def synthetic_lines():
    # reach 0: two-vertex line ending at the junction (0, 0)
    # reach 1: three-vertex line, digitized BACKWARDS (starts at junction)
    # reach 2: outlet, from the junction to (0, -1500)
    return [
        [(-1000.0, 0.0), (0.0, 0.0)],
        [(0.0, 0.0), (500.0, 1000.0), (1000.0, 2000.0)],
        [(0.0, 0.0), (0.0, -1500.0)],
    ]


@pytest.mark.domainless
def test_export_polyline_block(
    synthetic_params, synthetic_run_dir, synthetic_lines, tmp_path
):
    from pywatershed.utils.network_hydraulics import (
        export_network_hydraulics,
    )

    shp = tmp_path / "segs.shp"
    # shuffle the shapefile row order to prove matching is by id
    _write_segments_shp(
        shp,
        [synthetic_lines[2], synthetic_lines[0], synthetic_lines[1]],
        [103, 101, 102],
    )
    ds = xr.open_dataset(
        export_network_hydraulics(
            synthetic_params,
            synthetic_run_dir,
            tmp_path / "net.nc",
            segment_shp_file=shp,
        )
    )
    assert ds.attrs["n_unconnected"] == 0
    assert "5070" in ds.attrs["crs_wkt"] or "Albers" in ds.attrs["crs_wkt"]
    np.testing.assert_array_equal(ds["reach_vertex_count"], [2, 3, 2])
    np.testing.assert_array_equal(ds["reach_vertex_start"], [0, 2, 5])
    assert ds.sizes["vertex"] == 7
    vx = ds["vertex_x"].values
    vy = ds["vertex_y"].values
    vd = ds["vertex_dist"].values
    # reach 0 as digitized
    np.testing.assert_allclose(vx[0:2], [-1000.0, 0.0])
    np.testing.assert_allclose(vd[0:2], [0.0, 1000.0])
    # reach 1 was reversed so that it ends at the junction
    np.testing.assert_allclose(vx[2:5], [1000.0, 500.0, 0.0])
    np.testing.assert_allclose(vy[2:5], [2000.0, 1000.0, 0.0])
    seg = np.hypot(500.0, 1000.0)
    np.testing.assert_allclose(vd[2:5], [0.0, seg, 2 * seg])
    # reach 2 (outlet) untouched
    np.testing.assert_allclose(vy[5:7], [0.0, -1500.0])
    np.testing.assert_allclose(vd[5:7], [0.0, 1500.0])
    # midpoints at half arc length
    np.testing.assert_allclose(ds["x_mid"], [-500.0, 500.0, 0.0])
    np.testing.assert_allclose(ds["y_mid"], [0.0, 1000.0, -750.0])
    assert ds["vertex_dist"].attrs["units"] == "m"
    ds.close()


@pytest.mark.domainless
def test_export_polyline_unconnected_counted(
    synthetic_params, synthetic_run_dir, synthetic_lines, tmp_path
):
    from pywatershed.utils.network_hydraulics import (
        export_network_hydraulics,
    )

    lines = [list(ll) for ll in synthetic_lines]
    lines[0] = [(-1000.0, 50.0), (0.0, 50.0)]  # displaced by 50 m
    shp = tmp_path / "segs.shp"
    _write_segments_shp(shp, lines, [101, 102, 103])
    with pytest.warns(UserWarning, match="1 reach polyline"):
        out = export_network_hydraulics(
            synthetic_params,
            synthetic_run_dir,
            tmp_path / "net.nc",
            segment_shp_file=shp,
        )
    ds = xr.open_dataset(out)
    assert ds.attrs["n_unconnected"] == 1
    ds.close()


@pytest.mark.domainless
def test_export_polyline_id_mismatch_raises(
    synthetic_params, synthetic_run_dir, synthetic_lines, tmp_path
):
    from pywatershed.utils.network_hydraulics import (
        export_network_hydraulics,
    )

    shp = tmp_path / "segs.shp"
    _write_segments_shp(shp, synthetic_lines, [101, 102, 999])
    with pytest.raises(ValueError, match="nsegment_v"):
        export_network_hydraulics(
            synthetic_params,
            synthetic_run_dir,
            tmp_path / "net.nc",
            segment_shp_file=shp,
        )
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `cd autotest && /home/rmcd/miniforge3/envs/pws/bin/python -m pytest test_network_hydraulics.py -m domainless -v -k polyline`
Expected: 3 FAIL with `NotImplementedError: added in the next task`

- [ ] **Step 3: Implement `_polyline_block`**

Replace the stub in `pywatershed/utils/network_hydraulics.py` with:

```python
def _polyline_block(
    segment_shp_file, shp_id_col, reach_id, to_index, connect_tol
):
    """Vertex arrays for each reach's polyline, oriented downstream."""
    from .optional_import import import_optional_dependency

    gpd = import_optional_dependency("geopandas")
    gdf = gpd.read_file(segment_shp_file)
    if shp_id_col not in gdf.columns:
        raise ValueError(
            f"Column {shp_id_col} not in {segment_shp_file}; "
            f"columns are {list(gdf.columns)}"
        )
    shp_ids = gdf[shp_id_col].to_numpy().astype(np.int64)
    if set(shp_ids) != set(reach_id) or len(shp_ids) != len(reach_id):
        raise ValueError(
            f"Shapefile column {shp_id_col} identifiers do not match the "
            "parameters' nhm_seg identifiers"
        )
    order = {rid: ii for ii, rid in enumerate(shp_ids)}
    geoms = [gdf.geometry.iloc[order[rid]] for rid in reach_id]
    coords = [np.asarray(gg.coords, dtype=float)[:, :2] for gg in geoms]

    # orient each line so its end is nearest its downstream reach
    def _min_end_dist(point, line_coords):
        return min(
            np.hypot(*(point - line_coords[0])),
            np.hypot(*(point - line_coords[-1])),
        )

    for ii, down in enumerate(to_index):
        if down < 0:
            continue
        d_start = _min_end_dist(coords[ii][0], coords[down])
        d_end = _min_end_dist(coords[ii][-1], coords[down])
        if d_start < d_end:
            coords[ii] = coords[ii][::-1]

    n_unconnected = 0
    for ii, down in enumerate(to_index):
        if down < 0:
            continue
        if np.hypot(*(coords[ii][-1] - coords[down][0])) > connect_tol:
            n_unconnected += 1

    counts = np.array([len(cc) for cc in coords], dtype=np.int32)
    starts = np.concatenate([[0], np.cumsum(counts)[:-1]]).astype(np.int64)
    vertex_x = np.concatenate([cc[:, 0] for cc in coords])
    vertex_y = np.concatenate([cc[:, 1] for cc in coords])
    dists = []
    x_mid = np.empty(len(coords))
    y_mid = np.empty(len(coords))
    for ii, cc in enumerate(coords):
        step = np.hypot(np.diff(cc[:, 0]), np.diff(cc[:, 1]))
        dist = np.concatenate([[0.0], np.cumsum(step)])
        dists.append(dist)
        half = dist[-1] / 2.0
        x_mid[ii] = np.interp(half, dist, cc[:, 0])
        y_mid[ii] = np.interp(half, dist, cc[:, 1])
    vertex_dist = np.concatenate(dists)

    crs_units = "m"
    crs_wkt = gdf.crs.to_wkt() if gdf.crs is not None else ""

    def vvar(values, dims, units, long_name):
        return xr.DataArray(
            values, dims=dims, attrs={"units": units, "long_name": long_name}
        )

    poly_vars = {
        "vertex_x": vvar(vertex_x, ("vertex",), crs_units, "vertex x in the CRS"),
        "vertex_y": vvar(vertex_y, ("vertex",), crs_units, "vertex y in the CRS"),
        "vertex_dist": vvar(vertex_dist, ("vertex",), "m", "cumulative arc length from the reach's upstream end"),
        "reach_vertex_start": vvar(starts, ("reach",), "-", "index of the reach's first vertex"),
        "reach_vertex_count": vvar(counts, ("reach",), "-", "number of vertices in the reach polyline"),
        "x_mid": vvar(x_mid, ("reach",), crs_units, "x at half the polyline arc length"),
        "y_mid": vvar(y_mid, ("reach",), crs_units, "y at half the polyline arc length"),
    }
    return poly_vars, n_unconnected, crs_wkt
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `cd autotest && /home/rmcd/miniforge3/envs/pws/bin/python -m pytest test_network_hydraulics.py test_hydraulic_geometry_utils.py -m domainless -v`
Expected: 11 + 9 PASS

- [ ] **Step 5: Smoke-run the exporter on the real DRB notebook output**

The `02` notebook has already been run locally, so `examples/02_prms_legacy_models/nhm/` has the routed flows but not the geometry variables. Build a temporary run dir that adds them from the width-only defaults, purely to exercise the DRB shapefile path:

```bash
cd /home/rmcd/projects/pywatershed && /home/rmcd/miniforge3/envs/pws/bin/python - <<'EOF'
import shutil, pathlib as pl, numpy as np, xarray as xr, warnings
import pywatershed as pws
from pywatershed.hydrology.prms_hydraulic_geometry import CFS_TO_CMS
from pywatershed.utils.network_hydraulics import export_network_hydraulics
src = pl.Path("examples/02_prms_legacy_models/nhm")
tmp = pl.Path("/tmp/claude-1000/-home-rmcd-projects-pywatershed/fc5eaa5f-83c2-41d5-bdd0-c94cf01de40f/scratchpad/drb_run")
tmp.mkdir(exist_ok=True)
for nm in ["seg_outflow", "seg_inflow", "seg_res_time", "seg_tave_water"]:
    shutil.copy(src / f"{nm}.nc", tmp / f"{nm}.nc")
p = pws.parameters.PrmsParameters.load(pws.constants.__pywatershed_root__ / "data/drb_2yr/myparam.param")
q = xr.open_dataarray(src / "seg_outflow.nc")
qc = q * CFS_TO_CMS
w = p.parameters["width_alpha"] * qc ** p.parameters["width_m"]
d = 0.27 * qc ** 0.39
v = xr.where(w * d > 1e-6, qc / (w * d), 0.0)
for nm, da, units in [("seg_flow_width", w, "meters"), ("seg_flow_depth", d, "meters"), ("seg_flow_velocity", v, "meters per second")]:
    da.name = nm; da.attrs = {"units": units}; da.to_netcdf(tmp / f"{nm}.nc")
with warnings.catch_warnings(record=True) as rec:
    warnings.simplefilter("always")
    out = export_network_hydraulics(p, tmp, tmp / "drb_network.nc", segment_shp_file=pws.utils.get_gis_dir("drb_2yr") / "Segments_subset.shp")
    print([str(r.message) for r in rec])
ds = xr.open_dataset(out)
print(ds)
print("n_unconnected:", ds.attrs["n_unconnected"], "outlets:", int(ds.is_outlet.sum()), "vertices:", ds.sizes["vertex"])
EOF
```
Expected: dataset prints with 456 reaches, 182 times, a `vertex` dimension in the tens of thousands, 6 outlets, `n_unconnected` of 4 or fewer (orientation may repair some of the 4 seen during screening), and one warning naming that count if it is above zero.

- [ ] **Step 6: Lint, format, commit**

```bash
cd /home/rmcd/projects/pywatershed && ruff format pywatershed/utils/network_hydraulics.py autotest/test_network_hydraulics.py && ruff check pywatershed autotest
git add pywatershed/utils/network_hydraulics.py autotest/test_network_hydraulics.py
git commit -m "feat: polyline vertex block in export_network_hydraulics

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 5: Package exports, docs, API baseline, maintenance ledger

**Files:**
- Modify: `pywatershed/utils/__init__.py`, `doc/api/utils.rst`, `doc/whats-new.rst`, `autotest/api_surface.txt`, `MAINTENANCE.md`

**Interfaces:**
- Consumes: the four public names from Tasks 1-4.
- Produces: `pws.utils.at_a_station_hydraulic_geometry`, `pws.utils.export_network_hydraulics`, `pws.utils.shear_velocity`, `pws.utils.calculate_seg_mid_elevations`.

- [ ] **Step 1: Write the failing test (API import)**

Append to `autotest/test_network_hydraulics.py`:

```python
@pytest.mark.domainless
def test_public_exports():
    import pywatershed as pws

    for name in (
        "at_a_station_hydraulic_geometry",
        "export_network_hydraulics",
        "shear_velocity",
        "calculate_seg_mid_elevations",
    ):
        assert callable(getattr(pws.utils, name))
        assert name in pws.utils.__all__
```

Run: `cd autotest && /home/rmcd/miniforge3/envs/pws/bin/python -m pytest test_network_hydraulics.py::test_public_exports -m domainless -v`
Expected: FAIL with `AttributeError: module 'pywatershed.utils' has no attribute 'at_a_station_hydraulic_geometry'`

- [ ] **Step 2: Export the names**

In `pywatershed/utils/__init__.py`, add after `from .gis_files import get_gis_dir`:

```python
from .hydraulic_geometry import at_a_station_hydraulic_geometry
```

and after `from .netcdf_utils import NetCdfRead, NetCdfWrite`:

```python
from .network_hydraulics import (
    calculate_seg_mid_elevations,
    export_network_hydraulics,
    shear_velocity,
)
```

Add to `__all__` (keep it as a tuple of strings; insert alphabetically near the matching neighbors):

```python
    "at_a_station_hydraulic_geometry",
    "calculate_seg_mid_elevations",
    "export_network_hydraulics",
    "shear_velocity",
```

Run: `cd autotest && /home/rmcd/miniforge3/envs/pws/bin/python -m pytest test_network_hydraulics.py::test_public_exports -m domainless -v`
Expected: PASS. Also run `/home/rmcd/miniforge3/envs/pws/bin/python -c "import pywatershed"` to prove there is no circular import.

- [ ] **Step 3: API docs page**

In `doc/api/utils.rst`, add these lines to the autosummary list, keeping the list alphabetical after `MmrToMf6Dfw`:

```
    utils.at_a_station_hydraulic_geometry
    utils.calculate_seg_mid_elevations
    utils.export_network_hydraulics
    utils.shear_velocity
```

- [ ] **Step 4: whats-new entry**

In `doc/whats-new.rst`, under `v3.1.0 (Unreleased)` → `New Features`, add after the existing migration-guide entry:

```rst
- Network hydraulics for 1D river-network transport. 
  :func:`~pywatershed.utils.at_a_station_hydraulic_geometry` derives
  per-segment ``width_alpha``, ``width_m``, ``depth_alpha`` and ``depth_m``
  from the bankfull ``seg_width`` and ``seg_depth`` (Manning bankfull
  discharge, Leopold-Maddock at-a-station exponents), replacing the
  one-size-fits-all PRMS default depth relation.
  :func:`~pywatershed.utils.export_network_hydraulics` writes a
  model-agnostic NetCDF of reach topology, optional planform polylines,
  and per-reach time series of flow, velocity, depth, width, shear
  velocity and residence time in SI units, consumed by the particle
  tracker in the ``fluvial-particle`` package. Helpers
  :func:`~pywatershed.utils.shear_velocity` and
  :func:`~pywatershed.utils.calculate_seg_mid_elevations` (the latter
  refactored out of :class:`MmrToMf6Dfw`, behavior unchanged) are public.
  New example notebook ``examples/02a_network_hydraulics_export.ipynb``
  demonstrates both on the Delaware River Basin.
  (:pull:`XXX`) By `Richard McDonald <https://github.com/rmcd-mscb>`_.
```

Confirm the GitHub handle by running `git log --format='%an <%ae>' -1 develop` and `gh api user -q .login` if `gh` is authenticated; otherwise use the handle from the most recent whats-new entry authored by Richard McDonald, or leave `rmcd-mscb` and flag it in the PR.

- [ ] **Step 5: Regenerate the API-surface baseline and check the diff is additive**

```bash
cd /home/rmcd/projects/pywatershed && /home/rmcd/miniforge3/envs/pws/bin/python .github/scripts/api_surface.py --write && git diff --stat autotest/api_surface.txt && git diff autotest/api_surface.txt | grep -E '^-[^-]' ; echo "exit(removed lines above if any)"
```
Expected: only added lines (the `grep` for removed lines prints nothing). Then:

`cd autotest && /home/rmcd/miniforge3/envs/pws/bin/python -m pytest test_api_surface.py -m domainless -v` → PASS.

- [ ] **Step 6: Maintenance ledger entries**

In `MAINTENANCE.md`, under `## Open`, append two items before `## Done` (then run `doctoc MAINTENANCE.md` if `doctoc` is installed, or let pre-commit regenerate the TOC):

```markdown
### fluvial-particle: 1D network particle solver consuming the network hydraulics export

- **Blocked on:** this repo's `export_network_hydraulics` PR merged to
  `develop` (check: `pywatershed/utils/network_hydraulics.py` exists on
  `develop` via the GitHub contents API).
- **Action:** in `/home/rmcd/projects/fluvial-particle`, write the solver
  spec (passive tracer first; reach index + distance-from-upstream-end
  convention documented in the export's `conventions_note` attribute) and
  implement a file-backed hydraulics provider for the export schema in
  `docs/superpowers/specs/2026-09-11-network-hydraulics-export-design.md`.
- **Notes:** keep the export variable names as the future BMI vocabulary.

### NWM transformer to the network hydraulics schema

- **Blocked on:** nothing external; follow-on to the export PR.
- **Action:** add a transformer from NWM `RouteLink_CONUS.nc` plus
  CHRTOUT (`streamflow`, `velocity`, `q_lateral`) to the same schema:
  depth by inverting Manning for the trapezoid (`BtmWdth`, `ChSlp`,
  `So`, `n`), width = `BtmWdth + 2*ChSlp*depth`, `flow_in = streamflow -
  q_lateral`, outlets where `to == 0`, planform from NHDPlus v2
  flowlines by COMID. Sources and field lists are in the spec above.
```

- [ ] **Step 7: Lint, format, commit**

```bash
cd /home/rmcd/projects/pywatershed && ruff format pywatershed/utils/__init__.py autotest/test_network_hydraulics.py && ruff check pywatershed autotest
git add pywatershed/utils/__init__.py doc/api/utils.rst doc/whats-new.rst autotest/api_surface.txt MAINTENANCE.md autotest/test_network_hydraulics.py
git commit -m "docs: export network hydraulics utilities, whats-new, API baseline, ledger

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 6: Example notebook `02a_network_hydraulics_export.ipynb`

**Files:**
- Create: `examples/02a_network_hydraulics_export.ipynb` (generated by a scratchpad script; only the `.ipynb` is committed, without outputs)
- Test: `autotest_exs/test_notebooks.py` (existing; globs `[0-9]*.ipynb`)

**Interfaces:**
- Consumes: `pws.utils.at_a_station_hydraulic_geometry`, `pws.utils.export_network_hydraulics`, `pws.PRMSHydraulicGeometryFull`, `pws.analysis.ProcessPlot.plot_seg_var` (accepts any object supporting `process[var_name]`, so a plain dict works for arrays read from the export file).

- [ ] **Step 1: Write the notebook-builder script in the scratchpad**

Create `/tmp/claude-1000/-home-rmcd-projects-pywatershed/fc5eaa5f-83c2-41d5-bdd0-c94cf01de40f/scratchpad/build_02a.py`:

```python
import nbformat as nbf

nb = nbf.v4.new_notebook()
cells = []
md = lambda s: cells.append(nbf.v4.new_markdown_cell(s))  # noqa: E731
code = lambda s: cells.append(nbf.v4.new_code_cell(s))  # noqa: E731

md("""# Network hydraulics export for 1D particle tracking

This notebook is an add-on to `02_prms_legacy_models.ipynb`. It runs the same NHM configuration on the Delaware River Basin (DRB) with two changes:

1. A more realistic per-segment **hydraulic geometry**. The DRB parameter file carries bankfull width (`seg_width`) and depth (`seg_depth`), originally set from the Bieger et al. (2015) regional curves, but its width power law is nearly flat (`width_m = 0.015`) and it has no depth power law at all, so PRMS falls back to a single default depth relation for every segment. Here we anchor at-a-station power laws through each segment's bankfull point using Manning's equation and the Leopold and Maddock (1953) exponents (width 0.26, depth 0.40, velocity 0.34).
2. An **export** of the routed flows and their hydraulic geometry to one model-agnostic NetCDF file, the interface to the 1D river-network particle tracker in the `fluvial-particle` package.

## Prerequisites
As for the other example notebooks. This notebook regenerates the CBH NetCDF inputs it needs into its own output directory, so it does not depend on `02` having been run.""")

code("""import pathlib as pl
from shutil import rmtree

import matplotlib.pyplot as plt
import numpy as np
import pywatershed as pws
import xarray as xr
from pywatershed.utils import gis_files

gis_files.download()  # GIS files for plotting and for the export

domain_dir = pws.constants.__pywatershed_root__ / "data/drb_2yr"
nb_output_dir = pl.Path("./02a_network_hydraulics_export")
nb_output_dir.mkdir(exist_ok=True)""")

md("""## Preprocess CBH files to NetCDF
Identical to notebook `02`.""")

code("""cbh_nc_dir = nb_output_dir / "drb_2yr_cbh_files"
if cbh_nc_dir.exists():
    rmtree(cbh_nc_dir)
cbh_nc_dir.mkdir(parents=True)

renamer = {
    "prcp": "prcp",
    "tmax": "tmax",
    "tmin": "tmin",
    "rhavg": "humidity_hru",
}
params = pws.parameters.PrmsParameters.load(domain_dir / "myparam.param")
for stem in renamer:
    pws.utils.cbh_file_to_netcdf(
        domain_dir / f"{stem}.cbh",
        params,
        cbh_nc_dir / f"{renamer[stem]}.nc",
        rename_vars=renamer,
    )""")

md("""## Derive at-a-station hydraulic geometry from bankfull parameters

`at_a_station_hydraulic_geometry` computes each segment's bankfull discharge with Manning's equation for a rectangular section of `seg_width` by `seg_depth` on `seg_slope` with roughness `mann_n`, then sets `width_alpha`, `width_m`, `depth_alpha`, `depth_m` so that `alpha * Q**m` returns the bankfull width and depth at that discharge. It returns a new `Parameters` object; the original is untouched. With `return_bankfull=True` we also get the bankfull discharge and velocity for plotting.""")

code("""params_hg, bankfull = pws.utils.at_a_station_hydraulic_geometry(
    params, return_bankfull=True
)
print("velocity exponent:", bankfull["velocity_exp"])
for name in ["width_alpha", "width_m", "depth_alpha", "depth_m"]:
    vals = params_hg.parameters[name]
    print(f"{name:12s} min={vals.min():.4g} median={np.median(vals):.4g} max={vals.max():.4g}")""")

code("""gis_dir = pws.utils.get_gis_dir("drb_2yr")
proc_plot = pws.analysis.ProcessPlot(gis_dir)

# ProcessPlot.plot_seg_var only needs item access by variable name, so a
# dict of arrays stands in for a Process.
bankfull_arrays = {
    "bankfull_flow": bankfull["bankfull_flow"],
    "bankfull_velocity": bankfull["bankfull_velocity"],
}
proc_plot.plot_seg_var(
    "bankfull_flow",
    bankfull_arrays,
    title="Manning bankfull discharge (m3/s)",
    value_transform=np.log10,
)
proc_plot.plot_seg_var(
    "bankfull_velocity", bankfull_arrays, title="Manning bankfull velocity (m/s)"
)""")

md("""## Run the NHM with the full hydraulic geometry process

The process list is the one from notebook `02` with `PRMSHydraulicGeometryFull` in place of `PRMSHydraulicGeometryWidthOnly`, so both depth and width come from the derived power laws. We add the geometry variables to the output list so the exporter can read them, and run the same six months as `02`.""")

code("""nhm_processes = [
    pws.PRMSSolarGeometry,
    pws.PRMSAtmosphere,
    pws.PRMSCanopy,
    pws.PRMSSnow,
    pws.PRMSRunoff,
    pws.PRMSSoilzone,
    pws.PRMSGroundwater,
    pws.PRMSChannel,
    pws.PRMSHydraulicGeometryFull,
    pws.PRMSStreamTempHumidityCBH,
]

control = pws.Control.load_prms(
    domain_dir / "nhm_stream_temp.control", warn_unused_options=False
)
control.edit_end_time(np.datetime64("1979-07-01T00:00:00"))
run_dir = nb_output_dir / "nhm"
if run_dir.exists():
    rmtree(run_dir)
control.options["netcdf_output_var_names"] += [
    "seg_flow_width",
    "seg_flow_depth",
    "seg_flow_area",
    "seg_flow_velocity",
    "seg_res_time",
]
control.options = control.options | {
    "input_dir": cbh_nc_dir,
    "imbalance_behavior": "warn",
    "calc_method": "numba",
    "netcdf_output_dir": run_dir,
}""")

code("""%%time
nhm = pws.Model(nhm_processes, control=control, parameters=params_hg)
nhm.run(finalize=True)""")

md("""## What changed relative to the PRMS-default geometry?

Notebook `02` used the width-only process: constant width and the PRMS default depth relation (`0.27 * Q**0.39`) for every segment. We recompute that default geometry from the routed flow and compare last-day velocity and depth with the new parameterization. Flows are identical in both cases because the geometry does not feed back on routing.""")

code("""from pywatershed.hydrology.prms_hydraulic_geometry import CFS_TO_CMS

q_last = nhm.processes["PRMSChannel"]["seg_outflow"] * CFS_TO_CMS
w_default = params.parameters["width_alpha"] * q_last ** params.parameters["width_m"]
d_default = 0.27 * q_last**0.39
v_default = np.where(w_default * d_default > 1e-6, q_last / (w_default * d_default), 0.0)

hg = nhm.processes["PRMSHydraulicGeometryFull"]
compare = {
    "velocity_default": v_default,
    "velocity_at_a_station": hg["seg_flow_velocity"],
    "depth_default": d_default,
    "depth_at_a_station": hg["seg_flow_depth"],
}
for name in compare:
    proc_plot.plot_seg_var(name, compare, title=f"{name} (last day)")""")

code("""fig, ax = plt.subplots(1, 2, figsize=(10, 4.5))
ax[0].scatter(v_default, hg["seg_flow_velocity"], s=8)
ax[0].plot([0, v_default.max()], [0, v_default.max()], "k--", lw=1)
ax[0].set_xlabel("PRMS default velocity (m/s)")
ax[0].set_ylabel("at-a-station velocity (m/s)")
ax[1].scatter(d_default, hg["seg_flow_depth"], s=8)
ax[1].plot([0, d_default.max()], [0, d_default.max()], "k--", lw=1)
ax[1].set_xlabel("PRMS default depth (m)")
ax[1].set_ylabel("at-a-station depth (m)")
fig.suptitle("Last-day hydraulic geometry, default vs bankfull-anchored")
plt.show()""")

md("""## Export network hydraulics for particle tracking

`export_network_hydraulics` reads the run directory and the parameters, adds the segment polylines from the GIS shapefile, and writes one NetCDF with:

- static per-reach fields: identifiers, downstream index (`to_index`, -1 at outlets), length, slope, Manning's n, midpoint elevation, bankfull width and depth;
- a `vertex` block of polyline coordinates with cumulative arc length per vertex, oriented upstream to downstream;
- per-reach daily time series in SI units: `flow_in`, `flow_out`, `velocity`, `depth`, `width`, `ustar` (shear velocity), `residence_time`, and `water_temperature`.

This file is the interface to the 1D network particle tracker in `fluvial-particle`. A particle there is a reach index plus a distance from the reach's upstream end; when it passes the reach length it moves to `to_index`, and it exits at an outlet. Map positions come from scaling that distance onto the polyline's `vertex_dist`.""")

code("""network_file = nb_output_dir / "drb_network_hydraulics.nc"
pws.utils.export_network_hydraulics(
    params_hg,
    run_dir,
    network_file,
    segment_shp_file=gis_dir / "Segments_subset.shp",
)
network = xr.open_dataset(network_file)
network""")

code("""last = network.isel(time=-1)
from_file = {
    "velocity": last["velocity"].values,
    "depth": last["depth"].values,
    "ustar": last["ustar"].values,
    "residence_time_hours": last["residence_time"].values / 3600.0,
}
for name in from_file:
    proc_plot.plot_seg_var(name, from_file, title=f"{name} from the export (last day)")

print("outlets:", int(network["is_outlet"].sum()))
print("unconnected polylines:", network.attrs["n_unconnected"])
print(
    "median residence time (h):",
    float(np.nanmedian(network["residence_time"].where(network["flow_out"] > 0)) / 3600.0),
)
network.close()""")

md("""## References
* Bieger, K., Rathjens, H., Allen, P.M., Arnold, J.G. (2015). Development and evaluation of bankfull hydraulic geometry relationships for the physiographic regions of the United States. JAWRA 51(3), 842-858.
* Leopold, L.B., Maddock, T. (1953). The hydraulic geometry of stream channels and some physiographic implications. USGS Professional Paper 252.
* Regan, R.S., Markstrom, S.L., LaFontaine, J.H., Norton, P.A., 2022, PRMS version 5.2.1: Precipitation-Runoff Modeling System (PRMS): U.S. Geological Survey Software Release, 02/10/2022.""")

nb["cells"] = cells
nb["metadata"] = {
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python"},
}
nbf.write(nb, "/home/rmcd/projects/pywatershed/examples/02a_network_hydraulics_export.ipynb")
print("written")
```

Run: `/home/rmcd/miniforge3/envs/pws/bin/python /tmp/claude-1000/-home-rmcd-projects-pywatershed/fc5eaa5f-83c2-41d5-bdd0-c94cf01de40f/scratchpad/build_02a.py`
Expected: prints `written`; `examples/02a_network_hydraulics_export.ipynb` exists.

- [ ] **Step 2: Execute the notebook the way CI does**

Run: `cd autotest_exs && MPLBACKEND=Agg /home/rmcd/miniforge3/envs/pws/bin/python -m pytest test_notebooks.py -k 02a -v -s`
Expected: PASS (several minutes: the NHM run takes about as long as notebook `02`). The test converts the notebook to `examples/02a_network_hydraulics_export.py` and runs it with ipython; delete that generated `.py` afterwards (`rm examples/02a_network_hydraulics_export.py`) and do not commit it. Check `.gitignore` already ignores `examples/*.py` generated files; if the `.py` shows in `git status`, delete it.

If `plot_seg_var` rejects the dict (for example it calls a `Process` method besides item access), replace the dict-based calls with a small shim class in the notebook:

```python
class _Arrays(dict):
    pass
```

and if a method is still required, read `pywatershed/analysis/process_plot.py:90-158` and pass what it needs; do not change `ProcessPlot` in this task.

- [ ] **Step 3: Inspect the outputs for sanity**

```bash
cd /home/rmcd/projects/pywatershed && /home/rmcd/miniforge3/envs/pws/bin/python - <<'EOF'
import xarray as xr, numpy as np
ds = xr.open_dataset("examples/02a_network_hydraulics_export/drb_network_hydraulics.nc")
v = ds.velocity.where(ds.flow_out > 0)
print("velocity m/s: median %.3f max %.3f" % (float(v.median()), float(v.max())))
print("depth m: median %.3f max %.3f" % (float(ds.depth.where(ds.flow_out > 0).median()), float(ds.depth.max())))
print("reaches", ds.sizes["reach"], "times", ds.sizes["time"], "vertices", ds.sizes["vertex"], "unconnected", ds.attrs["n_unconnected"])
EOF
```
Expected: median velocity in the 0.3 to 0.8 m/s range and a maximum well below the 4.3 m/s seen with the default depth relation during screening (bankfull anchoring raises depth at high flow); 456 reaches, 182 times. Record the numbers for the PR body.

- [ ] **Step 4: Strip outputs and commit the notebook only**

```bash
cd /home/rmcd/projects/pywatershed && /home/rmcd/miniforge3/envs/pws/bin/python -m jupyter nbconvert --clear-output --inplace examples/02a_network_hydraulics_export.ipynb
git status --short examples/
git add examples/02a_network_hydraulics_export.ipynb
git commit -m "docs: 02a example notebook, at-a-station geometry and network export

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

`git status` must show only the new notebook staged; the `02a_network_hydraulics_export/` output directory should be ignored by the existing `examples/` ignore rules (check `.gitignore`; if it is not ignored, add `examples/02a_network_hydraulics_export/` to `.gitignore` in this commit).

---

### Task 7: Full verification and PR

**Files:**
- No new files; may touch `autotest/ci_local.sh` only if new test files need listing (they do not: the domainless job selects by marker).

- [ ] **Step 1: Run the whole domainless suite, lint, and format check**

```bash
cd /home/rmcd/projects/pywatershed && ruff check . && ruff format . --check --diff
cd autotest && /home/rmcd/miniforge3/envs/pws/bin/python -m pytest -m domainless -v --durations=10 2>&1 | tail -30
```
Expected: ruff clean; all domainless tests PASS (MF6-dependent ones may SKIP locally for a missing binary).

- [ ] **Step 2: Run the hydraulic geometry domain tests for regression**

```bash
cd autotest && /home/rmcd/miniforge3/envs/pws/bin/python -m pytest test_prms_hydraulic_geometry.py test_prms_stream_temp.py --domain drb_2yr -v 2>&1 | tail -15
```
Expected: PASS or the pre-existing conditional SKIPs; nothing in these tests changed, this confirms the refactor left the processes untouched. If the domain test data are not generated locally (see `DEVELOPER.md`), note that in the PR and rely on CI.

- [ ] **Step 3: Confirm CI wiring needs no change**

Read `.github/workflows/ci.yaml` lines 140-240 and `autotest/ci_local.sh`: the domainless job runs `pytest -m domainless`, so the two new test files are picked up without listing. The examples workflow globs numbered notebooks. State this in the PR body.

- [ ] **Step 4: Push and open the PR against `develop`**

Read `.github/PULL_REQUEST_TEMPLATE.md` fresh, then:

```bash
cd /home/rmcd/projects/pywatershed && git push -u origin feat_network_hydraulics_export
```

Draft the PR body to the template: summary on top (the two utilities, the notebook, the refactor, the spec and plan files, the export schema as the fluvial-particle contract, NWM applicability, and the velocity numbers recorded in Task 6 Step 3), then the template's full checklist under `## Checklist` with applicable items checked and inapplicable ones struck through (`- [ ] ~~item~~`), keeping its Docs section. End with `🤖 Generated with [Claude Code](https://claude.com/claude-code)`. Open it with `gh pr create --base develop --title "feat: network hydraulics export and at-a-station hydraulic geometry" --body-file <file>` and then replace `XXX` in `doc/whats-new.rst` with the PR number in a final commit:

```bash
git commit -am "docs: whats-new PR number

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>" && git push
```

- [ ] **Step 5: Update the cross-repo memory**

Append to `/home/rmcd/.claude/projects/-home-rmcd-projects-pywatershed/memory/fluvial-particle-1d-network.md` a line with the PR number and the export file path (`examples/02a_network_hydraulics_export/drb_network_hydraulics.nc` after running the notebook) so the fluvial-particle session can find its input.
