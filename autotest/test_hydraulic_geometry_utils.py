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
def test_at_a_station_returns_parameters_only_by_default(
    synthetic_params,
):
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
def test_at_a_station_nan_slope_raises(synthetic_params):
    from pywatershed.utils.hydraulic_geometry import (
        at_a_station_hydraulic_geometry,
    )

    dd = synthetic_params.to_dd()
    dd.data_vars["seg_slope"][1] = np.nan
    bad = Parameters(**dd.data)
    with pytest.raises(ValueError, match="seg_slope.*1 segment"):
        at_a_station_hydraulic_geometry(bad)


@pytest.mark.domainless
def test_at_a_station_negative_slope_raises(synthetic_params):
    from pywatershed.utils.hydraulic_geometry import (
        at_a_station_hydraulic_geometry,
    )

    dd = synthetic_params.to_dd()
    dd.data_vars["seg_slope"][1] = -0.01
    bad = Parameters(**dd.data)
    with pytest.raises(ValueError, match="seg_slope.*1 segment"):
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
    assert isinstance(new, pws.parameters.PrmsParameters)
    p = params.parameters
    q_cms = bankfull["bankfull_flow"]
    # mirror PRMSHydraulicGeometryFull: flow_cms = seg_outflow_cfs * CFS_TO_CMS
    q_cfs = q_cms / CFS_TO_CMS
    flow_cms = q_cfs * CFS_TO_CMS
    width = (
        new.parameters["width_alpha"] * flow_cms ** new.parameters["width_m"]
    )
    depth = (
        new.parameters["depth_alpha"] * flow_cms ** new.parameters["depth_m"]
    )
    np.testing.assert_allclose(width, p["seg_width"], rtol=1e-10)
    np.testing.assert_allclose(depth, p["seg_depth"], rtol=1e-10)
    assert np.all(q_cms > 0)
