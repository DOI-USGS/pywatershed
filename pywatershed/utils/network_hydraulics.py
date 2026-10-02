"""Network hydraulics helpers and a model-agnostic NetCDF export.

The export carries reach topology, optional planform geometry, and
per-reach, per-time-step hydraulics (flow, velocity, depth, width, shear
velocity, residence time, and optionally water temperature) in SI units
for consumers such as 1D network particle trackers. See
``docs/superpowers/specs/2026-09-11-network-hydraulics-export-design.md``.
"""

import datetime
import pathlib as pl
from warnings import warn

import numpy as np
import xarray as xr

from ..base import meta
from ..base.parameters import Parameters
from ..hydrology.prms_hydraulic_geometry import CFS_TO_CMS
from ..version import __version__

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

    Raises:
        ValueError: any slope is negative or not finite.
    """
    slope = np.asarray(slope, dtype=float)
    n_bad = int(np.sum(~(np.isfinite(slope) & (slope >= 0.0))))
    if n_bad:
        raise ValueError(
            f"slope must be non-negative and finite; {n_bad} value(s) are not"
        )
    slope_floored = np.maximum(slope, SLOPE_FLOOR)
    return np.sqrt(G * np.asarray(depth, dtype=float) * slope_floored)


def _validate_tosegment(tosegment: np.ndarray) -> np.ndarray:
    """Check ``tosegment`` and return it as a zero-based index array.

    Args:
        tosegment: one-based downstream segment for each segment, 0 at
            outlets.

    Returns:
        The zero-based downstream index, -1 at outlets.

    Raises:
        ValueError: any value is not an integer 0 or in ``1..nsegment``.
    """
    values = np.asarray(tosegment)
    nseg = values.size
    as_float = values.astype(float)
    valid = (as_float == np.floor(as_float)) & (
        (as_float == 0) | ((as_float >= 1) & (as_float <= nseg))
    )
    bad = np.where(~valid)[0]
    if bad.size:
        raise ValueError(
            f"tosegment must be 0 (outlet) or in 1..{nseg}; bad at "
            f"segments {bad.tolist()} with values {values[bad].tolist()}"
        )
    return values.astype(np.int64) - 1


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

    Raises:
        ValueError: ``tosegment`` is out of range or contains a cycle.
    """
    params = parameters.parameters
    seg_dy = params["seg_slope"] * params["seg_length"]
    nseg = len(seg_dy)
    seg_y = np.full(nseg, np.nan)  # elevation at the upstream end
    tosegment0 = _validate_tosegment(params["tosegment"])
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
            chain.append(int(ind))
            if len(chain) > nseg:
                raise ValueError(
                    "Cycle detected in tosegment starting at segment "
                    f"index {ss}: chain {chain}"
                )
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


_REQUIRED_PARAMS = (
    "nhm_seg",
    "tosegment",
    "seg_length",
    "seg_slope",
    "mann_n",
    "seg_width",
    "seg_depth",
    "hru_segment",
    "hru_elev",
)
"""Parameters the exporter requires."""

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

_ZERO_FLOW_NOTE = "0 where flow_out == 0; mask on flow_out > 0"

_ZERO_FLOW_VARS = ("velocity", "depth", "width", "residence_time")

# export name: (source name, units, long_name, method, scale factor)
_TIME_VARS = {
    "flow_out": (
        "seg_outflow",
        "m3 s-1",
        "flow leaving the reach",
        "routed",
        CFS_TO_CMS,
    ),
    "flow_in": (
        "seg_inflow",
        "m3 s-1",
        "flow entering the reach",
        "routed",
        CFS_TO_CMS,
    ),
    "velocity": (
        "seg_flow_velocity",
        "m s-1",
        "mean flow velocity",
        _GEOMETRY_METHOD,
        1.0,
    ),
    "depth": (
        "seg_flow_depth",
        "m",
        "mean flow depth",
        _GEOMETRY_METHOD,
        1.0,
    ),
    "width": (
        "seg_flow_width",
        "m",
        "flow width",
        _GEOMETRY_METHOD,
        1.0,
    ),
    "residence_time": (
        "seg_res_time",
        "s",
        "mean residence time of water in the reach",
        "area*length/flow_out",
        1.0,
    ),
    "water_temperature": (
        "seg_tave_water",
        "degC",
        "mean water temperature",
        "PRMS stream temperature",
        1.0,
    ),
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
    if start_time is not None and end_time is not None:
        if start_time > end_time:
            raise ValueError("start_time is after end_time")
    # seg_outflow is first, so its time axis is the reference for the rest
    names = list(REQUIRED_RUN_VARS) + [
        nm for nm in OPTIONAL_RUN_VARS if (run_dir / f"{nm}.nc").exists()
    ]
    result = {}
    reference_time = None
    for nm in names:
        path = run_dir / f"{nm}.nc"
        with xr.open_dataarray(path) as opened:
            da = opened.load()
        if "nhm_seg" not in da.coords:
            raise ValueError(
                f"{nm}.nc has no nhm_seg coordinate; cannot verify reach order"
            )
        if not np.array_equal(da["nhm_seg"].values, reach_id):
            raise ValueError(
                f"{nm}.nc coordinate nhm_seg does not match the "
                "parameters' nhm_seg order"
            )
        available = da["time"].values
        if reference_time is None:
            reference_time = available
        elif not np.array_equal(available, reference_time):
            raise ValueError(
                f"{nm}.nc time coordinate does not match seg_outflow.nc; "
                "run outputs must share one time axis"
            )
        expected_units = meta.get_vars(nm)[nm]["units"]
        units = da.attrs.get("units")
        if units is None:
            warn(
                f"{nm}.nc has no units attribute; assuming '{expected_units}'"
            )
        elif units != expected_units:
            raise ValueError(
                f"{nm}.nc units '{units}' differ from expected "
                f"'{expected_units}'"
            )
        if start_time is not None or end_time is not None:
            da = da.sel(time=slice(start_time, end_time))
            if da.sizes["time"] == 0:
                if len(available):
                    span = (
                        f"its times run from {available[0]} to {available[-1]}"
                    )
                else:
                    span = "the file has no time steps at all"
                raise ValueError(
                    f"{nm}.nc has no time steps between start_time "
                    f"{start_time} and end_time {end_time}; {span}"
                )
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

    Where ``flow_out`` is 0 the process outputs give ``velocity``,
    ``depth``, ``width`` and ``residence_time`` of 0 (not inf);
    consumers should mask on ``flow_out > 0``.

    Args:
        parameters: the run's parameters (needs ``nhm_seg``,
            ``tosegment``, ``seg_length``, ``seg_slope``, ``mann_n``,
            ``seg_width``, ``seg_depth``, ``hru_segment``,
            ``hru_elev``). ``tosegment_nhm`` is used for ``to_id`` when
            present; otherwise ``to_id`` is derived from ``tosegment``
            and ``nhm_seg``.
        run_dir: pywatershed NetCDF output directory containing
            ``seg_outflow``, ``seg_inflow``, ``seg_flow_width``,
            ``seg_flow_depth``, ``seg_flow_velocity`` and ``seg_res_time``
            (``seg_tave_water`` is included when present).
        out_file: path of the NetCDF file to write.
        segment_shp_file: optional shapefile of segment LineStrings; adds
            the ``vertex`` block and reach midpoints. Must use a
            projected CRS in meters: a geographic CRS or a projected
            CRS not in meters raises ``ValueError``; a missing CRS
            warns and coordinates are labeled with units "unknown".
            Each line is oriented so its downstream end is last. For a
            reach with a downstream neighbor, the line is reversed
            when its first vertex is nearer than its last vertex to
            the downstream reach's nearest end. For an outlet reach
            (no downstream neighbor), the reference point is instead
            the last vertices of the upstream reaches that drain to
            it, and the line is reversed when its last vertex is
            nearer that reference than its first vertex is.
            ``connect_tol`` does not affect this orientation; it only
            controls the ``n_unconnected`` count below.
        shp_id_col: shapefile column holding ``nhm_seg`` identifiers.
        connect_tol: distance (m) within which a reach's last
            vertex must meet its downstream reach's first vertex, used
            only to count (not fix) unconnected reaches; see
            ``n_unconnected`` in the global attributes, which is -1
            when no ``segment_shp_file`` is supplied (no polyline
            block was written) and otherwise the count of reaches
            still failing this tolerance after orientation. More than
            half of the reaches with a downstream reach failing it
            raises ``ValueError``; fewer only warn. Its value is
            written as the global attribute ``connect_tol``.
        start_time: optional first time to include.
        end_time: optional last time to include.

    Returns:
        ``out_file`` as a Path.

    Raises:
        FileNotFoundError: a required output file is missing from
            ``run_dir``; all missing names are listed.
        ValueError: a required parameter is missing; ``tosegment`` is
            out of range or contains a cycle; a run file has no
            ``nhm_seg`` coordinate or its order does not match the
            parameters; run files do not share one time axis; a run
            file's ``units`` attribute differs from the pywatershed
            metadata; the requested time selection is empty or
            ``start_time`` is after ``end_time``; the shapefile
            identifiers do not match ``nhm_seg``; a shapefile geometry
            is not a ``LineString``; the shapefile CRS is geographic or
            not in meters; or more than half of the reaches with a
            downstream reach fail ``connect_tol``.
    """
    params = parameters.parameters
    missing_params = [nm for nm in _REQUIRED_PARAMS if nm not in params]
    if missing_params:
        raise ValueError(
            "export_network_hydraulics requires parameters "
            f"{list(_REQUIRED_PARAMS)}; missing {missing_params}"
        )
    reach_id = np.asarray(params["nhm_seg"], dtype=np.int64)
    to_index = _validate_tosegment(params["tosegment"]).astype(np.int32)
    is_outlet = (to_index < 0).astype(np.int8)
    elevation_mid, _ = calculate_seg_mid_elevations(parameters)
    if "tosegment_nhm" in params:
        to_id = np.asarray(params["tosegment_nhm"], dtype=np.int64)
        to_id_source = "tosegment_nhm"
    else:
        to_id = reach_id[np.maximum(to_index, 0)]
        to_id_source = "derived from tosegment and nhm_seg"
    to_id = np.where(to_index >= 0, to_id, 0)

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
        "reach_id": static(
            reach_id, np.int64, "-", "reach identifier", "nhm_seg"
        ),
        "to_id": static(
            to_id,
            np.int64,
            "-",
            "downstream reach identifier (0 = outlet)",
            to_id_source,
        ),
        "to_index": static(
            to_index,
            np.int32,
            "-",
            "zero-based index of the downstream reach (-1 = outlet)",
            "tosegment",
        ),
        "is_outlet": static(
            is_outlet,
            np.int8,
            "-",
            "1 where the reach drains out of the network",
            "tosegment",
        ),
        "length": static(
            params["seg_length"],
            np.float64,
            "m",
            "reach hydraulic length",
            "seg_length",
        ),
        "slope": static(
            params["seg_slope"],
            np.float64,
            "m m-1",
            "reach slope",
            "seg_slope",
        ),
        "mann_n": static(
            params["mann_n"],
            np.float64,
            "s m-1/3",
            "Manning roughness",
            "mann_n",
        ),
        "elevation_mid": static(
            elevation_mid,
            np.float64,
            "m",
            "elevation at the reach midpoint, walked up from outlets",
            "seg_slope*seg_length, hru_elev",
        ),
        "bankfull_width": static(
            params["seg_width"],
            np.float64,
            "m",
            "bankfull width",
            "seg_width",
        ),
        "bankfull_depth": static(
            params["seg_depth"],
            np.float64,
            "m",
            "bankfull depth",
            "seg_depth",
        ),
    }

    slope = np.asarray(params["seg_slope"], dtype=float)
    for name, (src, units, long_name, method, scale) in _TIME_VARS.items():
        if src not in run_vars:
            continue
        values = run_vars[src].values * scale
        attrs = {
            "units": units,
            "long_name": long_name,
            "source_name": src,
            "method": method,
        }
        if name in _ZERO_FLOW_VARS:
            attrs["note"] = _ZERO_FLOW_NOTE
        data_vars[name] = xr.DataArray(
            values, dims=("time", "reach"), attrs=attrs
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
            "created": datetime.datetime.now(
                datetime.timezone.utc
            ).isoformat(),
            "n_unconnected": n_unconnected,
            "connect_tol": float(connect_tol),
            "crs_wkt": crs_wkt,
            "conventions_note": (
                "Particle state is (reach index, s) with 0 <= s <= length "
                "from the reach's upstream end. When s exceeds length the "
                "particle moves to to_index carrying the unused fraction of "
                "the time step; to_index == -1 is an outlet. Map position "
                "scales s/length onto the polyline's vertex_dist. "
                "Where flow_out is 0 the velocity, depth, width and "
                "residence_time are 0 (not inf); mask on flow_out > 0. "
                "n_unconnected is the number of reaches whose polyline end "
                "does not meet the downstream reach within connect_tol "
                "(-1 when no polyline block)."
            ),
        },
    )
    out_file = pl.Path(out_file)
    ds.to_netcdf(out_file)
    ds.close()
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
    if gdf.crs is None:
        crs_units = "unknown"
        crs_wkt = ""
        warn(
            "Segment shapefile has no CRS; coordinates are assumed to be "
            "in meters"
        )
    elif gdf.crs.is_geographic:
        raise ValueError(
            f"Segment shapefile CRS {gdf.crs.name!r} is geographic; the "
            "segment shapefile must use a projected CRS in meters "
            "(for example, reproject to EPSG:5070)"
        )
    else:
        unit_name = gdf.crs.axis_info[0].unit_name
        if unit_name not in ("metre", "meter", "m"):
            raise ValueError(
                f"Segment shapefile CRS {gdf.crs.name!r} uses units "
                f"{unit_name!r}; the segment shapefile must use a "
                "projected CRS in meters"
            )
        crs_units = "m"
        crs_wkt = gdf.crs.to_wkt()

    order = {rid: ii for ii, rid in enumerate(shp_ids)}
    geoms = [gdf.geometry.iloc[order[rid]] for rid in reach_id]
    coords = []
    for rid, gg in zip(reach_id, geoms):
        if gg.geom_type != "LineString":
            raise ValueError(
                f"Reach {rid} geometry is {gg.geom_type}; only LineString "
                "is supported (explode or merge multipart segments first)"
            )
        coords.append(np.asarray(gg.coords, dtype=float)[:, :2])

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

    # outlets are not downstream of anything, so the pass above never
    # orients them; instead point each outlet's line so its upstream
    # end (first vertex) is nearest the upstream reaches that drain to
    # it, leaving the true outlet end (last vertex) farthest away
    # (those upstream reaches are already oriented by the pass above)
    n = len(coords)
    for o, down in enumerate(to_index):
        if down >= 0:
            continue
        ups = [i for i in range(n) if to_index[i] == o]
        if not ups:
            warn(
                f"Outlet reach {reach_id[o]} has no upstream reach; its "
                "polyline orientation is unverified"
            )
            continue
        last_pts = [coords[i][-1] for i in ups]
        dist_to_end = min(np.hypot(*(pt - coords[o][-1])) for pt in last_pts)
        dist_to_start = min(np.hypot(*(pt - coords[o][0])) for pt in last_pts)
        if dist_to_end < dist_to_start:
            coords[o] = coords[o][::-1]

    n_unconnected = 0
    n_with_downstream = 0
    for ii, down in enumerate(to_index):
        if down < 0:
            continue
        n_with_downstream += 1
        if np.hypot(*(coords[ii][-1] - coords[down][0])) > connect_tol:
            n_unconnected += 1
    if n_unconnected > n_with_downstream / 2:
        raise ValueError(
            f"{n_unconnected} of {n_with_downstream} reach polylines do "
            "not meet their downstream reach within connect_tol; this "
            "usually means an id, CRS or units mismatch"
        )

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
        if dist[-1] == 0:
            x_mid[ii], y_mid[ii] = cc[0, 0], cc[0, 1]
        else:
            half = dist[-1] / 2.0
            x_mid[ii] = np.interp(half, dist, cc[:, 0])
            y_mid[ii] = np.interp(half, dist, cc[:, 1])
    vertex_dist = np.concatenate(dists)

    def vvar(values, dims, units, long_name):
        return xr.DataArray(
            values, dims=dims, attrs={"units": units, "long_name": long_name}
        )

    poly_vars = {
        "vertex_x": vvar(
            vertex_x, ("vertex",), crs_units, "vertex x in the CRS"
        ),
        "vertex_y": vvar(
            vertex_y, ("vertex",), crs_units, "vertex y in the CRS"
        ),
        "vertex_dist": vvar(
            vertex_dist,
            ("vertex",),
            crs_units,
            "cumulative arc length from the reach's upstream end",
        ),
        "reach_vertex_start": vvar(
            starts, ("reach",), "-", "index of the reach's first vertex"
        ),
        "reach_vertex_count": vvar(
            counts,
            ("reach",),
            "-",
            "number of vertices in the reach polyline",
        ),
        "x_mid": vvar(
            x_mid, ("reach",), crs_units, "x at half the polyline arc length"
        ),
        "y_mid": vvar(
            y_mid, ("reach",), crs_units, "y at half the polyline arc length"
        ),
    }
    return poly_vars, n_unconnected, crs_wkt
