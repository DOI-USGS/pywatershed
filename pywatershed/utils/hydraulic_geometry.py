"""At-a-station hydraulic geometry anchored on bankfull parameters."""

import numpy as np

from ..base.parameters import Parameters
from .network_hydraulics import SLOPE_FLOOR

_REQUIRED = ("seg_width", "seg_depth", "seg_slope", "mann_n")

# units follow the registry entries for these PRMS parameters in
# pywatershed/static/metadata/parameters.yaml
_NEW_META = {
    "width_alpha": {
        "desc": "Alpha coefficient in power function for width calculation",
        "units": "unknown",
    },
    "width_m": {
        "desc": "M value in power function for width calculation",
        "units": "unknown",
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
        A new object of the same class as ``parameters``, a copy of the
        input with ``width_alpha``, ``width_m``, ``depth_alpha`` and
        ``depth_m`` set (overwriting any existing values). With
        ``return_bankfull=True`` a tuple of that object and a dict with
        ``bankfull_flow`` (m^3/s), ``bankfull_velocity`` (m/s) and
        ``velocity_exp``.

    Raises:
        ValueError: a required parameter is missing, ``seg_slope`` is
            negative or not finite, or ``seg_width``, ``seg_depth`` or
            ``mann_n`` is not positive.
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
    raw_slope = np.asarray(params["seg_slope"], dtype=float)
    mann_n = np.asarray(params["mann_n"], dtype=float)

    # the raw slope is checked before the floor is applied, which would
    # otherwise hide a negative (but not a NaN) slope
    n_bad_slope = int(np.sum(~(np.isfinite(raw_slope) & (raw_slope >= 0.0))))
    if n_bad_slope:
        raise ValueError(
            "Parameter seg_slope must be non-negative and finite; "
            f"{n_bad_slope} segment(s) are not"
        )
    slope = np.maximum(raw_slope, SLOPE_FLOOR)

    for name, arr in (
        ("seg_width", width),
        ("seg_depth", depth),
        ("mann_n", mann_n),
    ):
        n_bad = int(np.sum(~(np.isfinite(arr) & (arr > 0.0))))
        if n_bad:
            raise ValueError(
                f"Parameter {name} must be positive; "
                f"{n_bad} segment(s) are not"
            )

    area = width * depth
    radius = area / (width + 2.0 * depth)
    q_bf = area * radius ** (2.0 / 3.0) * np.sqrt(slope) / mann_n

    dd = parameters.to_dd()
    nseg = dd.data["dims"]["nsegment"]
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
    new_params = type(parameters)(**dd.data)

    if not return_bankfull:
        return new_params
    bankfull = {
        "bankfull_flow": q_bf,
        "bankfull_velocity": q_bf / area,
        "velocity_exp": 1.0 - width_exp - depth_exp,
    }
    return new_params, bankfull
