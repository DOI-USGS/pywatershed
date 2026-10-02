"""Derive PRMS cascade parameters from a PRMS parameter file.

PRMS computes the HRU-to-HRU (and HRU-to-segment) cascade routing at
startup, in basin.f90 (the HRU routing order) and cascade.f90
(``init_cascade`` and ``order_hrus``), and never writes the result.
pywatershed does that work here, once, as a preprocessing step, so the
cascade process classes can take the derived quantities as ordinary
parameters. :func:`preprocess_cascade_params` runs both stages and
returns the input :class:`Parameters` with these added:

* ``hru_route_order`` and ``active_hrus`` from
  :func:`calc_hru_route_order` (basin.f90, with its error checks);
* ``ncascade_hru``, ``hru_down``, ``hru_down_frac``,
  ``hru_down_fracwt`` and ``cascade_area`` from
  :func:`init_cascade_params` (cascade.f90). ``hru_type`` and
  ``hru_route_order`` are rewritten there as PRMS does: a land or
  glacier HRU (``hru_type`` 1 or 4) that passes no cascade flow becomes
  a swale (``hru_type`` 3), and the order is recomputed from the cascade
  links. Positive ``hru_down``
  entries are downslope HRU indices (1-based), negative ones are
  stream segments.

HRU cascades (control ``cascade_flag = 1``) are handled, and groundwater
reservoir cascades (``cascadegw_flag`` 1 or 2) are derived alongside by
:func:`init_gw_cascade_params`; the HRU-to-segment-only mode
(``cascade_flag = 2``) raises. The basin-area sums that basin.f90
computes alongside are not reproduced.

:class:`PRMSRunoffCascadesNoDprst` and
:class:`PRMSSoilzoneCascadesNoDprst` call
:func:`preprocess_cascade_params` themselves when any of the derived
parameters (``cascade_param_names``) is missing from their parameters,
so users need not; calling it
explicitly (or via :func:`~utils.separate_domain_params_dis_to_ncdf`
with a control) lets the result be inspected, saved, and shared by
both processes instead of derived twice.
"""

from typing import Union

import networkx as nx
import numpy as np
import xarray as xr

from ..base import Control
from ..base.data_model import DatasetDict
from ..constants import ACTIVE, HruType, one
from ..parameters import Parameters

# The parameters preprocess_cascade_params derives, which the cascade
# process classes declare. A Parameters object lacking any of them has
# not been (fully) preprocessed.
cascade_param_names = (
    "hru_route_order",
    "ncascade_hru",
    "hru_down",
    "hru_down_frac",
    "hru_down_fracwt",
    "cascade_area",
)


def _ensure_cascade_params(
    control: Control,
    parameters: Parameters,
    verbose: Union[bool, None] = None,
) -> Parameters:
    """Return parameters with the derived cascade parameters present.

    The derived cascade parameters are not in a PRMS parameter file, so
    the cascade process classes derive them here when any is missing
    rather than require them.
    """
    if all(kk in parameters.parameters for kk in cascade_param_names):
        return parameters
    return preprocess_cascade_params(
        control, parameters, verbosity=int(bool(verbose))
    )


def _verbosity_msg(msg: str, verbosity: int) -> None:
    """Print a diagnostic PRMS writes only when Print_debug = 13.

    Messages PRMS prints unconditionally use a bare print.
    """
    if verbosity >= 1:
        print(msg, flush=True)


def check_no_lake_hrus(hru_type: np.ndarray, process_name: str) -> None:
    """Raise if any HRU is a lake (hru_type = 2).

    PRMS 5.2.1 gives lake HRUs their own cascade handling, which the
    cascade processes do not implement: srunoff sends upslope Hortonian
    flow to Hortonian_lakes (srunoff.f90:697-703) and soilzone applies
    lake_evap_adj and collects upslope flow in lakein_sz
    (soilzone.f90:961-982).
    """
    wh_lake = np.where(hru_type == HruType.LAKE.value)[0]
    if len(wh_lake):
        raise NotImplementedError(
            f"{process_name} does not support lake HRUs (hru_type = 2); "
            f"lake HRU indices (0-based): {wh_lake.tolist()}"
        )


def check_cascade_param_bounds(
    hru_up_id: np.ndarray,
    hru_down_id: np.ndarray,
    hru_strmseg_down_id: np.ndarray,
    hru_pct_up: np.ndarray,
    nhru: int,
    nsegment: int,
) -> None:
    """Raise if a cascade parameter is outside the bounds PRMS declares.

    PRMS rejects such a file when it reads the parameters (the 'bounded'
    declparam calls in cascade.f90); pywatershed enforces no parameter
    bounds. Without this check a negative hru_down_id is carried into
    hru_down, which the cascade kernels then use as a stream segment
    index without bounds checks.
    """
    bounds = {
        "hru_up_id": (hru_up_id, nhru),
        "hru_down_id": (hru_down_id, nhru),
        "hru_strmseg_down_id": (hru_strmseg_down_id, nsegment),
        "hru_pct_up": (hru_pct_up, 1),
    }
    for name, (vals, upper) in bounds.items():
        wh_bad = np.where((vals < 0) | (vals > upper))[0]
        if len(wh_bad):
            raise ValueError(
                f"{name} must be in [0, {upper}]; out of bounds at cascade "
                f"indices (0-based): {wh_bad.tolist()}"
            )


def preprocess_cascade_params(
    control: Control,
    parameters: Parameters,
    verbosity: int = 1,
) -> Parameters:
    """Preprocess to obtain all cascade parameters from PRMS parameter files.

    When the control option ``cascadegw_flag`` is set, the groundwater
    reservoir (GWR) cascade parameters are added as well
    (init_gw_cascade_params).

    Args:
      control: a Control object.
      parameters: a parameter object of class Parameters.
      verbosity: 0 prints only what PRMS prints unconditionally (a cascade
        ignored for hru_up_id < 1, an hru_type rewritten to swale); 1 adds
        the diagnostics PRMS writes to cascade.msgs when Print_debug = 13.

    Returns:
      Parameters: the input parameters with all cascade parameters added
    """
    new_params = calc_hru_route_order(parameters)

    # In PRMS, basin.f90 copies hru_type and hru_route_order to gwr_type and
    # gwr_route_order BEFORE cascade.f90::order_hrus edits them (swales,
    # routing order). initgw_cascade (cascadegw_flag=1) uses those copies.
    gwr_type = new_params.parameters["hru_type"].copy()
    gwr_route_order = new_params.parameters["hru_route_order"].copy()

    new_params = init_cascade_params(control, new_params, verbosity=verbosity)

    cascadegw_flag = control.options.get("cascadegw_flag", 0)
    if cascadegw_flag:
        new_params = init_gw_cascade_params(
            control,
            new_params,
            gwr_type=gwr_type,
            gwr_route_order=gwr_route_order,
            verbosity=verbosity,
        )

    # cascade.f90 declares cascade_min (REAL, default 0.000001 inches): flows
    # at or below it do not cascade. Supply the PRMS default when the
    # parameter file omits it.
    if "cascade_min" not in new_params.parameters.keys():
        ds = new_params.to_xr_ds()
        ds["cascade_min"] = xr.Variable(
            "scalar", np.array([float(np.float32(1.0e-6))])
        )
        new_params = Parameters.from_ds(ds)

    return new_params


def _max_links(up_id: np.ndarray, nunits: int) -> int:
    """Maximum number of cascade links from any one HRU/GWR (>= 1).

    From cascade.f90::cascdecl, where the result sizes the 2-D arrays
    (ndown) for HRUs and GWRs alike.
    """
    counts = np.zeros(nunits, dtype="int64")
    for k in up_id:
        if k > 0:
            counts[k - 1] += 1
    return max(1, int(counts.max()))


def calc_hru_route_order(parameters: Parameters) -> Parameters:
    """Calculate the HRU routing order.

    This is taken from basin.f90 with all its error trapping checks. The
    returned array contains 1-based indices.

    Args:
      parameters: A Parameters object for the domain which includes hru_type

    Returns:
      Parameters: the input parameters with hru_route_order (nhru, 1-based,
        the active HRUs first, zeros after) and active_hrus (scalar, their
        count) added
    """
    nhru = parameters.dims["nhru"]
    hru_type = parameters.parameters["hru_type"]
    # PRMS bounds-checks hru_type when it reads the parameters; pywatershed
    # enforces no parameter bounds, and an unknown value would pass through
    # the dispatch below (and ActiveHruMixin) as active land.
    valid_types = [tt.value for tt in HruType]
    wh_bad = np.where(~np.isin(hru_type, valid_types))[0]
    if len(wh_bad):
        raise ValueError(
            f"hru_type must be one of {valid_types}; invalid at HRU indices "
            f"(0-based): {wh_bad.tolist()}"
        )
    hru_route_order = np.zeros(nhru, dtype=np.int32)

    nlake = parameters.dims.get("nlake", 0)
    numlakes_check = 0
    if nlake > 0:
        lake_hru_id = parameters.parameters["lake_hru_id"]

    active_hrus = 0
    for ii in range(nhru):
        if hru_type[ii] == HruType.INACTIVE.value:
            continue

        if hru_type[ii] == HruType.LAKE.value:
            if nlake == 0:
                msg = (
                    f"ERROR, hru_type = 2 for HRU: {ii} "
                    "and dimension nlake = 0"
                )
                raise ValueError(msg)
            lakeid = lake_hru_id[ii]
            if lakeid > 0:
                if lakeid > numlakes_check:
                    numlakes_check = lakeid
            else:
                msg = f"ERROR, hru_type = 2 for HRU: {ii} and lake_hru_id = 0"
                raise ValueError(msg)

        else:
            if nlake > 0 and lake_hru_id[ii] > 0:
                msg = (
                    f"ERROR, HRU: {ii} specifed to be a lake by lake_hru_id "
                    "but hru_type not equal 2"
                )
                raise ValueError(msg)

            # PRMS frozen_flag (CFGI) is not supported; its check that a
            # swale HRU cannot be frozen is omitted.

        # <<<
        active_hrus += 1
        hru_route_order[active_hrus - 1] = ii + 1

    # <
    if nlake > 0:
        if numlakes_check != nlake:
            msg = (
                "ERROR, number of lakes specified in lake_hru_id "
                f"does not equal dimension nlake: {nlake} number of "
                f"lakes: {numlakes_check}. For PRMS lake routing each lake "
                "must be a single HRU."
            )
            raise ValueError(msg)

    new_params = parameters.to_xr_ds()
    new_params["hru_route_order"] = xr.Variable("nhru", hru_route_order)
    new_params["active_hrus"] = xr.Variable(
        "scalar", np.array([active_hrus], dtype="int64")
    )
    return Parameters.from_ds(new_params)


def init_cascade_params(
    control: Control,
    parameters: Parameters,
    verbosity: int = 1,
) -> Parameters:
    """init_cascade from cascade.f90

    Args:
      control: a Control object.
      parameters: a parameter object of class Parameters.
      verbosity: 0 prints only what PRMS prints unconditionally (a cascade
        ignored for hru_up_id < 1, an hru_type rewritten to swale); 1 adds
        the diagnostics PRMS writes to cascade.msgs when Print_debug = 13.

    Returns:
      Parameters: the input parameters with the cascade parameters from
        init_cascade added
    """

    params = parameters.to_dd()

    nhru = params.dims["nhru"]
    nsegment = params.dims["nsegment"]
    ncascade = params.dims["ncascade"]
    hru_type = params.data_vars["hru_type"]
    cascade_tol = params.data_vars["cascade_tol"][0]
    active_hrus = params.data_vars["active_hrus"][0]
    hru_route_order = params.data_vars["hru_route_order"]
    circle_switch = params.data_vars["circle_switch"][0]
    ndown = 1

    cascade_flg = params.data_vars["cascade_flg"][0]
    cascade_flag = control.options["cascade_flag"]

    ncascade_hru = np.zeros([nhru], dtype="int64")
    hru_frac = np.zeros([nhru], dtype="double")

    # NOTE: because negative indices are used, we keep 1-based indexing
    # everywhere and subtract inside square brackets ***
    hru_up_id = params.data_vars["hru_up_id"]
    hru_down_id = params.data_vars["hru_down_id"]
    hru_strmseg_down_id = params.data_vars["hru_strmseg_down_id"]
    hru_pct_up = params.data_vars["hru_pct_up"]
    hru_area = params.data_vars["hru_area"]
    check_cascade_param_bounds(
        hru_up_id, hru_down_id, hru_strmseg_down_id, hru_pct_up, nhru, nsegment
    )

    # cascade_hru_segment is a constant = 2. This is the case :
    #   "2=simple cascades defined by parameter hru_segment"
    cascade_hru_segment = 2
    if cascade_flag == cascade_hru_segment:
        msg = "simple cascades defined by param hru_segment not implemented"
        raise ValueError(msg)

    #  figure out the maximum number of cascades links from all HRUs, to
    # set dimensions for 2-D arrays
    for i in range(ncascade):
        k = hru_up_id[i]
        if k > 0:
            ncascade_hru[k - 1] = ncascade_hru[k - 1] + 1
            if ncascade_hru[k - 1] > ndown:
                ndown = ncascade_hru[k - 1]

    # cascade.f90::cascdecl sizes ndown over the GWR links too, so the HRU
    # and GWR 2-D arrays share one ndown dimension.
    if control.options.get("cascadegw_flag", 0) == 1:
        ndown = max(ndown, _max_links(params.data_vars["gw_up_id"], nhru))

    if ndown > 15:
        msg = f"possible ndown issue: {ndown=}"
        _verbosity_msg(msg, verbosity)

    hru_down = np.zeros([ndown, nhru], dtype="int64")
    cascade_area = np.zeros([ndown, nhru], dtype="double")
    hru_down_frac = np.zeros([ndown, nhru], dtype="double")
    hru_down_fracwt = np.zeros([ndown, nhru], dtype="double")
    # hru_frac declared above

    # reset ncascade_hru
    ncascade_hru[:] = 0

    # these are indices not ids, they are used as indexes below
    # note per above that loaded "indices" are kept as 1-based until used
    # inside square brackets because of negative indexing in the original code.
    # <<<<
    for ii in range(ncascade):
        kup = hru_up_id[ii]
        if kup < 1:
            msg = f"Cascade ignored as hru_up_id<1, {ii+1=}, hru_up_id: {kup=}"
            print(msg)
            continue

        jdn = hru_down_id[ii]
        frac = hru_pct_up[ii]
        if frac > 0.9998:
            frac = 1.0

        istrm = hru_strmseg_down_id[ii]

        diag_msg = (
            f"\nCascade: {ii+1=}; up HRU: {kup=}; down HRU: {jdn=}; "
            f"\nup fraction: {frac=}; stream segment: {istrm=}"
        )

        # only the last of these ifs does anything before end of loop, so
        # a "continue" is not necessary except in that last case.
        if frac < 0.00001:
            msg = "Cascade ignored as hru_pct_up = 0.0, " + diag_msg
            _verbosity_msg(msg, verbosity)
        elif istrm > nsegment:
            msg = "Cascade ignored as isegment > nsegment-1, " + diag_msg
            _verbosity_msg(msg, verbosity)
        elif (kup < 1) and (jdn == 0):
            msg = "Cascade ignored as up and down HRU <0, " + diag_msg
            _verbosity_msg(msg, verbosity)
        elif (istrm == 0) and (jdn == 0):
            msg = "Cascade ignored as down HRU and segment < 0, " + diag_msg
            _verbosity_msg(msg, verbosity)
        elif hru_type[kup - 1] == HruType.INACTIVE.value:
            msg = "Cascade ignored as up HRU is inactive, " + diag_msg
            _verbosity_msg(msg, verbosity)
        elif hru_type[kup - 1] == HruType.SWALE.value:
            msg = "Cascade ignored as up HRU is a swale, " + diag_msg
            _verbosity_msg(msg, verbosity)
        elif (hru_type[kup - 1] == HruType.LAKE.value) and (istrm < 1):
            msg = (
                "Cascade ignored as lake HRU cannot cascade to an HRU"
                + diag_msg
            )
            _verbosity_msg(msg, verbosity)
        else:
            if (jdn > 0) and (istrm < 1):
                if hru_type[jdn - 1] == HruType.INACTIVE.value:
                    msg = (
                        "Cascade ignored as down HRU is inactive, " + diag_msg
                    )
                    _verbosity_msg(msg, verbosity)
                    continue

            # <
            carea = frac * hru_area[kup - 1]

            # ! get rid of small cascades, redistribute fractions
            if (carea < cascade_tol) and (frac < 0.075):
                msg = (
                    "*** WARNING, ignoring small cascade: carea<cascade_tol\n"
                    f"Cascade:  {ii+1=}; "
                    f"HRU up:  {kup=}; "
                    f"HRU down:  {jdn=}; "
                    f"fraction up:  {frac*100.0=}; "
                    f"cascade areea:  {carea=}"
                )
                _verbosity_msg(msg, verbosity)

            elif cascade_flg == 1:
                # This forces 1 to 1 cascades
                if frac > hru_frac[kup - 1]:
                    hru_frac[kup - 1] = frac
                    ncascade_hru[kup - 1] = 1
                    hru_down_frac[0, kup - 1] = frac
                    if istrm > 0:
                        hru_down[0, kup - 1] = -istrm
                    else:
                        hru_down[0, kup - 1] = jdn

            # <<<
            else:
                hru_frac[kup - 1] = hru_frac[kup - 1] + frac
                if hru_frac[kup - 1] > one:
                    if hru_frac[kup - 1] > 1.00001:
                        msg = (
                            "Addition of cascade link makes contributing area "
                            "\nadd up to > 1.0, thus fraction reduced: "
                            f"\nCascade: {ii+1=}; up HRU: {kup=}; "
                            f" down HRU: {jdn=};"
                            f" up fraction: {hru_frac[kup-1]=};"
                            f" stream segment: {istrm=}"
                        )
                        _verbosity_msg(msg, verbosity)

                    # <
                    frac = frac + 1.0 - hru_frac[kup - 1]
                    hru_frac[kup - 1] = 1.0

                # <
                ncascade_hru[kup - 1] = ncascade_hru[kup - 1] + 1
                kk = ncascade_hru[kup - 1]
                hru_down_frac[kk - 1, kup - 1] = frac
                if istrm > 0:
                    hru_down[kk - 1, kup - 1] = -istrm
                else:
                    hru_down[kk - 1, kup - 1] = jdn

    # < end of for loop

    for ii in range(active_hrus):
        i = hru_route_order[ii]
        num = ncascade_hru[i - 1]
        if num == 0:
            continue

        for k in range(num):
            frac = hru_down_frac[k, i - 1]
            hru_down_frac[k, i - 1] = (
                frac + frac * (1.0 - hru_frac[i - 1]) / hru_frac[i - 1]
            )

        # <
        k = 0
        for kk in range(num):
            dnhru = hru_down[kk, i - 1]
            if dnhru == 0:
                continue

            hru_down_frac[k, i - 1] = hru_down_frac[kk, i - 1]
            hru_down[k, i - 1] = dnhru
            j = num

            while (j - 1) > kk:
                if dnhru == hru_down[j - 1, i - 1]:
                    hru_down[j - 1, i - 1] = 0
                    hru_down_frac[k, i - 1] = (
                        hru_down_frac[k, i - 1] + hru_down_frac[j - 1, i - 1]
                    )
                    if hru_down_frac[k, i - 1] > 1.00001:
                        msg = (
                            "combining cascade links makes contributing area "
                            "add up to > 1.0, thus fraction reduced."
                            f"up hru: {i}, down hru: {dnhru}"
                        )
                        _verbosity_msg(msg, verbosity)
                        hru_down_frac[k, i - 1] = 1.0

                    # <
                    if dnhru < 0:
                        #  two cascades to same stream segment, combine
                        msg = (
                            "Combined multiple cascade paths from "
                            f"HRU: {i=} to stream segment, {abs(dnhru)=}"
                        )
                        _verbosity_msg(msg, verbosity)
                    else:
                        #  two cascades to same hru, combine
                        msg = (
                            "Combined multiple cascade paths from "
                            f"HRU: {i=}, downslope hru, {dnhru=}"
                        )
                        _verbosity_msg(msg, verbosity)

                    # <
                    ncascade_hru[i - 1] = ncascade_hru[i - 1] - 1

                # <
                j -= 1

            # <
            cascade_area[k, i - 1] = hru_down_frac[k, i - 1] * hru_area[i - 1]
            if dnhru > 0:
                hru_down_fracwt[k, i - 1] = (
                    cascade_area[k, i - 1] / hru_area[dnhru - 1]
                )

            # <
            k += 1
        # < end of while
    # < end of do

    # The port's own invariants, checked before the arrays are used as
    # indices: every cascade of an HRU names a target, and the fractions
    # of an HRU's cascades sum to one.
    for ii in range(active_hrus):
        i = hru_route_order[ii]
        num = ncascade_hru[i - 1]
        if num == 0:
            continue
        if (hru_down[:num, i - 1] == 0).any():
            raise RuntimeError(
                f"hru_down has a zero among the {num} cascades of HRU {i}"
            )
        frac_sum = hru_down_frac[:num, i - 1].sum()
        if abs(frac_sum - one) > 1e-6:
            raise RuntimeError(
                f"hru_down_frac of HRU {i} sums to {frac_sum}, not 1"
            )

    hru_type, hru_route_order = order_hrus(
        nhru,
        active_hrus,
        hru_route_order,
        ncascade_hru,
        hru_down,
        hru_type,
        circle_switch,
        verbosity=verbosity,
    )

    msg = f"{hru_route_order=}"
    _verbosity_msg(msg, verbosity)

    new_params = parameters.to_xr_ds()
    new_params["hru_type"] = xr.Variable("nhru", hru_type)
    new_params["hru_route_order"] = xr.Variable("nhru", hru_route_order)
    new_params["ncascade_hru"] = xr.Variable("nhru", ncascade_hru)

    new_params["cascade_area"] = xr.Variable(["ndown", "nhru"], cascade_area)
    new_params["hru_down"] = xr.Variable(["ndown", "nhru"], hru_down)
    new_params["hru_down_frac"] = xr.Variable(["ndown", "nhru"], hru_down_frac)
    new_params["hru_down_fracwt"] = xr.Variable(
        ["ndown", "nhru"], hru_down_fracwt
    )

    return Parameters.from_ds(new_params)


def order_hrus(
    nhru: int,
    active_hrus: int,
    hru_route_order: np.ndarray,
    ncascade_hru: np.ndarray,
    hru_down: np.ndarray,
    hru_type: np.ndarray,
    circle_switch: int,
    verbosity: int = 1,
) -> tuple:
    """From cascade.f90::order_hrus.

    Rewrites a land or glacier HRU with no cascade to a swale and
    recomputes the routing order so every HRU follows all of its upslope
    HRUs; checks for circular cascades when circle_switch is 1.

    Args:
      nhru: number of HRUs.
      active_hrus: number of active HRUs.
      hru_route_order: 1-based, the active HRUs first, zeros after.
        Modified in place.
      ncascade_hru: number of cascades of each HRU.
      hru_down: (ndown, nhru) 1-based downslope HRU (positive) or stream
        segment (negative) of each cascade.
      hru_type: modified in place (swale rewrite).
      circle_switch: 1 to raise on a circular cascade.
      verbosity: as for :func:`init_cascade_params`.

    Returns:
      tuple: (hru_type, hru_route_order), the two arrays modified in place.

    Raises:
      ValueError: on a circular cascade, or when the ordering cannot place
        every active HRU.
    """

    # up_id_count equals number of upslope HRUs an HRU has.
    # dn_id_count equals number of downslope HRUs an HRU has.
    # ncascade_hru equals number of downslope HRUs and stream segments
    # an HRU has.
    max_up_id_count = 0

    up_id_count = np.zeros(nhru, dtype="int64")
    dn_id_count = np.zeros(nhru, dtype="int64")
    roots = np.zeros(nhru, dtype="int64")
    is_hru_on_list = np.zeros(nhru, dtype="int64")

    for ii in range(active_hrus):
        i = hru_route_order[ii]
        for k in range(ncascade_hru[i - 1]):
            dnhru = hru_down[k, i - 1]
            if dnhru > 0:
                dn_id_count[i - 1] = dn_id_count[i - 1] + 1
                up_id_count[dnhru - 1] = up_id_count[dnhru - 1] + 1
                # determine the maximum up_id_count
                if up_id_count[dnhru - 1] > max_up_id_count:
                    max_up_id_count = up_id_count[dnhru - 1]

    # <<<<
    hrus_up_list = np.zeros([max_up_id_count, nhru], dtype="int64")
    # get the list of HRUs upslope of each HRU and root HRUs
    up_id_cnt = up_id_count.copy()

    nroots = 0

    for ii in range(active_hrus):
        i = hru_route_order[ii]
        if dn_id_count[i - 1] == 0:
            nroots = nroots + 1
            roots[nroots - 1] = i
        # <
        if up_id_count[i - 1] == 0:
            # hru does not receive or cascade flow - swale
            if (
                (hru_type[i - 1] == HruType.LAND.value)
                or (hru_type[i - 1] == HruType.GLACIER.value)
            ) and ncascade_hru[i - 1] == 0:
                msg = (
                    f"HRU {i} does not cascade or receive flow "
                    "and was specified as hru_type = 1. "
                    "hru_type was changed to 3 (swale)"
                )
                print(msg)
                hru_type[i - 1] = HruType.SWALE.value
                continue

        # <<
        if (
            (hru_type[i - 1] == HruType.LAND.value)
            or (hru_type[i - 1] == HruType.GLACIER.value)
        ) and (ncascade_hru[i - 1] == 0):
            # hru does not cascade flow - swale
            msg = (
                f"HRU {i=} receives flow but does not cascade and was "
                "specified as hru_type 1. hru_type was changed to 3 (swale)"
            )
            print(msg)
            hru_type[i - 1] = HruType.SWALE.value
            continue
        else:
            for k in range(ncascade_hru[i - 1]):
                dnhru = hru_down[k, i - 1]
                if dnhru > 0:
                    hrus_up_list[up_id_cnt[dnhru - 1] - 1, dnhru - 1] = i
                    up_id_cnt[dnhru - 1] = up_id_cnt[dnhru - 1] - 1

    # <<<< End of for loop

    # check for circles when circle_switch = 1. cascade.f90 walks up from
    # each root recursively (up_tree/check_path); a directed-graph cycle
    # search is equivalent and also catches a cycle with no root above it.
    if circle_switch == ACTIVE:
        graph = nx.DiGraph()
        for ii in range(active_hrus):
            i = hru_route_order[ii]
            for k in range(ncascade_hru[i - 1]):
                dnhru = hru_down[k, i - 1]
                if dnhru > 0:
                    graph.add_edge(i, dnhru)
        try:
            cycle = nx.find_cycle(graph)
        except nx.NetworkXNoCycle:
            pass
        else:
            msg = (
                "Circular cascading path specified among HRUs: "
                f"{[edge[0] for edge in cycle]}"
            )
            raise ValueError(msg)

    # <<

    # determine hru routing order
    hru_route_order[:] = 0
    iorder = 0  # number of hrus added to hru_route_order
    while iorder < (active_hrus):
        added = 0
        for i in range(nhru):
            if hru_type[i] == HruType.INACTIVE.value:
                continue

            if is_hru_on_list[i] == 0:
                goes_on_list = 1
                for j in range(up_id_count[i]):
                    up_hru_id = hrus_up_list[j, i]
                    # if upslope hru not on list, can't add hru i
                    if is_hru_on_list[up_hru_id - 1] == 0:
                        goes_on_list = 0

                        break

                # <<
                # add hru to list
                if goes_on_list == 1:
                    is_hru_on_list[i] = 1
                    iorder = iorder + 1
                    hru_route_order[iorder - 1] = i + 1  # keep it 1-based
                    added = 1

        # <<<
        if added == 0:
            not_in_order_list = []
            for i in range(nhru):
                if is_hru_on_list[i] == 0:
                    not_in_order_list.append(i)

            msg = ""
            if len(not_in_order_list):
                msg = f"indices of hrus not in order: {not_in_order_list}\n\n"

            msg += (
                "No HRUs added to routing order on last pass through \n"
                "cascades, possible circles. \n"
                f"{hru_route_order=}"
            )
            raise ValueError(msg)

    # <<
    msg = (
        f"{nroots=} HRUs do not cascade to another HRU (roots)\n"
        f"{roots[0:nroots]=}"
    )
    _verbosity_msg(msg, verbosity)

    if iorder != active_hrus:
        list_missing_hrus = []
        list_inactive_hrus = []
        for i in range(nhru):
            if is_hru_on_list[i] == 0:
                if hru_type[i] != HruType.INACTIVE.value:
                    list_missing_hrus.append(i)
                else:
                    # PRMS lists inactive HRUs separately, not as missing
                    list_inactive_hrus.append(i)

        # <<<
        msg = (
            "Not all HRUs are included in the cascading pattern,\n"
            "likely circle or inactive HRUs.\n"
            f"Number of HRUs in pattern: {iorder=}\n"
            f"Number of HRUs: {nhru=}\n"
            f"Number of active HRUs: {active_hrus=}\n"
            f"HRUs missing: {list_missing_hrus}\n"
            f"HRUs inactive: {list_inactive_hrus}\n"
        )
        raise ValueError(msg)

    return hru_type, hru_route_order


def init_gw_cascade_params(
    control: Control,
    parameters: Parameters,
    gwr_type: np.ndarray,
    gwr_route_order: np.ndarray,
    verbosity: int = 1,
) -> Parameters:
    """Groundwater reservoir (GWR) cascades from cascade.f90::cascinit.

    Requires the HRU cascade parameters (init_cascade_params) to be present.
    Control option ``cascadegw_flag``: 1 derives the GWR cascades from the
    parameters gw_up_id, gw_down_id, gw_pct_up and gw_strmseg_down_id
    (initgw_cascade + order_gwrs); 2 copies the HRU cascades.

    Args:
      control: a Control object.
      parameters: a Parameters object with the HRU cascade parameters.
      gwr_type: hru_type as it was BEFORE init_cascade_params edited it
        (PRMS copies it in basin.f90). Only used for cascadegw_flag=1.
      gwr_route_order: hru_route_order as it was BEFORE init_cascade_params
        (basin.f90 order). Only used for cascadegw_flag=1.
      verbosity: Currently an integer in [0, 1], boolean.

    Returns:
      Parameters: the input parameters with gwr_route_order, ncascade_gwr,
        gwr_down, gwr_down_frac, and cascade_gwr_area added.
    """
    cascadegw_flag = control.options.get("cascadegw_flag", 0)
    gwr_swale_flag = control.options.get("gwr_swale_flag", 0)
    if gwr_swale_flag != 0:
        msg = "gwr_swale_flag != 0 is not implemented in pywatershed"
        raise NotImplementedError(msg)

    params = parameters.to_dd()
    nhru = params.dims["nhru"]
    if "ngw" in params.dims.keys() and params.dims["ngw"] != nhru:
        msg = f"ngw ({params.dims['ngw']}) must equal nhru ({nhru})"
        raise ValueError(msg)

    if cascadegw_flag == 2:
        # CASCADEGW_SAME: GWR cascades set to the (ordered) HRU cascades
        gwr_type = params.data_vars["hru_type"]
        gwr_route_order = params.data_vars["hru_route_order"].copy()
        ncascade_gwr = params.data_vars["ncascade_hru"].copy()
        gwr_down = params.data_vars["hru_down"].copy()
        gwr_down_frac = params.data_vars["hru_down_frac"].copy()
        cascade_gwr_area = params.data_vars["cascade_area"].copy()

    elif cascadegw_flag == 1:
        (
            ncascade_gwr,
            gwr_down,
            gwr_down_frac,
            cascade_gwr_area,
            gwr_type,
            gwr_route_order,
        ) = _initgw_cascade(
            params,
            gwr_type.copy(),
            gwr_route_order.copy(),
            ndown=params.data_vars["hru_down"].shape[0],
            verbosity=verbosity,
        )

    else:
        msg = f"invalid cascadegw_flag value: {cascadegw_flag}"
        raise ValueError(msg)

    # cascinit: GWRs cannot be swales when gwr_swale_flag = 0
    active_gwrs = int((gwr_type != HruType.INACTIVE.value).sum())
    for ii in range(active_gwrs):
        i = gwr_route_order[ii]
        if gwr_type[i - 1] == HruType.SWALE.value:
            msg = f"GWR is a swale when gwr_swale_flag = 0, GWR: {i}"
            raise ValueError(msg)

    new_params = parameters.to_xr_ds()
    new_params["gwr_route_order"] = xr.Variable("nhru", gwr_route_order)
    new_params["ncascade_gwr"] = xr.Variable("nhru", ncascade_gwr)
    new_params["gwr_down"] = xr.Variable(["ndown", "nhru"], gwr_down)
    new_params["gwr_down_frac"] = xr.Variable(["ndown", "nhru"], gwr_down_frac)
    new_params["cascade_gwr_area"] = xr.Variable(
        ["ndown", "nhru"], cascade_gwr_area
    )

    return Parameters.from_ds(new_params)


def _initgw_cascade(
    params: DatasetDict,
    gwr_type: np.ndarray,
    gwr_route_order: np.ndarray,
    ndown: int,
    verbosity: int = 1,
) -> tuple:
    """From cascade.f90::initgw_cascade (cascadegw_flag=1).

    The same link-building as init_cascade_params for HRUs, with these
    differences taken from the Fortran: gw_pct_up is capped at 1.0 (not
    0.9998), there is no lake check, and no gwr_down_fracwt.
    """

    def verbosity_msg(msg: str, verbosity_thresh: int = 1) -> None:
        if verbosity >= verbosity_thresh:
            print(msg, flush=True)

    nhru = params.dims["nhru"]
    nsegment = params.dims["nsegment"]
    ncascdgw = params.dims["ncascdgw"]
    cascade_tol = params.data_vars["cascade_tol"][0]
    cascade_flg = params.data_vars["cascade_flg"][0]
    circle_switch = params.data_vars["circle_switch"][0]
    hru_area = params.data_vars["hru_area"]
    active_gwrs = int((gwr_type != HruType.INACTIVE.value).sum())

    # 1-based indices kept as in init_cascade_params (negative = segment)
    gw_up_id = params.data_vars["gw_up_id"]
    gw_down_id = params.data_vars["gw_down_id"]
    gw_strmseg_down_id = params.data_vars["gw_strmseg_down_id"]
    gw_pct_up = params.data_vars["gw_pct_up"]

    ncascade_gwr = np.zeros([nhru], dtype="int64")
    gwr_frac = np.zeros([nhru], dtype="double")
    gwr_down = np.zeros([ndown, nhru], dtype="int64")
    gwr_down_frac = np.zeros([ndown, nhru], dtype="double")
    cascade_gwr_area = np.zeros([ndown, nhru], dtype="double")

    for ii in range(ncascdgw):
        kup = gw_up_id[ii]
        if kup < 1:
            msg = f"Cascade ignored as gw_up_id<1, cascade: {ii + 1}, {kup=}"
            print(msg)
            continue

        jdn = gw_down_id[ii]
        frac = gw_pct_up[ii]
        if frac > 1.0:
            frac = 1.0
        istrm = gw_strmseg_down_id[ii]

        diag_msg = (
            f"\nCascade: {ii+1=}; up GWR: {kup=}; down GWR: {jdn=}; "
            f"\nup fraction: {frac=}; stream segment: {istrm=}"
        )

        if frac < 0.00001:
            print("Cascade ignored as gw_pct_up=0.0, " + diag_msg)
        elif istrm > nsegment:
            print("Cascade ignored as segment>nsegment, " + diag_msg)
        elif (kup < 1) and (jdn == 0):
            print("Cascade ignored as up and down GWR = 0, " + diag_msg)
        elif (istrm == 0) and (jdn == 0):
            print("Cascade ignored as down GWR and segment = 0, " + diag_msg)
        elif gwr_type[kup - 1] == HruType.INACTIVE.value:
            print("Cascade ignored as up GWR is inactive, " + diag_msg)
        elif gwr_type[kup - 1] == HruType.SWALE.value:
            # gwr_swale_flag == 0 is guaranteed by the caller
            msg = (
                "ERROR, Cascade as up GWR specified as a swale and "
                "gwr_swale_flag = 0" + diag_msg
            )
            raise ValueError(msg)
        else:
            if (jdn > 0) and (istrm < 1):
                if gwr_type[jdn - 1] == HruType.INACTIVE.value:
                    msg = "Cascade ignored as down GWR is inactive, "
                    print(msg + diag_msg)
                    continue

            carea = frac * hru_area[kup - 1]

            # get rid of small cascades, redistribute fractions
            if (carea < cascade_tol) and (frac < 0.075):
                msg = (
                    "*** WARNING, ignoring small GWR cascade: "
                    "carea<cascade_tol\n"
                    f"Cascade: {ii+1=}; GWR up: {kup=}; GWR down: {jdn=}; "
                    f"fraction up: {frac*100.0=}; cascade area: {carea=}"
                )
                print(msg)

            elif cascade_flg == 1:
                # This forces 1 to 1 cascades
                if frac > gwr_frac[kup - 1]:
                    gwr_frac[kup - 1] = frac
                    ncascade_gwr[kup - 1] = 1
                    gwr_down_frac[0, kup - 1] = frac
                    if istrm > 0:
                        gwr_down[0, kup - 1] = -istrm
                    else:
                        gwr_down[0, kup - 1] = jdn

            else:
                gwr_frac[kup - 1] = gwr_frac[kup - 1] + frac
                if gwr_frac[kup - 1] > one:
                    if gwr_frac[kup - 1] > 1.00001:
                        msg = (
                            "Addition of GWR cascade link makes contributing "
                            "area add up to > 1.0, thus fraction reduced: "
                            f"\nCascade: {ii+1=}; up GWR: {kup=}; "
                            f" down GWR: {jdn=};"
                            f" up fraction: {gwr_frac[kup-1]=};"
                            f" stream segment: {istrm=}"
                        )
                        print(msg)

                    frac = frac + 1.0 - gwr_frac[kup - 1]
                    gwr_frac[kup - 1] = 1.0

                ncascade_gwr[kup - 1] = ncascade_gwr[kup - 1] + 1
                kk = ncascade_gwr[kup - 1]
                gwr_down_frac[kk - 1, kup - 1] = frac
                if istrm > 0:
                    gwr_down[kk - 1, kup - 1] = -istrm
                else:
                    gwr_down[kk - 1, kup - 1] = jdn

    # < end of link loop

    for ii in range(active_gwrs):
        i = gwr_route_order[ii]
        num = ncascade_gwr[i - 1]
        if num == 0:
            continue

        for k in range(num):
            frac = gwr_down_frac[k, i - 1]
            gwr_down_frac[k, i - 1] = (
                frac + frac * (1.0 - gwr_frac[i - 1]) / gwr_frac[i - 1]
            )

        k = 0
        for kk in range(num):
            dngwr = gwr_down[kk, i - 1]
            if dngwr == 0:
                continue

            gwr_down_frac[k, i - 1] = gwr_down_frac[kk, i - 1]
            gwr_down[k, i - 1] = dngwr
            j = num

            while (j - 1) > kk:
                if dngwr == gwr_down[j - 1, i - 1]:
                    gwr_down[j - 1, i - 1] = 0
                    gwr_down_frac[k, i - 1] = (
                        gwr_down_frac[k, i - 1] + gwr_down_frac[j - 1, i - 1]
                    )
                    if gwr_down_frac[k, i - 1] > 1.00001:
                        msg = (
                            "combining GWR cascade links makes contributing "
                            "area add up to > 1.0, thus fraction reduced."
                            f"up GWR: {i}, down GWR: {dngwr}"
                        )
                        verbosity_msg(msg)
                        gwr_down_frac[k, i - 1] = 1.0

                    if dngwr < 0:
                        msg = (
                            "Combined multiple cascade paths from "
                            f"GWR: {i=} to stream segment, {abs(dngwr)=}"
                        )
                    else:
                        msg = (
                            "Combined multiple cascade paths from "
                            f"GWR: {i=}, downslope GWR, {dngwr=}"
                        )
                    print(msg)

                    ncascade_gwr[i - 1] = ncascade_gwr[i - 1] - 1

                j -= 1

            cascade_gwr_area[k, i - 1] = (
                gwr_down_frac[k, i - 1] * hru_area[i - 1]
            )
            k += 1

    # < end of combine loop

    gwr_type, gwr_route_order = order_gwrs(
        nhru,
        active_gwrs,
        gwr_route_order,
        ncascade_gwr,
        gwr_down,
        gwr_type,
        circle_switch,
    )

    verbosity_msg(f"{gwr_route_order=}")

    return (
        ncascade_gwr,
        gwr_down,
        gwr_down_frac,
        cascade_gwr_area,
        gwr_type,
        gwr_route_order,
    )


def order_gwrs(
    nhru: int,
    active_gwrs: int,
    gwr_route_order: np.ndarray,
    ncascade_gwr: np.ndarray,
    gwr_down: np.ndarray,
    gwr_type: np.ndarray,
    circle_switch: int,
) -> tuple:
    """From cascade.f90::order_gwrs, for gwr_swale_flag = 0.

    Differs from order_hrus: a GWR that does not cascade is an error (with
    gwr_swale_flag > 0 PRMS would change its type to swale instead), and
    there is no lake handling.
    """
    max_up_id_count = 0
    up_id_count = np.zeros(nhru, dtype="int64")
    dn_id_count = np.zeros(nhru, dtype="int64")
    roots = np.zeros(nhru, dtype="int64")
    is_gwr_on_list = np.zeros(nhru, dtype="int64")

    for ii in range(active_gwrs):
        i = gwr_route_order[ii]
        for k in range(ncascade_gwr[i - 1]):
            dngwr = gwr_down[k, i - 1]
            if dngwr > 0:
                dn_id_count[i - 1] = dn_id_count[i - 1] + 1
                up_id_count[dngwr - 1] = up_id_count[dngwr - 1] + 1
                if up_id_count[dngwr - 1] > max_up_id_count:
                    max_up_id_count = up_id_count[dngwr - 1]

    gwrs_up_list = np.zeros([max_up_id_count, nhru], dtype="int64")
    up_id_cnt = up_id_count.copy()

    nroots = 0
    no_cascade_gwrs = []
    for ii in range(active_gwrs):
        i = gwr_route_order[ii]
        if dn_id_count[i - 1] == 0:
            nroots = nroots + 1
            roots[nroots - 1] = i

        if ncascade_gwr[i - 1] == 0:
            # GWR does not cascade flow: an error when gwr_swale_flag = 0
            # (Fortran formats 9010/9011 then Iret = 1)
            no_cascade_gwrs.append(i)
            continue

        for k in range(ncascade_gwr[i - 1]):
            dngwr = gwr_down[k, i - 1]
            if dngwr > 0:
                gwrs_up_list[up_id_cnt[dngwr - 1] - 1, dngwr - 1] = i
                up_id_cnt[dngwr - 1] = up_id_cnt[dngwr - 1] - 1

    if len(no_cascade_gwrs):
        msg = (
            "GWRs that do not cascade flow with gwr_swale_flag = 0: "
            f"{no_cascade_gwrs}"
        )
        raise ValueError(msg)

    del up_id_cnt

    # check for circles when circle_switch = 1 (see order_hrus)
    if circle_switch == ACTIVE:
        graph = nx.DiGraph()
        for ii in range(active_gwrs):
            i = gwr_route_order[ii]
            for k in range(ncascade_gwr[i - 1]):
                dngwr = gwr_down[k, i - 1]
                if dngwr > 0:
                    graph.add_edge(i, dngwr)
        try:
            cycle = nx.find_cycle(graph)
        except nx.NetworkXNoCycle:
            pass
        else:
            msg = (
                "Circular cascading path specified among GWRs: "
                f"{[edge[0] for edge in cycle]}"
            )
            raise ValueError(msg)

    # determine GWR routing order
    gwr_route_order[:] = 0
    igworder = 0
    while igworder < active_gwrs:
        added = 0
        for i in range(nhru):
            if gwr_type[i] == HruType.INACTIVE.value:
                continue

            if is_gwr_on_list[i] == 0:
                goes_on_list = 1
                for j in range(up_id_count[i]):
                    up_gwr_id = gwrs_up_list[j, i]
                    if is_gwr_on_list[up_gwr_id - 1] == 0:
                        goes_on_list = 0
                        break

                if goes_on_list == 1:
                    is_gwr_on_list[i] = 1
                    igworder = igworder + 1
                    gwr_route_order[igworder - 1] = i + 1  # keep it 1-based
                    added = 1

        if added == 0:
            not_in_order_list = [
                i for i in range(nhru) if is_gwr_on_list[i] == 0
            ]
            msg = (
                f"indices of GWRs not in order: {not_in_order_list}\n\n"
                "No GWRs added to routing order on last pass through \n"
                "cascades, possible circles. \n"
                f"{gwr_route_order=}"
            )
            raise ValueError(msg)

    msg = (
        f"{nroots=} GWRs do not cascade to another GWR (roots)\n"
        f"{roots[0:nroots]=}"
    )
    print(msg)

    if igworder != active_gwrs:
        list_missing = [
            i
            for i in range(nhru)
            if is_gwr_on_list[i] == 0 and gwr_type[i] != HruType.INACTIVE.value
        ]
        msg = (
            "Not all GWRs are included in the cascading pattern,\n"
            "likely circle or inactive GWRs.\n"
            f"Number of GWRs in pattern: {igworder=}\n"
            f"Number of GWRs: {nhru=}\n"
            f"Number of active GWRs: {active_gwrs=}\n"
            f"GWRs missing: {list_missing}\n"
        )
        raise ValueError(msg)

    return gwr_type, gwr_route_order
