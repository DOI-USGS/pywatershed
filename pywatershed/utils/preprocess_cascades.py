import networkx as nx
import numpy as np
import xarray as xr

from ..base import Control
from ..base.data_model import DatasetDict
from ..constants import ACTIVE, HruType, one
from ..parameters import Parameters

# NOTES:
# * A preprocess needs to return new parametr objects or files for
#   PRMSRunoff, PRMSSoilzone, and MAYBE PRMSGroundwater
# * This combines functionality from cascade.f90 and basin.f90
# * Neglecting/commenting basin calculations below.
# * Is it strage that hru_route_order is calculated here in basing.f90,
#   hru_route_order is also calculated or edited in cascade.f90::order_hrus

# Is this a discretization edit?


def preprocess_cascade_params(
    control: Control,
    parameters: Parameters,
    verbosity: int = 1,
) -> Parameters:
    """Preprocess to obtain all cascade parameters from PRMS parameter files.

    This function combines the legacy calc_hru_route_order and
    init_cascade_params routines into one pre-processing step.

    Args:
      control: a Control object.
      parameters: a parameter object of class Parameters.
      verbosity: Currently an integer in [0, 1], boolean.

    """
    new_params = calc_hru_route_order(parameters)
    return init_cascade_params(control, new_params, verbosity=verbosity)


def calc_hru_route_order(parameters: Parameters) -> Parameters:
    """Calculate the HRU routing order.

    This is taken from basin.f90 with all its error trapping checks. The
    returned array contains 1-based indices.

    Args:
      parameters: A Parameters object for the domain which includes hru_type

    """
    nhru = parameters.dims["nhru"]
    hru_type = parameters.parameters["hru_type"]
    hru_route_order = np.zeros(nhru, dtype=np.int32)

    nlake = parameters.dims.get("nlake", 0)
    numlake_hrus = 0  # to verify we have all the lakes
    numlakes_check = 0
    if nlake > 0:
        lake_hru_id = parameters.parameters["lake_hru_id"]

    frozen_flag = HruType.INACTIVE.value  # until further notice

    active_hrus = 0
    for ii in range(nhru):
        if hru_type[ii] == HruType.INACTIVE.value:
            continue

        # ?? need to fix for lakes with multiple HRUs and PRMS lake routing ??
        if hru_type[ii] == HruType.LAKE.value:
            numlake_hrus = numlake_hrus + 1
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

            if frozen_flag == ACTIVE:
                if hru_type[ii] == HruType.SWALE.value:
                    msg = (
                        "ERROR, a swale HRU cannot be frozen for CFGI, "
                        f"HRU: {ii}"
                    )
                    raise ValueError(msg)

        # <<<
        active_hrus += 1
        hru_route_order[active_hrus - 1] = ii + 1

        if hru_type[ii] == HruType.LAKE.value:
            continue

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
    return Parameters.from_dataset_dict(DatasetDict.from_ds(new_params))


def init_cascade_params(
    control: Control,
    parameters: Parameters,
    verbosity: int = 1,
) -> Parameters:
    """init_cascade from cascade.f90

    Args:
      control: a Control object.
      parameters: a parameter object of class Parameters.
      verbosity: Currently an integer in [0, 1], boolean.

    """

    def verbosity_msg(msg, verbosity_thresh: int = 1, warn: bool = False):
        if verbosity >= verbosity_thresh:
            print(msg, flush=True)

    # This is changed by order_hrus but may be printed diagnostically here
    iorder = 0

    # since we are doing the BAD thing of editing the parameters, they have to
    # be exported first to a DatasetDict
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

    # brilliant
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

    # cascade_hru_segment is a constant = 2. This is the case :
    #   "2=simple cascades defined by parameter hru_segment"
    cascade_hru_segment = 2
    if cascade_flag == cascade_hru_segment:
        msg = "simple cascades defined by param hru_segment not implemented"
        raise ValueError(msg)
    else:
        #  figure out the maximum number of cascades links from all HRUs, to
        # set dimensions for 2-D arrays
        ncascade_hru[:] = 0
        for i in range(ncascade):
            k = hru_up_id[i]
            if k > 0:
                jdn = hru_down_id[i]  # this line does anything?
                ncascade_hru[k - 1] = ncascade_hru[k - 1] + 1
                if ncascade_hru[k - 1] > ndown:
                    ndown = ncascade_hru[k - 1]

    if ndown > 15:
        msg = f"possible ndown issue: {ndown=}"
        verbosity_msg(msg)

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

        msg = ""
        # only the last of these ifs does anything before end of loop, so
        # a "continue" is not necessary except in that last case.
        if frac < 0.00001:
            msg = "Cascade ignored as hru_pct_up = 0.0, " + diag_msg
            print(msg)
        elif istrm > nsegment:
            msg = "Cascade ignored as isegment > nsegment-1, " + diag_msg
            print(msg)
        elif (kup < 1) and (jdn == 0):
            msg = "Cascade ignored as up and down HRU <0, " + diag_msg
            print(msg)
        elif (istrm == 0) and (jdn == 0):
            msg = "Cascade ignored as down HRU and segment < 0, " + diag_msg
            print(msg)
        elif hru_type[kup - 1] == HruType.INACTIVE.value:
            msg = "Cascade ignored as up HRU is inactive, " + diag_msg
            print(msg)
        elif hru_type[kup - 1] == HruType.SWALE.value:
            msg = "Cascade ignored as up HRU is a swale, " + diag_msg
            print(msg)
        elif (hru_type[kup - 1] == HruType.LAKE.value) and (istrm < 1):
            msg = (
                "Cascade ignored as lake HRU cannot cascade to an HRU"
                + diag_msg
            )
            print(msg)
        else:
            if (jdn > 0) and (istrm < 1):
                if hru_type[jdn - 1] == HruType.INACTIVE.value:
                    msg = (
                        "Cascade ignored as down HRU is inactive, " + diag_msg
                    )
                    print(msg)
                    continue

            # <
            # TODO: This logic is horrible, no need to be in this else. FIX
            carea = frac * hru_area[kup - 1]

            # ! get rid of small cascades, redistribute fractions
            if (carea < cascade_tol) and (frac < 0.075):
                msg = (
                    "*** WARNING, ignoring small cascade: carea<cascade_tol\n"
                    # "carea < cascade_tol and  frac < 0.075: "
                    f"Cascade:  {ii+1=}; "
                    f"HRU up:  {kup=}; "
                    f"HRU down:  {jdn=}; "
                    f"fraction up:  {frac*100.0=}; "
                    f"cascade areea:  {carea=}"
                )
                print(msg)

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
                        print(msg)

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

    # how do we route headwater HRUs to stream segment rather than
    # across valleys**********************RSR???
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
        # import pdb

        # if i == 90:
        #     pdb.set_trace()
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
                        verbosity_msg(msg)
                        hru_down_frac[k, i - 1] = 1.0

                    # <
                    if dnhru < 0:
                        #  two cascades to same stream segment, combine
                        msg = (
                            "Combined multiple cascade paths from "
                            f"HRU: {i=} to stream segment, {abs(dnhru)=}"
                        )
                        print(msg)
                    else:
                        #  two cascades to same hru, combine
                        msg = (
                            "Combined multiple cascade paths from "
                            f"HRU: {i=}, downslope hru, {dnhru=}"
                        )
                        print(msg)

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

    (
        iorder,
        hru_type,
        hru_route_order,
    ) = order_hrus(
        nhru,
        active_hrus,
        hru_route_order,
        ncascade_hru,
        hru_down,
        hru_type,
        circle_switch,
    )

    msg = f"{hru_route_order=}"
    verbosity_msg(msg)

    new_params = parameters.to_xr_ds()
    del new_params["hru_type"]
    new_params["hru_type"] = xr.Variable("nhru", hru_type)
    new_params["hru_route_order"] = xr.Variable("nhru", hru_route_order)
    new_params["ncascade_hru"] = xr.Variable("nhru", ncascade_hru)

    new_params["cascade_area"] = xr.Variable(["ndown", "nhru"], cascade_area)
    new_params["hru_down"] = xr.Variable(["ndown", "nhru"], hru_down)
    new_params["hru_down_frac"] = xr.Variable(["ndown", "nhru"], hru_down_frac)
    new_params["hru_down_fracwt"] = xr.Variable(
        ["ndown", "nhru"], hru_down_fracwt
    )

    # This is a hack of convenience. Once again, a process on one
    # discretization needs to know about the discretization of another process
    # instead of just passing on information. Runoff/soilzone should not
    # need to know about nsegments. We'll look for a way to remove this in the
    # future.
    nseg_dum = np.arange(params.dims["nsegment"])
    new_params["nsegment_dum"] = xr.Variable("nsegment", nseg_dum)

    return Parameters.from_dataset_dict(DatasetDict.from_ds(new_params))


def order_hrus(
    nhru: int,
    active_hrus: int,
    hru_route_order: np.ndarray,
    ncascade_hru: np.ndarray,
    hru_down: np.ndarray,
    hru_type: np.ndarray,
    circle_switch: int,
) -> tuple:
    """From cascade.f90::order_hrus."""

    # up_id_count equals number of upslope HRUs an HRU has.
    # dn_id_count equals number of downslope HRUs an HRU has.
    # ncascade_hru equals number of downslope HRUs and stream segments
    # an HRU has.
    max_up_id_count = 0

    # ALLOCATE (up_id_count(Nhru), dn_id_count(Nhru), roots(Nhru))
    # ALLOCATE (path(Nhru), is_hru_on_list(Nhru))
    # for i in range(nhru): # Unecessary loop
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
    # up_id_cnt = np.zeros(nhru, dtype="int64")
    up_id_cnt = up_id_count.copy()

    nroots = 0
    # type_flag = 0

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
                # type_flag = 1
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
            # type_flag = 1
            continue
        else:
            for k in range(ncascade_hru[i - 1]):
                dnhru = hru_down[k, i - 1]
                if dnhru > 0:
                    hrus_up_list[up_id_cnt[dnhru - 1] - 1, dnhru - 1] = i
                    up_id_cnt[dnhru - 1] = up_id_cnt[dnhru - 1] - 1

    # <<<< End of for loop

    del up_id_cnt

    # if type_flag==1:
    # not going to write the file in the type_flag ==1 case. We are returning
    # a new parameter object here.

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
            # huh? this section is a bit of a head scratcher
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
            # iret = 0  # pointless
            raise ValueError(msg)

    # <<
    msg = (
        f"{nroots=} HRUs do not cascade to another HRU (roots)\n"
        f"{roots[0:nroots]=}"
    )
    print(msg)

    if iorder != active_hrus:
        list_missing_hrus = []
        list_inactive_hrus = []
        for i in range(nhru):
            if is_hru_on_list[i] == 0:
                if hru_type[i] != HruType.INACTIVE.value:
                    list_missing_hrus.append(i)
                else:
                    list_inactive_hrus.append(i)
                    # apparently not an error in this case?

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

    return iorder, hru_type, hru_route_order
