import pathlib as pl
from typing import Literal, Union
from warnings import warn

import numpy as np

from ..base.adapter import adaptable
from ..base.control import Control
from ..constants import cubic_ft_per_acre_in, nan, zero
from ..parameters import Parameters
from ..utils.preprocess_cascades import preprocess_cascade_params
from .prms_groundwater import PRMSGroundwater


class PRMSGroundwaterCascadesNoDprst(PRMSGroundwater):
    """PRMS groundwater reservoir with cascades and no depression storage.

    Implementation based on PRMS 5.2.1 with theoretical documentation given in
    the PRMS-IV documentation:

    `Markstrom, S. L., Regan, R. S., Hay, L. E., Viger, R. J., Webb, R. M.,
    Payn, R. A., & LaFontaine, J. H. (2015). PRMS-IV, the
    precipitation-runoff modeling system, version 4. US Geological Survey
    Techniques and Methods, 6, B7.
    <https://pubs.usgs.gov/tm/6b7/pdf/tm6-b7.pdf>`__

    Groundwater reservoirs (GWRs) are processed in the cascade routing order
    (parameter gwr_route_order): outflow from an upslope GWR is added to the
    inflow of its downslope GWRs (gw_upslope) or to the stream segment inflow
    (stream_seg_in) in the same timestep, as in gwflow.f90 (rungw_cascade).
    The cascade parameters are derived from the PRMS parameter file by
    :func:`~pywatershed.utils.preprocess_cascades.preprocess_cascade_params`
    when the parameters do not already hold gwr_route_order.

    Args:
        control: a Control object
        discretization: a discretization of class Parameters
        parameters: a parameter object of class Parameters
        soil_to_gw: Portion of excess flow to the capillary reservoir that
            drains to the associated GWR for each HRU
        ssr_to_gw: Drainage from the gravity-reservoir to the associated GWR
            for each HRU
        stream_seg_in: Flow into each stream segment from cascading flow
            (cfs), accumulated across HRUs during the timestep
        imbalance_behavior: one of ["defer", None, "warn", "error"]
            with "defer" being the default and defering to
            control.options["imbalance_behavior"] when available. When
            control.options["imbalance_behavior"] is not avaiable,
            imbalance_behavior is set to "warn".
        calc_method: one of ["numba", "numpy"]. None defaults to
            "numba".
        verbose: Print extra information or not?
        restart_read:
            May be boolean or a Pathlib.Path. If False, control.options
            will be examined for this key. If True, the working
            directory is searched for restart files. If a Pathlib.Path, this
            specifies an alternative directory to search for restart files.
            Files searched for are of the pattern YYYY-mm-dd-varname.nc where
            the date is the control.init_time. The timestamp on the file is the
            valid time of the states in the file with the exception of
            processes with sub-daily timesteps. For example, the outflow_ts
            variable of PRMSChannel is instantaneous and valid at the 23rd hour
            of the timestampped day whereas its variable seg_outflow is the
            daily averge value over the timestampped day.
        restart_write:
            As for restart_read but for writing. The directory in either
            case will be attempted to be created if it does not exist.
        restart_write_freq:
            If False, then control.options is examined for this key. The
            follwing values set the frequency of restart output with "y" for
            yearly, "m" for monthly, "d" for daily, or "f" for final. "Final"
            means that restart files are written with the states at
            control.end_time to files timestampped with control.end_time.
            Yearly and monthly restart options write files with timestamps on
            the last day of each year or month during the run. If daily,
            restarts are written every day. If restart_write is not False and
            restart_write_freq is False, the default of "f" is used.
    """

    def __init__(
        self,
        control: Control,
        discretization: Parameters,
        parameters: Parameters,
        soil_to_gw: adaptable,
        ssr_to_gw: adaptable,
        stream_seg_in: adaptable,
        imbalance_behavior: Literal["defer", None, "warn", "error"] = "defer",
        calc_method: Literal["numba", "numpy"] = None,
        input_aliases: dict = None,
        verbose: bool = None,
        restart_read: Union[pl.Path, bool] = False,
        restart_write: Union[pl.Path, bool] = False,
        restart_write_freq: Literal["y", "m", "d", "f", False] = False,
    ) -> None:
        self._dprst_flag = False

        if "gwr_route_order" not in parameters.parameters.keys():
            parameters = preprocess_cascade_params(
                control, parameters, verbosity=int(bool(verbose))
            )

        super().__init__(
            control=control,
            discretization=discretization,
            parameters=parameters,
            soil_to_gw=soil_to_gw,
            ssr_to_gw=ssr_to_gw,
            dprst_seep_hru=None,
            stream_seg_in=stream_seg_in,
            imbalance_behavior=imbalance_behavior,
            calc_method=calc_method,
            input_aliases=input_aliases,
            verbose=verbose,
            restart_read=restart_read,
            restart_write=restart_write,
            restart_write_freq=restart_write_freq,
        )

        self.name = "PRMSGroundwaterCascadesNoDprst"
        self._wh_inactive_hrus = np.where(~self._active_hru_mask)[0]
        self._set_budget(active_mask=self._active_hru_mask)

        return

    @staticmethod
    def get_dimensions() -> tuple:
        return ("nhru", "nsegment")

    @staticmethod
    def get_parameters() -> tuple:
        return (
            "hru_area",
            "hru_type",
            "hru_in_to_cf",
            "gwflow_coef",
            "gwsink_coef",
            "gwstor_init",
            "gwstor_min",
            "cascade_min",
            "gwr_route_order",
            "ncascade_gwr",
            "gwr_down",
            "gwr_down_frac",
            "cascade_gwr_area",
        )

    @staticmethod
    def get_inputs() -> tuple:
        return (
            "soil_to_gw",
            "ssr_to_gw",
            "stream_seg_in",
        )

    @staticmethod
    def get_mass_budget_terms():
        return {
            "inputs": [
                "soil_to_gw",
                "ssr_to_gw",
                "gw_upslope_hru",
            ],
            "outputs": [
                "gwres_flow",
                "hru_gw_cascadeflow",
            ],
            "storage_changes": [
                "gwres_stor_change",
            ],
        }

    @staticmethod
    def get_init_values() -> dict:
        return {
            "gwres_flow": nan,
            "gwres_flow_vol": nan,
            "gwres_sink": nan,
            "gwres_stor": nan,
            "gwres_stor_old": nan,
            "gwres_stor_change": nan,
            "gw_upslope": zero,
            "gw_upslope_hru": zero,
            "hru_gw_cascadeflow": zero,
        }

    @staticmethod
    def get_restart_variables() -> list:
        # The cascade variables (gw_upslope, hru_gw_cascadeflow) are
        # re-accumulated from zero every timestep, so only the reservoir
        # storage is state (as PRMSGroundwater).
        return ["gwres_stor"]

    def _init_calc_method(self):
        if self._calc_method is None:
            self._calc_method = "numba"

        avail_methods = ["numpy", "numba"]

        if self._calc_method.lower() not in avail_methods:
            msg = (
                f"Invalid calc_method={self._calc_method} for {self.name}. "
                f"Setting calc_method to 'numba' for {self.name}"
            )
            warn(msg)
            self._calc_method = "numba"

        if self._calc_method.lower() == "numba":
            import numba as nb

            # The loop is order-dependent (upslope before downslope GWRs),
            # so the kernel is never parallelized.
            print(f"{self.name} jit compiling with numba ", flush=True)
            # no fastmath: it fuses gwflow / area * hru_in_to_cf and moves
            # gwres_flow_vol off the numpy result by more than 1e-13
            self._calculate_gw = nb.njit(fastmath=False, parallel=False)(
                self._calculate_numpy
            )

        else:
            self._calculate_gw = self._calculate_numpy

        return

    def _calculate(self, simulation_time) -> None:
        self._simulation_time = simulation_time
        cfs_conv = cubic_ft_per_acre_in / self.control.time_step_seconds

        (
            self.gwres_stor[:],
            self.gwres_flow[:],
            self.gwres_sink[:],
            self.gwres_stor_change[:],
            self.gwres_flow_vol[:],
            self.gw_upslope[:],
            self.gw_upslope_hru[:],
            self.hru_gw_cascadeflow[:],
            self.stream_seg_in[:],
        ) = self._calculate_gw(
            nactive_gwrs=self._nactive_hrus,
            gwr_route_order=self.gwr_route_order,
            wh_inactive_hrus=self._wh_inactive_hrus,
            gwarea=self.hru_area,
            soil_to_gw=self.soil_to_gw,
            ssr_to_gw=self.ssr_to_gw,
            gwres_stor=self.gwres_stor,
            gwflow_coef=self.gwflow_coef,
            gwsink_coef=self.gwsink_coef,
            gwres_stor_old=self.gwres_stor_old,
            hru_in_to_cf=self.hru_in_to_cf,
            cascade_min=self.cascade_min[0],
            ncascade_gwr=self.ncascade_gwr,
            gwr_down=self.gwr_down,
            gwr_down_frac=self.gwr_down_frac,
            cascade_gwr_area=self.cascade_gwr_area,
            stream_seg_in=self.stream_seg_in,
            cfs_conv=cfs_conv,
        )
        return

    @staticmethod
    def _calculate_numpy(
        nactive_gwrs,
        gwr_route_order,
        wh_inactive_hrus,
        gwarea,
        soil_to_gw,
        ssr_to_gw,
        gwres_stor,
        gwflow_coef,
        gwsink_coef,
        gwres_stor_old,
        hru_in_to_cf,
        cascade_min,
        ncascade_gwr,
        gwr_down,
        gwr_down_frac,
        cascade_gwr_area,
        stream_seg_in,
        cfs_conv,
    ):
        """gwflow.f90::gwflowrun with rungw_cascade, cascadegw_flag > 0.

        Volumes (acre-inches) are used inside the loop as in the Fortran;
        depths (inches) are returned. gw_upslope is kept in acre-inches as
        PRMS reports it; gw_upslope_hru is the same water as a depth on the
        receiving GWR for the mass budget.
        """
        nhru = len(gwarea)
        new_gwres_stor = np.full(nhru, np.nan)
        gwres_flow = np.full(nhru, np.nan)
        gwres_sink = np.full(nhru, np.nan)
        gw_upslope = np.zeros(nhru)
        hru_gw_cascadeflow = np.zeros(nhru)

        for jj in range(nactive_gwrs):
            ii = gwr_route_order[jj] - 1
            area = gwarea[ii]
            gwstor = gwres_stor[ii] * area
            gwin = (soil_to_gw[ii] + ssr_to_gw[ii]) * area + gw_upslope[ii]
            gwstor = gwstor + gwin

            gwflow = zero
            gwsink = zero
            if gwstor > zero:
                gwflow = gwstor * gwflow_coef[ii]
                gwstor = gwstor - gwflow
                if gwsink_coef[ii] > zero:
                    gwsink = min(gwstor * gwsink_coef[ii], gwstor)
                    gwstor = gwstor - gwsink
            elif gwstor < zero:
                gwstor = zero

            flow = gwflow / area  # inches
            gwres_sink[ii] = gwsink / area

            # rungw_cascade: don't cascade small flows
            if (flow > cascade_min) and (ncascade_gwr[ii] > 0):
                dnflow = zero
                for kk in range(ncascade_gwr[ii]):
                    jdn = gwr_down[kk, ii]
                    if jdn > 0:
                        # cascade contributes to a downslope GWR
                        gw_upslope[jdn - 1] = (
                            gw_upslope[jdn - 1]
                            + flow * cascade_gwr_area[kk, ii]
                        )
                        dnflow = dnflow + flow * gwr_down_frac[kk, ii]
                    elif jdn < 0:
                        # cascade contributes to a stream segment
                        stream_seg_in[-jdn - 1] = (
                            stream_seg_in[-jdn - 1]
                            + flow * cascade_gwr_area[kk, ii] * cfs_conv
                        )

                # gwres_flow reduced by cascading flow to GWRs
                flow = flow - dnflow
                if flow < zero:
                    flow = zero
                hru_gw_cascadeflow[ii] = dnflow

            gwres_flow[ii] = flow
            new_gwres_stor[ii] = gwstor / area

        gwres_stor_change = new_gwres_stor - gwres_stor_old
        gwres_flow_vol = gwres_flow * hru_in_to_cf
        gw_upslope_hru = gw_upslope / gwarea

        # The storage and flow arrays start as NaN, so their inactive entries
        # stay NaN; these three accumulate from zeros and need masking.
        if len(wh_inactive_hrus) > 0:
            gw_upslope[wh_inactive_hrus] = np.nan
            gw_upslope_hru[wh_inactive_hrus] = np.nan
            hru_gw_cascadeflow[wh_inactive_hrus] = np.nan

        return (
            new_gwres_stor,
            gwres_flow,
            gwres_sink,
            gwres_stor_change,
            gwres_flow_vol,
            gw_upslope,
            gw_upslope_hru,
            hru_gw_cascadeflow,
            stream_seg_in,
        )
