import pathlib as pl
from typing import Literal, Union
from warnings import warn

import numpy as np
from numba import prange

from ..base.adapter import adaptable
from ..base.conservative_process import ConservativeProcess
from ..base.control import Control
from ..base.hru_mixin import HruMixin
from ..constants import (
    HruType,
    dnearzero,
    nan,
    nearzero,
    numba_num_threads,
    zero,
)
from ..parameters import Parameters

RAIN = 0
SNOW = 1

BARESOIL = 0
GRASSES = 1

OFF = 0
ACTIVE = 1

LAND = HruType.LAND.value
LAKE = HruType.LAKE.value

# TODO: using through_rain and not net_rain and net_ppt is a WIP


class PRMSRunoff(ConservativeProcess, HruMixin):
    """PRMS surface runoff.

    A surface runoff representation from PRMS.

    Implementation based on PRMS 5.2.1 with theoretical documentation given in
    the PRMS-IV documentation:

    `Markstrom, S. L., Regan, R. S., Hay, L. E., Viger, R. J., Webb, R. M.,
    Payn, R. A., & LaFontaine, J. H. (2015). PRMS-IV, the
    precipitation-runoff modeling system, version 4. US Geological Survey
    Techniques and Methods, 6, B7.
    <https://pubs.usgs.gov/tm/6b7/pdf/tm6-b7.pdf>`__


    Args:
        control: a Control object
        discretization: a discretization of class Parameters
        parameters: a parameter object of class Parameters
        soil_lower_prev: Previous storage of lower reservoir for each HRU
        soil_rechr_prev: Previous storage of recharge reservoir for each HRU
        net_ppt: Precipitation (rain and/or snow) that falls through the
            canopy for each HRU
        net_rain: Rain that falls through canopy for each HRU
        net_snow: Snow that falls through canopy for each HRU
        potet: Potential ET for each HRU
        snowmelt: Snowmelt from snowpack on each HRU
        snow_evap: Evaporation and sublimation from snowpack on each HRU
        pkwater_equiv: Snowpack water equivalent on each HRU
        pptmix_nopack: Flag indicating that a mixed precipitation event has
            occurred with no snowpack
        snowcov_area: Snow-covered area on each HRU prior to melt and
            sublimation unless snowpack
        through_rain: Rain that passes through snow when no snow present
        hru_intcpevap: HRU area-weighted average evaporation from the
            canopy for each HRU
        intcp_changeover: Canopy throughfall caused by canopy density
            change from winter to summer
        dprst_flag: use depression storage or not? None uses value in control
            file, which otherwise defaults to True.
        intcp_changeover_in_net_rain: Boolean flag indicating whether
            intcp_changeover is included in net rain (GSFLOW 4.2.0 and PRMS
            6.0.0) or not (pywatershed and PRMS < 6.0.0).
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

    # Cascades make the HRU loop order-dependent. Subclasses that route
    # cascades set this False so numba never parallelizes the kernel.
    _nb_parallel_ok = True

    def __init__(
        self,
        control: Control,
        discretization: Parameters,
        parameters: Parameters,
        soil_lower_prev: adaptable,
        soil_rechr_prev: adaptable,
        net_ppt: adaptable,
        net_rain: adaptable,
        net_snow: adaptable,
        potet: adaptable,
        snowmelt: adaptable,
        snow_evap: adaptable,
        pkwater_equiv: adaptable,
        pptmix_nopack: adaptable,
        snowcov_area: adaptable,
        through_rain: adaptable,
        hru_intcpevap: adaptable,
        intcp_changeover: adaptable,
        dprst_flag: Union[bool, None] = None,
        intcp_changeover_in_net_rain: bool = False,
        imbalance_behavior: Literal["defer", None, "warn", "error"] = "defer",
        calc_method: Literal["numba", "numpy", None] = None,
        input_aliases: dict = None,
        verbose: Union[bool, None] = None,
        restart_read: Union[pl.Path, bool] = False,
        restart_write: Union[pl.Path, bool] = False,
        restart_write_freq: Literal["y", "m", "d", "f", False] = False,
    ) -> None:
        self._dprst_flag = dprst_flag
        if self._dprst_flag is None:
            self._dprst_flag = True

        if not hasattr(self, "name"):
            self.name = "PRMSRunoff"

        super().__init__(
            control=control,
            discretization=discretization,
            parameters=parameters,
            input_aliases=input_aliases,
            restart_read=restart_read,
            restart_write=restart_write,
            restart_write_freq=restart_write_freq,
        )

        self._set_active_hrus()
        self._mask_inactive_hrus()
        self._set_inputs(locals())
        self._set_options(locals())

        self._set_budget(active_mask=self._active_hru_mask)
        self._init_calc_method()

        if self._intcp_changeover_in_net_rain is None:
            if "intcp_changeover_in_net_rain" in self.control.options.keys():
                self._intcp_changeover_in_net_rain = self.control.options[
                    "intcp_changeover_in_net_rain"
                ]
            else:
                self._intcp_changeover_in_net_rain = False
                # raise ValueError(
                #     "intcp_changeover_in_net_rain must be specified for "
                #     "{self.name}"
                # )

        self.basin_init()

        if self._dprst_flag:
            self.dprst_init()

        if restart_read is not False or restart_write is not False:
            self.restart_read = False
            self.restart_write = False

        return

    def _set_initial_conditions(self):
        """Set initial conditions for variables not in get_init_values"""
        # These should probably be private
        self.dprst_in = np.zeros(self.nhru, dtype=float)
        self.dprst_vol_open_max = np.zeros(self.nhru, dtype=float)
        self.dprst_vol_clos_max = np.zeros(self.nhru, dtype=float)
        self.dprst_frac_clos = np.zeros(self.nhru, dtype=float)
        self.dprst_vol_thres_open = np.zeros(self.nhru, dtype=float)
        # self.basin_init()

        return

    def _init_diagnostic_vars(self):
        # if self._dprst_flag:
        #     self.dprst_init()
        pass

    @staticmethod
    def get_dimensions() -> tuple:
        return ("nhru",)

    @staticmethod
    def get_parameters() -> tuple:
        return (
            "hru_type",
            "hru_area",
            "hru_in_to_cf",
            "hru_percent_imperv",
            "imperv_stor_max",
            "carea_max",
            "smidx_coef",
            "smidx_exp",
            "soil_moist_max",
            "snowinfil_max",
            "dprst_depth_avg",
            "dprst_et_coef",
            "dprst_flow_coef",
            "dprst_frac",
            "dprst_frac_init",
            "dprst_frac_open",
            "dprst_seep_rate_clos",
            "dprst_seep_rate_open",
            "sro_to_dprst_imperv",
            "sro_to_dprst_perv",
            "va_open_exp",
            "va_clos_exp",
            "op_flow_thres",
        )

    @staticmethod
    def get_inputs() -> tuple:
        return (
            "soil_lower_prev",
            "soil_rechr_prev",
            "net_rain",
            "net_ppt",
            "net_snow",
            "potet",
            "snowmelt",
            "snow_evap",
            "pkwater_equiv",
            "pptmix_nopack",
            "snowcov_area",
            "through_rain",
            "hru_intcpevap",
            "intcp_changeover",
        )

    @staticmethod
    def get_init_values() -> dict:
        return {
            "contrib_fraction": zero,
            "infil": zero,
            "infil_hru": nan,
            "sroff": zero,
            "sroff_vol": zero,
            "hru_sroffp": zero,
            "hru_sroffi": zero,
            "imperv_stor": zero,
            "imperv_evap": zero,
            "hru_impervevap": zero,
            "hru_impervstor": zero,
            "hru_impervstor_old": zero,
            "hru_impervstor_change": zero,
            "dprst_vol_frac": zero,
            "dprst_vol_clos": zero,
            "dprst_vol_open": zero,
            "dprst_vol_clos_frac": zero,
            "dprst_vol_open_frac": zero,
            "dprst_area_clos": zero,
            "dprst_area_open": zero,
            "dprst_area_clos_max": zero,
            "dprst_area_open_max": zero,
            "dprst_sroff_hru": zero,
            "dprst_seep_hru": zero,
            "dprst_evap_hru": zero,
            "dprst_insroff_hru": zero,
            "dprst_stor_hru": zero,
            "dprst_stor_hru_old": zero,
            "dprst_stor_hru_change": zero,
            "dprst_vol_thres_open": zero,
        }

    @staticmethod
    def get_restart_variables() -> list:
        return [
            "imperv_stor",
            "hru_impervstor",
            "dprst_stor_hru",
            "dprst_area_open",
            "dprst_area_clos",
            "dprst_vol_open",
            "dprst_vol_clos",
            "dprst_vol_thres_open",
        ]

    @staticmethod
    def get_mass_budget_terms():
        return {
            "inputs": [
                "through_rain",
                "snowmelt",
                "intcp_changeover",
            ],
            "outputs": [
                # sroff = hru_sroffi + hru_sroffp + dprst_sroff_hru
                "hru_sroffi",
                "hru_sroffp",
                "dprst_sroff_hru",
                "infil_hru",
                "hru_impervevap",
                "dprst_seep_hru",
                "dprst_evap_hru",
            ],
            "storage_changes": [
                "hru_impervstor_change",
                "dprst_stor_hru_change",
            ],
        }

    def basin_init(self):
        """
        This is trying to replicate the prms basin.f90/basinit() function that
        calculates some of the variables needed here by runoff.  This should
        probably go somewhere else at some point as I suspect other components
        may need similar information.
        """

        self.hru_perv = np.zeros(self.nhru, float)
        self.hru_frac_perv = np.zeros(self.nhru, float)
        self.hru_imperv = np.zeros(self.nhru, float)
        self.dprst_area_max = np.zeros(self.nhru, float)

        self._dprst_open_flag = OFF
        self._dprst_clos_flag = OFF

        for k in range(self.nhru):
            i = k
            harea = self.hru_area[k]
            perv_area = harea
            if self.hru_percent_imperv[k] > 0.0:
                self.hru_imperv[i] = self.hru_percent_imperv[i] * harea
                perv_area = perv_area - self.hru_imperv[i]

            if self._dprst_flag == ACTIVE:
                self.dprst_area_max[i] = self.dprst_frac[i] * harea
                if self.dprst_area_max[i] > 0.0:
                    self.dprst_area_open_max[i] = (
                        self.dprst_area_max[i] * self.dprst_frac_open[i]
                    )
                    self.dprst_frac_clos[i] = 1.0 - self.dprst_frac_open[i]
                    self.dprst_area_clos_max[i] = (
                        self.dprst_area_max[i] - self.dprst_area_open_max[i]
                    )
                    if self.dprst_area_clos_max[i] > 0.0:
                        self._dprst_clos_flag = ACTIVE
                    if self.dprst_area_open_max[i] > 0.0:
                        self._dprst_open_flag = ACTIVE
                    perv_area = perv_area - self.dprst_area_max[i]

            self.hru_perv[i] = perv_area
            self.hru_frac_perv[i] = perv_area / harea

        # <
        if not hasattr(self, "hru_route_order"):
            # hru_route_order in cascades is 1-based index, keep it the same.
            if not hasattr(self, "_wh_active_hrus"):
                # subclasses (e.g. PRMSRunoffAg) which do not call
                # _set_active_hrus in their inits get the active HRU
                # information on demand.
                self._set_active_hrus()
            self.hru_route_order = self._wh_active_hrus + 1

        return

    def dprst_init(self):
        if self._dprst_clos_flag == OFF:
            # This is a BAD practice of editing parameters.
            # not possible if we use [:] on the LHS
            self.dprst_seep_rate_clos = self.dprst_seep_rate_clos * 0.0
            self.va_clos_exp = self.va_clos_exp * 0.0

        for j in range(self.nhru):
            i = j
            if self.dprst_frac[i] > 0.0:
                if self.dprst_depth_avg[i] == 0.0:
                    raise Exception(
                        f"dprst_fac > and dprst_depth_avg == 0 for HRU {i}"
                    )

                # calculate open and closed volumes (acre-inches) of depression
                # storage by HRU
                # Dprst_area_open_max is the maximum open depression area
                # (acres) that can generate surface runoff:
                self._dprst_clos_flag = ACTIVE
                if self._dprst_clos_flag == ACTIVE:
                    self.dprst_vol_clos_max[i] = (
                        self.dprst_area_clos_max[i] * self.dprst_depth_avg[i]
                    )
                self._dprst_open_flag = ACTIVE
                if self._dprst_open_flag == ACTIVE:
                    self.dprst_vol_open_max[i] = (
                        self.dprst_area_open_max[i] * self.dprst_depth_avg[i]
                    )

                # calculate the initial open and closed depression storage
                # volume:
                if not self._restart_read:
                    self._dprst_open_flag = ACTIVE
                    if self._dprst_open_flag == ACTIVE:
                        self.dprst_vol_open[i] = (
                            self.dprst_frac_init[i]
                            * self.dprst_vol_open_max[i]
                        )
                    self._dprst_clos_flag = ACTIVE
                    if self._dprst_clos_flag == ACTIVE:
                        self.dprst_vol_clos[i] = (
                            self.dprst_frac_init[i]
                            * self.dprst_vol_clos_max[i]
                        )

                # threshold volume is calculated as the % of maximum open
                # depression storage above which flow occurs *  total open
                # depression storage volume
                self.dprst_vol_thres_open[i] = (
                    self.op_flow_thres[i] * self.dprst_vol_open_max[i]
                )
                if self.dprst_vol_open[i] > 0.0:
                    open_vol_r = (
                        self.dprst_vol_open[i] / self.dprst_vol_open_max[i]
                    )
                    if open_vol_r < nearzero:
                        frac_op_ar = 0.0
                    elif open_vol_r > 1.0:
                        frac_op_ar = 1.0
                    else:
                        frac_op_ar = np.exp(
                            self.va_open_exp[i] * np.log(open_vol_r)
                        )
                    # <
                    self.dprst_area_open[i] = (
                        self.dprst_area_open_max[i] * frac_op_ar
                    )
                    if self.dprst_area_open[i] > self.dprst_area_open_max[i]:
                        self.dprst_area_open[i] = self.dprst_area_open_max[i]

                # Closed depression surface area for each HRU:
                if self.dprst_vol_clos[i] > 0.0:
                    clos_vol_r = (
                        self.dprst_vol_clos[i] / self.dprst_vol_clos_max[i]
                    )
                    if clos_vol_r < nearzero:
                        frac_cl_ar = 0.0
                    elif clos_vol_r > 1.0:
                        frac_cl_ar = 1.0
                    else:
                        frac_cl_ar = np.exp(
                            self.va_clos_exp[i] * np.log(clos_vol_r)
                        )
                    self.dprst_area_clos[i] = (
                        self.dprst_area_clos_max[i] * frac_cl_ar
                    )
                    if self.dprst_area_clos[i] > self.dprst_area_clos_max[i]:
                        self.dprst_area_clos[i] = self.dprst_area_clos_max[i]

                #
                # calculate basin open and closed depression storage volumes
                self.dprst_stor_hru[i] = (
                    self.dprst_vol_open[i] + self.dprst_vol_clos[i]
                ) / self.hru_area[i]
                self.dprst_stor_hru_old[i] = self.dprst_stor_hru[i]

                if (
                    self.dprst_vol_open_max[i] + self.dprst_vol_clos_max[i]
                    > 0.0
                ):
                    self.dprst_vol_frac[i] = (
                        self.dprst_vol_open[i] + self.dprst_vol_clos[i]
                    ) / (
                        self.dprst_vol_open_max[i] + self.dprst_vol_clos_max[i]
                    )
        return

    def _init_calc_method(self):
        if self._calc_method is None:
            self._calc_method = "numba"

        if self._calc_method.lower() not in ["numpy", "numba"]:
            msg = (
                f"Invalid calc_method={self._calc_method} for {self.name}. "
                f"Setting calc_method to 'numba' for {self.name}"
            )
            warn(msg, UserWarning)
            self._calc_method = "numba"

        if self._calc_method.lower() == "numba":
            import numba as nb

            numba_msg = f"{self.name} jit compiling with numba "
            nb_parallel = (
                (numba_num_threads is not None)
                and (numba_num_threads > 1)
                and self._nb_parallel_ok
            )
            if nb_parallel:
                numba_msg += f"and using {numba_num_threads} threads"
            print(numba_msg, flush=True)

            self.check_capacity = nb.njit(self.check_capacity)
            self.perv_comp = nb.njit(self.perv_comp)
            self.compute_infil = nb.njit(self.compute_infil)
            self.dprst_comp = nb.njit(self.dprst_comp)
            self.imperv_et = nb.njit(self.imperv_et)
            self._run_cascade_sroff = nb.njit(self._run_cascade_sroff)
            self._run_cascade_sroff_dummy = nb.njit(
                self._run_cascade_sroff_dummy
            )

            self._calculate_runoff = nb.njit(
                self._calculate_numpy, parallel=nb_parallel
            )

        else:
            self._calculate_runoff = self._calculate_numpy

        return

    def _advance_variables(self) -> None:
        self.hru_impervstor_old[:] = self.hru_impervstor
        self.dprst_stor_hru_old[:] = self.dprst_stor_hru
        return None

    def _calculate(self, time_length, vectorized=False):
        """Perform the core calculations"""
        zero_array_2d_int = np.zeros((2, 2), dtype="int32")
        nan_array = np.nan * self.infil
        nan_array_2d = np.zeros((2, 2)) * np.nan

        (
            self.infil[:],
            self.contrib_fraction[:],
            self.hru_sroffp[:],
            self.hru_sroffi[:],
            self.imperv_evap[:],
            self.hru_impervevap[:],
            self.imperv_stor[:],
            self.dprst_in[:],
            self.dprst_vol_open[:],
            self.dprst_vol_clos[:],
            self.dprst_sroff_hru[:],
            self.dprst_evap_hru[:],
            self.dprst_seep_hru[:],
            self.dprst_insroff_hru[:],
            self.dprst_vol_open_frac[:],
            self.dprst_vol_clos_frac[:],
            self.dprst_vol_frac[:],
            self.dprst_stor_hru[:],
            self.sroff[:],
            _,
            _,
        ) = self._calculate_runoff(
            infil=self.infil,
            nhru=self.nhru,
            hru_area=self.hru_area,
            hru_perv=self.hru_perv,
            hru_frac_perv=self.hru_frac_perv,
            hru_sroffp=self.hru_sroffp,
            contrib_fraction=self.contrib_fraction,
            hru_percent_imperv=self.hru_percent_imperv,
            hru_sroffi=self.hru_sroffi,
            imperv_evap=self.imperv_evap,
            hru_imperv=self.hru_imperv,
            hru_impervevap=self.hru_impervevap,
            potet=self.potet,
            snow_evap=self.snow_evap,
            hru_intcpevap=self.hru_intcpevap,
            soil_lower_prev=self.soil_lower_prev,
            soil_rechr_prev=self.soil_rechr_prev,
            soil_moist_max=self.soil_moist_max,
            carea_max=self.carea_max,
            smidx_coef=self.smidx_coef,
            smidx_exp=self.smidx_exp,
            pptmix_nopack=self.pptmix_nopack,
            net_rain=self.net_rain,
            net_ppt=self.net_ppt,
            imperv_stor=self.imperv_stor,
            imperv_stor_max=self.imperv_stor_max,
            snowmelt=self.snowmelt,
            snowinfil_max=self.snowinfil_max,
            net_snow=self.net_snow,
            pkwater_equiv=self.pkwater_equiv,
            hru_type=self.hru_type,
            intcp_changeover=self.intcp_changeover,
            dprst_in=self.dprst_in,
            dprst_seep_hru=self.dprst_seep_hru,
            dprst_area_max=self.dprst_area_max,
            dprst_vol_open=self.dprst_vol_open,
            dprst_vol_clos=self.dprst_vol_clos,
            dprst_sroff_hru=self.dprst_sroff_hru,
            dprst_evap_hru=self.dprst_evap_hru,
            dprst_insroff_hru=self.dprst_insroff_hru,
            dprst_vol_open_frac=self.dprst_vol_open_frac,
            dprst_vol_clos_frac=self.dprst_vol_clos_frac,
            dprst_vol_frac=self.dprst_vol_frac,
            dprst_stor_hru=self.dprst_stor_hru,
            dprst_area_clos_max=self.dprst_area_clos_max,
            dprst_area_clos=self.dprst_area_clos,
            dprst_vol_open_max=self.dprst_vol_open_max,
            dprst_area_open_max=self.dprst_area_open_max,
            dprst_area_open=self.dprst_area_open,
            sro_to_dprst_perv=self.sro_to_dprst_perv,
            sro_to_dprst_imperv=self.sro_to_dprst_imperv,
            dprst_frac_open=self.dprst_frac_open,
            dprst_frac_clos=self.dprst_frac_clos,
            va_open_exp=self.va_open_exp,
            dprst_vol_clos_max=self.dprst_vol_clos_max,
            va_clos_exp=self.va_clos_exp,
            snowcov_area=self.snowcov_area,
            dprst_et_coef=self.dprst_et_coef,
            dprst_seep_rate_open=self.dprst_seep_rate_open,
            dprst_vol_thres_open=self.dprst_vol_thres_open,
            dprst_flow_coef=self.dprst_flow_coef,
            dprst_seep_rate_clos=self.dprst_seep_rate_clos,
            sroff=self.sroff,
            hru_impervstor=self.hru_impervstor,
            through_rain=self.through_rain,
            dprst_flag=self._dprst_flag,
            intcp_changeover_in_net_rain=self._intcp_changeover_in_net_rain,
            ncascade_hru=nan_array,
            nactive_hrus=self._nactive_hrus,
            hru_route_order=self.hru_route_order,
            hru_down=zero_array_2d_int,
            hru_down_frac=nan_array_2d,
            hru_down_fracwt=nan_array_2d,
            cascade_area=nan_array_2d,
            hortonian_flow=nan_array,
            upslope_hortonian=nan_array,
            stream_seg_in=nan_array,
            cfs_conv=nan_array,
            # functions at end
            check_capacity=self.check_capacity,
            perv_comp=self.perv_comp,
            compute_infil=self.compute_infil,
            dprst_comp=self.dprst_comp,
            imperv_et=self.imperv_et,
            run_cascade_sroff=self._run_cascade_sroff_dummy,
        )

        self.infil_hru[:] = self.infil * self.hru_frac_perv

        self.hru_impervstor_change[:] = (
            self.hru_impervstor - self.hru_impervstor_old
        )
        self.dprst_stor_hru_change[:] = (
            self.dprst_stor_hru - self.dprst_stor_hru_old
        )

        self.sroff_vol[:] = self.sroff * self.hru_in_to_cf

        return

    @staticmethod
    def _calculate_numpy(
        infil,
        nhru,
        hru_area,
        hru_perv,
        hru_frac_perv,
        hru_sroffp,
        contrib_fraction,
        hru_percent_imperv,
        hru_sroffi,
        imperv_evap,
        hru_imperv,
        hru_impervevap,
        potet,
        snow_evap,
        hru_intcpevap,
        soil_lower_prev,
        soil_rechr_prev,
        soil_moist_max,
        carea_max,
        smidx_coef,
        smidx_exp,
        pptmix_nopack,
        net_rain,
        net_ppt,
        imperv_stor,
        imperv_stor_max,
        snowmelt,
        snowinfil_max,
        net_snow,
        pkwater_equiv,
        hru_type,
        intcp_changeover,
        dprst_in,
        dprst_seep_hru,
        dprst_area_max,
        dprst_vol_open,
        dprst_vol_clos,
        dprst_sroff_hru,
        dprst_evap_hru,
        dprst_insroff_hru,
        dprst_vol_open_frac,
        dprst_vol_clos_frac,
        dprst_vol_frac,
        dprst_stor_hru,
        dprst_area_clos_max,
        dprst_area_clos,
        dprst_vol_open_max,
        dprst_area_open_max,
        dprst_area_open,
        sro_to_dprst_perv,
        sro_to_dprst_imperv,
        dprst_frac_open,
        dprst_frac_clos,
        va_open_exp,
        dprst_vol_clos_max,
        va_clos_exp,
        snowcov_area,
        dprst_et_coef,
        dprst_seep_rate_open,
        dprst_vol_thres_open,
        dprst_flow_coef,
        dprst_seep_rate_clos,
        sroff,
        hru_impervstor,
        through_rain,
        dprst_flag,
        intcp_changeover_in_net_rain,
        ncascade_hru,
        nactive_hrus,
        hru_route_order,
        hru_down,
        hru_down_frac,
        hru_down_fracwt,
        cascade_area,
        hortonian_flow,
        upslope_hortonian,
        stream_seg_in,
        cfs_conv,
        # functions at end
        check_capacity,
        perv_comp,
        compute_infil,
        dprst_comp,
        imperv_et,
        run_cascade_sroff,
    ):
        hru_horton_cascflow = np.zeros(nhru, dtype="float64")
        ncascade_hru_active = ~np.isnan(ncascade_hru).all()
        if ncascade_hru_active:
            # PRMS zeroes both every timestep in srunoffrun (srunoff.f90).
            # Soilzone adds to stream_seg_in later in the same step.
            upslope_hortonian[:] = zero
            stream_seg_in[:] = zero

        dprst_chk = 0
        infil[:] = 0.0

        soil_moist_prev = soil_lower_prev + soil_rechr_prev

        for k in prange(nactive_hrus):
            # TODO: remove duplicated vars
            # TODO: move setting constants outside the loop.
            # note that i is 0-based index

            i = hru_route_order[k] - 1

            runoff = zero

            hruarea = hru_area[i]
            perv_area = hru_perv[i]
            perv_frac = hru_frac_perv[i]
            srp = zero
            sri = zero
            hru_sroffp[i] = zero
            contrib_fraction[i] = zero
            hruarea_imperv = hru_imperv[i]
            imperv_frac = zero
            if hruarea_imperv > zero:
                imperv_frac = hru_percent_imperv[i]
                hru_sroffi[i] = zero
                imperv_evap[i] = zero
                hru_impervevap[i] = zero

            avail_et = potet[i] - snow_evap[i] - hru_intcpevap[i]
            availh2o = intcp_changeover[i] + net_rain[i]

            (
                sri,
                srp,
                imperv_stor[i],
                infil[i],
                contrib_fraction[i],
            ) = compute_infil(
                contrib_fraction=contrib_fraction[i],
                soil_moist_prev=soil_moist_prev[i],
                soil_moist_max=soil_moist_max[i],
                carea_max=carea_max[i],
                smidx_coef=smidx_coef[i],
                smidx_exp=smidx_exp[i],
                pptmix_nopack=pptmix_nopack[i],
                net_rain=net_rain[i],
                net_ppt=net_ppt[i],
                imperv_stor=imperv_stor[i],
                imperv_stor_max=imperv_stor_max[i],
                snowmelt=snowmelt[i],
                snowinfil_max=snowinfil_max[i],
                net_snow=net_snow[i],
                pkwater_equiv=pkwater_equiv[i],
                infil=infil[i],
                hru_type=hru_type[i],
                intcp_changeover=intcp_changeover[i],
                hruarea_imperv=hruarea_imperv,
                sri=sri,
                srp=srp,
                check_capacity=check_capacity,
                perv_comp=perv_comp,
                through_rain=through_rain[i],
                intcp_changeover_in_net_rain=intcp_changeover_in_net_rain,
                ncascade_hru=ncascade_hru,
                ncascade_hru_active=ncascade_hru_active,
                upslope_hortonian=upslope_hortonian,
                ihru=i,
            )

            frzen = OFF  # cdl todo: hardwired

            if dprst_flag == ACTIVE:
                dprst_in[i] = 0.0
                dprst_chk = OFF
                # JLM: can this logic be moved inside dprst_comp?
                if dprst_area_max[i] > 0.0:
                    dprst_chk = ACTIVE
                    if frzen == OFF:
                        (
                            dprst_in[i],
                            dprst_vol_open[i],
                            dprst_area_open[i],
                            avail_et,
                            dprst_vol_clos[i],
                            dprst_sroff_hru[i],
                            srp,
                            sri,
                            dprst_evap_hru[i],
                            dprst_seep_hru[i],
                            dprst_insroff_hru[i],
                            dprst_vol_open_frac[i],
                            dprst_vol_clos_frac[i],
                            dprst_vol_frac[i],
                            dprst_stor_hru[i],
                        ) = dprst_comp(
                            dprst_vol_clos=dprst_vol_clos[i],
                            dprst_area_clos_max=dprst_area_clos_max[i],
                            dprst_area_clos=dprst_area_clos[i],
                            dprst_vol_open_max=dprst_vol_open_max[i],
                            dprst_vol_open=dprst_vol_open[i],
                            dprst_area_open_max=dprst_area_open_max[i],
                            dprst_sroff_hru=dprst_sroff_hru[i],
                            sro_to_dprst_perv=sro_to_dprst_perv[i],
                            sro_to_dprst_imperv=sro_to_dprst_imperv[i],
                            dprst_evap_hru=dprst_evap_hru[i],
                            pptmix_nopack=pptmix_nopack[i],
                            snowmelt=snowmelt[i],
                            pkwater_equiv=pkwater_equiv[i],
                            net_snow=net_snow[i],
                            hru_area=hru_area[i],
                            dprst_insroff_hru=dprst_insroff_hru[i],
                            dprst_frac_open=dprst_frac_open[i],
                            dprst_frac_clos=dprst_frac_clos[i],
                            va_open_exp=va_open_exp[i],
                            dprst_vol_clos_max=dprst_vol_clos_max[i],
                            dprst_vol_clos_frac=dprst_vol_clos_frac[i],
                            va_clos_exp=va_clos_exp[i],
                            potet=potet[i],
                            snowcov_area=snowcov_area[i],
                            dprst_et_coef=dprst_et_coef[i],
                            dprst_seep_rate_open=dprst_seep_rate_open[i],
                            dprst_vol_thres_open=dprst_vol_thres_open[i],
                            dprst_flow_coef=dprst_flow_coef[i],
                            dprst_seep_rate_clos=dprst_seep_rate_clos[i],
                            avail_et=avail_et,
                            net_rain=availh2o,
                            dprst_in=dprst_in[i],
                            srp=srp,
                            sri=sri,
                            imperv_frac=imperv_frac,
                            perv_frac=perv_frac,
                        )
                        runoff = runoff + dprst_sroff_hru[i] * hruarea

            # cdl -- the upper part of this block needs to be done to calculate
            #        runoff and srunoff
            # Compute runoff for pervious and impervious area, and depression
            # storage area
            srunoff = zero
            if hru_type[i] == LAND:
                runoff = runoff + srp * perv_area + sri * hruarea_imperv
                srunoff = runoff / hruarea

                if ncascade_hru_active:
                    hru_sroff_down = zero
                    if srunoff > zero:
                        hru_horton_cascflow[i] = zero
                        if ncascade_hru[i] > 0:
                            (
                                srunoff,
                                hru_horton_cascflow[i],
                                stream_seg_in[:],
                                upslope_hortonian[:],
                            ) = run_cascade_sroff(
                                i,
                                ncascade_hru[i],
                                upslope_hortonian,
                                srunoff,
                                hru_sroff_down,
                                hru_down,
                                hru_down_frac,
                                hru_down_fracwt,
                                cascade_area,
                                stream_seg_in,
                                cfs_conv,
                            )

                    # <<
                    else:
                        hru_horton_cascflow[i] = zero
                # <
                hru_sroffp[i] = srp * perv_frac

            # <
            # cdl -- the guts of this was implemented in the python code below
            # Compute evaporation from impervious area
            if hruarea_imperv > 0.0:
                if imperv_stor[i] > 0.0:
                    imperv_stor[i], imperv_evap[i] = imperv_et(
                        imperv_stor[i],
                        potet[i],
                        imperv_evap[i],
                        snowcov_area[i],
                        avail_et,
                        imperv_frac,
                    )
                    hru_impervevap[i] = imperv_evap[i] * imperv_frac
                    avail_et = avail_et - hru_impervevap[i]
                    if avail_et < 0.0:
                        hru_impervevap[i] = hru_impervevap[i] + avail_et
                        if hru_impervevap[i] < 0.0:
                            hru_impervevap[i] = 0.0
                        imperv_evap[i] = imperv_evap[i] / imperv_frac
                        imperv_stor[i] = (
                            imperv_stor[i] - avail_et / imperv_frac
                        )
                        avail_et = 0.0

                    # <
                    hru_impervstor[i] = imperv_stor[i] * imperv_frac

                # <
                hru_sroffi[i] = sri * imperv_frac

            # <
            # cdl -- saving sroff here
            if dprst_chk == ACTIVE:
                dprst_stor_hru[i] = (
                    dprst_vol_open[i] + dprst_vol_clos[i]
                ) / hruarea
            # <
            sroff[i] = srunoff
            hortonian_flow[i] = srunoff

        # <
        return (
            infil,
            contrib_fraction,
            hru_sroffp,
            hru_sroffi,
            imperv_evap,
            hru_impervevap,
            imperv_stor,
            dprst_in,
            dprst_vol_open,
            dprst_vol_clos,
            dprst_sroff_hru,
            dprst_evap_hru,
            dprst_seep_hru,
            dprst_insroff_hru,
            dprst_vol_open_frac,
            dprst_vol_clos_frac,
            dprst_vol_frac,
            dprst_stor_hru,
            sroff,
            hru_horton_cascflow,
            hortonian_flow,
        )

    @staticmethod
    def compute_infil(
        contrib_fraction,
        soil_moist_prev,
        soil_moist_max,
        carea_max,
        smidx_coef,
        smidx_exp,
        pptmix_nopack,
        net_rain,
        net_ppt,
        imperv_stor,
        imperv_stor_max,
        snowmelt,
        snowinfil_max,
        net_snow,
        pkwater_equiv,
        infil,
        hru_type,
        intcp_changeover,
        hruarea_imperv,
        sri,
        srp,
        check_capacity,
        perv_comp,
        through_rain,
        intcp_changeover_in_net_rain,
        ncascade_hru,
        ncascade_hru_active,
        upslope_hortonian,
        ihru,
    ):
        isglacier = False  # todo -- hardwired
        hru_flag = 0
        if hru_type == LAND or isglacier:
            hru_flag = 1

        if ncascade_hru_active:
            # compute runoff from cascading Hortonian flow
            avail_water = upslope_hortonian[ihru]
            if avail_water > zero:
                infil = infil + avail_water

                if hru_flag == 1:
                    infil, srp, contrib_fraction = perv_comp(
                        soil_moist_prev,
                        carea_max,
                        smidx_coef,
                        smidx_exp,
                        avail_water,
                        avail_water,
                        infil,
                        srp,
                    )
        # <<<
        else:
            avail_water = 0.0

        # compute runoff from canopy changeover water
        if intcp_changeover > 0.0 and not intcp_changeover_in_net_rain:
            avail_water = avail_water + intcp_changeover
            infil = infil + intcp_changeover
            if hru_flag == 1:
                infil, srp, contrib_fraction = perv_comp(
                    soil_moist_prev,
                    carea_max,
                    smidx_coef,
                    smidx_exp,
                    intcp_changeover,
                    intcp_changeover,
                    infil,
                    srp,
                )

        # if rain/snow event with no antecedent snowpack,
        # compute the runoff from the rain first and then proceed with the
        # snowmelt computations
        # double_counting = 0
        cond2 = pptmix_nopack != 0
        # if pptmix_nopack == ACTIVE:
        if cond2:
            # double_counting += 1
            avail_water = avail_water + through_rain
            infil = infil + through_rain
            if hru_flag == 1:
                infil, srp, contrib_fraction = perv_comp(
                    soil_moist_prev,
                    carea_max,
                    smidx_coef,
                    smidx_exp,
                    through_rain,
                    through_rain,
                    infil,
                    srp,
                )

        # If precipitation on snowpack, all water available to the surface is
        # considered to be snowmelt, and the snowmelt infiltration
        # procedure is used.  If there is no snowpack and no precip,
        # then check for melt from last of snowpack.  If rain/snow mix
        # with no antecedent snowpack, compute snowmelt portion of runoff.
        # cond3 = snowmelt < nearzero
        cond4 = pkwater_equiv < dnearzero
        # cond1 = net_ppt > zero
        cond6 = net_snow < nearzero

        if snowmelt > 0.0:
            # if not cond3:  # this results in less precision accuracy
            # There was no snowmelt but a snowpack may exist.  If there is
            # no snowpack then check for rain on a snowfree HRU.
            avail_water = avail_water + snowmelt
            infil = infil + snowmelt
            if hru_flag == 1:
                # The condition below determines whether to use check_capacity
                # (infiltration-based) vs perv_comp (contributing area-based)
                # runoff calculation. The logic differs based on how
                # intcp_changeover water is handled:
                #
                # When intcp_changeover_in_net_rain = False (default PRMS):
                #   - intcp_changeover is added separately to avail_water above
                #   - net_rain does NOT include intcp_changeover
                #   - Use: (net_rain < nearzero) to check for rain-on-snow
                #
                # When intcp_changeover_in_net_rain = True (alternative):
                #   - intcp_changeover is already included in net_rain
                #   - Must check if (net_ppt - net_snow) > 0 for rain presence
                #   - Use: not (net_ppt - net_snow > 0.0)
                #
                # These conditions are NOT equivalent because net_rain may
                # differ from (net_ppt - net_snow) depending on how
                # intcp_changeover is accounted for.
                if intcp_changeover_in_net_rain:
                    check_condition = not (net_ppt - net_snow > 0.0)
                else:
                    check_condition = net_rain < nearzero

                if (pkwater_equiv > 0.0) or check_condition:
                    # Pervious area computations
                    infil, srp = check_capacity(
                        soil_moist_prev,
                        soil_moist_max,
                        snowinfil_max,
                        infil,
                        srp,
                    )
                else:
                    # Snowmelt occurred and depleted the snowpack
                    # this frequently gets counted along with pptmix_nopack
                    # double_counting += 1
                    # if double_counting > 1:
                    #     print("snowmelt")

                    infil, srp, contrib_fraction = perv_comp(
                        soil_moist_prev,
                        carea_max,
                        smidx_coef,
                        smidx_exp,
                        snowmelt,
                        net_ppt,
                        infil,
                        srp,
                    )

        # elif pkwater_equiv < Dnearzero:
        elif cond4:
            # If no snowmelt and no snowpack but there was net snow then
            # snowpack was small and was lost to sublimation.
            # if net_snow < nearzero and net_rain > 0.0:
            if cond6 and through_rain > 0.0:
                # if net_snow < 1.0e-6 and net_rain > 0.0:
                # cond3 & cond4 & cond6 & cond1
                # this is through_rain's top/most narrow case

                avail_water = avail_water + through_rain
                infil = infil + through_rain

                # double_counting += 1
                # if double_counting > 1:
                #     print("cond4")
                if hru_flag == 1:
                    infil, srp, contrib_fraction = perv_comp(
                        soil_moist_prev,
                        carea_max,
                        smidx_coef,
                        smidx_exp,
                        through_rain,
                        through_rain,
                        infil,
                        srp,
                    )

        # Snowpack exists, check to see if infil exceeds maximum daily
        # snowmelt infiltration rate. Infil results from rain snow mix
        # on a snowfree surface.
        elif infil > 0.0:
            if hru_flag == 1:
                infil, srp = check_capacity(
                    soil_moist_prev,
                    soil_moist_max,
                    snowinfil_max,
                    infil,
                    srp,
                )

        # if double_counting > 1:
        #     assert False

        if hruarea_imperv > 0.0:
            imperv_stor = imperv_stor + avail_water
            if hru_flag == 1:
                if imperv_stor > imperv_stor_max:
                    sri = imperv_stor - imperv_stor_max
                    imperv_stor = imperv_stor_max

        return sri, srp, imperv_stor, infil, contrib_fraction

    @staticmethod
    def dprst_comp(
        dprst_vol_clos,
        dprst_area_clos_max,
        dprst_area_clos,
        dprst_vol_open_max,
        dprst_vol_open,
        dprst_area_open_max,
        dprst_sroff_hru,
        sro_to_dprst_perv,
        sro_to_dprst_imperv,
        dprst_evap_hru,
        pptmix_nopack,
        snowmelt,
        pkwater_equiv,
        net_snow,
        hru_area,
        dprst_insroff_hru,
        dprst_frac_open,
        dprst_frac_clos,
        va_open_exp,
        dprst_vol_clos_max,
        dprst_vol_clos_frac,
        va_clos_exp,
        potet,
        snowcov_area,
        dprst_et_coef,
        dprst_seep_rate_open,
        dprst_vol_thres_open,
        dprst_flow_coef,
        dprst_seep_rate_clos,
        avail_et,
        net_rain,
        dprst_in,
        srp,
        sri,
        imperv_frac,
        perv_frac,
    ):
        cascade_flag = OFF  # cdl -- todo: hardwired
        if cascade_flag > OFF:
            raise Exception("i am brokin")
        else:
            inflow = 0.0

        if pptmix_nopack:
            inflow = inflow + net_rain

        # I f precipitation on snowpack all water available to the surface is
        # considered to be snowmelt
        # If there is no snowpack and no precip,then check for melt from last
        # of snowpack.
        # If rain/snow mix with no antecedent snowpack, compute snowmelt
        # portion of runoff.
        if snowmelt > zero:
            inflow = inflow + snowmelt

        # !******There was no snowmelt but a snowpack may exist.  If there is
        # !******no snowpack then check for rain on a snowfree HRU.
        elif pkwater_equiv < dnearzero:
            # !      If no snowmelt and no snowpack but there was net snow then
            # !      snowpack was small and was lost to sublimation.
            if net_snow < nearzero and net_rain > 0.0:
                inflow = inflow + net_rain

        dprst_add_water_use = OFF
        if dprst_add_water_use == ACTIVE:
            raise Exception("not supported")

        dprst_in = 0.0
        if dprst_area_open_max > 0.0:
            dprst_in = inflow * dprst_area_open_max
            dprst_vol_open = dprst_vol_open + dprst_in

        if dprst_area_clos_max > 0.0:
            tmp1 = inflow * dprst_area_clos_max
            dprst_vol_clos = dprst_vol_clos + tmp1
            dprst_in = dprst_in + tmp1
        dprst_in = dprst_in / hru_area

        # add any pervious surface runoff fraction to depressions
        dprst_srp = 0.0
        dprst_sri = 0.0
        if srp > 0.0:
            tmp = srp * perv_frac * sro_to_dprst_perv * hru_area
            if dprst_area_open_max > 0.0:
                dprst_srp_open = tmp * dprst_frac_open
                dprst_srp = dprst_srp_open / hru_area
                dprst_vol_open = dprst_vol_open + dprst_srp_open
            if dprst_area_clos_max > 0.0:
                dprst_srp_clos = tmp * dprst_frac_clos
                dprst_srp = dprst_srp + dprst_srp_clos / hru_area
                dprst_vol_clos = dprst_vol_clos + dprst_srp_clos
            srp = srp - dprst_srp / perv_frac
            if srp < zero:
                srp = zero
                # if srp < -nearzero: this should raise an error

        if sri > 0.0:
            tmp = sri * imperv_frac * sro_to_dprst_imperv * hru_area
            if dprst_area_open_max > 0.0:
                dprst_sri_open = tmp * dprst_frac_open
                dprst_sri = dprst_sri_open / hru_area
                dprst_vol_open = dprst_vol_open + dprst_sri_open
            if dprst_area_clos_max > 0.0:
                dprst_sri_clos = tmp * dprst_frac_clos
                dprst_sri = dprst_sri + dprst_sri_clos / hru_area
                dprst_vol_clos = dprst_vol_clos + dprst_sri_clos
            sri = sri - dprst_sri / imperv_frac
            if sri < 0.0:
                # if sri < -nearzero: warn??
                sri = 0.0

        # <<<
        dprst_insroff_hru = dprst_srp + dprst_sri

        dprst_area_open = 0.0
        if dprst_vol_open > 0.0:
            open_vol_r = dprst_vol_open / dprst_vol_open_max
            if open_vol_r < nearzero:
                frac_op_ar = 0.0
            elif open_vol_r > 1.0:
                frac_op_ar = 1.0
            else:
                frac_op_ar = np.exp(va_open_exp * np.log(open_vol_r))
            dprst_area_open = dprst_area_open_max * frac_op_ar
            if dprst_area_open > dprst_area_open_max:
                dprst_area_open = dprst_area_open_max

        if dprst_area_clos_max > 0.0:
            dprst_area_clos = 0.0
            if dprst_vol_clos > 0.0:
                clos_vol_r = dprst_vol_clos / dprst_vol_clos_max
                if clos_vol_r < nearzero:
                    frac_cl_ar = 0.0
                elif clos_vol_r > 1.0:
                    frac_cl_ar = 1.0
                else:
                    frac_cl_ar = np.exp(va_clos_exp * np.log(clos_vol_r))
                dprst_area_clos = dprst_area_clos_max * frac_cl_ar
                if dprst_area_clos > dprst_area_clos_max:
                    dprst_area_clos = dprst_area_clos_max
                # if dprst_area_clos < nearzero:
                #     dprst_area_clos = 0.0

        #
        # evaporate water from depressions based on snowcov_area
        # dprst_evap_open & dprst_evap_clos = inches-acres on the HRU
        unsatisfied_et = avail_et
        dprst_avail_et = potet * (1.0 - snowcov_area) * dprst_et_coef
        dprst_evap_hru = 0.0
        if dprst_avail_et > 0.0:
            dprst_evap_open = 0.0
            dprst_evap_clos = 0.0
            if dprst_area_open > 0.0:
                dprst_evap_open = min(
                    dprst_area_open * dprst_avail_et, dprst_vol_open
                )
                if dprst_evap_open / hru_area > unsatisfied_et:
                    dprst_evap_open = unsatisfied_et * hru_area
                if dprst_evap_open > dprst_vol_open:
                    dprst_evap_open = dprst_vol_open
                unsatisfied_et = unsatisfied_et - dprst_evap_open / hru_area
                dprst_vol_open = dprst_vol_open - dprst_evap_open

            if dprst_area_clos > 0.0:
                dprst_evap_clos = min(
                    dprst_area_clos * dprst_avail_et, dprst_vol_clos
                )
                if dprst_evap_clos / hru_area > unsatisfied_et:
                    dprst_evap_clos = unsatisfied_et * hru_area
                if dprst_evap_clos > dprst_vol_clos:
                    dprst_evap_clos = dprst_vol_clos
                dprst_vol_clos = dprst_vol_clos - dprst_evap_clos

            dprst_evap_hru = (dprst_evap_open + dprst_evap_clos) / hru_area

        # compute seepage
        dprst_seep_hru = 0.0
        if dprst_vol_open > 0.0:
            seep_open = dprst_vol_open * dprst_seep_rate_open
            dprst_vol_open = dprst_vol_open - seep_open
            if dprst_vol_open < 0.0:
                seep_open = seep_open + dprst_vol_open
                dprst_vol_open = 0.0
            dprst_seep_hru = seep_open / hru_area

        # compute open surface runoff
        dprst_sroff_hru = 0.0
        if dprst_vol_open > 0.0:
            dprst_sroff_hru = max(0.0, dprst_vol_open - dprst_vol_open_max)
            dprst_sroff_hru = dprst_sroff_hru + max(
                0.0,
                (dprst_vol_open - dprst_sroff_hru - dprst_vol_thres_open)
                * dprst_flow_coef,
            )
            dprst_vol_open = dprst_vol_open - dprst_sroff_hru
            dprst_sroff_hru = dprst_sroff_hru / hru_area
            if dprst_vol_open < 0.0:
                dprst_vol_open = 0.0

        if dprst_area_clos_max > 0.0:
            if dprst_area_clos > nearzero:
                seep_clos = dprst_vol_clos * dprst_seep_rate_clos
                dprst_vol_clos = dprst_vol_clos - seep_clos
                if dprst_vol_clos < 0.0:
                    seep_clos = seep_clos + dprst_vol_clos
                    dprst_vol_clos = 0.0
                dprst_seep_hru = dprst_seep_hru + seep_clos / hru_area

        avail_et = avail_et - dprst_evap_hru
        if dprst_vol_open_max > 0.0:
            dprst_vol_open_frac = dprst_vol_open / dprst_vol_open_max
        if dprst_vol_clos_max > 0.0:
            dprst_vol_clos_frac = dprst_vol_clos / dprst_vol_clos_max
        if dprst_vol_open_max + dprst_vol_clos_max > 0.0:
            dprst_vol_frac = (dprst_vol_open + dprst_vol_clos) / (
                dprst_vol_open_max + dprst_vol_clos_max
            )
        dprst_stor_hru = (dprst_vol_open + dprst_vol_clos) / hru_area

        return (
            dprst_in,
            dprst_vol_open,
            dprst_area_open,
            avail_et,
            dprst_vol_clos,
            dprst_sroff_hru,
            srp,
            sri,
            dprst_evap_hru,
            dprst_seep_hru,
            dprst_insroff_hru,
            dprst_vol_open_frac,
            dprst_vol_clos_frac,
            dprst_vol_frac,
            dprst_stor_hru,
        )

    @staticmethod
    def perv_comp(
        soil_moist_prev,
        carea_max,
        smidx_coef,
        smidx_exp,
        pptp,
        ptc,
        infil,
        srp,
    ):
        """Pervious area computations."""
        smidx_module = True
        if smidx_module:
            smidx = soil_moist_prev + 0.5 * ptc
            if smidx > 25.0:
                ca_fraction = carea_max
            else:
                ca_fraction = smidx_coef * 10.0 ** (smidx_exp * smidx)
        else:
            raise Exception("you did a bad thing...")

        if ca_fraction > carea_max:
            ca_fraction = carea_max

        srpp = ca_fraction * pptp
        infil = infil - srpp
        srp = srp + srpp

        return infil, srp, ca_fraction

    @staticmethod
    def check_capacity(
        soil_moist_prev, soil_moist_max, snowinfil_max, infil, srp
    ):
        """
        Fill soil to soil_moist_max, if more than capacity restrict
        infiltration by snowinfil_max, with excess added to runoff
        """
        capacity = soil_moist_max - soil_moist_prev
        excess = infil - capacity
        if excess > snowinfil_max:
            srp = srp + excess - snowinfil_max
            infil = snowinfil_max + capacity

        return infil, srp

    @staticmethod
    def imperv_et(imperv_stor, potet, imperv_evap, sca, avail_et, imperv_frac):
        if sca < 1.0:
            if potet < imperv_stor:
                imperv_evap = potet * (1.0 - sca)
            else:
                imperv_evap = imperv_stor * (1.0 - sca)
            if imperv_evap * imperv_frac > avail_et:
                imperv_evap = avail_et / imperv_frac
            imperv_stor = imperv_stor - imperv_evap
        return imperv_stor, imperv_evap

    @staticmethod
    def _run_cascade_sroff(
        ihru: int,
        ncascade_hru_i: int,
        upslope_hortonian: np.ndarray,
        runoff: float,
        hru_sroff_down: float,
        hru_down: np.ndarray,
        hru_down_frac: np.ndarray,
        hru_down_fracwt: np.ndarray,
        cascade_area: np.ndarray,
        stream_seg_in: np.ndarray,
        cfs_conv: float,
    ):
        # ihru is already a 0-based index
        for kk in range(ncascade_hru_i):
            jj = hru_down[kk, ihru]  # a 1-based index
            jjabs = abs(jj)

            #  if hru_down(k, Ihru) > 0, cascade contributes to a downslope HRU
            if jj > 0:
                upslope_hortonian[jjabs - 1] = (
                    upslope_hortonian[jjabs - 1]
                    + runoff * hru_down_fracwt[kk, ihru]
                )

                hru_sroff_down = (
                    hru_sroff_down + runoff * hru_down_frac[kk, ihru]
                )
            elif jj < 0:
                # if hru_down(k, Ihru) < 0, cascade contributes to a stream
                stream_seg_in[jjabs - 1] = (
                    stream_seg_in[jjabs - 1]
                    + runoff * cascade_area[kk, ihru] * cfs_conv
                )

        # <<
        # reset Sroff as it accumulates flow to streams
        runoff = runoff - hru_sroff_down
        return (runoff, hru_sroff_down, stream_seg_in, upslope_hortonian)

    @staticmethod
    def _run_cascade_sroff_dummy(
        ihru: int,
        ncascade_hru_i: int,
        upslope_hortonian: np.ndarray,
        runoff: float,
        hru_sroff_down: float,
        hru_down: np.ndarray,
        hru_down_frac: np.ndarray,
        hru_down_fracwt: np.ndarray,
        cascade_area: np.ndarray,
        stream_seg_in: np.ndarray,
        cfs_conv: float,
    ):
        return (runoff, hru_sroff_down, stream_seg_in, upslope_hortonian)
