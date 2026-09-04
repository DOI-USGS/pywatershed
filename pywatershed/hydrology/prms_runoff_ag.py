import pathlib as pl
from typing import Literal, Union
from warnings import warn

import numpy as np

from ..base.adapter import adaptable
from ..base.conservative_process import ConservativeProcess
from ..base.control import Control
from ..constants import HruType, dnearzero, nearzero, numba_num_threads, zero
from ..parameters import Parameters
from .prms_runoff import PRMSRunoff

RAIN = 0
SNOW = 1

BARESOIL = 0
GRASSES = 1

OFF = 0
ACTIVE = 1

LAND = HruType.LAND.value
LAKE = HruType.LAKE.value


class PRMSRunoffAg(PRMSRunoff):
    """PRMS surface runoff with agricultural infiltration.

    Extension of PRMSRunoff that calculates infiltration for both pervious
    and agricultural areas separately. This is required for agricultural
    soilzone simulations (PRMSSoilzoneAg).

    Implementation based on GSFLOW with agricultural extensions, following
    the same logic as PRMSRunoff but adding parallel calculations for
    agricultural areas.

    Restart Variables
    -----------------
    PRMSRunoffAg inherits get_restart_variables() from PRMSRunoff without
    modification because it has no additional storage state variables beyond
    the parent class. The agricultural-specific outputs (infil_ag,
    infil_ag_hru,
    hru_sroff_ag) are flux variables that are recalculated each timestep, not
    state variables that need to be saved for restart.

    The ag calculations use:

    - ag_soil_moist_prev and ag_soil_rechr_prev: These come from PRMSSoilzoneAg
      as inputs, not from runoff itself
    - ag_area: Derived from the ag_frac parameter and recalculated in
      _update_ag_areas() as needed

    All storage states (imperv_stor, dprst_vol_open, dprst_vol_clos, etc.) are
    inherited from the parent class and properly saved/restored during restart.

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
        ag_soil_moist_prev: Previous timestep water storage for upper portion
            in the capillary reservoir of the irrigated area for each HRU.
        ag_soil_rechr_prev: Previous timestep water storage for upper portion
            in the capillary reservoir of the irrigated area for each HRU that
            is available for both evaporation and transpiration.
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
        ag_soil_moist_prev: adaptable,
        ag_soil_rechr_prev: adaptable,
        ag_frac: adaptable,
        dprst_flag: Union[bool, None] = None,
        intcp_changeover_in_net_rain: bool | None = None,
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

        # Call ConservativeProcess.__init__ directly (grandparent)
        ConservativeProcess.__init__(
            self,
            control=control,
            discretization=discretization,
            parameters=parameters,
            restart_read=restart_read,
            restart_write=restart_write,
            restart_write_freq=restart_write_freq,
            input_aliases=input_aliases,
        )

        self.name = "PRMSRunoffAg"

        self._set_inputs(locals())
        self._set_options(locals())

        self._set_budget()
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

        if restart_read is not False or restart_write is not False:
            self.restart_read = False
            self.restart_write = False

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
            nb_parallel = (numba_num_threads is not None) and (
                numba_num_threads > 1
            )
            if nb_parallel:
                numba_msg += f"and using {numba_num_threads} threads"
            print(numba_msg, flush=True)

            self._calculate_runoff = nb.njit(
                self._calculate_numpy_ag, parallel=nb_parallel
            )
            self.check_capacity = nb.njit(self.check_capacity)
            self.perv_comp = nb.njit(self.perv_comp)
            self.compute_infil = nb.njit(self.compute_infil)
            self._compute_infil_ag = nb.njit(self._compute_infil_ag)
            self.dprst_comp_ag = nb.njit(self.dprst_comp_ag)
            self.imperv_et = nb.njit(self.imperv_et)
            self.ag_comp = nb.njit(self.ag_comp)
            self.check_capacity_ag = nb.njit(self.check_capacity_ag)

        else:
            self._calculate_runoff = self._calculate_numpy_ag

        return

    @staticmethod
    def get_inputs() -> tuple:
        return (
            "soil_lower_prev",
            "soil_rechr_prev",
            "ag_soil_moist_prev",
            "ag_soil_rechr_prev",
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
            "ag_frac",
        )

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
            "ag_soil_moist_max",
            "ag_soil_rechr_max_frac",
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
            "sat_threshold",
        )

    @staticmethod
    def get_init_values() -> dict:
        return {
            "contrib_fraction": zero,
            "infil": zero,
            "infil_hru": zero,
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
            "infil_ag": zero,
            "infil_ag_hru": zero,
            "infil_perv_hru": zero,
            "hru_sroff_ag": zero,
            "intcp_changeover_budget": zero,
        }

    @staticmethod
    def get_mass_budget_terms():
        """Get mass budget terms for agricultural runoff.

        In GSFLOW with agriculture, the pervious area (hru_perv) is reduced
        by ag_area. We break out infiltration and surface runoff into
        separate pervious and agricultural components for budget clarity:

        Infiltration components:
        - infil_perv_hru: pervious infiltration only = infil * hru_frac_perv
        - infil_ag_hru: agricultural infiltration = infil_ag * ag_frac
        - infil_hru: combined total (pervious + ag) for compatibility with
          PRMS source code

        Surface runoff is split into three components:
        - hru_sroffi: impervious runoff
        - hru_sroffp: pervious runoff
        - hru_sroff_ag: agricultural runoff
        """
        return {
            "inputs": [
                "through_rain",
                "snowmelt",
                "intcp_changeover_budget",
            ],
            "outputs": [
                # sroff = hru_sroffi + hru_sroffp + hru_sroff_ag
                # + dprst_sroff_hru
                "hru_sroffi",
                "hru_sroffp",
                "hru_sroff_ag",  # Agricultural surface runoff
                "dprst_sroff_hru",
                "infil_perv_hru",  # Pervious infiltration only
                "infil_ag_hru",  # Agricultural infiltration
                "hru_impervevap",
                "dprst_seep_hru",
                "dprst_evap_hru",
            ],
            "storage_changes": [
                "hru_impervstor_change",
                "dprst_stor_hru_change",
            ],
        }

    def _set_initial_conditions(self):
        """Set initial conditions for variables not in get_init_values"""
        super()._set_initial_conditions()
        self.infil_ag = self["infil_ag"]
        self.infil_ag_hru = self["infil_ag_hru"]
        self.hru_sroff_ag = self["hru_sroff_ag"]

        # Calculate ag_soil_rechr_max from ag_soil_moist_max and fraction
        self.ag_soil_rechr_max = (
            self.ag_soil_moist_max * self.ag_soil_rechr_max_frac
        )

        return

    def _update_ag_areas(self):
        """Update ag_area and related quantities when ag_frac changes.

        This method recalculates derived areas that depend on ag_frac,
        which may change dynamically via an adapter.

        For PRMSRunoffAg, we only need to update area fractions - there's
        no water storage to redistribute (unlike PRMSSoilzoneAg).
        """
        # Recalculate agricultural area
        new_ag_area = self.ag_frac * self.hru_area

        # Recalculate pervious area to exclude agricultural area
        # Start from hru_area, subtract imperv, dprst (if active), and ag
        new_hru_perv = self.hru_area - self.hru_imperv - new_ag_area
        if self._dprst_flag:
            new_hru_perv = new_hru_perv - self.dprst_area_max

        # Update the instance variables
        self.ag_area = new_ag_area
        self.hru_perv = new_hru_perv
        self.hru_frac_perv = self.hru_perv / self.hru_area

        return

    def _calculate(self, time_length, vectorized=False):
        """Calculate runoff with agricultural infiltration."""

        if self.control.itime_step == 0:
            self.basin_init()
            if self._dprst_flag:
                self.dprst_init()
            # Calculate ag_area from ag_frac
            # Following PRMSSoilzoneAg line 447-451
            self.ag_area = self.ag_frac * self.hru_area
            # Adjust pervious area to exclude agricultural area
            self.hru_perv = self.hru_perv - self.ag_area
            self.hru_frac_perv = self.hru_perv / self.hru_area

        (
            self.infil[:],
            self.infil_ag[:],
            self.contrib_fraction[:],
            self.hru_sroffp[:],
            self.hru_sroffi[:],
            self.hru_sroff_ag[:],
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
        ) = self._calculate_runoff(
            infil=self.infil,
            infil_ag=self.infil_ag,
            nhru=self.nhru,
            hru_area=self.hru_area,
            hru_perv=self.hru_perv,
            hru_frac_perv=self.hru_frac_perv,
            hru_sroffp=self.hru_sroffp,
            hru_sroff_ag=self.hru_sroff_ag,
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
            ag_soil_moist_prev=self.ag_soil_moist_prev,
            ag_soil_rechr_prev=self.ag_soil_rechr_prev,
            soil_moist_max=self.soil_moist_max,
            ag_soil_moist_max=self.ag_soil_moist_max,
            ag_soil_rechr_max=self.ag_soil_rechr_max,
            ag_area=self.ag_area,
            ag_frac=self.ag_frac,
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
            check_capacity=self.check_capacity,
            perv_comp=self.perv_comp,
            compute_infil=self.compute_infil,
            dprst_comp_ag=self.dprst_comp_ag,
            imperv_et=self.imperv_et,
            ag_comp=self.ag_comp,
            check_capacity_ag=self.check_capacity_ag,
            compute_infil_ag=self._compute_infil_ag,
            through_rain=self.through_rain,
            dprst_flag=self._dprst_flag,
            intcp_changeover_in_net_rain=self._intcp_changeover_in_net_rain,
        )

        # Update ag_area and related quantities in case ag_frac changed
        self._update_ag_areas()

        # Calculate infiltration components for budget clarity
        self.infil_perv_hru[:] = self.infil * self.hru_frac_perv
        self.infil_ag_hru[:] = self.infil_ag * self.ag_frac
        # Calculate combined infiltration (maintains PRMS source code parity)
        self.infil_hru[:] = self.infil_perv_hru + self.infil_ag_hru

        self.hru_impervstor_change[:] = (
            self.hru_impervstor - self.hru_impervstor_old
        )
        self.dprst_stor_hru_change[:] = (
            self.dprst_stor_hru - self.dprst_stor_hru_old
        )

        # The budget input term for canopy changeover water. When changeover
        # is included in net_rain (GSFLOW 4.2.0/PRMS 5.2.1.1 convention), it
        # is already counted in through_rain, so the budget term is zero.
        if self._intcp_changeover_in_net_rain:
            self.intcp_changeover_budget[:] = 0.0
        else:
            self.intcp_changeover_budget[:] = self.intcp_changeover

        self.sroff_vol[:] = self.sroff * self.hru_in_to_cf

        return

    @staticmethod
    def _calculate_numpy_ag(
        infil,
        infil_ag,
        nhru,
        hru_area,
        hru_perv,
        hru_frac_perv,
        hru_sroffp,
        hru_sroff_ag,
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
        ag_soil_moist_prev,
        ag_soil_rechr_prev,
        soil_moist_max,
        ag_soil_moist_max,
        ag_soil_rechr_max,
        ag_area,
        ag_frac,
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
        check_capacity,
        perv_comp,
        compute_infil,
        dprst_comp_ag,
        imperv_et,
        ag_comp,
        check_capacity_ag,
        compute_infil_ag,
        through_rain,
        dprst_flag,
        intcp_changeover_in_net_rain,
    ):
        """Modified calculation that includes agricultural runoff in depression
        storage routing.
        """

        dprst_chk = 0
        infil[:] = 0.0
        infil_ag[:] = 0.0

        soil_moist_prev = soil_lower_prev + soil_rechr_prev

        for i in range(nhru):
            runoff = zero
            hruarea = hru_area[i]
            perv_area = hru_perv[i]
            perv_frac = hru_frac_perv[i]
            srp = zero
            sri = zero
            sroff_ag = zero
            hru_sroffp[i] = zero
            hru_sroff_ag[i] = zero
            contrib_fraction[i] = zero
            hruarea_imperv = hru_imperv[i]
            imperv_frac = zero
            if hruarea_imperv > zero:
                imperv_frac = hru_percent_imperv[i]
                hru_sroffi[i] = zero
                imperv_evap[i] = zero
                hru_impervevap[i] = zero

            avail_et = potet[i] - snow_evap[i] - hru_intcpevap[i]

            # Calculate pervious infiltration
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
                # cascades are not active for ag, arrays are unused
                ncascade_hru=net_rain,
                ncascade_hru_active=False,
                upslope_hortonian=net_rain,
                ihru=i,
            )

            # Calculate agricultural infiltration (if ag area exists)
            if ag_area[i] > 0.0:
                infil_ag[i], sroff_ag = compute_infil_ag(
                    ag_soil_moist_prev=ag_soil_moist_prev[i],
                    ag_soil_rechr_prev=ag_soil_rechr_prev[i],
                    ag_soil_moist_max=ag_soil_moist_max[i],
                    ag_soil_rechr_max=ag_soil_rechr_max[i],
                    carea_max=carea_max[i],
                    smidx_coef=smidx_coef[i],
                    smidx_exp=smidx_exp[i],
                    snowinfil_max=snowinfil_max[i],
                    pptmix_nopack=pptmix_nopack[i],
                    net_rain=net_rain[i],
                    net_ppt=net_ppt[i],
                    snowmelt=snowmelt[i],
                    net_snow=net_snow[i],
                    pkwater_equiv=pkwater_equiv[i],
                    hru_type=hru_type[i],
                    intcp_changeover=intcp_changeover[i],
                    through_rain=through_rain[i],
                    ag_comp=ag_comp,
                    check_capacity_ag=check_capacity_ag,
                    intcp_changeover_in_net_rain=intcp_changeover_in_net_rain,
                )

            frzen = OFF

            # Route through depression storage including agricultural runoff
            if dprst_flag == ACTIVE:
                dprst_in[i] = 0.0
                dprst_chk = OFF
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
                            sroff_ag,
                            dprst_evap_hru[i],
                            dprst_seep_hru[i],
                            dprst_insroff_hru[i],
                            dprst_vol_open_frac[i],
                            dprst_vol_clos_frac[i],
                            dprst_vol_frac[i],
                            dprst_stor_hru[i],
                        ) = dprst_comp_ag(
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
                            through_rain=through_rain[i],
                            intcp_changeover=intcp_changeover[i],
                            intcp_changeover_in_net_rain=(
                                intcp_changeover_in_net_rain
                            ),
                            snowmelt=snowmelt[i],
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
                            dprst_in=dprst_in[i],
                            srp=srp,
                            sri=sri,
                            sroff_ag=sroff_ag,
                            imperv_frac=imperv_frac,
                            perv_frac=perv_frac,
                            ag_frac=ag_frac[i],
                        )
                        runoff = runoff + dprst_sroff_hru[i] * hruarea

            # Compute runoff for pervious, impervious, and agricultural areas
            srunoff = zero
            if hru_type[i] == LAND:
                runoff = runoff + srp * perv_area + sri * hruarea_imperv
                if ag_area[i] > 0.0:
                    runoff = runoff + sroff_ag * ag_area[i]
                srunoff = runoff / hruarea
                hru_sroffp[i] = srp * perv_frac
                hru_sroff_ag[i] = sroff_ag * ag_frac[i]

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

                    hru_impervstor[i] = imperv_stor[i] * imperv_frac

                hru_sroffi[i] = sri * imperv_frac

            if dprst_chk == ACTIVE:
                dprst_stor_hru[i] = (
                    dprst_vol_open[i] + dprst_vol_clos[i]
                ) / hruarea

            sroff[i] = srunoff

        return (
            infil,
            infil_ag,
            contrib_fraction,
            hru_sroffp,
            hru_sroffi,
            hru_sroff_ag,
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
        )

    @staticmethod
    def _compute_infil_ag(
        ag_soil_moist_prev,
        ag_soil_rechr_prev,
        ag_soil_moist_max,
        ag_soil_rechr_max,
        carea_max,
        smidx_coef,
        smidx_exp,
        snowinfil_max,
        pptmix_nopack,
        net_rain,
        net_ppt,
        snowmelt,
        net_snow,
        pkwater_equiv,
        hru_type,
        intcp_changeover,
        through_rain,
        ag_comp,
        check_capacity_ag,
        intcp_changeover_in_net_rain,
    ):
        """Calculate agricultural infiltration and runoff for a single HRU."""
        infil_ag = 0.0
        sroff_ag = 0.0

        # Process intcp_changeover
        if intcp_changeover > 0.0 and not intcp_changeover_in_net_rain:
            infil_ag = infil_ag + intcp_changeover
            if hru_type == LAND:
                infil_ag, sroff_ag = ag_comp(
                    ag_soil_moist_prev,
                    ag_soil_rechr_prev,
                    carea_max,
                    smidx_coef,
                    smidx_exp,
                    intcp_changeover,
                    intcp_changeover,
                    infil_ag,
                    sroff_ag,
                )

        # Process pptmix_nopack
        if pptmix_nopack != 0:
            infil_ag = infil_ag + through_rain
            if hru_type == LAND:
                infil_ag, sroff_ag = ag_comp(
                    ag_soil_moist_prev,
                    ag_soil_rechr_prev,
                    carea_max,
                    smidx_coef,
                    smidx_exp,
                    through_rain,
                    through_rain,
                    infil_ag,
                    sroff_ag,
                )

        # Process snowmelt
        if snowmelt > 0.0:
            infil_ag = infil_ag + snowmelt
            if hru_type == LAND:
                if (pkwater_equiv > 0.0) or (net_rain < nearzero):
                    # Check capacity
                    infil_ag, sroff_ag = check_capacity_ag(
                        ag_soil_moist_prev,
                        ag_soil_moist_max,
                        snowinfil_max,
                        infil_ag,
                        sroff_ag,
                    )
                else:
                    # Snowmelt occurred and depleted the snowpack
                    infil_ag, sroff_ag = ag_comp(
                        ag_soil_moist_prev,
                        ag_soil_rechr_prev,
                        carea_max,
                        smidx_coef,
                        smidx_exp,
                        snowmelt,
                        net_ppt,
                        infil_ag,
                        sroff_ag,
                    )

        elif pkwater_equiv < dnearzero:
            # No snowpack
            if net_snow < nearzero and through_rain > 0.0:
                infil_ag = infil_ag + through_rain
                if hru_type == LAND:
                    infil_ag, sroff_ag = ag_comp(
                        ag_soil_moist_prev,
                        ag_soil_rechr_prev,
                        carea_max,
                        smidx_coef,
                        smidx_exp,
                        through_rain,
                        through_rain,
                        infil_ag,
                        sroff_ag,
                    )

        elif infil_ag > 0.0:
            # Snowpack exists, check capacity
            if hru_type == LAND:
                infil_ag, sroff_ag = check_capacity_ag(
                    ag_soil_moist_prev,
                    ag_soil_moist_max,
                    snowinfil_max,
                    infil_ag,
                    sroff_ag,
                )

        return infil_ag, sroff_ag

    @staticmethod
    def dprst_comp_ag(
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
        through_rain,
        intcp_changeover,
        intcp_changeover_in_net_rain,
        snowmelt,
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
        dprst_in,
        srp,
        sri,
        sroff_ag,
        imperv_frac,
        perv_frac,
        ag_frac,
    ):
        """Depression storage computation with agricultural runoff routing.

        This is a modified version of dprst_comp that also routes agricultural
        runoff through depression storage, following the Fortran implementation
        in srunoff.f90 lines 1687-1700.

        """
        inflow = through_rain + snowmelt
        # Canopy changeover water reaches depression storage as in the
        # parent's dprst_comp (via availh2o = intcp_changeover + net_rain);
        # when changeover is in net_rain it is already part of through_rain.
        if not intcp_changeover_in_net_rain:
            inflow = inflow + intcp_changeover

        dprst_in = 0.0
        if dprst_area_open_max > 0.0:
            dprst_in = inflow * dprst_area_open_max
            dprst_vol_open = dprst_vol_open + dprst_in

        if dprst_area_clos_max > 0.0:
            tmp1 = inflow * dprst_area_clos_max
            dprst_vol_clos = dprst_vol_clos + tmp1
            dprst_in = dprst_in + tmp1
        dprst_in = dprst_in / hru_area

        # Add pervious surface runoff fraction to depressions
        dprst_srp = 0.0
        dprst_sri = 0.0
        dprst_sra = 0.0

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
                sri = 0.0

        # Add pervious and impervious contributions first
        dprst_insroff_hru = dprst_srp + dprst_sri

        # Add agricultural surface runoff fraction to depressions
        # Following Fortran srunoff.f90 lines 1687-1700
        if sroff_ag > 0.0:
            tmp = sroff_ag * ag_frac * sro_to_dprst_perv * hru_area
            if dprst_area_open_max > 0.0:
                dprst_sra_open = tmp * dprst_frac_open
                dprst_sra = dprst_sra_open / hru_area
                dprst_vol_open = dprst_vol_open + dprst_sra_open
            if dprst_area_clos_max > 0.0:
                dprst_sra_clos = tmp * dprst_frac_clos
                dprst_sra = dprst_sra + dprst_sra_clos / hru_area
                dprst_vol_clos = dprst_vol_clos + dprst_sra_clos
            sroff_ag = sroff_ag - dprst_sra / ag_frac
            if sroff_ag < 0.0:
                sroff_ag = 0.0
            dprst_insroff_hru = dprst_insroff_hru + dprst_sra

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

        # Evaporate water from depressions
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

        # Compute seepage
        dprst_seep_hru = 0.0
        if dprst_vol_open > 0.0:
            seep_open = dprst_vol_open * dprst_seep_rate_open
            dprst_vol_open = dprst_vol_open - seep_open
            if dprst_vol_open < 0.0:
                seep_open = seep_open + dprst_vol_open
                dprst_vol_open = 0.0
            dprst_seep_hru = seep_open / hru_area

        if dprst_area_clos_max > 0.0:
            if dprst_vol_clos > 0.0:
                seep_clos = dprst_vol_clos * dprst_seep_rate_clos
                dprst_vol_clos = dprst_vol_clos - seep_clos
                if dprst_vol_clos < 0.0:
                    seep_clos = seep_clos + dprst_vol_clos
                    dprst_vol_clos = 0.0
                dprst_seep_hru = dprst_seep_hru + seep_clos / hru_area

        # Compute open surface runoff
        # Note: In rare cases, accumulated floating-point errors can cause
        # dprst_vol_open to become slightly negative (O(1e-9) or smaller).
        # This is a known issue inherited from PRMS 5.2.1. While physically
        # impossible, these tiny negative values generally don't affect model
        # results significantly. However, they can cause bit-level differences
        # in restart tests due to how the values are serialized/deserialized.
        # A comprehensive fix would require clamping all depression storage
        # volumes to >= 0.0 after each operation, but this would break
        # regression testing against PRMS Fortran outputs.
        # TODO: Consider adding systematic clamping in a future major version.
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

        # Update fractions
        dprst_stor_hru = (dprst_vol_open + dprst_vol_clos) / hru_area
        if dprst_vol_open_max > 0.0:
            dprst_vol_open_frac = dprst_vol_open / dprst_vol_open_max
        else:
            dprst_vol_open_frac = 0.0

        if dprst_vol_clos_max > 0.0:
            dprst_vol_clos_frac = dprst_vol_clos / dprst_vol_clos_max
        else:
            dprst_vol_clos_frac = 0.0

        if (dprst_vol_open_max + dprst_vol_clos_max) > 0.0:
            dprst_vol_frac = (dprst_vol_open + dprst_vol_clos) / (
                dprst_vol_open_max + dprst_vol_clos_max
            )
        else:
            dprst_vol_frac = 0.0

        return (
            dprst_in,
            dprst_vol_open,
            dprst_area_open,
            avail_et,
            dprst_vol_clos,
            dprst_sroff_hru,
            srp,
            sri,
            sroff_ag,
            dprst_evap_hru,
            dprst_seep_hru,
            dprst_insroff_hru,
            dprst_vol_open_frac,
            dprst_vol_clos_frac,
            dprst_vol_frac,
            dprst_stor_hru,
        )

    @staticmethod
    def ag_comp(
        ag_soil_moist_prev,
        ag_soil_rechr_prev,
        carea_max,
        smidx_coef,
        smidx_exp,
        pptp,
        ptc,
        infil_ag,
        sroff_ag,
    ):
        """Agricultural area runoff computations.

        Similar to perv_comp but uses agricultural soil moisture.
        """
        smidx_module = True
        if smidx_module:
            # Use total ag soil moisture
            smidx = ag_soil_moist_prev + 0.5 * ptc
            if smidx > 25.0:
                ca_fraction = carea_max
            else:
                ca_fraction = smidx_coef * 10.0 ** (smidx_exp * smidx)
        else:
            raise Exception("carea method not implemented for ag_comp")

        if ca_fraction > carea_max:
            ca_fraction = carea_max

        srpp = ca_fraction * pptp
        infil_ag = infil_ag - srpp
        sroff_ag = sroff_ag + srpp

        return infil_ag, sroff_ag

    @staticmethod
    def check_capacity_ag(
        ag_soil_moist_prev,
        ag_soil_moist_max,
        snowinfil_max,
        infil_ag,
        sroff_ag,
    ):
        """
        Fill agricultural soil to ag_soil_moist_max, if more than capacity
        restrict infiltration by snowinfil_max, with excess added to runoff.
        """
        capacity = ag_soil_moist_max - ag_soil_moist_prev
        excess = infil_ag - capacity
        if excess > snowinfil_max:
            sroff_ag = sroff_ag + excess - snowinfil_max
            infil_ag = snowinfil_max + capacity

        return infil_ag, sroff_ag
