import pathlib as pl
from typing import Literal, Union

from ..base.adapter import adaptable
from ..base.control import Control
from ..constants import nan, zero
from ..parameters import Parameters
from .prms_soilzone import PRMSSoilzone

ONETHIRD = 1 / 3
TWOTHIRDS = 2 / 3


class PRMSSoilzoneNoDprst(PRMSSoilzone):
    """PRMS soil zone.

    Implementation based on PRMS 5.2.1 with theoretical documentation given in
    the PRMS-IV documentation:

    `Markstrom, S. L., Regan, R. S., Hay, L. E., Viger, R. J., Webb, R. M.,
    Payn, R. A., & LaFontaine, J. H. (2015). PRMS-IV, the
    precipitation-runoff modeling system, version 4. US Geological Survey
    Techniques and Methods, 6, B7.
    <https://pubs.usgs.gov/tm/6b7/pdf/tm6-b7.pdf>`__

    Note that as of pywatershed version 2.0.0, pref_flow_infil_frac is a
    required parameters which is optional in PRMS. Specifying zeros for all
    HRUs gives the same behavior as not supplying the parameter to PRMS.

    Args:
        control: a Control object
        discretization: a discretization of class Parameters
        parameters: a parameter object of class Parameters
        hru_impervevap: HRU area-weighted average evaporation from impervious
            area for each HRU
        hru_intcpevap: HRU area-weighted average evaporation from the
            canopy for each HRU
        infil_hru: Infiltration to the capillary and preferential-flow
            reservoirs, depth on HRU area
        sroff: Surface runoff to the stream network for each HRU
        sroff_vol: Surface runoff volume to the stream network for each HRU
        potet: Potential ET for each HRU
        transp_on: Flag indicating whether transpiration is occurring
            (0=no;1=yes)
        snow_evap: Evaporation and sublimation from snowpack on each HRU
        snowcov_area: Snow-covered area on each HRU prior to melt and
            sublimation unless snowpack
        imbalance_behavior: one of ["defer", None, "warn", "error"]
            with "defer" being the default and defering to
            control.options["imbalance_behavior"] when available. When
            control.options["imbalance_behavior"] is not avaiable,
            imbalance_behavior is set to "warn".
        calc_method: one of ["fortran", "numba", "numpy"]. None defaults to
            "numba".
        adjust_parameters: one of ["warn", "error", "no"]. Default is "warn",
            the code edits the parameters and issues a warning. If "error" is
            selected the the code issues warnings about all edited parameters
            before raising the error to give you information. If "no" is
            selected then no parameters are adjusted and there will be no
            warnings or errors.
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
        hru_impervevap: adaptable,
        hru_intcpevap: adaptable,
        infil_hru: adaptable,  # in /pywatershed/analysis/budget_soilzone.py
        sroff: adaptable,
        sroff_vol: adaptable,
        potet: adaptable,
        transp_on: adaptable,
        snow_evap: adaptable,
        snowcov_area: adaptable,
        imbalance_behavior: Literal["defer", None, "warn", "error"] = "defer",
        calc_method: Literal["numba", "numpy"] = None,
        adjust_parameters: Literal["warn", "error", "no"] = "warn",
        input_aliases: dict = None,
        verbose: bool = None,
        restart_read: Union[pl.Path, bool] = False,
        restart_write: Union[pl.Path, bool] = False,
        restart_write_freq: Literal["y", "m", "d", "f", False] = False,
    ) -> None:
        self._dprst_flag = False

        super().__init__(
            control=control,
            discretization=discretization,
            parameters=parameters,
            dprst_evap_hru=None,
            dprst_seep_hru=None,
            hru_impervevap=hru_impervevap,
            hru_intcpevap=hru_intcpevap,
            infil_hru=infil_hru,  # in /pywatershed/analysis/budget_soilzone.py
            sroff=sroff,
            sroff_vol=sroff_vol,
            potet=potet,
            transp_on=transp_on,
            snow_evap=snow_evap,
            snowcov_area=snowcov_area,
            dprst_flag=False,
            imbalance_behavior=imbalance_behavior,
            calc_method=calc_method,
            adjust_parameters=adjust_parameters,
            input_aliases=input_aliases,
            verbose=verbose,
            restart_read=restart_read,
            restart_write=restart_write,
            restart_write_freq=restart_write_freq,
        )

        self.name = "PRMSSoilzoneNoDprst"
        self._set_budget(active_mask=self._active_hru_mask)

        return

    @staticmethod
    def get_dimensions() -> tuple:
        return ("nhru",)

    @staticmethod
    def get_parameters() -> tuple:
        return (
            "cov_type",
            "fastcoef_lin",
            "fastcoef_sq",
            "hru_area",
            "hru_in_to_cf",
            "hru_percent_imperv",
            "hru_type",
            "pref_flow_den",
            "pref_flow_infil_frac",
            "sat_threshold",
            "slowcoef_lin",
            "slowcoef_sq",
            "soil_moist_max",
            "soil_moist_init_frac",
            "soil_rechr_init_frac",
            "soil_rechr_max_frac",
            "soil_type",
            "soil2gw_max",
            "ssr2gw_exp",
            "ssr2gw_rate",
            "ssstor_init_frac",
        )

    @staticmethod
    def get_inputs() -> tuple:
        return (
            "hru_impervevap",  # JLM ??
            "hru_intcpevap",  # JLM ???
            "infil_hru",
            "sroff",
            "sroff_vol",
            "potet",
            # hru_ppt => model_precip%hru_ppt, & # JLM ??
            "transp_on",
            "snow_evap",
            "snowcov_area",
        )

    @staticmethod
    def get_init_values() -> dict:
        return {
            "cap_infil_tot": zero,
            "cap_waterin": zero,
            "dunnian_flow": zero,
            "hru_actet": zero,
            "perv_actet": zero,
            "perv_actet_hru": zero,
            "potet_lower": zero,
            "potet_rechr": zero,
            "pref_flow": zero,
            "pref_flow_in": zero,
            "pref_flow_infil": zero,
            "pref_flow_max": zero,
            "pref_flow_stor": zero,
            "pref_flow_stor_change": nan,
            "pref_flow_stor_prev": nan,
            "pref_flow_thrsh": zero,
            "recharge": zero,
            "slow_flow": zero,
            "slow_stor": zero,
            "slow_stor_change": zero,
            "slow_stor_prev": nan,
            "soil_lower": nan,  # completely set later
            "soil_lower_change": nan,
            "soil_lower_change_hru": nan,
            "soil_lower_prev": nan,
            "soil_lower_ratio": zero,
            "soil_lower_max": nan,  # completely set later
            "soil_moist": nan,  # sm_climateflow
            "soil_moist_tot": nan,  # completely set later
            "soil_rechr": nan,  # sm_climateflow
            "soil_rechr_change": nan,  # sm_climateflow
            "soil_rechr_change_hru": nan,  # sm_climateflow
            "soil_rechr_prev": nan,  # sm_climateflow
            "soil_to_gw": zero,
            "soil_to_ssr": zero,
            "soil_zone_max": nan,  # this is completely later
            "ssr_to_gw": zero,
            "ssres_flow": zero,  # todo: privatize keep vol public
            "ssres_flow_vol": nan,
            "ssres_in": zero,
            "ssres_stor": nan,  # sm_soilzone
            "swale_actet": zero,
            "unused_potet": zero,
        }

    @staticmethod
    def get_restart_variables() -> list:
        raise NotImplementedError(
            "Restart capability not implemented for PRMSSoilzoneNoDprst"
        )

    @staticmethod
    def get_mass_budget_terms():
        return {
            "inputs": [
                "infil_hru",
            ],
            "outputs": [
                "perv_actet_hru",
                "soil_to_gw",
                "ssr_to_gw",
                "slow_flow",
                "dunnian_flow",
                "pref_flow",
            ],
            "storage_changes": [
                "soil_rechr_change_hru",
                "soil_lower_change_hru",
                "slow_stor_change",
                "pref_flow_stor_change",
            ],
        }

    def _calculate(self, simulation_time):
        zero_array = self.soil_to_gw * zero

        (
            self.soil_to_gw[:],
            self.soil_to_ssr[:],
            self.ssr_to_gw[:],
            self.slow_flow[:],
            self.ssres_flow[:],
            self.potet_rechr[:],
            self.potet_lower[:],
            self.cap_waterin[:],
            self.soil_moist[:],
            self.soil_rechr[:],
            self.hru_actet[:],
            self.cap_infil_tot[:],
            self.slow_stor[:],
            self.pref_flow_in[:],
            self.pref_flow_stor[:],
            self.perv_actet[:],
            self.soil_lower[:],
            self.dunnian_flow[:],
            self.perv_actet_hru[:],
            self.pref_flow[:],
            self.pref_flow_stor_change[:],
            self.recharge[:],
            self.slow_stor_change[:],
            self.soil_lower_change[:],
            self.soil_lower_change_hru[:],
            self.soil_lower_ratio[:],
            self.soil_moist_tot[:],
            self.soil_rechr_change[:],
            self.soil_rechr_change_hru[:],
            self.sroff[:],
            self.ssres_flow_vol[:],
            self.ssres_in[:],
            self.ssres_stor[:],
            self.swale_actet[:],
            self.unused_potet[:],
            # cascade returns:
            _,
            _,
            _,
        ) = self._calculate_soilzone(
            _pref_flow_flag=self._pref_flow_flag,
            _snow_free=self._snow_free,
            _soil2gw_flag=self._soil2gw_flag,
            cap_infil_tot=self.cap_infil_tot,
            cap_waterin=self.cap_waterin,
            compute_gwflow=self._compute_gwflow,
            compute_interflow=self._compute_interflow,
            compute_soilmoist=self._compute_soilmoist,
            compute_szactet=self._compute_szactet,
            cov_type=self.cov_type,
            current_time=self.control.current_time,
            dprst_evap_hru=zero_array.copy(),
            dprst_flag=False,
            dprst_seep_hru=zero_array.copy(),
            dunnian_flow=self.dunnian_flow,
            fastcoef_lin=self.fastcoef_lin,
            fastcoef_sq=self.fastcoef_sq,
            hru_actet=self.hru_actet,
            hru_frac_perv=self.hru_frac_perv,
            hru_impervevap=self.hru_impervevap,
            hru_in_to_cf=self.hru_in_to_cf,
            hru_intcpevap=self.hru_intcpevap,
            hru_type=self.hru_type,
            infil_hru=self.infil_hru,
            nhru=self.nhru,
            perv_actet=self.perv_actet,
            perv_actet_hru=self.perv_actet_hru,
            potet=self.potet,
            potet_lower=self.potet_lower,
            potet_rechr=self.potet_rechr,
            pref_flow=self.pref_flow,
            pref_flow_den=self.pref_flow_den,
            pref_flow_in=self.pref_flow_in,
            pref_flow_infil=self.pref_flow_infil,
            pref_flow_infil_frac=self.pref_flow_infil_frac,
            pref_flow_max=self.pref_flow_max,
            pref_flow_stor=self.pref_flow_stor,
            pref_flow_stor_change=self.pref_flow_stor_change,
            pref_flow_stor_prev=self.pref_flow_stor_prev,
            pref_flow_thrsh=self.pref_flow_thrsh,
            recharge=self.recharge,
            sat_threshold=self._sat_threshold,
            slow_flow=self.slow_flow,
            slow_stor=self.slow_stor,
            slow_stor_change=self.slow_stor_change,
            slow_stor_prev=self.slow_stor_prev,
            slowcoef_lin=self.slowcoef_lin,
            slowcoef_sq=self.slowcoef_sq,
            snow_evap=self.snow_evap,
            snowcov_area=self.snowcov_area,
            soil2gw_max=self.soil2gw_max,
            soil_lower=self.soil_lower,
            soil_lower_change=self.soil_lower_change,
            soil_lower_change_hru=self.soil_lower_change_hru,
            soil_lower_max=self.soil_lower_max,
            soil_lower_prev=self.soil_lower_prev,
            soil_lower_ratio=self.soil_lower_ratio,
            soil_moist=self.soil_moist,
            soil_moist_max=self.soil_moist_max,
            soil_moist_tot=self.soil_moist_tot,
            soil_rechr=self.soil_rechr,
            soil_rechr_change=self.soil_rechr_change,
            soil_rechr_change_hru=self.soil_rechr_change_hru,
            soil_rechr_max=self.soil_rechr_max,
            soil_rechr_prev=self.soil_rechr_prev,
            soil_to_gw=self.soil_to_gw,
            soil_to_ssr=self.soil_to_ssr,
            soil_type=self.soil_type,
            sroff=self.sroff,
            ssr2gw_exp=self.ssr2gw_exp,
            ssr2gw_rate=self.ssr2gw_rate,
            ssr_to_gw=self.ssr_to_gw,
            ssres_flow=self.ssres_flow,
            ssres_flow_vol=self.ssres_flow_vol,
            ssres_in=self.ssres_in,
            ssres_stor=self.ssres_stor,
            swale_actet=self.swale_actet,
            transp_on=self.transp_on,
            unused_potet=self.unused_potet,
            ncascade_hru=None,
            nactive_hrus=self._nactive_hrus,
            hru_route_order=self.hru_route_order,
            hru_down=None,
            hru_down_frac=None,
            hru_down_fracwt=None,
            cascade_area=None,
            upslope_dunnianflow=None,
            upslope_interflow=None,
            hru_sz_cascadeflow=None,
            stream_seg_in=None,
            cfs_conv=None,
            _compute_cascades=self._compute_cascades,
        )

        self.sroff_vol[:] = self.sroff * self.hru_in_to_cf

        return
