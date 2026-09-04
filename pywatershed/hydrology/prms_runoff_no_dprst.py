import pathlib as pl
from typing import Literal, Union

import numpy as np

from ..base.adapter import adaptable
from ..base.control import Control
from ..constants import HruType, nan, zero
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

# TODO: using through_rain and not net_rain and net_ppt is a WIP


class PRMSRunoffNoDprst(PRMSRunoff):
    """PRMS surface runoff without depression storage.

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
        imbalance_behavior: one of ["defer", None, "warn", "error"]
            with "defer" being the default and defering to
            control.options["imbalance_behavior"] when available. When
            control.options["imbalance_behavior"] is not avaiable,
            imbalance_behavior is set to "warn".
        calc_method: one of ["fortran", "numba", "numpy"]. None defaults to
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
        intcp_changeover_in_net_rain: bool = False,
        imbalance_behavior: Literal["defer", None, "warn", "error"] = "defer",
        calc_method: Literal["numba", "numpy"] = None,
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
            soil_lower_prev=soil_lower_prev,
            soil_rechr_prev=soil_rechr_prev,
            net_ppt=net_ppt,
            net_rain=net_rain,
            net_snow=net_snow,
            potet=potet,
            snowmelt=snowmelt,
            snow_evap=snow_evap,
            pkwater_equiv=pkwater_equiv,
            pptmix_nopack=pptmix_nopack,
            snowcov_area=snowcov_area,
            through_rain=through_rain,
            hru_intcpevap=hru_intcpevap,
            intcp_changeover=intcp_changeover,
            dprst_flag=False,
            imbalance_behavior=imbalance_behavior,
            calc_method=calc_method,
            input_aliases=input_aliases,
            verbose=verbose,
            restart_read=restart_read,
            restart_write=restart_write,
            restart_write_freq=restart_write_freq,
            intcp_changeover_in_net_rain=intcp_changeover_in_net_rain,
        )

        self.name = "PRMSRunoffNoDprst"

        self._set_inputs(locals())
        self._set_options(locals())

        self._set_budget(active_mask=self._active_hru_mask)

        self.basin_init()

        return

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
            "infil_hru": zero,
            "sroff": zero,  # todo: privatize and only make vol public
            "sroff_vol": zero,
            "hru_sroffp": zero,
            "hru_sroffi": zero,
            "imperv_stor": zero,
            "imperv_evap": zero,
            "hru_impervevap": zero,
            "hru_impervstor": zero,
            "hru_impervstor_old": zero,
            "hru_impervstor_change": zero,
        }

    @staticmethod
    def get_restart_variables() -> list:
        return [
            "imperv_stor",
            "hru_impervstor",
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
                # sroff = hru_sroffi + hru_sroffp
                "hru_sroffi",
                "hru_sroffp",
                "infil_hru",
                "hru_impervevap",
            ],
            "storage_changes": [
                "hru_impervstor_change",
            ],
        }

    def _advance_variables(self) -> None:
        self.hru_impervstor_old[:] = self.hru_impervstor
        return None

    def _calculate(self, time_length, vectorized=False):
        """Perform the core calculations"""

        zero_array = zero * self.infil
        zero_array_2d_int = np.zeros((2, 2), dtype="int32")
        nan_array = nan * self.infil
        nan_array_2d = np.zeros((2, 2)) * nan

        (
            self.infil[:],
            self.contrib_fraction[:],
            self.hru_sroffp[:],
            self.hru_sroffi[:],
            self.imperv_evap[:],
            self.hru_impervevap[:],
            self.imperv_stor[:],
            _,
            _,
            _,
            _,
            _,
            _,
            _,
            _,
            _,
            _,
            _,
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
            dprst_in=zero_array.copy(),
            dprst_seep_hru=zero_array.copy(),
            dprst_area_max=zero_array.copy(),
            dprst_vol_open=zero_array.copy(),
            dprst_vol_clos=zero_array.copy(),
            dprst_sroff_hru=zero_array.copy(),
            dprst_evap_hru=zero_array.copy(),
            dprst_insroff_hru=zero_array.copy(),
            dprst_vol_open_frac=zero_array.copy(),
            dprst_vol_clos_frac=zero_array.copy(),
            dprst_vol_frac=zero_array.copy(),
            dprst_stor_hru=zero_array.copy(),
            dprst_area_clos_max=zero_array.copy(),
            dprst_area_clos=zero_array.copy(),
            dprst_vol_open_max=zero_array.copy(),
            dprst_area_open_max=zero_array.copy(),
            dprst_area_open=zero_array.copy(),
            sro_to_dprst_perv=zero_array.copy(),
            sro_to_dprst_imperv=zero_array.copy(),
            dprst_frac_open=zero_array.copy(),
            dprst_frac_clos=zero_array.copy(),
            va_open_exp=zero_array.copy(),
            dprst_vol_clos_max=zero_array.copy(),
            va_clos_exp=zero_array.copy(),
            snowcov_area=self.snowcov_area,
            dprst_et_coef=zero_array.copy(),
            dprst_seep_rate_open=zero_array.copy(),
            dprst_vol_thres_open=zero_array.copy(),
            dprst_flow_coef=zero_array.copy(),
            dprst_seep_rate_clos=zero_array.copy(),
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

        self.sroff_vol[:] = self.sroff * self.hru_in_to_cf

        return
