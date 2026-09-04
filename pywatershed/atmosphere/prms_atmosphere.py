import pathlib as pl
from typing import Literal, Union
from warnings import warn

import numpy as np

from pywatershed.base.hru_mixin import HruMixin
from pywatershed.base.process import Process
from pywatershed.utils.netcdf_utils import NetCdfWrite

from ..base.adapter import adaptable
from ..base.control import Control
from ..constants import inch2cm, nan, nearzero, one, zero
from ..parameters import Parameters
from ..utils.prms_dyn_param import PrmsDynamicParameter
from ..utils.time_utils import (
    datetime_day_of_month,
    datetime_jsol,
    datetime_month,
)
from .solar_constants import solf


# may not use this if they cant be called with jit
# if it can, put it in a common utility.
def tile_space_to_time(arr: np.ndarray, n_time) -> np.ndarray:
    return np.tile(arr, (n_time, 1))


def tile_time_to_space(arr: np.ndarray, n_space) -> np.ndarray:
    return np.transpose(np.tile(arr, (n_space, 1)))


class PRMSAtmosphere(Process, HruMixin):
    """PRMS atmospheric boundary layer model.

    Implementation based on PRMS 5.2.1 with theoretical documentation given in
    the PRMS-IV documentation:

    `Markstrom, S. L., Regan, R. S., Hay, L. E., Viger, R. J., Webb, R. M.,
    Payn, R. A., & LaFontaine, J. H. (2015). PRMS-IV, the
    precipitation-runoff modeling system, version 4. US Geological Survey
    Techniques and Methods, 6, B7.
    <https://pubs.usgs.gov/tm/6b7/pdf/tm6-b7.pdf>`__

    This representation uses precipitation and temperature inputs. Relative
    humidity could be added as well.

    The boundary layer calculates and manages the following variables (given
    by PRMSAtmosphere.get_variables()):

    * tmaxf, tminf, prmx, hru_ppt, hru_rain, hru_snow, swrad, potet, transp_on

    PRMS adjustments to temperature and precipitation are applied here to
    the inputs. Shortwave radiation (using degree day method) and potential
    evapotranspiration (Jensen and Haise ,1963) and a temperature based
    transpiration flag (transp_on) are also calculated.

    Note that all variables are calculated for all time upon the first advance
    and that all calculated variables are written to NetCDF (when netcdf output
    is requested) the first time output is requested. This is effectively a
    complete preprocessing of the input CBH files to the fields the model
    actually uses on initialization. For an example of preprocessing the
    variables in PRMSAtmosphere, see
    `this notebook <https://github.com/DOI-USGS/pywatershed/tree/main/examples/04_preprocess_atm.ipynb>`_.

    The full time version of a variable is given by the "private" version of
    the variable which is named with a single-leading underscore (eg tmaxf for
    all time is _tmaxf).

    This full-time initialization may not be tractable for large domains and/or
    long periods of time and require changes to batch the processing of the
    variables. The benefits of full-time initialization are 1) the code is
    vectorized and fast for such a large calculation, 2) the initialization of
    this class effectively preprocess all the inputs to the rest of the model
    and can then be skipped in subsequent model calls (unless the parameters
    are changing).

    Args:
        control: a Control object
        discretization: a discretization of class Parameters
        parameters: a parameter object of class Parameters

        prcp: daily precipitation
        tmax: daily maximum temperature
        tmin: daily minimum temperature
        soltab_potsw: the solar table of potential shortwave radiation
        soltab_horad_potsw: the solar table of potential shortwave
            radiation on a horizontal plane
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
        prcp: Union[str, pl.Path],
        tmax: Union[str, pl.Path],
        tmin: Union[str, pl.Path],
        soltab_potsw: adaptable,
        soltab_horad_potsw: adaptable,
        input_aliases: dict = None,
        verbose: bool = False,
        restart_read: Union[pl.Path, bool] = False,
        restart_write: Union[pl.Path, bool] = False,
        restart_write_freq: Literal["y", "m", "d", False] = False,
    ) -> None:
        # Defering handling batch handling of time chunks but self.n_time_chunk
        # is a dimension used in the metadata/variables dimensions.
        # TODO: make time chunking options work (esp with output)
        # if n_time_chunk <= 0:
        #     self.n_time_chunk = control.n_times
        # else:
        #     self.n_time_chunk = n_time_chunk

        # Initialize full time with nans
        self._time = np.full(control.n_times, nan, dtype="datetime64[s]")

        metadata_patches = {
            kk: {"dims": ("ntime", "nhru")} for kk in self.variables
        }

        super().__init__(
            control=control,
            discretization=discretization,
            parameters=parameters,
            input_aliases=input_aliases,
            metadata_patches=metadata_patches,
            metadata_patch_conflicts="left",
            restart_read=restart_read,
            restart_write=restart_write,
            restart_write_freq=restart_write_freq,
        )
        self.name = "PRMSAtmosphere"
        self._set_active_hrus()
        self._set_inputs(locals())
        self._set_options(locals())

        self.calculate_transp = self.calc_transp_tindex

        self._calculated = False
        self._netcdf_initialized = False

        return

    def _calculate_all_time(self):
        if self._calculated:
            return

        # Eventually refactor to work at specific chunks of time
        for input in ["prcp", "tmax", "tmin"]:
            # # this is a bit of a mess: ._dataset.dataset
            input_time = self._input_variables_dict[input]._nc_read.times
            # dont do this...
            # self[input][:] = ds.dataset[input][:].data
            if np.isnan(self._time[0]):
                self._time[:] = input_time
            else:
                assert (input_time == self._time).all()

        start_time_ind = np.where(self._time == self.control._start_time)[0]
        if not len(start_time_ind):
            msg = "Control start_time is not in the input data time"
            raise ValueError(msg)
        self._init_time_ind = start_time_ind[0]

        # Solve all variables for all time
        self._month_ind_12 = datetime_month(self._time) - 1  # (time)
        self._month_ind_1 = np.zeros(self._time.shape, dtype=int)  # (time)
        self._month = datetime_month(self._time)  # (time)
        self._dom = datetime_day_of_month(self._time)  # (time)

        self.adjust_temperature()
        self.adjust_precip()
        self.calculate_sw_rad_degree_day()
        self.calculate_potential_et_jh()
        self.calculate_transp()

        self._mask_inactive_hrus()

        # JLM todo: delete large variables on self for memory management
        self._calculated = True

        return

    @staticmethod
    def get_dimensions():
        return ("nhru", "nmonth", "ndoy", "ntime")

    @staticmethod
    def get_parameters():
        return (
            "doy",
            "radadj_intcp",
            "radadj_slope",
            "tmax_index",
            "dday_slope",
            "dday_intcp",
            "radmax",
            "ppt_rad_adj",
            "tmax_allsnow",
            "tmax_allrain_offset",
            "hru_slope",
            "hru_type",
            "radj_sppt",
            "radj_wppt",
            "hru_lat",
            "hru_area",
            "hru_aspect",
            "jh_coef",
            "jh_coef_hru",
            "tmax_cbh_adj",
            "tmin_cbh_adj",
            "tmax_allsnow",
            "tmax_allrain_offset",
            "snow_cbh_adj",
            "rain_cbh_adj",
            "adjmix_rain",
            "transp_beg",
            "transp_end",
            "transp_tmax",
            "radadj_intcp",  # below are solar params used by Atmosphere
            "radadj_slope",
            "tmax_index",
            "dday_slope",
            "dday_intcp",
            "radmax",
            "ppt_rad_adj",
            "tmax_allsnow",
            "tmax_allrain_offset",
            "hru_slope",
            "radj_sppt",
            "radj_wppt",
            "hru_lat",
            "hru_area",
            "temp_units",
        )

    @staticmethod
    def get_inputs() -> tuple:
        return (
            "prcp",
            "tmax",
            "tmin",
            "soltab_potsw",
            "soltab_horad_potsw",
        )

    @staticmethod
    def get_init_values() -> dict:
        return {
            "tmaxf": nan,
            "tminf": nan,
            "prmx": nan,
            "hru_ppt": nan,
            "hru_rain": nan,
            "hru_snow": nan,
            "swrad": nan,
            "potet": nan,
            # "hru_actet": zero,
            # "available_potet": nan,
            "transp_on": 0,
            "tmax_sum": zero,
            "tmaxc": nan,
            "tavgc": nan,
            "tminc": nan,
            "pptmix": -9999,
            "orad_hru": nan,
        }

    @staticmethod
    def get_restart_variables() -> list:
        # The restart capability in PRMS does not work, PRMS fails daily
        # restart test starting on many days, including 1979-12-31
        # as documented on
        # nueva: ~/usgs/pywatershed/autotest/prms_restart_transp_on
        # so a more in-depth solution is necessary.
        # raise NotImplementedError(
        #     "Restart capability not implemented for PRMSAtmosphere"
        # )
        return ["tmax_sum", "transp_on"]

    def _init_diagnostic_vars(self) -> None:
        return

    def _set_initial_conditions(self):
        return

    def _advance_variables(self):
        if not self._calculated:
            self._calculate_all_time()
        for vv in self.variables:
            self[vv].advance()
        return

    def _calculate(self, time_length):
        return

    def adjust_temperature(self):
        """Input temperature adjustments using calibrated parameters."""

        # throw an error if these have different shapes
        if self["tmax_cbh_adj"].shape != self["tmin_cbh_adj"].shape:
            msg = (
                "Not implemented: tmin/tmax cbh adj parameters "
                "with different shapes"
            )
            raise NotImplementedError(msg)

        if self["tmax_cbh_adj"].shape[0] == self.nmonth:
            month_ind = self._month_ind_12
        elif self["tmax_cbh_adj"].shape[0] == 1:
            month_ind = self._month_ind_1
        else:
            msg = (
                "Unexpected month dimension for cbh "
                "temperature adjustment params"
            )
            raise ValueError(msg)

        # (time, space) dimensions on these variables
        ivd = self._input_variables_dict
        self.tmaxf.data[:] = ivd["tmax"].data + self.tmax_cbh_adj[month_ind]
        self.tminf.data[:] = ivd["tmin"].data + self.tmin_cbh_adj[month_ind]
        self.tminc.data[:] = (self["tminf"].data - 32.0) * (5 / 9)
        self.tmaxc.data[:] = (self["tmaxf"].data - 32.0) * (5 / 9)
        self.tavgc.data[:] = (self["tmaxc"].data + self["tminc"].data) / 2.0

        return

    def adjust_precip(self):
        """Input precipitation adjustments using calibrated parameters.

        Snow/rain partitioning of total precip depends on adjusted temperature
        in addition to depending on additonal parameters.

        Returns:
            None
        """

        ivd = self._input_variables_dict

        # throw an error shapes are inconsistent
        shape_list = np.array(
            [
                self.tmax_allsnow.shape[0],
                self.tmax_allrain_offset.shape[0],
                self.snow_cbh_adj.shape[0],
                self.rain_cbh_adj.shape[0],
                self.adjmix_rain.shape[0],
            ]
        )
        if not (shape_list == self.nmonth).all():
            msg = (
                "Not implemented: cbh precip adjustment parameters with "
                " different shapes"
            )
            raise NotImplementedError(msg)

        tmax_allrain = self.tmax_allsnow + self.tmax_allrain_offset

        if self.tmax_allsnow.shape[0] == self.nmonth:
            month_ind = self._month_ind_12
        elif self.tmax_allsnow.shape[0] == 1:
            month_ind = self._month_ind_1
        else:
            msg = "Unexpected month dimension for cbh precip adjustment params"
            raise ValueError(msg)

        # Order MATTERS in calculating the prmx mask
        # The logic in PRMS is if(all_snow),elif(all_rain),else(mixed)
        # so we set the mask in the reverse order
        # Calculate the mix everywhere, then set the precip/rain/snow amounts
        # from the conditions.
        tdiff = self.tmaxf.data - self.tminf.data
        tdiff = np.where(tdiff < nearzero, 1.0e-4, tdiff)
        self.prmx.data[:] = (
            (self.tmaxf.data - self.tmax_allsnow[month_ind]) / tdiff
        ) * self.adjmix_rain[month_ind]

        self.prmx.data[:] = np.where(
            self.prmx.data < zero, zero, self.prmx.data
        )
        self.prmx.data[:] = np.where(self.prmx.data > one, one, self.prmx.data)
        del tdiff

        wh_all_rain = np.where(
            np.logical_or(
                self.tminf.data > self.tmax_allsnow[month_ind],
                self.tmaxf.data >= tmax_allrain[month_ind],
            )
        )
        self.prmx.data[wh_all_rain] = one

        wh_all_snow = np.where(self.tmaxf.data <= self.tmax_allsnow[month_ind])
        self.prmx.data[wh_all_snow] = zero

        # This is in climate_hru as a condition of calling climateflow
        # (eye roll)
        self.prmx.data[:] = np.where(
            ivd["prcp"].data <= zero, zero, self.prmx.data
        )

        # Recalculate/redefine these now based on prmx instead of the
        # temperature logic
        wh_all_snow = np.where(self.prmx.data <= zero)
        wh_all_rain = np.where(self.prmx.data >= one)

        # Mixed case (everywhere, to be overwritten by the all-snow/rain-fall
        # cases)
        self.hru_ppt.data[:] = ivd["prcp"].data * self.snow_cbh_adj[month_ind]
        self.hru_rain.data[:] = self.prmx.data * self.hru_ppt.data
        self.hru_snow.data[:] = self.hru_ppt.data - self.hru_rain.data

        # All precip is snow case
        # The condition to be used later:
        self.hru_ppt.data[wh_all_snow] = (
            ivd["prcp"].data * self.snow_cbh_adj[month_ind]
        )[wh_all_snow]
        self.hru_snow.data[wh_all_snow] = self.hru_ppt.data[wh_all_snow]
        self.hru_rain.data[wh_all_snow] = zero

        # All precip is rain case
        # The condition to be used later:
        self.hru_ppt.data[wh_all_rain] = (
            ivd["prcp"].data * self.rain_cbh_adj[month_ind]
        )[wh_all_rain]
        self.hru_rain.data[wh_all_rain] = self.hru_ppt.data[wh_all_rain]
        self.hru_snow.data[wh_all_rain] = zero

        cond = (
            (self.hru_ppt.data > zero)
            & (self.tmaxf.data > self.tmax_allsnow[month_ind])
            & (
                (self.tminf.data <= self.tmax_allsnow[month_ind])
                & (self.tmaxf.data < tmax_allrain[month_ind])
            )
            & (self.prmx.data < one)
        )
        self.pptmix.data[:] = np.where(cond, 1, 0)

        return

    def calculate_sw_rad_degree_day(self) -> None:
        """Calculate shortwave radiation using the degree day method.

        Returns:
            None

        """

        solar_params = {}
        solar_param_names = (
            "radadj_intcp",
            "radadj_slope",
            "tmax_index",
            "dday_slope",
            "dday_intcp",
            "radmax",
            "ppt_rad_adj",
            "tmax_allsnow",
            "tmax_allrain_offset",
            "hru_slope",
            "radj_sppt",
            "radj_wppt",
            "hru_lat",
            "hru_area",
        )
        for name in solar_param_names:
            solar_params[name] = self[name]

        ivd = self._input_variables_dict
        self.swrad.data[:], self.orad_hru.data[:] = self._ddsolrad_run(
            dates=self._time,
            tmax_hru=self.tmaxf.data,
            hru_ppt=self.hru_ppt.data,
            soltab_potsw=ivd["soltab_potsw"].data,
            soltab_horad_potsw=ivd["soltab_horad_potsw"].data,
            **solar_params,
            nmonth=self.nmonth,
        )

        return

    # @jit
    @staticmethod
    def _ddsolrad_run(
        dates: np.ndarray,  # [n_time]
        tmax_hru: np.ndarray,  # [n_time, n_hru]
        hru_ppt: np.ndarray,  # [n_time, n_hru]
        soltab_potsw: np.ndarray,  # [n_time, n_hru] ??
        soltab_horad_potsw: np.ndarray,  # [n_time, n_hru] ??
        radadj_intcp,  # param [12, n_hru]
        radadj_slope,  # "    "
        tmax_index,
        dday_slope,
        dday_intcp,
        radmax,
        ppt_rad_adj,
        tmax_allsnow,
        tmax_allrain_offset,
        hru_slope,  # pram [n_hru]
        radj_sppt,
        radj_wppt,
        hru_lat,
        hru_area,
        nmonth,
    ) -> (np.ndarray, np.ndarray):  # [n_time, n_hru]
        # https://github.com/nhm-usgs/prms/blob/6.0.0_dev/src/prmslib/physics/sm_solar_radiation_degday.f90
        n_time, n_hru = tmax_hru.shape

        # Transforms of time
        doy = (dates - dates.astype("datetime64[Y]")).astype(
            "timedelta64[h]"
        ).astype(int) / 24 + 1
        doy = tile_time_to_space(doy, n_hru).astype(int)
        month = (dates.astype("datetime64[M]").astype(int) % nmonth) + 1
        month = tile_time_to_space(month, n_hru)

        # is_summer
        # https://github.com/nhm-usgs/prms/blob/92f3c470bbf10e37ee23f015f60d42f6a028cf48/src/prmslib/misc/sm_prms_time.f90#L114
        is_summer = (doy >= 79) & (doy <= 265)
        northern_hemisphere = hru_lat > zero
        if not any(northern_hemisphere):
            northern_hemisphere = tile_space_to_time(
                northern_hemisphere, n_time
            )
            msg = "Implementation not checked"
            raise NotImplementedError(msg)
            is_summer = is_summer & northern_hemisphere

        hru_cossl = tile_space_to_time(np.cos(np.arctan(hru_slope)), n_time)

        def monthly_to_daily(arr):
            return arr[month[:, 0] - 1, :]

        def doy_to_daily(arr):
            return arr[doy[:, 0] - 1, :]

        # This is the start of the loop over hru & time
        # https://github.com/nhm-usgs/prms/blob/92f3c470bbf10e37ee23f015f60d42f6a028cf48/src/prmslib/physics/sm_solar_radiation_degday.f90#L103

        # dday
        dday_slope_day = monthly_to_daily(dday_slope)
        dday_intcp_day = monthly_to_daily(dday_intcp)
        dday = (dday_slope_day * tmax_hru) + dday_intcp_day + one
        dday[np.where(dday < one)] = one
        del dday_slope_day, dday_intcp_day

        # radadj
        radmax_day = monthly_to_daily(radmax)
        radadj = monthly_to_daily(radmax)  # the else condition
        wh_dday_lt_26 = np.where(dday < 26.0)
        if len(wh_dday_lt_26[0]):
            kp = dday.astype(int)[wh_dday_lt_26]  # dddayi = float(kp)
            radadj[wh_dday_lt_26] = solf[kp - 1] + (
                (solf[kp] - solf[kp - 1]) * (dday[wh_dday_lt_26] - kp)
            )
            wh_radadj_gt_max = np.where(radadj > radmax_day)
            if len(wh_radadj_gt_max[0]):
                radadj[wh_radadj_gt_max] = radmax_day[wh_radadj_gt_max]
            del kp, wh_radadj_gt_max
        del radmax_day, wh_dday_lt_26

        # pptadj
        pptadj = np.ones((n_time, n_hru))
        radadj_intcp_day = monthly_to_daily(radadj_intcp)
        radadj_slope_day = monthly_to_daily(radadj_slope)
        tmax_index_day = monthly_to_daily(tmax_index)

        ppt_rad_adj_day = monthly_to_daily(ppt_rad_adj)

        # * if
        # This is the outer if. All of the following conditions work on this
        cond_ppt_gt_rad_adj = hru_ppt > ppt_rad_adj_day
        cond_if = cond_ppt_gt_rad_adj
        wh_if = np.where(cond_if)
        if len(wh_if[0]):
            # * if.else
            pptadj[wh_if] = (
                radadj_intcp_day
                + radadj_slope_day * (tmax_hru - tmax_index_day)
            )[wh_if]

            # * if.else.if
            cond_ppt_adj_gt_one = pptadj > one
            cond_if_else_if = cond_if & cond_ppt_adj_gt_one
            wh_if_else_if = np.where(cond_if_else_if)
            if len(wh_if_else_if[0]):
                pptadj[wh_if_else_if] = one
            del radadj_intcp_day, radadj_slope_day, tmax_index_day

            # * if.if
            tmax_index_day = monthly_to_daily(tmax_index)
            cond_tmax_lt_index = tmax_hru < tmax_index_day
            cond_if_if = cond_if & cond_tmax_lt_index
            wh_if_if = np.where(cond_if_if)
            if len(wh_if_if[0]):
                # The logic equiv to but changed from the original
                radj_sppt_day = tile_space_to_time(radj_sppt, n_time)
                pptadj[wh_if_if] = radj_sppt_day[wh_if_if]

                # if.if.else: negate the if.if.if
                radj_wppt_day = tile_space_to_time(radj_wppt, n_time)
                tmax_allrain_day = monthly_to_daily(
                    tmax_allrain_offset + tmax_allsnow
                )
                cond_tmax_lt_allrain = tmax_hru < tmax_allrain_day
                cond_if_if_else = cond_if_if & cond_tmax_lt_allrain
                wh_if_if_else = np.where(cond_if_if_else)
                if len(wh_if_if_else[0]):
                    pptadj[wh_if_if_else] = radj_wppt_day[wh_if_if_else]

                # if.if.if(.if): the cake is taken!
                cond_tmax_gt_allrain_and_not_summer = (
                    tmax_hru >= tmax_allrain_day
                ) & (~is_summer)
                cond_if_if_if = (
                    cond_if_if & cond_tmax_gt_allrain_and_not_summer
                )
                wh_if_if_if = np.where(cond_if_if_if)
                if len(wh_if_if_if[0]):
                    pptadj[wh_if_if_if] = radj_wppt_day[wh_if_if_if]

        radadj = radadj * pptadj
        wh_radadj_lt_2_tenths = np.where(radadj < 0.2)
        if len(wh_radadj_lt_2_tenths[0]):
            radadj[wh_radadj_lt_2_tenths] = 0.2

        swrad = doy_to_daily(soltab_potsw) * radadj / hru_cossl
        orad_hru = radadj * doy_to_daily(soltab_horad_potsw)
        return swrad, orad_hru

    # https://github.com/nhm-usgs/prms/blob/6.0.0_dev/src/prmslib/physics/sm_potet_jh.f90

    def calculate_potential_et_jh(self) -> None:
        """Calculate potential evapotranspiration following Jensen and Haise

        Jensen and Haise (1963)

        Returns:
            None
        """
        self.potet.data[:] = self._potet_jh_run(
            dates=self._time,
            tavgc=self.tavgc.data,
            swrad=self.swrad.data,
            jh_coef=self.jh_coef,
            jh_coef_hru=self.jh_coef_hru,
            nmonth=self.nmonth,
        )
        return

    @staticmethod
    def _potet_jh_run(dates, tavgc, swrad, jh_coef, jh_coef_hru, nmonth):
        """This code mixes celcius and fahrenheit units in its calculation"""

        n_time, n_hru = tavgc.shape

        # The vector
        month = (dates.astype("datetime64[M]").astype(int) % nmonth) + 1
        month = tile_time_to_space(month, n_hru)
        tavgf = (tavgc * 9 / 5) + 32

        elh = (597.3 - (0.5653 * tavgc)) * inch2cm

        # JLM: move this out?
        def monthly_to_daily(arr):
            return arr[month[:, 0] - 1, :]

        jh_coef_day = monthly_to_daily(jh_coef)
        jh_coef_hru_day = tile_space_to_time(jh_coef_hru, n_time)

        potet = jh_coef_day * (tavgf - jh_coef_hru_day) * swrad / elh
        cond_potet_lt_zero = potet < zero
        wh_cond = np.where(cond_potet_lt_zero)
        if len(wh_cond[0]):
            potet[wh_cond] = zero

        return potet

    # # Track the amount of potential ET used at a given timestep
    # # JLM: is this strange to track here? I suppose not.
    # def consume_pot_et(self, requested_et):
    #     et = requested_et
    #     available_et = self.pot_et_current - self.pot_et_consumed
    #     if et > available_et:
    #         et = available_et
    #     self.pot_et_consumed += et
    #     return et

    def calc_transp_tindex(self):
        # INIT: Process_flag==INIT
        # transp_on inited to 0 everywhere above

        # candidate for worst code lines
        if self._params.parameters["temp_units"] == 0:
            transp_tmax_f = self.transp_tmax
        else:
            transp_tmax_f = (self.transp_tmax * (9.0 / 5.0)) + 32.0

        transp_check = self.transp_on.current.copy()  # dim nhrus only
        start_day = self.control.start_doy
        start_month = self.control.start_month

        motmp = start_month + self.nmonth

        # time zero calculations
        for hh in range(self.nhru):
            if start_month == self.transp_beg[hh]:
                # rsr, why 10? if transp_tmax < 300, should be < 10
                if start_day > 10:
                    self.transp_on.data[0, hh] = 1
                else:
                    transp_check[hh] = 1

            elif self.transp_end[hh] > self.transp_beg[hh]:
                if (start_month > self.transp_beg[hh]) and (
                    start_month < self.transp_end[hh]
                ):
                    self.transp_on.data[0, hh] = 1

            else:
                if (start_month > self.transp_beg[hh]) or (
                    motmp < self.transp_end[hh] + self.nmonth
                ):
                    self.transp_on.data[0, hh] = 1

        # vectorize
        ntime = self.transp_on.data.shape[0]
        # _transp_check = self.transp_on.data.copy()
        # self._transp_beg = tile_space_to_time(self.transp_beg, ntime)
        # self._transp_end = tile_space_to_time(self.transp_end, ntime)

        # RUN: Process_flag == RUN
        # Set switch for active transpiration period
        for tt in range(ntime):
            for hh in range(self.nhru):
                if tt > 0:
                    self.transp_on.data[tt, hh] = self.transp_on.data[
                        tt - 1, hh
                    ]
                    self.tmax_sum.data[tt, hh] = self.tmax_sum.data[tt - 1, hh]

                # check for month to turn check switch on or
                # transpiration switch off
                if self._dom[tt] == 1:
                    # check for end of period
                    if self._month[tt] == self.transp_end[hh]:
                        self.transp_on.data[tt, hh] = 0
                        transp_check[hh] = 0
                        self.tmax_sum.data[tt, hh] = zero

                    # <
                    # check for month to turn transpiration switch on or off
                    if self._month[tt] == self.transp_beg[hh]:
                        transp_check[hh] = 1
                        self.tmax_sum.data[tt, hh] = zero

                # <<
                # If in checking period, then for each day
                # sum maximum temperature until greater than temperature index
                # parameter, at which time, turn transpiration switch on, check
                # switch off freezing temperature assumed to be 32 degrees
                # Fahrenheit
                if transp_check[hh] == 1:
                    if self.tmaxf.data[tt, hh] > 32.0:
                        self.tmax_sum.data[tt, hh] = (
                            self.tmax_sum.data[tt, hh]
                            + self.tmaxf.data[tt, hh]
                        )

                    # <
                    if self.tmax_sum.data[tt, hh] > transp_tmax_f[hh]:
                        self.transp_on.data[tt, hh] = 1
                        transp_check[hh] = 0
                        self.tmax_sum.data[tt, hh] = 0.0

        # <<<
        return

    def _write_netcdf_timeseries(self) -> None:
        if not self._netcdf_initialized:
            return

        if self._netcdf_separate:
            for var in self.variables:
                if var not in self._netcdf_output_vars:
                    continue
                nc_path = self._netcdf_output_dir / f"{var}.nc"

                nc = NetCdfWrite(
                    nc_path,
                    self._params.coords,
                    [var],
                    {var: self.meta[var]},
                )
                nc.add_all_data(
                    var,
                    self[var].data,
                    self._time,
                )
                nc.close()
                assert nc_path.exists()
                print(f"Wrote file: {nc_path}")

        else:
            nc_path = self._netcdf_output_dir / f"{self.name}.nc"
            nc = NetCdfWrite(
                nc_path,
                self._params.coords,
                self._netcdf_output_vars,
                self.meta,
            )
            for var in self.variables:
                if var not in self._netcdf_output_vars:
                    continue
                nc.add_all_data(
                    var,
                    self[var].data,
                    self._time,
                )

            nc.close()
            assert nc_path.exists()
            print(f"Wrote file: {nc_path}")

        self._finalize_netcdf()
        return

    def initialize_netcdf(
        self,
        output_dir: [str, pl.Path] = None,
        separate_files: bool = None,
        output_vars: list = None,
        **kwargs,
    ):
        if (
            self._netcdf_initialized
            and "verbosity" in self.control.options.keys()
            and self.control.options["verbosity"] > 5
        ):
            msg = (
                f"{self.name} class previously initialized netcdf output "
                f"in {self._netcdf_output_dir}"
            )
            warn(msg)
            return

        if (
            "verbosity" in self.control.options.keys()
            and self.control.options["verbosity"] > 5
        ):
            print(f"initializing netcdf output for: {self.name}")

        (
            output_dir,
            output_vars,
            separate_files,
        ) = self._reconcile_nc_args_w_control_opts(
            output_dir, output_vars, separate_files
        )

        # apply defaults if necessary
        if output_dir is None:
            msg = (
                "An output directory is required to be specified for netcdf"
                "initialization."
            )
            raise ValueError(msg)

        if separate_files is None:
            separate_files = True

        self._netcdf_separate = separate_files

        self._netcdf_initialized = True
        self._netcdf_output_dir = pl.Path(output_dir)

        if output_vars is None:
            self._netcdf_output_vars = self.variables
        else:
            self._netcdf_output_vars = list(
                set(output_vars).intersection(set(self.variables))
            )
            if len(self._netcdf_output_vars) == 0:
                self._netcdf_initialized = False

        return

    def _finalize_netcdf(self) -> None:
        self._netcdf_initialized = False
        return

    def output(self):
        if self._netcdf_initialized:
            if self._verbose:
                print(
                    f"Writing FULL timeseries output for: {self.name}",
                    flush=True,
                )
            # <
            self._write_netcdf_timeseries()

        # logic for when to write restarts in the function
        self._output_restart()

        return


class PRMSAtmosphereTranspFrost(PRMSAtmosphere):
    """PRMS atmospheric boundary layer model with a frost transpiration model.

    In this subclass, the `PRMSAtmosphere` transpiration model using
    temperature index is replaced by a specified active period between the
    parameteres of (last) spring_frost and (first, killing) fall frost. This is
    as implemented in the PRMS module transp_frost.f90.

    See Also
    --------
    PRMSAtmosphere

    """

    def __init__(
        self,
        control: Control,
        discretization: Parameters,
        parameters: Parameters,
        prcp: Union[str, pl.Path],
        tmax: Union[str, pl.Path],
        tmin: Union[str, pl.Path],
        soltab_potsw: adaptable,
        soltab_horad_potsw: adaptable,
        input_aliases: dict = None,
        verbose: bool = False,
        restart_read: Union[pl.Path, bool] = False,
        restart_write: Union[pl.Path, bool] = False,
        restart_write_freq: Literal["y", "m", "d", False] = False,
    ) -> None:
        super().__init__(
            control=control,
            discretization=discretization,
            parameters=parameters,
            prcp=prcp,
            tmax=tmax,
            tmin=tmin,
            soltab_potsw=soltab_potsw,
            soltab_horad_potsw=soltab_horad_potsw,
            input_aliases=input_aliases,
            verbose=verbose,
            restart_read=restart_read,
            restart_write=restart_write,
            restart_write_freq=restart_write_freq,
        )

        self.calculate_transp = self.calc_transp_frost

        return

    @staticmethod
    def get_parameters():
        return (
            "doy",
            "radadj_intcp",
            "radadj_slope",
            "tmax_index",
            "dday_slope",
            "dday_intcp",
            "radmax",
            "ppt_rad_adj",
            "tmax_allsnow",
            "tmax_allrain_offset",
            "hru_slope",
            "hru_type",
            "radj_sppt",
            "radj_wppt",
            "hru_lat",
            "hru_area",
            "hru_aspect",
            "jh_coef",
            "jh_coef_hru",
            "tmax_cbh_adj",
            "tmin_cbh_adj",
            "tmax_allsnow",
            "tmax_allrain_offset",
            "snow_cbh_adj",
            "rain_cbh_adj",
            "adjmix_rain",
            "fall_frost",
            "spring_frost",
            "radadj_intcp",  # below are solar params used by Atmosphere
            "radadj_slope",
            "tmax_index",
            "dday_slope",
            "dday_intcp",
            "radmax",
            "ppt_rad_adj",
            "tmax_allsnow",
            "tmax_allrain_offset",
            "hru_slope",
            "radj_sppt",
            "radj_wppt",
            "hru_lat",
            "hru_area",
            "temp_units",
        )

    def _calculate_all_time(self):
        super()._calculate_all_time()

    def calc_transp_frost(self) -> None:
        self._jsol = tile_time_to_space(
            datetime_jsol(self._time), self.nhru
        )  # (time)
        spring_frost = tile_space_to_time(self.spring_frost, self.ntime)
        fall_frost = tile_space_to_time(self.fall_frost, self.ntime)

        self.transp_on.data[:, :] = np.where(
            (self._jsol >= spring_frost) & (self._jsol <= fall_frost), 1, 0
        )

        return


class PRMSAtmosphereTranspFrostDynamic(PRMSAtmosphereTranspFrost):
    """PRMSAtmosphereTranspFrost with dynamic (time-varying) frost dates.

    In this subclass, the fall_frost and spring_frost solar dates bounding
    the transpiration period may each be supplied as a PRMS dynamic parameter
    file (or PrmsDynamicParameter object) instead of the static parameters
    used by PRMSAtmosphereTranspFrost. This reproduces PRMS/GSFLOW runs with
    dyn_fallfrost_flag and/or dyn_springfrost_flag set (dynamic_param_read.f90
    updating transp_frost.f90).

    The sparse (typically annual) updates in a dynamic parameter file are
    forward-filled to daily values, starting from the first date in the file,
    exactly as PRMS applies updates: values from dates at or before the model
    start time are in effect at the start, then each new date in the file
    takes effect on that day. The model time window must begin at or after
    the first date in each dynamic parameter file.

    When neither dynamic argument is supplied, behavior is identical to
    PRMSAtmosphereTranspFrost using the static fall_frost and spring_frost
    parameters. When only one is supplied, the other uses its static
    parameter.

    Args:
        fall_frost_dyn: a PRMS dynamic parameter file (e.g. fall_frost.dyn)
            or a PrmsDynamicParameter object giving time-varying solar dates
            of the first killing frost of the fall. If None (default), the
            static fall_frost parameter is used.
        spring_frost_dyn: as for fall_frost_dyn but for the solar date of
            the last killing frost of the spring, using the static
            spring_frost parameter if None.

    See PRMSAtmosphereTranspFrost and PRMSAtmosphere for all other arguments.

    See Also
    --------
    PRMSAtmosphereTranspFrost

    """

    def __init__(
        self,
        control: Control,
        discretization: Parameters,
        parameters: Parameters,
        prcp: Union[str, pl.Path],
        tmax: Union[str, pl.Path],
        tmin: Union[str, pl.Path],
        soltab_potsw: adaptable,
        soltab_horad_potsw: adaptable,
        fall_frost_dyn: Union[str, pl.Path, PrmsDynamicParameter] = None,
        spring_frost_dyn: Union[str, pl.Path, PrmsDynamicParameter] = None,
        input_aliases: dict = None,
        verbose: bool = False,
        restart_read: Union[pl.Path, bool] = False,
        restart_write: Union[pl.Path, bool] = False,
        restart_write_freq: Literal["y", "m", "d", False] = False,
    ) -> None:
        super().__init__(
            control=control,
            discretization=discretization,
            parameters=parameters,
            prcp=prcp,
            tmax=tmax,
            tmin=tmin,
            soltab_potsw=soltab_potsw,
            soltab_horad_potsw=soltab_horad_potsw,
            input_aliases=input_aliases,
            verbose=verbose,
            restart_read=restart_read,
            restart_write=restart_write,
            restart_write_freq=restart_write_freq,
        )

        self._dyn_frost = {}
        dyn_args = {
            "fall_frost": fall_frost_dyn,
            "spring_frost": spring_frost_dyn,
        }
        for name, arg in dyn_args.items():
            if arg is None:
                continue
            if isinstance(arg, (str, pl.Path)):
                arg = PrmsDynamicParameter.load(arg, dtype="int")
            if not isinstance(arg, PrmsDynamicParameter):
                msg = (
                    f"{name}_dyn must be a path to a PRMS dynamic parameter "
                    f"file or a PrmsDynamicParameter, got {type(arg)}"
                )
                raise TypeError(msg)
            self._dyn_frost[name] = arg

        return

    def _frost_all_time(self, name: str) -> np.ndarray:
        """Frost dates over all model time, (ntime, nhru).

        From the dynamic source when supplied, forward-filled to daily from
        the first date in the dynamic parameter file and subset to model
        time. Otherwise the static parameter tiled over time.
        """
        if name not in self._dyn_frost:
            return tile_space_to_time(self[name], self.ntime)

        dyn = self._dyn_frost[name]
        dyn.daily_start_date = None  # daily data start at first file date
        dyn.daily_end_date = self._time[-1]
        window = dyn.daily_data_array.sel(
            time=slice(self._time[0], self._time[-1])
        )
        window_time = window["time"].values.astype("datetime64[s]")
        if (window_time.shape != self._time.shape) or (
            window_time != self._time
        ).any():
            if len(window_time):
                data_range = f"{window_time[0]} to {window_time[-1]}"
            else:
                data_range = "empty"
            msg = (
                f"Dynamic {name} data do not cover the model time window: "
                f"model is {self._time[0]} to {self._time[-1]} but "
                f"daily-filled dynamic data are {data_range}"
            )
            raise ValueError(msg)

        return window.values

    def calc_transp_frost(self) -> None:
        self._jsol = tile_time_to_space(
            datetime_jsol(self._time), self.nhru
        )  # (time)
        spring_frost = self._frost_all_time("spring_frost")
        fall_frost = self._frost_all_time("fall_frost")

        self.transp_on.data[:, :] = np.where(
            (self._jsol >= spring_frost) & (self._jsol <= fall_frost), 1, 0
        )

        return
