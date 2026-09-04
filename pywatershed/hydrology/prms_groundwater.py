import pathlib as pl
from typing import Literal, Union
from warnings import warn

import numpy as np

from ..base.adapter import adaptable, adapter_factory
from ..base.conservative_process import ConservativeProcess
from ..base.control import Control
from ..base.hru_mixin import HruMixin
from ..constants import nan, numba_num_threads
from ..parameters import Parameters


class PRMSGroundwater(ConservativeProcess, HruMixin):
    """PRMS groundwater reservoir.

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
        soil_to_gw: Portion of excess flow to the capillary reservoir that
            drains to the associated GWR for each HRU
        ssr_to_gw: Drainage from the gravity-reservoir to the associated GWR
            for each HRU
        dprst_seep_hru: Seepage from surface-depression storage to associated
            GWR for each HRU
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
        dprst_seep_hru: adaptable,
        dprst_flag: bool = None,
        imbalance_behavior: Literal["defer", None, "warn", "error"] = "defer",
        calc_method: Literal["numba", "numpy"] = None,
        input_aliases: dict = None,
        verbose: bool = None,
        restart_read: Union[pl.Path, bool] = False,
        restart_write: Union[pl.Path, bool] = False,
        restart_write_freq: Literal["y", "m", "d", "f", False] = False,
    ) -> None:
        super().__init__(
            control=control,
            discretization=discretization,
            parameters=parameters,
            input_aliases=input_aliases,
            restart_read=restart_read,
            restart_write=restart_write,
            restart_write_freq=restart_write_freq,
        )
        self.name = "PRMSGroundwater"
        self._set_active_hrus()
        self._mask_inactive_hrus()
        self._wh_inactive_hrus = np.where(~self._active_hru_mask)[0]

        self._set_inputs(locals())
        self._set_options(locals())

        if self._dprst_flag is None:
            self._dprst_flag = True

        # This is a hacky dprst_flag == False approach. Improve design to
        # get rid of these inputs.
        if not self._dprst_flag:
            for kk, vv in self._input_variables_dict.items():
                if vv is not None:
                    continue
                self._input_variables_dict[kk] = adapter_factory(
                    np.zeros(self._params.dimensions["nhru"]),
                    variable_name=kk,
                    control=self.control,
                )

        self._set_budget(active_mask=self._active_hru_mask)
        self._init_calc_method()

        return

    @staticmethod
    def get_dimensions() -> tuple:
        return ("nhru",)

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
        )

    @staticmethod
    def get_inputs() -> tuple:
        return (
            "soil_to_gw",
            "ssr_to_gw",
            "dprst_seep_hru",
        )

    @staticmethod
    def get_mass_budget_terms():
        return {
            "inputs": [
                "soil_to_gw",
                "ssr_to_gw",
                "dprst_seep_hru",
            ],
            "outputs": [
                "gwres_flow",
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
        }

    @staticmethod
    def get_restart_variables() -> list:
        return ["gwres_stor"]

    def _set_initial_conditions(self):
        # initialize groundwater reservoir storage
        self.gwres_stor[:] = self.gwstor_init.copy()
        self.gwres_stor_old[:] = self.gwstor_init.copy()
        return

    def _init_diagnostic_vars(self) -> None:
        return

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

            numba_msg = f"{self.name} jit compiling with numba "
            nb_parallel = (numba_num_threads is not None) and (
                numba_num_threads > 1
            )
            if nb_parallel:
                numba_msg += f"and using {numba_num_threads} threads"
            print(numba_msg, flush=True)

            self._calculate_gw = nb.njit(
                nb.types.UniTuple(nb.float64[:], 5)(
                    nb.types.Array(nb.types.int64, 1, "C", readonly=True),
                    nb.types.Array(nb.types.float64, 1, "C", readonly=True),
                    nb.float64[:],
                    nb.float64[:],
                    nb.float64[:],
                    nb.float64[:],
                    nb.types.Array(nb.types.float64, 1, "C", readonly=True),
                    nb.types.Array(nb.types.float64, 1, "C", readonly=True),
                    nb.float64[:],
                    nb.types.Array(nb.types.float64, 1, "C", readonly=True),
                ),
                fastmath=True,
                parallel=False,
            )(self._calculate_numpy)

        else:
            self._calculate_gw = self._calculate_numpy

        return

    def _advance_variables(self) -> None:
        self.gwres_stor_old[:] = self.gwres_stor
        return

    def _calculate(self, simulation_time) -> None:
        self._simulation_time = simulation_time
        (
            self.gwres_stor[:],
            self.gwres_flow[:],
            self.gwres_sink[:],
            self.gwres_stor_change[:],
            self.gwres_flow_vol[:],
        ) = self._calculate_gw(
            self._wh_inactive_hrus,
            self.hru_area,
            self.soil_to_gw,
            self.ssr_to_gw,
            self.dprst_seep_hru,
            self.gwres_stor,
            self.gwflow_coef,
            self.gwsink_coef,
            self.gwres_stor_old,
            self.hru_in_to_cf,
        )
        return

    @staticmethod
    def _calculate_numpy(
        wh_inactive_hrus,
        gwarea,
        soil_to_gw,
        ssr_to_gw,
        dprst_seep_hru,
        gwres_stor,
        gwflow_coef,
        gwsink_coef,
        gwres_stor_old,
        hru_in_to_cf,
    ):
        soil_to_gw_vol = soil_to_gw * gwarea
        ssr_to_gw_vol = ssr_to_gw * gwarea
        dprst_seep_hru_vol = dprst_seep_hru * gwarea

        # todo: what about route order

        _gwres_stor = gwres_stor * gwarea
        _gwres_stor += soil_to_gw_vol + ssr_to_gw_vol + dprst_seep_hru_vol

        _gwres_flow = _gwres_stor * gwflow_coef
        _gwres_stor -= _gwres_flow

        _gwres_sink = _gwres_stor * gwsink_coef
        idx = np.where(_gwres_sink > _gwres_stor)
        _gwres_sink[idx] = _gwres_stor[idx]
        _gwres_stor -= _gwres_sink

        # convert most units back to self variables
        # output variables
        gwres_stor = _gwres_stor / gwarea
        # for some stupid reason this is left in acre-inches
        gwres_flow = _gwres_flow / gwarea
        gwres_sink = _gwres_sink / gwarea

        gwres_stor_change = gwres_stor - gwres_stor_old
        gwres_flow_vol = gwres_flow * hru_in_to_cf

        if len(wh_inactive_hrus) > 0:
            gwres_flow[wh_inactive_hrus] = np.nan
            gwres_flow_vol[wh_inactive_hrus] = np.nan
            gwres_sink[wh_inactive_hrus] = np.nan
            gwres_stor[wh_inactive_hrus] = np.nan
            gwres_stor_change[wh_inactive_hrus] = np.nan

        return (
            gwres_stor,
            gwres_flow,
            gwres_sink,
            gwres_stor_change,
            gwres_flow_vol,
        )
