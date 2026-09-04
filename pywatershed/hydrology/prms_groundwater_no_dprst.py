import pathlib as pl
from typing import Literal, Union

from ..base.adapter import adaptable
from ..base.control import Control
from ..constants import nan, zero
from ..parameters import Parameters
from .prms_groundwater import PRMSGroundwater


class PRMSGroundwaterNoDprst(PRMSGroundwater):
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
        soil_to_gw: adaptable,
        ssr_to_gw: adaptable,
        imbalance_behavior: Literal["defer", None, "warn", "error"] = "defer",
        calc_method: Literal["fortran", "numba", "numpy"] = None,
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
            soil_to_gw=soil_to_gw,
            ssr_to_gw=ssr_to_gw,
            dprst_seep_hru=None,
            imbalance_behavior=imbalance_behavior,
            calc_method=calc_method,
            input_aliases=input_aliases,
            verbose=verbose,
            restart_read=restart_read,
            restart_write=restart_write,
            restart_write_freq=restart_write_freq,
        )

        self.name = "PRMSGroundwaterNoDprst"
        self._set_budget(active_mask=self._active_hru_mask)

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
        )

    @staticmethod
    def get_mass_budget_terms():
        return {
            "inputs": [
                "soil_to_gw",
                "ssr_to_gw",
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

    def _calculate(self, simulation_time):
        zero_array = self.gwres_stor * zero
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
            zero_array,
            self.gwres_stor,
            self.gwflow_coef,
            self.gwsink_coef,
            self.gwres_stor_old,
            self.hru_in_to_cf,
        )
        return
