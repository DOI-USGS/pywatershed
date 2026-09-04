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
    CovType,
    HruType,
    dnearzero,
    nearzero,
    numba_num_threads,
    zero,
)
from ..parameters import Parameters

# set constants (may need .value for enum to be used in > comparisons)
BARESOIL = CovType.BARESOIL.value
GRASSES = CovType.GRASSES.value
LAND = HruType.LAND.value
LAKE = HruType.LAKE.value
RAIN = 0
SNOW = 1
OFF = 0
ACTIVE = 1


class PRMSCanopy(ConservativeProcess, HruMixin):
    """PRMS canopy class.

    A canopy or vegetation representation from PRMS.

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
        pkwater_ante: Previous snowpack water equivalent on each HRU
        transp_on: Flag indicating whether transpiration is occurring
            (0=no;1=yes)
        hru_ppt: Precipitation on each HRU
        hru_rain: Rain on each HRU
        hru_snow: Snow on each HRU
        imbalance_behavior: one of ["defer", None, "warn", "error"]
            with "defer" being the default and defering to
            control.options["imbalance_behavior"] when available. When
            control.options["imbalance_behavior"] is not avaiable,
            imbalance_behavior is set to "warn".
        calc_method: one of ["numba", "numpy"]. None defaults to
            "numba".
        verbose: Print extra information or not?
        load_n_time_batches: not-implemented
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
        pkwater_ante: adaptable,
        transp_on: adaptable,
        hru_ppt: adaptable,
        hru_rain: adaptable,
        hru_snow: adaptable,
        potet: adaptable,
        pptmix: adaptable,
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
        self.name = "PRMSCanopy"
        self._set_active_hrus()
        self._mask_inactive_hrus()
        self._set_inputs(locals())
        self._set_options(locals())

        self._set_budget(active_mask=self._active_hru_mask)
        self._init_calc_method()

        return

    @staticmethod
    def get_dimensions() -> tuple:
        return ("nhru",)

    @staticmethod
    def get_parameters() -> tuple:
        return (
            "cov_type",
            "covden_sum",
            "covden_win",
            "srain_intcp",
            "wrain_intcp",
            "snow_intcp",
            "potet_sublim",
            "hru_type",
        )

    @staticmethod
    def get_inputs() -> tuple:
        return (
            "pkwater_ante",
            "transp_on",
            "hru_ppt",
            "hru_rain",
            "hru_snow",
            "potet",
            "pptmix",
        )

    @staticmethod
    def get_init_values() -> dict:
        return {
            "net_ppt": zero,
            "net_rain": zero,
            "net_snow": zero,
            "intcp_changeover": zero,
            "intcp_evap": zero,
            "intcp_form": 0,  # = RAIN bool would have to match RAIN/SNOW
            "intcp_stor": zero,
            "intcp_transp_on": 0,  # could make boolean
            "hru_intcpevap": zero,
            "hru_intcpstor": zero,
            "hru_intcpstor_change": zero,
            "hru_intcpstor_old": zero,
        }

    @staticmethod
    def get_restart_variables() -> list:
        return [
            "hru_intcpstor",
            "intcp_stor",
            "intcp_transp_on",
        ]

    @staticmethod
    def get_mass_budget_terms():
        return {
            "inputs": ["hru_rain", "hru_snow"],
            "outputs": [
                "net_rain",
                "net_snow",
                "hru_intcpevap",
                "intcp_changeover",  # ?? always zero
            ],
            "storage_changes": ["hru_intcpstor_change"],
        }

    def _set_initial_conditions(self):
        """Set initial conditions for variables not in get_init_values"""
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

        if self._calc_method.lower() in ["numba"]:
            import numba as nb

            numba_msg = f"{self.name} jit compiling with numba "
            nb_parallel = (numba_num_threads is not None) and (
                numba_num_threads > 1
            )
            if nb_parallel:
                numba_msg += f"and using {numba_num_threads} threads"
            print(numba_msg, flush=True)

            # JLM: note. I gave up on specifying signatures because it
            #      appears impossible/undocumented how to specify the type
            #      of a passed function. The work is not in vain as then
            #      types will be required for f90. The signatures remain
            #      here commented for that reason and in case we can get it
            #      to work in the future.

            # self._intercept_numba = nb.njit(
            #     nb.types.Tuple(
            #         (
            #             nb.float64[:],  # intcp_stor
            #             nb.float64[:],  # net_precip
            #         )
            #     )(
            #         nb.float64[:],  # precip
            #         nb.float64[:],  # stor_max
            #         nb.float64[:],  # cov
            #         nb.float64[:],  # intcp_stor
            #         nb.float64[:],  # net_precip
            #     ),
            #     fastmath=True,
            # )(self._intercept)
            self._intercept = nb.njit(fastmath=True)(self._intercept)

            # self._calculate_numba = nb.njit(
            #     nb.types.Tuple(
            #         (
            #             nb.float64[:],  # intcp_evap
            #             nb.float64[:],  # intcp_stor
            #             nb.float64[:],  # net_rain
            #             nb.float64[:],  # net_snow
            #             nb.float64[:],  # net_ppt
            #             nb.float64[:],  # hru_intcpstor
            #             nb.float64[:],  # hru_intcpevap
            #             nb.float64[:],  # intcp_changeover
            #             nb.int32[:],  # intcp_transp_on
            #         )
            #     )(
            #         nb.int32,  # np.int32(self.nhru)
            #         nb.int64[:],  # cov_type
            #         nb.float64[:],  # covden_sum
            #         nb.float64[:],  # covden_win
            #         nb.float64[:],  # hru_intcpstor
            #         nb.float64[:],  # hru_intcpevap
            #         nb.float64[:],  # hru_ppt
            #         nb.float64[:],  # hru_rain
            #         nb.float64[:],  # hru_snow
            #         nb.float64[:],  # intcp_changeover
            #         nb.float64[:],  # intcp_evap
            #         nb.int32[:],  # intcp_form
            #         nb.float64[:],  # intcp_stor
            #         nb.int32[:],  # intcp_transp_on
            #         nb.float64[:],  # net_ppt
            #         nb.float64[:],  # net_rain
            #         nb.float64[:],  # net_snow
            #         nb.float64[:],  # self.pkwater_ante
            #         nb.float64[:],  # potet
            #         nb.float64[:],  # potet_sublim
            #         nb.float64[:],  # snow_intcp
            #         nb.float64[:],  # srain_intcp
            #         nb.float64[:],  # transp_on
            #         nb.float64[:],  # wrain_intcp
            #         nb.float64,  # np.float64(time_length),
            #         nb.int64[:],  # self._hru_type
            #         nb.float64,  # nearzero
            #         nb.float64,  # dnearzero,
            #         nb.int32,  # BARESOIL,
            #         nb.int32,  # GRASSES,
            #         nb.int32,  # LAND,
            #         nb.int32,  # LAKE,
            #         nb.int32,  # RAIN,
            #         nb.int32,  # SNOW,
            #         nb.int32,  # OFF,
            #         nb.int32,  # ACTIVE,
            #         nb.typeof(self._intercept_numba),  # function.
            #     ),
            #     fastmath=True,
            # )(self._calculate_procedural)
            self._calculate_canopy = nb.njit(
                fastmath=True, parallel=nb_parallel
            )(self._calculate_numpy)

        else:
            self._calculate_canopy = self._calculate_numpy

        return

    def _advance_variables(self) -> None:
        """Advance canopy variables"""
        self.hru_intcpstor_old[:] = self.hru_intcpstor
        return

    def _calculate(self, time_length) -> None:
        """Calculate canopy terms for a time step

        Args:
            simulation_time: current simulation time

        Returns:
            None

        """
        (
            self.intcp_evap[:],
            self.intcp_form[:],
            self.intcp_stor[:],
            self.net_rain[:],
            self.net_snow[:],
            self.pptmix[:],
            self.net_ppt[:],
            self.hru_intcpstor[:],
            self.hru_intcpevap[:],
            self.intcp_changeover[:],
            self.intcp_transp_on[:],
        ) = self._calculate_canopy(
            nactive_hrus=self._nactive_hrus,
            wh_active_hrus=self._wh_active_hrus,
            cov_type=self.cov_type,
            covden_sum=self.covden_sum,
            covden_win=self.covden_win,
            hru_intcpstor=self.hru_intcpstor,
            hru_intcpevap=self.hru_intcpevap,
            hru_ppt=self.hru_ppt,
            hru_rain=self.hru_rain,
            hru_snow=self.hru_snow,
            intcp_changeover=self.intcp_changeover,
            intcp_evap=self.intcp_evap,
            intcp_stor=self.intcp_stor,
            intcp_transp_on=self.intcp_transp_on,
            net_ppt=self.net_ppt,
            net_rain=self.net_rain,
            net_snow=self.net_snow,
            pptmix=self.pptmix,
            pkwater_ante=self.pkwater_ante,
            potet=self.potet,
            potet_sublim=self.potet_sublim,
            snow_intcp=self.snow_intcp,
            srain_intcp=self.srain_intcp,
            transp_on=self.transp_on,
            wrain_intcp=self.wrain_intcp,
            time_length=time_length,
            hru_type=self.hru_type,
            nearzero=nearzero,
            dnearzero=dnearzero,
            baresoil=np.int32(BARESOIL),
            grasses=np.int32(GRASSES),
            land=np.int32(LAND),
            lake=np.int32(LAKE),
            rain=np.int32(RAIN),
            snow=np.int32(SNOW),
            off=np.int32(OFF),
            active=np.int32(ACTIVE),
            intercept=self._intercept,
        )

        self.hru_intcpstor_change[:] = (
            self.hru_intcpstor - self.hru_intcpstor_old
        )

        return

    @staticmethod
    def _calculate_numpy(
        nactive_hrus,
        wh_active_hrus,
        cov_type,
        covden_sum,
        covden_win,
        hru_intcpstor,
        hru_intcpevap,
        hru_ppt,
        hru_rain,
        hru_snow,
        intcp_changeover,
        intcp_evap,
        intcp_stor,
        intcp_transp_on,
        net_ppt,
        net_rain,
        net_snow,
        pptmix,
        pkwater_ante,
        potet,
        potet_sublim,
        snow_intcp,
        srain_intcp,
        transp_on,
        wrain_intcp,
        time_length,
        hru_type,
        nearzero,
        dnearzero,
        baresoil,
        grasses,
        land,
        lake,
        rain,
        snow,
        off,
        active,
        intercept,
    ):
        # TODO: would be nice to alphabetize the arguments
        #       probably while keeping constants at the end.
        #       intcp_form and intcp_transp_on also do not appear to be
        #       actual inputs
        #       Keep the f90 call signature consistent with the args in
        #       python/numba.

        intcp_form = np.full_like(hru_rain, -9999, dtype="int32")

        for aa in prange(nactive_hrus):
            i = wh_active_hrus[aa]

            netrain = hru_rain[i]
            netsnow = hru_snow[i]

            if transp_on[i] == ACTIVE:
                cov = covden_sum[i]
                stor_max_rain = srain_intcp[i]
            else:
                cov = covden_win[i]
                stor_max_rain = wrain_intcp[i]

            intcp_form[i] = RAIN
            if hru_snow[i] > 0.0:
                intcp_form[i] = SNOW

            intcpstor = intcp_stor[i]
            intcpevap = 0.0
            changeover = 0.0
            extra_water = 0.0

            # lake or bare ground hrus
            if hru_type[i] == LAKE or cov_type[i] == BARESOIL:
                if cov_type[i] == BARESOIL and intcpstor > 0.0:
                    extra_water = intcp_stor[i]
                intcpstor = 0.0

            # ***** go from summer to winter cover density
            if transp_on[i] == OFF and intcp_transp_on[i] == ACTIVE:
                intcp_transp_on[i] = OFF
                if intcpstor > 0.0:
                    diff = covden_sum[i] - cov
                    changeover = intcpstor * diff
                    if cov > 0.0:
                        if changeover < 0.0:
                            intcpstor = intcpstor * covden_sum[i] / cov
                            changeover = 0.0
                    else:
                        intcpstor = 0.0

            # **** go from winter to summer cover density, excess = throughfall
            elif transp_on[i] == ACTIVE and intcp_transp_on[i] == OFF:
                intcp_transp_on[i] = ACTIVE
                if intcpstor > 0.0:
                    diff = covden_win[i] - cov
                    changeover = intcpstor * diff
                    if cov > 0.0:
                        if changeover < 0.0:
                            intcpstor = intcpstor * covden_win[i] / cov
                            changeover = 0.0
                    else:
                        intcpstor = 0.0

            # *****Determine the amount of interception from rain
            # IF ( Hru_type(i)/=LAKE .AND. Cov_type(i)/=BARESOIL ) THEN
            if hru_type[i] != LAKE and cov_type[i] != BARESOIL:
                if hru_rain[i] > 0.0:
                    if cov > 0.0:
                        # IF ( Cov_type(i)>GRASSES ) THEN
                        if cov_type[i] > GRASSES:
                            # intercept(
                            #     Hru_rain(i), stor_max_rain, cov, intcpstor,
                            #     netrain)
                            intcpstor, netrain = intercept(
                                hru_rain[i],
                                stor_max_rain,
                                cov,
                                intcpstor,
                                netrain,
                            )
                        elif cov_type[i] == GRASSES:
                            # if there is no snowpack and no snowfall, then
                            # apparently, grasses can intercept rain.
                            # intcp.f90:416
                            # IF ( Pkwater_equiv(i)<DNEARZERO .AND.
                            #      netsnow<NEARZERO ) THEN
                            # intcp runs before snowcomp, so the
                            # Pkwater_equiv it sees is the previous
                            # timestep's value, which is what snowcomp
                            # publishes as pkwater_ante
                            # (Pkwater_ante = Pkwater_equiv,
                            # snowcomp.f90:946).
                            if (
                                pkwater_ante[i] < dnearzero
                                and netsnow < nearzero
                            ):
                                intcpstor, netrain = intercept(
                                    hru_rain[i],
                                    stor_max_rain,
                                    cov,
                                    intcpstor,
                                    netrain,
                                )

            # ******Determine amount of interception from snow
            if hru_snow[i] > 0.0:
                if cov > 0.0:
                    if cov_type[i] > GRASSES:
                        intcpstor, netsnow = intercept(
                            hru_snow[i],
                            snow_intcp[i],
                            cov,
                            intcpstor,
                            netsnow,
                        )
                        if netsnow < nearzero:
                            netrain = netrain + netsnow
                            netsnow = 0.0
                            # todo: deal with newsnow and pptmix?
                            # Newsnow(i) = OFF
                            pptmix[i] = 0

            # todo: canopy application of irrigation water based on irr_type

            # ******compute evaporation or sublimation of interception

            # if precipitation assume no evaporation or sublimation
            if intcpstor > 0.0:
                if hru_ppt[i] < nearzero:
                    epan_coef = 1.0
                    evrn = potet[i] / epan_coef
                    evsn = potet[i] * potet_sublim[i]

                    # todo: pan et
                    # IF ( Use_pandata==ACTIVE ) THEN
                    # evrn = Pan_evap(Hru_pansta(i))
                    # IF ( evrn<0.0 ) evrn = 0.0
                    # ENDIF

                    if intcp_form[i] == SNOW:
                        z = intcpstor - evsn
                        if z > 0:
                            intcpstor = z
                            intcpevap = evsn
                        else:
                            intcpevap = intcpstor
                            intcpstor = 0.0
                    else:
                        d = intcpstor - evrn
                        if d > 0.0:
                            intcpstor = d
                            intcpevap = evrn
                        else:
                            intcpevap = intcpstor
                            intcpstor = 0.0

            if intcpevap * cov > potet[i]:
                last = intcpevap
                if cov > 0.0:
                    intcpevap = potet[i] / cov
                else:
                    intcpevap = 0.0
                intcpstor = intcpstor + last - intcpevap

            # Store calculated values in output variables
            intcp_evap[i] = intcpevap
            intcp_stor[i] = intcpstor
            net_rain[i] = netrain
            net_snow[i] = netsnow
            net_ppt[i] = netrain + netsnow
            hru_intcpstor[i] = intcpstor * cov
            hru_intcpevap[i] = intcpevap * cov

            intcp_changeover[i] = changeover + extra_water

        return (
            intcp_evap,
            intcp_form,
            intcp_stor,
            net_rain,
            net_snow,
            pptmix,
            net_ppt,
            hru_intcpstor,
            hru_intcpevap,
            intcp_changeover,
            intcp_transp_on,
        )

    @staticmethod
    def _intercept(precip, stor_max, cov, intcp_stor, net_precip):
        net_precip = precip * (1.0 - cov)
        intcp_stor = intcp_stor + precip
        if intcp_stor > stor_max:
            net_precip = net_precip + (intcp_stor - stor_max) * cov
            intcp_stor = stor_max
        return intcp_stor, net_precip

    @staticmethod
    def update_net_precip(
        precip, stor_max, covden, intcp_stor, net_precip, idx
    ):
        net_precip[idx] = precip[idx] * (1.0 - covden[idx])
        intcp_stor[idx] += precip[idx]
        for i in idx[0]:
            if intcp_stor[i] > stor_max[i]:
                net_precip[i] += (intcp_stor[i] - stor_max[i]) * covden[i]
                intcp_stor[i] = stor_max[i]
        return
