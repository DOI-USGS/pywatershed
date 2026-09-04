import pathlib as pl
import warnings
from typing import Literal, Tuple, Union

import numpy as np

from pywatershed.base.hru_mixin import HruMixin
from pywatershed.base.process import Process
from pywatershed.utils.netcdf_utils import NetCdfWrite

from ..base.control import Control
from ..constants import dnearzero, nan, one, zero
from ..parameters import Parameters
from ..utils.prms5util import load_soltab_debug
from .solar_constants import ndoy, pi, pi_12, r1, solar_declination, two_pi

doy = np.arange(ndoy) + 1

# Dimensions
# The cyclic time dimension is ndoy (known apriori)
# The spatial dimension n_hru (only known on init from parameters)


def tile_space_to_time(arr: np.ndarray) -> np.ndarray:
    return np.tile(arr, (ndoy, 1))


# def tile_time_to_space(arr: np.ndarray, n_hru) -> np.ndarray:
#    return np.transpose(np.tile(arr, (n_hru, 1)))


class PRMSSolarGeometry(Process, HruMixin):
    """PRMS solar geometry.

    Implementation based on PRMS 5.2.1 with theoretical documentation given in
    the PRMS-IV documentation:

    `Markstrom, S. L., Regan, R. S., Hay, L. E., Viger, R. J., Webb, R. M.,
    Payn, R. A., & LaFontaine, J. H. (2015). PRMS-IV, the
    precipitation-runoff modeling system, version 4. US Geological Survey
    Techniques and Methods, 6, B7.
    <https://pubs.usgs.gov/tm/6b7/pdf/tm6-b7.pdf>`__

    Implements Swift's daily potential solar radiation and number of hours
    of duration on a sloping surface in [cal/cm2/day].
    Swift, 1976, equation 6.

    Primary reference: Appendix E of Dingman, S. L., 1994, Physical Hydrology.
    Englewood Cliffs, NJ: Prentice Hall, 575 p.

    Args:
        control: a Control object
        discretization: a discretization of class Parameters
        parameters: a parameter object of class Parameters
        verbose: Print extra information or not?
        from_prms_file: Load from a PRMS output file?
        from_nc_files_dir: [str, pl.Path] = None,
        restart_read: If not False will issue warning that PRMSSolarGeometry
          has no restart variables to read or write.
        restart_write: As for restart_read.
        restart_write_freq: As for restart_read.
    """

    def __init__(
        self,
        control: Control,
        discretization: Parameters,
        parameters: Parameters,
        input_aliases: dict = None,
        verbose: bool = False,
        from_prms_file: [str, pl.Path] = None,
        from_nc_files_dir: [str, pl.Path] = None,
        restart_read: Union[pl.Path, bool] = False,
        restart_write: Union[pl.Path, bool] = False,
        restart_write_freq: Literal["y", "m", "d", False] = False,
    ):
        if (restart_read is not False) or (restart_write is not False):
            warnings.warn(
                "PRMSSolarGeometry has no restart variables to read or write."
            )
        # self._time is needed by Process for timeseries arrays
        # TODO: this is redundant because the parameter doy is set
        #       on load of prms file. Could pass the name to use for
        #       self._time to super or come up with some other work around.
        self._time = doy

        super().__init__(
            control=control,
            discretization=discretization,
            parameters=parameters,
            input_aliases=input_aliases,
        )
        self.name = "PRMSSolarGeometry"
        self._set_active_hrus()
        self._set_inputs(locals())
        self._set_options(locals())

        if from_prms_file:
            (
                self.soltab_potsw.data[:],
                self.soltab_horad_potsw.data[:],
                self.soltab_sunhrs.data[:],
            ) = load_soltab_debug(from_prms_file)

        elif from_nc_files_dir:
            raise NotImplementedError()

        self._netcdf_initialized = False
        self._calculated = False

        return

    @staticmethod
    def get_inputs() -> tuple:
        return ()

    @staticmethod
    def get_variables():
        return (
            "soltab_horad_potsw",
            "soltab_potsw",
            "soltab_sunhrs",
        )

    @staticmethod
    def get_init_values() -> dict:
        return {
            "soltab_horad_potsw": nan,
            "soltab_potsw": nan,
            "soltab_sunhrs": nan,
        }

    @staticmethod
    def get_dimensions() -> tuple:
        return ("nhru", "ndoy")

    @staticmethod
    def get_parameters() -> tuple:
        return (
            "doy",
            "hru_slope",
            "radj_sppt",
            "radj_wppt",
            "hru_lat",
            "hru_area",
            "hru_aspect",
            "hru_type",
        )

    @staticmethod
    def get_restart_variables() -> list:
        return []

    def _init_diagnostic_vars(self) -> None:
        return

    def _set_initial_conditions(self):
        return

    def _calculate_all_time(self):
        self._hru_cossl = np.cos(np.arctan(self["hru_slope"]))
        # The potential radiation on horizontal surfce
        self.soltab_horad_potsw.data[:], _ = self.compute_soltab(
            np.zeros(self["nhru"]),
            np.zeros(self["nhru"]),
            self["hru_lat"],
            self.compute_t,
            self.func3,
        )

        # The potential radiaton given slope and aspect
        (
            self.soltab_potsw.data[:],
            self.soltab_sunhrs.data[:],
        ) = self.compute_soltab(
            self["hru_slope"],
            self["hru_aspect"],
            self["hru_lat"],
            self.compute_t,
            self.func3,
        )

        self._mask_inactive_hrus()

        self._calculated = True
        return

    def _advance_variables(self):
        if not self._calculated:
            self._calculate_all_time()
        for vv in self.variables:
            self[vv].advance()
        return

    def _calculate(self, time_length):
        return

    # @jit
    @staticmethod
    def compute_soltab(
        slopes: np.ndarray,
        aspects: np.ndarray,
        lats: np.ndarray,
        compute_t: callable,
        func3: callable,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Calculate the solar table
        Args:
          cossl: cos(atan(hru_slope)) [nhru] ?
          slope: slope [nhru]
          aspect: aspect [nhru]
          latitude: latitude [nhru]

        Returns: (solt, sunh)
          solt: Swift's potential solar radiation on a sloping surface in
            [cal/cm2/day]. Swift, 1976, equation 6.
            Dimensions: [ndoy, nhru]
          sunh: The number of hours of direct sunlight
            Dimensions: [ndoy, nhru]
        """
        nhru = len(slopes)

        # Slope derived quantities
        sl = np.arctan(slopes)
        sl_sin = np.sin(sl)
        sl_cos = np.cos(sl)

        # Aspect derived quantities
        aspects_rad = np.radians(aspects)
        aspects_cos = np.cos(aspects_rad)

        # Latitude derived quantities
        x0 = np.radians(lats)
        # errors in PRMS ingesting latitudes can be seen here (or earlier)
        x0_cos = np.cos(x0)

        # x1 latitude of equivalent slope
        # This is equation 13 from Lee, 1963
        x1 = np.arcsin(sl_cos * np.sin(x0) + sl_sin * x0_cos * aspects_cos)

        # d1 is the denominator of equation 12, Lee, 1963
        d1 = sl_cos * x0_cos - sl_sin * np.sin(x0) * aspects_cos
        d1 = np.where(np.abs(d1) < dnearzero, dnearzero, d1)

        # x2 is the difference in longitude between the location of
        # the HRU and the equivalent horizontal surface expressed in angle hour
        # This is equation 12 from Lee, 1963
        x2 = np.arctan(sl_sin * np.sin(aspects_rad) / d1)
        wh_d1_lt_zero = np.where(d1 < zero)
        if len(wh_d1_lt_zero[0]) > 0:
            x2[wh_d1_lt_zero] = x2[wh_d1_lt_zero] + pi

        # The hour angle from the local meridian (local solar noon) to the
        # sunrise (negative) or sunset (positive)
        # t6: is the hour angle of sunrise on the equivalent slope
        # t7: is the hour angle of sunset on the equivalent slope
        tt = compute_t(x1, solar_declination)
        t6 = (-1 * tt) - x2
        t7 = tt - x2

        # Hours of sunrise and sunset on a horizontal surface at lat
        # t0: is the hour angle of sunrise on a hroizontal surface at the HRU
        # t1: is the hour angle of sunset on a hroizontal surface at the HRU
        tt = compute_t(x0, solar_declination)
        t0 = -1 * tt
        t1 = tt

        # For HRUs that have an east or west direction component to their
        # aspect, the longitude adjustment (moving the effective slope east
        # or west) will cause either:
        # (1) sunrise to be earlier than at the horizontal plane at the HRU
        # (2) sunset to be later than at the horizontal plane at the HRU
        # This is not possible. The if statements below check for this and
        # adjust the sunrise/sunset angle hours on the equivalent slopes as
        # necessary.

        # t2: is the hour angle of sunset on the slope at the HRU
        # t3: is the hour angle of sunrise on the slope at the HRU
        t3 = t7
        wh_t7_gt_t1 = np.where(t7 > t1)
        if len(wh_t7_gt_t1[0]) > 0:
            t3[wh_t7_gt_t1] = t1[wh_t7_gt_t1]

        t2 = t6
        wh_t6_lt_t0 = np.where(t6 < t0)
        if len(wh_t6_lt_t0[0]) > 0:
            t2[wh_t6_lt_t0] = t0[wh_t6_lt_t0]

        # JLM: vectorizing requires are reverse looped order

        t6 = t6 + two_pi
        t7 = t7 - two_pi
        wh_t3_lt_t2 = np.where(t3 < t2)
        if len(wh_t3_lt_t2[0]):
            t2[wh_t3_lt_t2] = zero
            t3[wh_t3_lt_t2] = zero

        # This is if no other conditions are met
        solt = func3(x2, x1, t3, t2)
        sunh = (t3 - t2) * pi_12

        # t7 > t0
        wh_t7_gt_t0 = np.where(t7 > t0)
        if len(wh_t7_gt_t0[0]):
            solt[wh_t7_gt_t0] = (
                func3(x2, x1, t3, t2)[wh_t7_gt_t0]
                + func3(x2, x1, t7, t0)[wh_t7_gt_t0]
            )
            sunh[wh_t7_gt_t0] = (t3 - t2 + t7 - t0)[wh_t7_gt_t0] * pi_12

        # t6 < t1
        wh_t6_lt_t1 = np.where(t6 < t1)
        if len(wh_t6_lt_t1[0]):
            solt[wh_t6_lt_t1] = (
                func3(x2, x1, t3, t2)[wh_t6_lt_t1]
                + func3(x2, x1, t1, t6)[wh_t6_lt_t1]
            )
            sunh[wh_t6_lt_t1] = (t3 - t2 + t1 - t6)[wh_t6_lt_t1] * pi_12

        # The first condition checked
        mask_sl_lt_dnearzero = tile_space_to_time(np.abs(sl)) < dnearzero
        solt = np.where(
            mask_sl_lt_dnearzero, func3(np.zeros(nhru), x0, t1, t0), solt
        )

        sunh = np.where(mask_sl_lt_dnearzero, (t1 - t0) * pi_12, sunh)

        mask_sunh_lt_dnearzero = sunh < dnearzero
        sunh = np.where(mask_sunh_lt_dnearzero, zero, sunh)

        wh_solt_lt_zero = np.where(solt < zero)
        if len(wh_solt_lt_zero[0]):
            solt[wh_solt_lt_zero] = zero
            warnings.warn(
                f"{len(wh_solt_lt_zero[0])}/{np.product(solt.shape)} "
                f"locations-times with negative "
                f"potential solar radiation."
            )

        return solt, sunh

    @staticmethod
    def compute_t(
        lats: np.ndarray, solar_declination: np.ndarray
    ) -> np.ndarray:
        # JLM: why is division by earth's angular velocity not done here?
        """The "sunrise" equation

        Args:
            lats - latitudes of the hrus
            solar_declination: the declination of the sun on all julian days
        Returns:
            The angle hour from the local meridian (local solar noon) to
            the sunrise (negative) or sunset (positive).  The Earth rotates at
            the angular speed of 15 degrees/hour (2 pi / 24 hour in radians)
            and, therefore, result*(24/(2pi) radians gives the time of sunrise
            as the number of hours before the local noon, or the time of sunset
            as the number of hours after the local noon. Here the term local
            noon indicates the local time when the sun is exactly to the
            south or north or exactly overhead.

        See reference: https://github.com/nhm-usgs/prms/blob/6.0.0_dev/src/prmslib/physics/sm_solar_radiation.f90
        """
        nhru = len(lats)
        lats_mat = np.tile(-1 * np.tan(lats), (ndoy, 1))
        sol_dec_mat = np.transpose(
            np.tile(np.tan(solar_declination), (nhru, 1))
        )
        tx = lats_mat * sol_dec_mat
        # result = np.copy(tx)
        # result[np.where((tx >= (-1 * one)) & (tx <= one))] = np.arccos(
        #    tx[np.where((tx >= (-1 * one)) & (tx <= one))]
        # )
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore", r"invalid value encountered in arccos"
            )
            result = np.arccos(np.copy(tx))

        result[np.where(tx < (-1 * one))] = pi
        result[np.where(tx > one)] = zero
        return result

    @staticmethod
    def func3(
        v: np.ndarray,
        w: np.ndarray,
        x: np.ndarray,
        y: np.ndarray,
    ) -> np.ndarray:
        """Potential solar radiation on the surface cal/cm2/day

        Radian angle version of FUNC3 (eqn 6) from Swift, 1976
        or Lee, 1963 equation 5. (Names in parens below from Swift?)

        Args:
            v: (L2) latitude angle hour offset between actual and equivalent
               slope [nhru]
            w: (L1) latitude of the equivalent slope [nhru]
            x: (T3) hour angle of sunset on equivalent slope [ndoy, nhru]
            y: (T2) hour angle of sunrise on equivalent slope [ndoy, nhru]

        Returns:
            (R4) is potential solar radiation on the surface cal/cm2/day
            [ndoy, nhru]

        Constants
            r1: solar constant for 60 minutes [ndoy]
            solar_declination: declination of sun [ndoy]

        See also: https://github.com/nhm-usgs/prms/blob/6.0.0_dev/src/prmslib/physics/sm_solar_radiation.f90
        """

        # Must alter the dimensions of the inputs to get the outputs.
        nhru = len(v)
        vv = np.tile(v, (ndoy, 1))
        ww = np.tile(w, (ndoy, 1))
        # These are known at init time, not sure they are worth saving in self
        # and passing
        rr = np.transpose(np.tile(r1, (nhru, 1)))
        dd = np.transpose(np.tile(solar_declination, (nhru, 1)))

        f3 = (
            rr
            * pi_12
            * (
                np.sin(dd) * np.sin(ww) * (x - y)
                + np.cos(dd) * np.cos(ww) * (np.sin(x + vv) - np.sin(y + vv))
            )
        )

        return f3

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
                    self.doy,
                    time_coord="doy",
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
                    self.doy,
                    time_coord="doy",
                )

            nc.close()
            assert nc_path.exists()
            print(f"Wrote file: {nc_path}")

        self._finalize_netcdf()
        return

    def _finalize_netcdf(self) -> None:
        self._netcdf_initialized = False
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
            warnings.warn(msg)
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

    def output(self):
        if self._netcdf_initialized:
            if self._verbose:
                print(f"writing FULL timeseries output for: {self.name}")
            self._write_netcdf_timeseries()
        return
