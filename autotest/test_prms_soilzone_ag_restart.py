"""Restart tests for PRMSSoilzoneAg.

This test module validates the restart capability of PRMSSoilzoneAg by
confirming that a continuous run produces identical results to a run that
was restarted from a saved state.

Test Domains
------------
The shipped/CI domain is **fgr_ag_2yr** (populated
into test_data/ from the oracle by test_data/generate/oracle_populate.py;
this module skips all other domains), with two configurations:

1. **spinup.control** (open-loop): Tests restart with basic dual-area
   soil moisture accounting without iterative AET matching.

2. **analysis.control** (closed-loop): Tests restart with full iterative
   AET matching with observed AET data.

(The tests were developed on the original larger UCB domains,
ucb_ag_spinup_2yr / ucb_ag_analysis_2yr -- too large for CI; may
return.)

Test Strategy
-------------
Perfect restart test:
- Run 1 "ac": Starts at time a, writes restart at time b, ends at time c
- Run 2 "bc": Starts at time b (reading restart), ends at time c'
- Confirm all variables match: c == c' (bit-for-bit)

"""

import pathlib as pl
from typing import Any, Optional

import numpy as np
import pytest

from pywatershed.base.adapter import adapter_factory
from pywatershed.base.control import Control
from pywatershed.base.timeseries import TimeseriesArray
from pywatershed.hydrology.prms_soilzone_ag import PRMSSoilzoneAg
from pywatershed.hydrology.prms_soilzone_ag_obs_et import PRMSSoilzoneAgObsET
from pywatershed.parameters import Parameters, PrmsParameters

dt_1d = np.timedelta64(24, "h")

# Restart frequency options and corresponding init times
# The "a", "b", "c" times for each restart frequency
# Restarts with "y" and "m" are always written on the last day of the period.
# The "f" option is tested separately since it writes restarts only at the end.

# For analysis domain (starts 2000-01-01)
init_times_dict: dict[str, dict[str, np.datetime64]] = {
    "d": {
        "a": np.datetime64("2000-12-20") - dt_1d,
        "b": np.datetime64("2000-12-25") - dt_1d,
        "c": np.datetime64("2000-12-31") - dt_1d,
    },
    "m": {
        "a": np.datetime64("2000-10-31") - dt_1d,
        "b": np.datetime64("2000-11-01") - dt_1d,
        "c": np.datetime64("2000-12-01") - dt_1d,
    },
    "y": {
        "a": np.datetime64("2000-12-31") - dt_1d,
        "b": np.datetime64("2001-01-01") - dt_1d,
        "c": np.datetime64("2001-12-31") - dt_1d,
    },
}

restart_freqs: list[str] = list(init_times_dict.keys())


def get_init_times_dict(
    simulation: dict[str, Any],
) -> dict[str, dict[str, np.datetime64]]:
    """Get the appropriate init times dict based on domain."""
    domain_name = simulation["name"].split(":")[0]
    if "fgr_ag_2yr" not in domain_name:
        pytest.skip(f"Only testing for fgr_ag_2yr domain, not {domain_name}")

    return init_times_dict


def get_control(
    simulation: dict[str, Any],
    init_time: Optional[np.datetime64] = None,
    end_time: Optional[np.datetime64] = None,
) -> Control:
    """Load control and optionally adjust time bounds."""
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        control = Control.load_prms(
            simulation["control_file"],
            warn_unused_options=False,
            keep_unused_options=True,
        )

    # Check this is a GSFLOW domain with soilzone_ag
    if "executable_desc" in control.options.keys():
        exe_desc = control.options["executable_desc"][0].lower()
    else:
        exe_desc = "prms"

    if "gsflow" not in exe_desc:
        pytest.skip(
            "Only testing PRMSSoilzoneAg restart for domains run with GSFLOW."
        )

    # Check soilzone_module is soilzone_ag
    soilzone_module = control.options.get("soilzone_module", [None])[0]
    if soilzone_module != "soilzone_ag":
        pytest.skip(
            f"Only testing PRMSSoilzoneAg restart for soilzone_ag module, "
            f"got '{soilzone_module}'."
        )

    # Check AET_cbh_file based on spinup vs analysis mode
    domain_name = simulation["name"].split(":")[0]
    aet_cbh_file = control.options.get("AET_cbh_file", [None])[0]

    if "spinup" in domain_name:
        if aet_cbh_file is not None:
            pytest.skip("Spinup mode should not have AET_cbh_file set.")
    elif "analysis" in domain_name:
        if aet_cbh_file is None:
            pytest.skip("Analysis mode should have AET_cbh_file set.")

    if init_time is not None:
        control.edit_init_start_times(init_time)
    if end_time is not None:
        control.edit_end_time(end_time)

    return control


@pytest.fixture(scope="function")
def SoilzoneAg(simulation):
    """Select appropriate SoilzoneAg class based on control file settings."""
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ctl = Control.load_prms(
            simulation["control_file"],
            warn_unused_options=False,
            keep_unused_options=True,
        )

    if "executable_desc" in ctl.options.keys():
        exe_desc = ctl.options["executable_desc"][0].lower()
    else:
        exe_desc = "prms"

    if "gsflow" not in exe_desc:
        pytest.skip(
            "Only testing PRMSSoilzoneAg for domains run with a GSFLOW exe."
        )

    # Choose class based on iter_aet_flag
    iter_aet_flag = ctl.options.get("iter_aet_flag", None)
    if iter_aet_flag:
        return PRMSSoilzoneAgObsET
    else:
        return PRMSSoilzoneAg


@pytest.fixture(scope="function")
def discretization(simulation: dict[str, Any]) -> Parameters:
    dis_hru_file = simulation["dir"] / "parameters_dis_hru.nc"
    return Parameters.from_netcdf(dis_hru_file, encoding=False)


@pytest.fixture(scope="function")
def parameters(simulation: dict[str, Any]) -> PrmsParameters:
    control = get_control(simulation)
    param_file = simulation["dir"] / control.options["parameter_file"]
    return PrmsParameters.load(param_file)


def get_input_variables(
    simulation: dict[str, Any],
    control: Control,
    parameters: PrmsParameters,
    SoilzoneAg,
) -> dict[str, Any]:
    """Build input variables dict for PRMSSoilzoneAg."""
    output_dir = simulation["output_dir"]

    input_variables: dict[str, Any] = {}
    for key in SoilzoneAg.get_inputs():
        nc_path = output_dir / f"{key}.nc"

        if not nc_path.exists():
            if key == "AET_external":
                # Use AET file from control options if set, otherwise zeros
                aet_cbh_file = control.options.get("AET_cbh_file", [None])[0]
                if aet_cbh_file is not None:
                    nc_path = simulation["dir"] / pl.Path(
                        aet_cbh_file
                    ).with_suffix(".nc")
                else:
                    nc_path = adapter_factory(
                        np.zeros(parameters.dims["nhru"]),
                        key,
                        control,
                    )
            elif key == "ag_frac":
                # Check for dynamic parameter file first
                dyn_ag_frac_file = simulation["dir"] / "dyn_ag_frac.param"
                if dyn_ag_frac_file.exists():
                    nc_path = dyn_ag_frac_file
                else:
                    nc_path = adapter_factory(
                        parameters.parameters[key].copy(),
                        key,
                        control,
                    )
            else:
                nc_path = None

        input_variables[key] = nc_path

    return input_variables


@pytest.mark.parametrize("restart_freq", restart_freqs)
def test_restart(
    simulation: dict[str, Any],
    discretization: Parameters,
    parameters: PrmsParameters,
    SoilzoneAg,
    tmp_path: pl.Path,
    restart_freq: str,
) -> None:
    """Perfect restart test for PRMSSoilzoneAg.

    Test outline/goals:
    run 1, "ac": a -----> c starts at a, restart written at time b, ends at c
                     |
    run 2, "bc":     b -> c'
    confirm c == c' in all variables (bit-for-bit match).
    """
    init_times_dict = get_init_times_dict(simulation)
    times = init_times_dict[restart_freq]

    restart_dir = tmp_path / "restarts"

    # Run ac: continuous run from a to c, writing restart at b
    control_ac = get_control(simulation, times["a"], times["c"])
    input_variables = get_input_variables(
        simulation, control_ac, parameters, SoilzoneAg
    )

    run_args: dict[str, Any] = {
        "control": control_ac,
        "discretization": discretization,
        "parameters": parameters,
        **input_variables,
    }
    run_args["restart_write"] = restart_dir
    run_args["restart_write_freq"] = restart_freq

    proc_ac = SoilzoneAg(**run_args)

    for istep in range(control_ac.n_times):
        control_ac.advance()
        proc_ac.advance()
        proc_ac.calculate(float(istep))
        proc_ac.output()

    proc_ac.finalize()

    # Run bc: restart from b to c
    control_bc = get_control(simulation, times["b"], times["c"])
    input_variables = get_input_variables(
        simulation, control_bc, parameters, SoilzoneAg
    )

    run_args = {
        "control": control_bc,
        "discretization": discretization,
        "parameters": parameters,
        **input_variables,
    }
    run_args["restart_read"] = restart_dir

    proc_bc = SoilzoneAg(**run_args)

    for istep in range(control_bc.n_times):
        control_bc.advance()
        proc_bc.advance()
        proc_bc.calculate(float(istep))

    proc_bc.finalize()

    # Verify both runs ended at the same time
    assert control_ac.current_time == control_bc.current_time, (
        f"End times don't match: {control_ac.current_time} vs "
        f"{control_bc.current_time}"
    )

    # Compare all variables - should be bit-for-bit identical
    for vv in proc_ac.variables:
        ac_result = proc_ac[vv]
        bc_result = proc_bc[vv]

        if isinstance(ac_result, TimeseriesArray):
            ac_result = ac_result.current
            bc_result = bc_result.current

        np.testing.assert_equal(
            ac_result,
            bc_result,
            err_msg=f"Variable {vv} differs between continuous and "
            f"restarted runs at time {control_bc.current_time}",
        )


def test_restart_f(
    simulation: dict[str, Any],
    discretization: Parameters,
    parameters: PrmsParameters,
    SoilzoneAg,
    tmp_path: pl.Path,
) -> None:
    """Test the "f" restart frequency option for PRMSSoilzoneAg.

    The "f" option writes restarts only at the end of a run, so it requires
    a different test structure:
    run 1, "ac": a ------> c starts at a, ends at c
    run 2, "ab": a -> b'
                      | restart files written at end
    run 3, "bc":      b -> c'
    confirm c == c' in all variables.
    """
    # Use fixed times that work for both domains
    domain_name = simulation["name"].split(":")[0]
    if "fgr_ag_2yr" in domain_name:
        init_times = {
            "a": np.datetime64("2000-06-01"),
            "b": np.datetime64("2000-06-15"),
            "c": np.datetime64("2000-06-30"),
        }
    else:
        pytest.skip(f"Only testing for fgr_ag_2yr domain, not {domain_name}")

    restart_dir = tmp_path / "restarts"

    def run_init_end(
        init_time: np.datetime64,
        end_time: np.datetime64,
        restart_write: bool = False,
        restart_read: bool = False,
    ):
        """Helper to run PRMSSoilzoneAg for a time period."""
        control = get_control(simulation, init_time, end_time)
        input_variables = get_input_variables(
            simulation, control, parameters, SoilzoneAg
        )

        run_args: dict[str, Any] = {
            "control": control,
            "discretization": discretization,
            "parameters": parameters,
            **input_variables,
        }

        if restart_write:
            run_args["restart_write"] = restart_dir
            run_args["restart_write_freq"] = "f"

        if restart_read:
            run_args["restart_read"] = restart_dir

        proc = SoilzoneAg(**run_args)

        for istep in range(control.n_times):
            control.advance()
            proc.advance()
            proc.calculate(float(istep))
            proc.output()

        proc.finalize()
        return proc

    # Run ac: continuous from a to c
    proc_ac = run_init_end(init_times["a"], init_times["c"])

    # Run ab: from a to b, write restart at end
    _ = run_init_end(init_times["a"], init_times["b"], restart_write=True)

    # Run bc: from b to c, reading restart
    proc_bc = run_init_end(init_times["b"], init_times["c"], restart_read=True)

    # Verify both runs ended at the same time
    assert proc_ac.control.current_time == proc_bc.control.current_time, (
        f"End times don't match: {proc_ac.control.current_time} vs "
        f"{proc_bc.control.current_time}"
    )

    # Compare all variables - should be bit-for-bit identical
    for vv in proc_ac.variables:
        ac_result = proc_ac[vv]
        bc_result = proc_bc[vv]

        if isinstance(ac_result, TimeseriesArray):
            ac_result = ac_result.current
            bc_result = bc_result.current

        np.testing.assert_equal(
            ac_result,
            bc_result,
            err_msg=f"Variable {vv} differs between continuous and "
            f"restarted runs at time {proc_bc.control.current_time}",
        )
