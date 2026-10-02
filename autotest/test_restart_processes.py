"""Perfect-restart tests for individual PRMS process classes.

Each process is run in isolation, forced by the PRMS output in the
simulation's output directory. Which classes run is decided from the
control file: PRMSSolarGeometry, PRMSAtmosphere, PRMSCanopy and PRMSSnow
always; runoff/soilzone/groundwater as the plain, NoDprst or CascadesNoDprst
variants from ``dprst_flag``, ``cascade_flag`` and ``cascadegw_flag``;
PRMSChannel for the muskingum ``streamflow_module`` values (``strmflow`` has
no routing and ``strmflow_in_out`` has no channel parameters). Skipped for
GSFLOW/agricultural domains, which have their own restart tests
(test_prms_runoff_ag_restart.py, test_prms_soilzone_ag_restart.py).

Test strategy:
- Run 1 "ac": starts at a, writes restart at b, ends at c
- Run 2 "bc": starts at b reading the restart, ends at c
- All variables match bit-for-bit at c.

The "f" (final) restart frequency needs a third run since it only writes
at the end: "ac" continuous, "ab" writing at b, "bc" reading at b.

The a/b/c dates all restart across the first Dec 31 that follows the
control's start_time (see get_init_times), so any domain spanning at least
Dec 31 of the next year works; "d"/"m"/"y" check daily/monthly/yearly
writes.

Future work:
- Fold the two GSFLOW/agricultural restart tests into this module.
- A module of full-model (multi-process) restart tests; today every
  restart test is process-level.
"""

import pathlib as pl
from typing import Any

import numpy as np
import pytest
from conftest import collect_simulations

import pywatershed as pws
from pywatershed.base.control import Control
from pywatershed.base.timeseries import TimeseriesArray
from pywatershed.parameters import Parameters, PrmsParameters

dt_1d = np.timedelta64(24, "h")

imbalance_behavior = "error"

restart_freqs = ["d", "m", "y"]


def load_control(control_file: pl.Path) -> Control:
    return Control.load_prms(control_file, warn_unused_options=False)


def get_processes(control: Control) -> list:
    """The process classes this control file selects (see module doc)."""
    opts = control.options
    dprst = bool(opts.get("dprst_flag", False))
    cascades = bool(opts.get("cascade_flag", False))
    gw_cascades = bool(opts.get("cascadegw_flag", False))

    processes = [
        pws.PRMSSolarGeometry,
        pws.PRMSAtmosphere,
        pws.PRMSCanopy,
        pws.PRMSSnow,
    ]
    if dprst and cascades:
        raise NotImplementedError("No dprst-with-cascades classes exist")
    elif dprst:
        processes += [pws.PRMSRunoff, pws.PRMSSoilzone, pws.PRMSGroundwater]
    elif cascades:
        processes += [
            pws.PRMSRunoffCascadesNoDprst,
            pws.PRMSSoilzoneCascadesNoDprst,
            (
                pws.PRMSGroundwaterCascadesNoDprst
                if gw_cascades
                else pws.PRMSGroundwaterNoDprst
            ),
        ]
    else:
        processes += [
            pws.PRMSRunoffNoDprst,
            pws.PRMSSoilzoneNoDprst,
            pws.PRMSGroundwaterNoDprst,
        ]

    if opts["streamflow_module"] in ("muskingum", "muskingum_mann"):
        processes += [pws.PRMSChannel]

    return processes


def pytest_generate_tests(metafunc):
    # The process list depends on the control file, so (simulation, Process)
    # pairs are parametrized together here instead of taking the simulation
    # fixture from conftest.
    if "sim_process" not in metafunc.fixturenames:
        return
    if "domainless" == metafunc.config.option.markexpr:
        return

    domain_list = metafunc.config.getoption("domain")
    if not len(domain_list):
        raise ValueError("test_restart_processes requires --domain")
    control_pattern_list = metafunc.config.getoption("control_pattern")
    simulations = collect_simulations(domain_list, control_pattern_list)

    pairs, ids = [], []
    for name, sim in simulations.items():
        for Process in get_processes(load_control(sim["control_file"])):
            pairs.append((sim, Process))
            ids.append(f"{name}-{Process.__name__}")

    metafunc.parametrize("sim_process", pairs, ids=ids)


def get_control(
    simulation: dict[str, Any],
    init_time: np.datetime64 | None = None,
    end_time: np.datetime64 | None = None,
) -> Control:
    control = load_control(simulation["control_file"])
    opts = control.options

    exe_desc = opts.get("executable_desc", ["prms"])[0].lower()
    if "gsflow" in exe_desc:
        pytest.skip("GSFLOW/ag domains have their own restart tests")

    control.options["imbalance_behavior"] = imbalance_behavior

    if init_time is not None:
        control.edit_init_start_times(init_time)
    if end_time is not None:
        control.edit_end_time(end_time)

    return control


def get_init_times(control: Control) -> dict[str, dict[str, np.datetime64]]:
    """The a/b/c init times for each restart frequency.

    Restarts with "y" and "m" are written on the last day of the period, so
    the "b" init time is that last day. Every case restarts across the
    first year boundary after start_time, on purpose: PRMS has year-based
    logic (e.g. in transpiration and snow) most likely to trip a restart
    there.
    """
    start = control.start_time
    year = start.astype("datetime64[Y]").astype(int) + 1970
    if start > np.datetime64(f"{year}-12-29"):
        year += 1
    y1 = year + 1

    def d64(s):
        return np.datetime64(s)

    times = {
        "d": {
            "a": d64(f"{year}-12-30"),
            "b": d64(f"{year}-12-31"),
            "c": d64(f"{y1}-01-01"),
        },
        "m": {
            "a": d64(f"{year}-12-30"),
            "b": d64(f"{year}-12-31"),
            "c": d64(f"{y1}-01-31"),
        },
        "y": {
            "a": d64(f"{year}-12-30"),
            "b": d64(f"{year}-12-31"),
            "c": d64(f"{y1}-12-30"),
        },
        "f": {
            "a": d64(f"{y1}-01-03"),
            "b": d64(f"{y1}-01-09"),
            "c": d64(f"{y1}-01-19"),
        },
    }
    for tt in times.values():
        assert tt["a"] >= control.init_time
        assert tt["c"] <= control.end_time
    return times


def get_input_variables(
    simulation: dict[str, Any], Process: type
) -> dict[str, pl.Path]:
    output_dir = simulation["output_dir"]
    input_variables = {}
    for kk in Process.get_inputs():
        nc_pth = output_dir / f"{kk}.nc"
        if not nc_pth.exists():
            # the "PRMS model" inputs (PRMSAtmosphere) are one level up
            nc_pth = output_dir.parent / f"{kk}.nc"
        input_variables[kk] = nc_pth
    return input_variables


def get_dis_params(
    simulation: dict[str, Any], control: Control, Process: type
) -> tuple[Parameters, PrmsParameters]:
    """Discretization and parameters for one process.

    The per-process parameter file (parameters_<ClassName>.nc, written by
    the domain's parameter separation) is used when present, else the
    PRMS parameter file: gridded sagehen's PRMS parameter file has no
    channel parameters. Segment discretization is merged in only for
    processes with an nsegment dimension.
    """
    dom_dir = simulation["dir"]
    discretization = Parameters.from_netcdf(
        dom_dir / "parameters_dis_hru.nc", encoding=False
    )
    dis_seg_file = dom_dir / "parameters_dis_seg.nc"
    if "nsegment" in Process.get_dimensions() and dis_seg_file.exists():
        discretization = Parameters.merge(
            discretization,
            Parameters.from_netcdf(dis_seg_file, encoding=False),
        )

    proc_param_file = dom_dir / f"parameters_{Process.__name__}.nc"
    if proc_param_file.exists():
        parameters = PrmsParameters.from_netcdf(proc_param_file)
    else:
        param_file = dom_dir / control.options["parameter_file"]
        parameters = PrmsParameters.load(param_file)

    return discretization, parameters


def run_process(
    Process: type,
    control: Control,
    simulation: dict[str, Any],
    restart_write: pl.Path | bool = False,
    restart_write_freq: str | bool = False,
    restart_read: pl.Path | bool = False,
):
    discretization, parameters = get_dis_params(simulation, control, Process)

    run_args: dict[str, Any] = {
        "control": control,
        "discretization": discretization,
        "parameters": parameters,
        **get_input_variables(simulation, Process),
    }
    if restart_write is not False:
        run_args["restart_write"] = restart_write
        run_args["restart_write_freq"] = restart_write_freq
    if restart_read is not False:
        run_args["restart_read"] = restart_read

    proc = Process(**run_args)
    for istep in range(control.n_times):
        control.advance()
        proc.advance()
        proc.calculate(float(istep))
        proc.output()

    proc.finalize()
    return proc


def assert_all_variables_equal(proc_ac, proc_bc, control_bc) -> None:
    for vv in proc_ac.variables:
        ac_result = proc_ac[vv]
        bc_result = proc_bc[vv]
        if isinstance(ac_result, TimeseriesArray):
            ac_result = ac_result.current
            bc_result = bc_result.current

        np.testing.assert_equal(
            ac_result,
            bc_result,
            err_msg=(
                f"Variable {vv} differs between continuous and restarted "
                f"runs at time {control_bc.current_time}"
            ),
        )


@pytest.mark.parametrize("restart_freq", restart_freqs)
def test_restart(sim_process, tmp_path: pl.Path, restart_freq: str) -> None:
    """Perfect restart test.

    run 1, "ac": a -----> c starts at a, restart written at time b, ends at c
                     |
    run 2, "bc":     b -> c'
    confirm c == c' in all variables (bit-for-bit match).
    """
    simulation, Process = sim_process
    times = get_init_times(get_control(simulation))[restart_freq]
    restart_dir = tmp_path / "restarts"

    control_ac = get_control(simulation, times["a"], times["c"])
    proc_ac = run_process(
        Process,
        control_ac,
        simulation,
        restart_write=restart_dir,
        restart_write_freq=restart_freq,
    )

    control_bc = get_control(simulation, times["b"], times["c"])
    proc_bc = run_process(
        Process, control_bc, simulation, restart_read=restart_dir
    )

    assert control_ac.current_time == control_bc.current_time
    assert_all_variables_equal(proc_ac, proc_bc, control_bc)


def test_restart_f(sim_process, tmp_path: pl.Path) -> None:
    """Test the "f" (final) restart frequency.

    run 1, "ac": a ------> c starts at a, ends at c
    run 2, "ab": a -> b'
                      | restart files written at end
    run 3, "bc":      b -> c'
    confirm c == c' in all variables (bit-for-bit match).
    """
    simulation, Process = sim_process
    times = get_init_times(get_control(simulation))["f"]
    restart_dir = tmp_path / "restarts"

    control_ac = get_control(simulation, times["a"], times["c"])
    proc_ac = run_process(Process, control_ac, simulation)

    control_ab = get_control(simulation, times["a"], times["b"])
    _ = run_process(
        Process,
        control_ab,
        simulation,
        restart_write=restart_dir,
        restart_write_freq="f",
    )

    control_bc = get_control(simulation, times["b"], times["c"])
    proc_bc = run_process(
        Process, control_bc, simulation, restart_read=restart_dir
    )

    assert control_ac.current_time == control_bc.current_time
    assert_all_variables_equal(proc_ac, proc_bc, control_bc)
