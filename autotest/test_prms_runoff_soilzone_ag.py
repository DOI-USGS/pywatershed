"""Tests for PRMSSoilzoneAg against PRMS/GSFLOW Fortran outputs.

This test module validates the PRMSSoilzoneAg implementation by comparing
Python outputs against pre-computed Fortran (PRMS/GSFLOW) outputs for
agricultural soilzone simulations.

Test Domains
------------
The shipped/CI domain is **fgr_ag_2yr** (populated
into test_data/ from the oracle by test_data/generate/oracle_populate.py),
with two configurations:

1. **spinup.control** (open-loop): Tests basic dual-area soil moisture
   accounting without iterative AET matching (iter_aet_flag=False,
   PRMSSoilzoneAg).

2. **analysis.control** (closed-loop): Tests full iterative AET matching
   with observed AET data (iter_aet_flag=True, PRMSSoilzoneAgObsET).

Historical/offline: **ucb_ag_spinup_2yr** / **ucb_ag_analysis_2yr**, the
original larger UCB domains these tests were developed on (too large for
CI; may return). The HRU-time exceptions in ag_exceptions.py remain
keyed to them.

Tolerance Strategy
------------------
- **Default tolerances**: rtol=1e-5, atol=1e-5 for most variables
- **Variable-specific exceptions**: Some variables accumulate floating-point
  errors over time or involve precision-sensitive calculations. These have
  relaxed tolerances defined in `var_tolerance_exceptions`.
- **HRU-time exceptions**: Specific HRU-timestep combinations where Python
  and Fortran diverge due to threshold crossings (e.g., pcts < 0.5 for LOAM
  soil, snow_free < 0.01 for ET type). At these points, Python values are
  replaced with Fortran values to allow the test to continue. These live in
  ag_exceptions.py (shared with test_prms_soilzone_ag.py), which validates
  them at collection time and warns when any apply.

"""

import shutil

import numpy as np
import pytest
import xarray as xr
from ag_exceptions import get_hru_time_exceptions
from utils_compare import compare_in_memory, compare_netcdfs

from pywatershed.base.adapter import adapter_factory
from pywatershed.base.control import Control
from pywatershed.base.model import Model
from pywatershed.hydrology.prms_runoff_ag import PRMSRunoffAg
from pywatershed.hydrology.prms_soilzone_ag import PRMSSoilzoneAg
from pywatershed.hydrology.prms_soilzone_ag_obs_et import PRMSSoilzoneAgObsET

# from pywatershed.hydrology.prms_soilzone_no_dprst import PRMSSoilzoneNoDprst
from pywatershed.parameters import Parameters, PrmsParameters

# compare in memory (faster) or full output files? or both!
do_compare_output_files = True
do_compare_in_memory = True
invoke_styles = ("legacy", "pws")
calc_methods = ("numpy", "numba")
imbalance_behavior = "error"
# Default tolerances for most variables (depth-based)
default_rtol = 1.0e-5
default_atol = 1.0e-5

# Variable-specific tolerance exceptions
# ssres_flow_vol: the fortran output of ssres_flow is only single precision,
#     errors at that precision multiplied by hru_area can give larger relative
#     errors while the absolute errors are still near precision.
var_tolerance_exceptions = {
    # Runoff variable exceptions:
    "sroff_vol": {"atol": 1.0e-4, "rtol": 5.0e-5},
    "dprst_vol_open": {"atol": 3.0e-4, "rtol": 3.0e-4},
    # Soilzone variable exceptions
    "ssres_flow_vol": {"atol": 2.0, "rtol": 1.0e-2},  # cubic feet
    # ratio capping: small differences when soil_lower is near soil_lower_max
    "soil_lower_ratio": {"atol": 1.0e-4, "rtol": 1.0e-5},
    # Flow variables accumulate single-precision errors over time
    "slow_flow": {"atol": 2.0e-5, "rtol": 1.0e-5},
    "ssres_flow": {"atol": 2.0e-5, "rtol": 1.0e-5},
    # Storage variables accumulate errors over many timesteps
    "slow_stor": {"atol": 1.0e-4, "rtol": 1.0e-5},
    "slow_stor_prev": {"atol": 1.0e-4, "rtol": 1.0e-5},
    "ssres_stor": {"atol": 1.0e-4, "rtol": 1.0e-5},
    # Total soil moisture accumulates errors from multiple components
    "soil_moist_tot": {"atol": 1.0e-4, "rtol": 1.0e-5},
    # Flow/change variables with borderline precision issues
    "recharge": {"atol": 2.0e-5, "rtol": 1.0e-5},
    "slow_stor_change": {"atol": 2.0e-5, "rtol": 1.0e-5},
    "ssr_to_gw": {"atol": 2.0e-5, "rtol": 1.0e-5},
    "soil_lower_change": {"atol": 2.0e-5, "rtol": 1.0e-5},
    "soil_to_gw": {"atol": 2.0e-5, "rtol": 1.0e-5},
    # perv_soil_to_gw accumulates precision differences over time
    "perv_soil_to_gw": {"atol": 2.0e-5, "rtol": 1.0e-5},
    # Agricultural soil moisture change accumulates precision differences
    "ag_soil_moist_change": {"atol": 2.0e-5, "rtol": 1.0e-5},
}

# Domain-specific HRU-time exceptions for threshold-crossing divergences
# live in ag_exceptions.py (single source shared with
# test_prms_soilzone_ag.py; validated at collection time).


def mk_ag_frac_input(domain_dir, full_param_file, input_dir):
    """Not a fixture"""
    # deal with the ag frac input which may option be supplied to PRMS
    # but not pywatershed
    param_obj = PrmsParameters.load(full_param_file)
    ag_frac = xr.load_dataarray(domain_dir / "tmin.nc")
    ag_frac[:, :] = param_obj.parameters["ag_frac"]
    ag_frac.name = "ag_frac"
    ag_frac.to_netcdf(input_dir / "ag_frac.nc")


def setup_input_dir(simulation, control, param_obj, process_list):
    opts = control.options
    domain_dir = simulation["dir"]
    input_dir = opts["input_dir"]
    ag_frac_dyn_flag = opts.get("dyn_ag_frac_flag", [False])[0]
    if not ag_frac_dyn_flag:
        mk_ag_frac_input(
            domain_dir,
            simulation["control_file"].parent
            / control.options["parameter_file"],
            input_dir,
        )
    output_dir = simulation["output_dir"]
    input_list = [ii for proc in process_list for ii in proc.get_inputs()]
    for ii in input_list:
        fname = f"{ii}.nc"
        out_path = output_dir / fname
        in_path = input_dir / fname
        if ii == "ag_frac":
            if ag_frac_dyn_flag:
                fname = opts["ag_frac_dynamic"][0]
                out_path = output_dir / f"../{fname}"
                in_path = input_dir / "ag_frac.param"
            else:
                continue
        elif ii == "aet_observed":
            out_path = output_dir / f"../{fname}"

        # <
        shutil.copyfile(out_path, in_path)  # not copy: answers are read-only


@pytest.fixture(scope="function", params=invoke_styles)
def invoke_style(request):
    return request.param


@pytest.fixture(scope="function", params=calc_methods)
def calc_method(request):
    return request.param


@pytest.fixture(scope="function")
def control(simulation, tmp_path, calc_method):
    control = Control.load_prms(
        simulation["control_file"], warn_unused_options=False
    )
    control.options["calc_method"] = calc_method
    control.options["imbalance_behavior"] = imbalance_behavior
    # because we have to write ag_frac input, we need a custom
    # input dir
    control.options["input_dir"] = tmp_path / "input"
    control.options["input_dir"].mkdir()
    if do_compare_output_files:
        # populate the input dir later when the processes are known
        control.options["netcdf_output_dir"] = tmp_path / "run_output"
        control.options["netcdf_output_dir"].mkdir()

    control.options["intcp_changeover_in_net_rain"] = (
        "gsflow" in control.options.get("executable_desc", ["PRMS"])[0].lower()
    )

    return control


@pytest.fixture(scope="function")
def SoilzoneAg(simulation):
    import warnings

    with warnings.catch_warnings():
        # This is the only way to silence "invalid" options.
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

    if "dprst_flag" in ctl.options.keys() and ctl.options["dprst_flag"]:
        # Choose class based on iter_aet_flag
        iter_aet_flag = ctl.options.get("iter_aet_flag", None)
        if iter_aet_flag:
            SoilzoneAg = PRMSSoilzoneAgObsET
        else:
            SoilzoneAg = PRMSSoilzoneAg
    else:
        pytest.skip("Not testing PRMSSoilzoneNoDprstAg")
        # SoilzoneAg = PRMSSoilzoneNoDprstAg

    return SoilzoneAg


@pytest.fixture(scope="function")
def discretization(simulation):
    dis_hru_file = simulation["dir"] / "parameters_dis_hru.nc"
    return Parameters.from_netcdf(dis_hru_file, encoding=False)


@pytest.fixture(scope="function")
def parameters_runoff(simulation, control, invoke_style):
    if invoke_style == "legacy":
        param_file = simulation["dir"] / control.options["parameter_file"]
        params = PrmsParameters.load(param_file)

    else:
        # Load runoff parameters
        param_file = simulation["dir"] / "parameters_PRMSRunoffAg.nc"
        params = PrmsParameters.from_netcdf(param_file)

    if abs(params.parameters["sat_threshold"]).min() < 999.0:
        pytest.skip(
            "test_prms_runoff_ag only valid when sat_threshold >= 999 "
            "(or some amount) which causes zero dunnian_flow"
        )

    return params


@pytest.fixture(scope="function")
def parameters_soilzone(simulation, control, invoke_style):
    if invoke_style == "legacy":
        param_file = simulation["dir"] / control.options["parameter_file"]
        params = PrmsParameters.load(param_file)
    else:
        param_file = simulation["dir"] / "parameters_PRMSSoilzoneAg.nc"
        params = PrmsParameters.from_netcdf(param_file)

    return params


@pytest.fixture(scope="function")
def model_args_init_legacy(simulation, control, SoilzoneAg):
    process_list = [PRMSRunoffAg, SoilzoneAg]
    param_file = simulation["dir"] / control.options["parameter_file"]
    parameters = PrmsParameters.load(param_file)
    args = {
        "process_list_or_model_dict": process_list,
        "control": control,
        "parameters": parameters,
    }
    setup_input_dir(simulation, control, parameters, process_list)
    return args


@pytest.fixture(scope="function")
def model_args_init_pws(
    simulation,
    control,
    discretization,
    parameters_runoff,
    parameters_soilzone,
    SoilzoneAg,
):
    process_list = [PRMSRunoffAg, SoilzoneAg]
    process_dict = {pp.__name__.lower(): pp for pp in process_list}
    model_dict = {
        "control": control,
        "dis_hru": discretization,
        "model_order": list(process_dict.keys()),
    }
    for proc_name, proc in process_dict.items():
        model_dict[proc_name] = {}
        proc_args = model_dict[proc_name]
        proc_args["class"] = proc
        if "runoff" in proc_name:
            params = parameters_runoff
        elif "soilzone" in proc_name:
            params = parameters_soilzone
        else:
            raise ValueError(f"Unknown process name: {proc_name}")
        # <
        proc_args["parameters"] = params  # PrmsParameters.from_netcdf(params)
        proc_args["dis"] = "dis_hru"

    args = {
        "process_list_or_model_dict": model_dict,
        "control": None,
        "parameters": None,
    }

    setup_input_dir(simulation, control, parameters_soilzone, process_list)
    return args


@pytest.fixture(scope="function")
def model(model_args_init_legacy, model_args_init_pws, invoke_style):
    if invoke_style == "legacy":
        return Model(**model_args_init_legacy)
    elif invoke_style == "pws":
        return Model(**model_args_init_pws)
    else:
        raise ValueError(f"Unknown invoke style: {invoke_style}")


def test_compare_prms(
    simulation,
    control,
    model,
    calc_method,
):
    process_list = list(model.processes.values())
    all_vars_set = set(
        [ii for proc in process_list for ii in proc.get_variables()]
    )

    # sroff is a runoff variable is edited by soilzone but the forcings are
    # from the output of soilzone, so checking it is kind of a tautology
    change_vars = {
        "ag_soil_moist_change",
        "ag_soil_rechr_change",
        "slow_stor_change",
        "soil_lower_change",
        "soil_rechr_change",
    }
    # These are not prms variables per se.
    # The _hru ones have non-hru equivalents being checked.
    # soil_zone_max and soil_lower_max would be nice to check but
    # prms5.2.1 wont write them as hru variables.
    not_prms_vars = {
        "perv_actet_hru",
        "soil_lower_change_hru",
        "soil_lower_max",
        "soil_rechr_change_hru",
        "soil_zone_max",  # not a prms variable?
    }
    other_vars_excl = {
        "dprst_vol_thres_open",  # not output by fortran nor post-processed
        "infil_ag_hru",  # currently not post-processed but infil_ag is
        "hru_sroff_ag",  # PRMS does can not write this variable
        "sroff_vol",  # errors for large HRUs, rely on sroff
        "intcp_changeover_budget",  # not a PRMS/GSFLOW variable
    }

    comparison_var_names = (
        all_vars_set - change_vars - not_prms_vars - other_vars_excl
    )

    control.options["netcdf_output_var_names"] = comparison_var_names

    # TODO: this is hacky, improve the design
    if (
        "dprst_flag" not in control.options.keys()
        or not control.options["dprst_flag"]
    ):
        comparison_var_names = {
            vv for vv in comparison_var_names if "dprst" not in vv
        }

    # Load answers for comparison and HRU-time exceptions
    prms_output_dir = simulation["output_dir"]
    answers = {}
    skipped_comp_vars = []
    for var in comparison_var_names:
        var_pth = prms_output_dir / f"{var}.nc"
        if not var_pth.exists():
            skipped_comp_vars += [var]
            continue
        # <
        answers[var] = adapter_factory(
            var_pth, variable_name=var, control=control
        )

    # <
    if len(skipped_comp_vars) > 0:
        print(f"Skipped comparison for variables: {skipped_comp_vars}")

    hru_time_exceptions = get_hru_time_exceptions(simulation)

    for istep in range(control.n_times):
        # control.advance()
        model.advance()

        # Because ag_mask is an input, this has to be after the first adva  nce
        if istep == 0 and do_compare_in_memory:
            first_proc = next(iter(model.processes.values()))
            ag_frac = first_proc["ag_frac"]
            ag_mask = np.where(ag_frac > 0.0)
            not_ag_mask = np.where(ag_frac <= 0.0)
            mask_dict = {}
            for vv in answers.keys():
                if "ag_" in vv:
                    mask_dict[vv] = ag_mask
                if vv in [
                    "soil_moist",
                    "soil_rechr",
                    "soil_lower",
                    "soil_moist_tot",
                    "soil_rechr_change",
                    "soil_lower_change",
                    "perv_actet",
                    "potet_rechr",
                    "potet_lower",
                    "cap_infil_tot",
                    "cap_waterin",
                ]:
                    mask_dict[vv] = not_ag_mask
                else:
                    mask_dict[vv] = None

        # Advance answers to current timestep
        for var in answers.values():
            var.advance()

        model.calculate()

        # Apply HRU-time exceptions: replace Python values with Fortran values
        # for HRUs that have diverged due to threshold crossings
        # This is done after calculate() but before output() so that both
        # in-memory and file-based comparisons see the corrected values
        for hru_idx, (
            start_time,
            affected_vars,
            reason,
        ) in hru_time_exceptions.items():
            if istep >= start_time:
                for var in affected_vars:
                    if var in answers:
                        for proc_name, proc in model.processes.items():
                            if var not in proc.variables:
                                continue
                            fortran_val = answers[var].current.data[hru_idx]
                            if isinstance(proc[var], np.ndarray):
                                proc[var][hru_idx] = fortran_val
                            else:
                                # Handle TimeseriesArray
                                proc[var].current[hru_idx] = fortran_val

        model.output()

        if do_compare_in_memory:
            # Build variable-specific tolerances: default for all,
            # then apply exceptions
            var_tolerances = {
                var: {"rtol": default_rtol, "atol": default_atol}
                for var in answers.keys()
            }
            for var, tols in var_tolerance_exceptions.items():
                if var in var_tolerances:
                    var_tolerances[var] = tols

            for proc_name, proc in model.processes.items():
                compare_in_memory(
                    proc,
                    answers,
                    mask_dict=mask_dict,
                    atol=default_atol,
                    rtol=default_rtol,
                    var_tolerances=var_tolerances,
                    skip_missing_ans=True,
                    fail_after_all_vars=True,
                    verbose=False,
                )

    model.finalize()

    if do_compare_output_files:
        # Filter out variables without answer files
        vars_with_answers = []
        for var in comparison_var_names:
            var_pth = simulation["output_dir"] / f"{var}.nc"
            if var_pth.exists():
                vars_with_answers.append(var)

        # Build variable-specific tolerances
        var_tolerances = {
            var: {"rtol": default_rtol, "atol": default_atol}
            for var in vars_with_answers
        }
        for var, tols in var_tolerance_exceptions.items():
            if var in var_tolerances:
                var_tolerances[var] = tols

        compare_netcdfs(
            vars_with_answers,
            control.options["netcdf_output_dir"],
            simulation["output_dir"],
            rtol=default_rtol,
            atol=default_atol,
            var_tolerances=var_tolerances,
            # fail_after_all_vars=False,
            verbose=True,
        )

    return
