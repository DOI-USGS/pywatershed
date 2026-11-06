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
  ag_exceptions.py (shared with test_prms_runoff_soilzone_ag.py), which
  validates them at collection time and warns when any apply.

"""

import pathlib as pl

import numpy as np
import pytest
from ag_exceptions import get_hru_time_exceptions
from utils_compare import compare_in_memory, compare_netcdfs

from pywatershed.base.adapter import adapter_factory
from pywatershed.base.control import Control
from pywatershed.hydrology.prms_soilzone_ag import PRMSSoilzoneAg
from pywatershed.hydrology.prms_soilzone_ag_obs_et import PRMSSoilzoneAgObsET

# from pywatershed.hydrology.prms_soilzone_no_dprst import PRMSSoilzoneNoDprst
from pywatershed.parameters import Parameters, PrmsParameters

# compare in memory (faster) or full output files? or both!
do_compare_output_files = True
do_compare_in_memory = True

# Default tolerances for most variables (depth-based)
default_rtol = 1.0e-5
default_atol = 1.0e-5

# Variable-specific tolerance exceptions
# ssres_flow_vol: the fortran output of ssres_flow is only single precision,
#     errors at that precision multiplied by hru_area can give larger relative
#     errors while the absolute errors are still near precision.
var_tolerance_exceptions = {
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
# test_prms_runoff_soilzone_ag.py; validated at collection time).

calc_methods = ("numpy", "numba")
params = ("params_sep", "params_one")
imbalance_behavior = "error"


@pytest.fixture(scope="function")
def control(simulation):
    control = Control.load_prms(
        simulation["control_file"], warn_unused_options=False
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


@pytest.fixture(scope="function", params=params)
def parameters(simulation, control, request):
    if request.param == "params_one":
        param_file = simulation["dir"] / control.options["parameter_file"]
        params = PrmsParameters.load(param_file)
    else:
        param_file = simulation["dir"] / "parameters_PRMSSoilzoneAg.nc"
        params = PrmsParameters.from_netcdf(param_file)

    return params


@pytest.fixture(scope="function")
def parameters_pref(simulation, control, request):
    """Parameters fixture with preferential flow enabled."""
    param_file = simulation["dir"] / "parameters_PRMSSoilzoneAg.nc"
    params = PrmsParameters.from_netcdf(param_file).to_xr_ds()
    # Enable preferential flow by directly modifying the underlying array
    params["pref_flow_den"][:] = 0.07
    params["pref_flow_infil_frac"][:] = 0.15
    params = Parameters.from_ds(params)

    return params


@pytest.mark.parametrize("calc_method", calc_methods)
def test_compare_prms(
    simulation,
    control,
    discretization,
    parameters,
    SoilzoneAg,
    tmp_path,
    calc_method,
):
    tmp_path = pl.Path(tmp_path)

    # sroff is a runoff variable is edited by soilzone but the forcings are
    # from the output of soilzone, so checking it is kind of a tautology
    change_vars = {
        "ag_soil_moist_change",
        "ag_soil_rechr_change",
        "slow_stor_change",
        "soil_lower_change",
        "soil_rechr_change",
    }
    comparison_var_names = list(
        set(SoilzoneAg.get_variables())
        # These are not prms variables per se.
        # The _hru ones have non-hru equivalents being checked.
        # soil_zone_max and soil_lower_max would be nice to check but
        # prms5.2.1 wont write them as hru variables.
        - {
            "perv_actet_hru",
            "soil_lower_change_hru",
            "soil_lower_max",
            "soil_rechr_change_hru",
            "soil_zone_max",  # not a prms variable?
        }
        - change_vars
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

    output_dir = simulation["output_dir"]

    input_variables = {}
    for key in SoilzoneAg.get_inputs():
        if key in ["aet_observed"]:
            nc_path = output_dir.parent / f"{key}.nc"

        elif key == "ag_frac":
            opts = control.options
            ag_frac_dyn_flag = opts.get("dyn_ag_frac_flag", [False])[0]
            ag_frac_dyn_file = opts.get("ag_frac_dynamic", [None])[0]
            if not ag_frac_dyn_flag:
                import xarray as xr

                af_da = xr.load_dataarray(
                    output_dir.parent / "ag_frac_static.nc"
                )
                nc_path = adapter_factory(
                    af_da.values,
                    key,
                    control,
                )
            else:
                # there is an adapter for dynamic param files.
                nc_path = output_dir.parent / ag_frac_dyn_file

        else:
            nc_path = output_dir / f"{key}.nc"
            if not nc_path.exists():
                nc_path = None

        input_variables[key] = nc_path

    if do_compare_output_files:
        nc_output_dir = tmp_path / simulation["name"].replace(":", "_")
        control.options["netcdf_output_dir"] = nc_output_dir

    soil = SoilzoneAg(
        control=control,
        discretization=discretization,
        parameters=parameters,
        **input_variables,
        imbalance_behavior=imbalance_behavior,
        calc_method=calc_method,
    )

    if do_compare_output_files:
        soil.initialize_netcdf()

    # Load answers for comparison and HRU-time exceptions
    answers = {}
    skipped_comp_vars = []
    for var in comparison_var_names:
        var_pth = output_dir / f"{var}.nc"
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

    if do_compare_in_memory:
        ag_mask = np.where(soil["ag_frac"] > 0.0)
        not_ag_mask = np.where(soil["ag_frac"] <= 0.0)
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

    hru_time_exceptions = get_hru_time_exceptions(simulation)

    for istep in range(control.n_times):
        control.advance()
        soil.advance()

        # Advance answers to current timestep
        for var in answers.values():
            var.advance()

        soil.calculate(1.0)

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
                        fortran_val = answers[var].current.data[hru_idx]
                        if isinstance(soil[var], np.ndarray):
                            soil[var][hru_idx] = fortran_val
                        else:
                            # Handle TimeseriesArray
                            soil[var].current[hru_idx] = fortran_val

        soil.output()

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

            compare_in_memory(
                soil,
                answers,
                mask_dict=mask_dict,
                atol=default_atol,
                rtol=default_rtol,
                var_tolerances=var_tolerances,
                skip_missing_ans=True,
                fail_after_all_vars=True,
            )

    soil.finalize()

    if do_compare_output_files:
        # Filter out variables without answer files
        vars_with_answers = []
        for var in comparison_var_names:
            var_pth = output_dir / f"{var}.nc"
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
            nc_output_dir,
            output_dir,
            atol=default_atol,
            rtol=default_rtol,
            var_tolerances=var_tolerances,
            # fail_after_all_vars=False,
            verbose=True,
        )

    return


@pytest.mark.parametrize("calc_method", calc_methods)
def test_mass_budget_pref_flow(
    simulation,
    control,
    discretization,
    parameters_pref,
    SoilzoneAg,
    calc_method,
):
    """Test mass budget with preferential flow enabled.

    This test only checks that the mass budget balances when preferential
    flow is enabled. It does not compare against Fortran outputs.
    """

    # sroff is a runoff variable edited by soilzone
    output_dir = simulation["output_dir"]

    input_variables = {}
    for key in SoilzoneAg.get_inputs():
        if key in ["aet_observed"]:
            nc_path = output_dir.parent / f"{key}.nc"

        elif key == "ag_frac":
            opts = control.options
            ag_frac_dyn_flag = opts.get("dyn_ag_frac_flag", [False])[0]
            ag_frac_dyn_file = opts.get("ag_frac_dynamic", [None])[0]
            if not ag_frac_dyn_flag:
                import xarray as xr

                af_da = xr.load_dataarray(
                    output_dir.parent / "ag_frac_static.nc"
                )
                nc_path = adapter_factory(
                    af_da.values,
                    key,
                    control,
                )
            else:
                nc_path = output_dir.parent / ag_frac_dyn_file

        else:
            nc_path = output_dir / f"{key}.nc"
            if not nc_path.exists():
                nc_path = None

        input_variables[key] = nc_path

    soil = SoilzoneAg(
        control=control,
        discretization=discretization,
        parameters=parameters_pref,
        **input_variables,
        imbalance_behavior="error",
        calc_method=calc_method,
    )

    # Run the simulation - mass budget will be checked automatically
    # because imbalance_behavior="error"
    for istep in range(control.n_times):
        control.advance()
        soil.advance()
        soil.calculate(1.0)
        soil.output()

    soil.finalize()

    return
