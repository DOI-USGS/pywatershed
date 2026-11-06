import pathlib as pl

import pytest
from utils_compare import compare_in_memory, compare_netcdfs

from pywatershed.base.adapter import adapter_factory
from pywatershed.base.control import Control
from pywatershed.hydrology.prms_runoff import PRMSRunoff
from pywatershed.hydrology.prms_runoff_no_dprst import PRMSRunoffNoDprst
from pywatershed.parameters import Parameters, PrmsParameters

# compare in memory (faster) or full output files? or both!
do_compare_output_files = False
do_compare_in_memory = True
rtol = atol = 1.0e-10

calc_methods = ("numpy", "numba")
params = ("params_sep", "params_one")


@pytest.fixture(scope="function")
def control(simulation):
    control = Control.load_prms(
        simulation["control_file"], warn_unused_options=False
    )

    return control


@pytest.fixture(scope="function")
def Runoff(control):
    if (
        "dprst_flag" in control.options.keys()
        and control.options["dprst_flag"]
    ):
        Runoff = PRMSRunoff
    else:
        Runoff = PRMSRunoffNoDprst

    return Runoff


@pytest.fixture(scope="function")
def discretization(simulation):
    dis_hru_file = simulation["dir"] / "parameters_dis_hru.nc"
    return Parameters.from_netcdf(dis_hru_file, encoding=False)


@pytest.fixture(scope="function", params=params)
def parameters(simulation, control, request):
    if request.param == "params_one":
        param_file = simulation["dir"] / control.options["parameter_file"]
        params = PrmsParameters.load(param_file)
        sat_threshold = params.parameters["sat_threshold"]

    else:
        param_file = simulation["dir"] / "parameters_PRMSRunoff.nc"
        params = PrmsParameters.from_netcdf(param_file)

        sz_param_file = simulation["dir"] / "parameters_PRMSSoilzone.nc"
        sz_params = PrmsParameters.from_netcdf(sz_param_file)
        sat_threshold = sz_params.parameters["sat_threshold"]

    if abs(sat_threshold).min() < 999.0:
        pytest.skip(
            "test_prms_runoff only valid when sat_threshold >= 999 (or some "
            "amount) which causes zero dunnian_flow"
        )

    return params


@pytest.mark.parametrize("calc_method", calc_methods)
def test_compare_prms(
    simulation,
    control,
    discretization,
    parameters,
    Runoff,
    tmp_path,
    calc_method,
):
    tmp_path = pl.Path(tmp_path)

    comparison_var_names = set(Runoff.get_variables()) - {
        "dprst_vol_thres_open",
    }
    control.options["netcdf_output_var_names"] = comparison_var_names

    output_dir = simulation["output_dir"]

    input_variables = {}
    for key in Runoff.get_inputs():
        nc_pth = output_dir / f"{key}.nc"
        input_variables[key] = nc_pth

    if do_compare_output_files:
        nc_output_dir = tmp_path / simulation["name"].replace(":", "_")
        control.options["netcdf_output_dir"] = nc_output_dir

    runoff = Runoff(
        control=control,
        discretization=discretization,
        parameters=parameters,
        **input_variables,
        imbalance_behavior="error",
        calc_method=calc_method,
    )

    if do_compare_output_files:
        runoff.initialize_netcdf()
        # test that init netcdf twice raises a warning
        with pytest.warns(UserWarning):
            runoff.initialize_netcdf()

    if do_compare_in_memory:
        answers = {}
        for var in comparison_var_names:
            var_pth = output_dir / f"{var}.nc"
            answers[var] = adapter_factory(
                var_pth, variable_name=var, control=control
            )

    for istep in range(control.n_times):
        control.advance()
        runoff.advance()
        runoff.calculate(1.0)
        runoff.output()

        if do_compare_in_memory:
            for var in answers.values():
                var.advance()
            compare_in_memory(
                runoff,
                answers,
                atol=atol,
                rtol=rtol,
                skip_missing_ans=True,
                fail_after_all_vars=True,
            )

    runoff.finalize()

    if do_compare_output_files:
        compare_netcdfs(
            comparison_var_names,
            nc_output_dir,
            output_dir,
            atol=atol,
            rtol=rtol,
        )

    return


def test_dprst_flag_default_true(simulation, control, discretization):
    """With dprst_flag given by neither the constructor nor the control,
    PRMSRunoff defaults to depression storage on, like PRMSSoilzone and
    PRMSGroundwater. It used to end up None (silently off) because
    _set_options ran after the default was applied."""
    control.options.pop("dprst_flag", None)
    param_file = simulation["dir"] / control.options["parameter_file"]
    parameters = PrmsParameters.load(param_file)
    input_variables = {
        key: simulation["output_dir"] / f"{key}.nc"
        for key in PRMSRunoff.get_inputs()
    }
    runoff = PRMSRunoff(
        control=control,
        discretization=discretization,
        parameters=parameters,
        **input_variables,
    )
    assert runoff._dprst_flag is True
