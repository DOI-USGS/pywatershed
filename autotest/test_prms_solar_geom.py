import pytest
from utils_compare import compare_in_memory, compare_netcdfs

from pywatershed.atmosphere.prms_solar_geometry import PRMSSolarGeometry
from pywatershed.base.adapter import adapter_factory
from pywatershed.base.control import Control
from pywatershed.base.parameters import Parameters
from pywatershed.parameters import PrmsParameters

# compare in memory (faster) or full output files? or both!
do_compare_output_files = False
do_compare_in_memory = True

# rtol = atol = 1.0e-10  # PRMS 5.2.1
rtol = atol = 5.0e-5  # GSFLOW 2.4.1

params = ("params_sep", "params_one")


@pytest.fixture(scope="function")
def tols(simulation):
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

    if "prms" in exe_desc:
        tols = {"atol": 1.0e-10, "rtol": 1.0e-10}
    elif "gsflow" in exe_desc:
        tols = {"atol": 5.0e-5, "rtol": 5.0e-5}
    else:
        raise ValueError(f"Unknown executable description: {exe_desc}")
    return tols


@pytest.fixture(scope="function")
def control(simulation):
    ctl = Control.load_prms(
        simulation["control_file"], warn_unused_options=False
    )
    del ctl.options["netcdf_output_dir"]
    return ctl


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
        param_file = simulation["dir"] / "parameters_PRMSSolarGeometry.nc"
        params = PrmsParameters.from_netcdf(param_file)

    return params


@pytest.mark.parametrize(
    "from_prms_file", (True, False), ids=("from_prms_file", "compute")
)
def test_compare_prms(
    simulation,
    control,
    discretization,
    parameters,
    tols,
    tmp_path,
    from_prms_file,
):
    output_dir = simulation["output_dir"]
    atol = tols["atol"]
    rtol = tols["rtol"]
    print(f"{atol=}")
    print(f"{rtol=}")

    # PRMS's raw solar table, shipped with the answers (print_debug 5)
    prms_soltab_file = simulation["output_dir"] / "soltab_debug"
    if from_prms_file:
        if not prms_soltab_file.exists():
            pytest.skip("No 'soltab_debug' file found in the output dir.")
        from_prms_file = prms_soltab_file
    else:
        from_prms_file = None

    solar_geom = PRMSSolarGeometry(
        control,
        discretization=discretization,
        parameters=parameters,
        from_prms_file=from_prms_file,
    )

    solar_geom.initialize_netcdf(output_dir=tmp_path)

    if do_compare_in_memory:
        answers = {}
        for var in PRMSSolarGeometry.get_variables():
            var_pth = output_dir / f"{var}.nc"
            answers[var] = adapter_factory(
                var_pth, variable_name=var, control=control
            )

        sunhrs_id = id(solar_geom.soltab_sunhrs)

        # Though the data is all calculate on the initial advance,
        # we step through it using the timeseries array.
        # we only need to output at time 0
        for ii in range(control.n_times):
            control.advance()
            solar_geom.advance()
            if ii == 0:
                solar_geom.output()
            solar_geom.calculate(1.0)

            if do_compare_in_memory:
                for var in answers.values():
                    var.advance()
                compare_in_memory(solar_geom, answers, atol=atol, rtol=rtol)

            assert id(solar_geom.soltab_sunhrs) == sunhrs_id

    if do_compare_output_files:
        compare_netcdfs(
            PRMSSolarGeometry.get_variables(),
            tmp_path,
            output_dir,
            atol=atol,
            rtol=rtol,
        )

    return
