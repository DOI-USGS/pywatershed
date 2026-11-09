import pathlib as pl
import shutil
from copy import deepcopy
from itertools import product

import numpy as np
import pytest
import xarray as xr

import pywatershed
from pywatershed.base.control import Control
from pywatershed.base.model import Model
from pywatershed.parameters import PrmsParameters
from pywatershed.utils.time_utils import datetime_doy as doy

# test for a few timesteps a model with both unit/cell and global balance
# budgets

# This probably dosent need varied over simulation

n_time_steps = 10


@pytest.fixture(scope="function")
def params(simulation, control):
    param_file = simulation["dir"] / control.options["parameter_file"]
    return PrmsParameters.load(param_file)


@pytest.fixture(scope="function")
def control(simulation):
    control = Control.load_prms(
        simulation["control_file"], warn_unused_options=False
    )
    control.edit_n_time_steps(n_time_steps)
    control.options["imbalance_behavior"] = "error"
    del control.options["netcdf_output_var_names"]
    del control.options["netcdf_output_dir"]
    return control


# Things to parameterize
# optional variables to processes
# optional variables to budgets

check_vars_dict = {
    "PRMSSolarGeometry": [
        "soltab_horad_potsw",
        "soltab_potsw",
    ],
    "PRMSAtmosphere": [
        "tminf",
        "potet",
        "swrad",
    ],
    "PRMSCanopy": [
        "hru_intcpstor",
        "hru_intcpstor_change",
        "net_snow",
    ],
    "PRMSChannel": [
        "channel_outflow_vol",
        "seg_stor_change",
        "seg_upstream_inflow",
    ],
}


@pytest.fixture(scope="function")
def check_vars():
    return deepcopy(check_vars_dict)


budget_sum_vars_all = ["inputs_sum", "outputs_sum", "storage_changes_sum"]
check_budget_sum_vars_params = [False, True, "some"]


@pytest.mark.filterwarnings("ignore:Budget for")
@pytest.mark.parametrize(
    "budget_sum_param",
    check_budget_sum_vars_params,
    ids=[str(ii) for ii in check_budget_sum_vars_params],
)
def test_process_budgets(
    simulation, control, params, tmp_path, budget_sum_param, check_vars
):
    tmp_dir = pl.Path(tmp_path)
    # print(tmp_dir)
    model_procs = [
        pywatershed.PRMSSolarGeometry,
        pywatershed.PRMSAtmosphere,
        pywatershed.PRMSCanopy,
        pywatershed.PRMSChannel,
    ]

    if control.options["streamflow_module"] == "strmflow":
        _ = model_procs.remove(pywatershed.PRMSChannel)
        del check_vars["PRMSChannel"]

    # setup input_dir with symlinked prms inputs and outputs
    domain_output_dir = simulation["output_dir"]
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    control.options["input_dir"] = input_dir

    # Could limit this to just the variables in model_procs
    for ff in domain_output_dir.resolve().glob("*.nc"):
        shutil.copyfile(ff, input_dir / ff.name)
    for ff in domain_output_dir.parent.resolve().glob("*.nc"):
        shutil.copyfile(ff, input_dir / ff.name)

    # Deal with parameter around what budget sum vars to write and check
    if budget_sum_param == "some":
        check_budget_sum_vars = budget_sum_vars_all[0:-1]
        budget_args = {"write_sum_vars": check_budget_sum_vars}
    elif budget_sum_param:
        check_budget_sum_vars = budget_sum_vars_all
        budget_args = {"write_sum_vars": budget_sum_param}
    elif not budget_sum_param:
        check_budget_sum_vars = []
        budget_args = {"write_sum_vars": budget_sum_param}
    else:
        raise ValueError("upexpected value")

    control.options["netcdf_output_dir"] = tmp_dir

    # TODO: Eliminate potet and other variables from being used
    model = Model(
        model_procs,
        control=control,
        parameters=params,
    )

    # we are going to harvest the data from memory and store here
    check_dict = {proc: {} for proc in check_vars.keys()}

    # test outputting specific vars by only using check_vars
    output_vars = [
        item for sublist in list(check_vars.values()) for item in sublist
    ]

    with pytest.raises(ValueError):
        model.initialize_netcdf(
            pl.Path("foo"),
            budget_args=budget_args,
            output_vars=output_vars,
        )

    model.initialize_netcdf(
        output_dir=tmp_dir,  # should allow a matching argument to control
        budget_args=budget_args,
        output_vars=output_vars,
    )

    for tt in range(n_time_steps):
        model.advance()
        model.calculate()
        model.output()

        for pp, pp_vars in check_vars.items():
            for vv in pp_vars:
                if tt == 0:
                    # use the output data to figure out the shape
                    if isinstance(
                        model.processes[pp][vv], pywatershed.TimeseriesArray
                    ):
                        spatial_len = model.processes[pp][vv].data.shape[1]
                    else:
                        spatial_len = model.processes[pp][vv].shape[0]

                    check_dict[pp][vv] = np.zeros((n_time_steps, spatial_len))

                if isinstance(
                    model.processes[pp][vv], pywatershed.TimeseriesArray
                ):
                    check_dict[pp][vv][tt, :] = model.processes[pp][vv].current
                else:
                    check_dict[pp][vv][tt, :] = model.processes[pp][vv]

            if pp in ["PRMSSolarGeometry", "PRMSAtmosphere"]:
                continue

            for bb in check_budget_sum_vars:
                if tt == 0:
                    # use the output data to figure out the shape
                    shp = model.processes[pp].mass_budget[f"_{bb}"].shape
                    if len(shp):
                        shp = shp[0]
                    else:
                        shp = 1
                    check_dict[pp][bb] = np.zeros((n_time_steps, shp))

                check_dict[pp][bb][tt, :] = model.processes[pp].mass_budget[
                    f"_{bb}"
                ]

    model.finalize()

    # read the data back in
    for pp, pp_vars in check_vars.items():
        for vv in pp_vars:
            nc_data = xr.open_dataarray(tmp_dir / f"{vv}.nc")
            if vv in pywatershed.PRMSSolarGeometry.get_variables():
                # these are on doy-basis in output, not a timeseries
                # start_ind
                start_doy = doy(control.start_time)
                start_ind = start_doy - 1
                end_ind = start_ind + n_time_steps
                assert np.allclose(
                    check_dict[pp][vv], nc_data[start_ind:end_ind, :]
                )
            else:
                assert np.allclose(check_dict[pp][vv], nc_data)

        if pp in ["PRMSSolarGeometry", "PRMSAtmosphere"]:
            continue

        for bb in check_budget_sum_vars:
            nc_data = xr.open_dataset(tmp_dir / f"{pp}_mass_budget.nc")[bb]
            assert np.allclose(check_dict[pp][bb], nc_data)

        if budget_sum_param == "some":
            nc_data = xr.open_dataset(tmp_dir / f"{pp}_mass_budget.nc")
            for nn in set(budget_sum_vars_all).difference(
                set(check_budget_sum_vars)
            ):
                assert nn not in nc_data.variables
        elif not budget_sum_param:
            assert not (tmp_dir / f"{pp}_mass_budget.nc").exists()

    return


separate_outputs = [False, True]

output_vars = [None, [var for kk, vv in check_vars_dict.items() for var in vv]]


@pytest.fixture(
    scope="function",
    params=list(product(separate_outputs, output_vars)),
)
def sep_vars(request):
    return (request.param[0], request.param[1])


def test_separate_together_var_list(
    simulation, control, params, tmp_path, sep_vars, check_vars
):
    separate = sep_vars[0]
    output_vars = sep_vars[1]

    tmp_dir = pl.Path(tmp_path)

    model_procs = [
        pywatershed.PRMSSolarGeometry,
        pywatershed.PRMSAtmosphere,
        pywatershed.PRMSCanopy,
        pywatershed.PRMSChannel,
    ]

    if control.options["streamflow_module"] == "strmflow":
        _ = model_procs.remove(pywatershed.PRMSChannel)
        del check_vars["PRMSChannel"]

    # setup input_dir with symlinked prms inputs and outputs
    domain_output_dir = simulation["output_dir"]
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    control.options["input_dir"] = input_dir
    control.options["netcdf_output_var_names"] = output_vars
    control.options["netcdf_output_separate_files"] = separate

    # Could limit this to just the variables in model_procs
    for ff in domain_output_dir.resolve().glob("*.nc"):
        shutil.copyfile(ff, input_dir / ff.name)
    for ff in domain_output_dir.parent.resolve().glob("*.nc"):
        shutil.copyfile(ff, input_dir / ff.name)

    model = Model(
        model_procs,
        control=control,
        parameters=params,
    )
    with pytest.raises(ValueError):
        # passing no output_dir arg and none in opts throws an error
        model.initialize_netcdf()

    test_output_dir = tmp_dir / "test_results"
    control.options["netcdf_output_dir"] = test_output_dir
    model = Model(
        model_procs,
        control=control,
        parameters=params,
    )

    for tt in range(n_time_steps):
        model.advance()
        model.calculate()
        model.output()

    model.finalize()

    if separate:
        for proc_key, proc in model.processes.items():
            for vv in proc.variables:
                if output_vars is not None and vv not in output_vars:
                    continue
                nc_file = test_output_dir / f"{vv}.nc"
                assert nc_file.exists()
                ds = xr.open_dataset(nc_file, decode_timedelta=False)
                if isinstance(
                    proc[vv], pywatershed.base.timeseries.TimeseriesArray
                ):
                    assert (ds[vv].values == proc[vv].data).all()
                else:
                    assert (ds[vv][-1, :] == proc[vv]).all()

                del ds

    else:
        for proc_key, proc in model.processes.items():
            # non-budget
            nc_file = test_output_dir / f"{proc_key}.nc"
            if output_vars is None or proc_key in check_vars.keys():
                assert nc_file.exists()

                ds = xr.open_dataset(nc_file, decode_timedelta=False)
                if output_vars is None:
                    proc_vars = set(proc.get_variables())
                else:
                    proc_vars = set(check_vars[proc_key])
                nc_vars = set(ds.data_vars)
                assert proc_vars == nc_vars
                for vv in proc.variables:
                    if output_vars is not None and vv not in output_vars:
                        continue

                    if isinstance(
                        proc[vv], pywatershed.base.timeseries.TimeseriesArray
                    ):
                        assert (ds[vv].values == proc[vv].data).all()

                    else:
                        assert (ds[vv][-1, :] == proc[vv]).all()

                del ds

            # budget
            # no budgets for solar or atmosphere
            if proc_key in ["PRMSSolarGeometry", "PRMSAtmosphere"]:
                continue
            nc_file = test_output_dir / f"{proc_key}_mass_budget.nc"
            ds = xr.open_dataset(nc_file)
            for ss in budget_sum_vars_all:
                assert (proc.mass_budget[ss] == ds[ss][-1, :]).all()

            del ds
    return
