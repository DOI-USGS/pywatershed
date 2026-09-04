import pathlib as pl
import shutil
from pprint import pprint

import numpy as np
import pytest

import pywatershed
from pywatershed.base.adapter import adapter_factory
from pywatershed.base.control import Control
from pywatershed.base.model import Model
from pywatershed.parameters import Parameters, PrmsParameters

# This test removes PRMSSnow from the full model test, removing its errorrs.
# Importantly, it tests that "sroff" is computed correctly as it is set by
# both PRMSRunoff and PRMSSoilzone.
# The (nearly) comprehensive suite of variables is verified against PRMS
# outputs. Several variables in soilzone are not output in the PRMS
# configuraton or perhaps not at all by PRMS and these are skipped.
# Importantly, this test and test_prms_above_snow test model
# instantiation/invocation three ways which is not tested elsewhere in the
# the test suite.

invoke_style = ("prms", "model_dict", "model_dict_from_yaml")
failfast = True
verbose = False

test_models = {
    "nhm": [
        pywatershed.PRMSRunoff,
        pywatershed.PRMSSoilzone,
        pywatershed.PRMSGroundwater,
        pywatershed.PRMSChannel,
    ],
    "nhm_no_dprst": [
        pywatershed.PRMSRunoffNoDprst,
        pywatershed.PRMSSoilzoneNoDprst,
        pywatershed.PRMSGroundwaterNoDprst,
        pywatershed.PRMSChannel,
    ],
    "sagehen_no_cascades": [
        pywatershed.PRMSRunoffNoDprst,
        pywatershed.PRMSSoilzoneNoDprst,
        pywatershed.PRMSGroundwaterNoDprst,
    ],
    "sagehen_no_gw_cascades": [
        pywatershed.PRMSRunoffCascadesNoDprst,
        pywatershed.PRMSSoilzoneCascadesNoDprst,
        pywatershed.PRMSGroundwaterNoDprst,
    ],
    "sagehen_gridded_cascades": [
        pywatershed.PRMSRunoffCascadesNoDprst,
        pywatershed.PRMSSoilzoneCascadesNoDprst,
        pywatershed.PRMSGroundwaterNoDprst,
    ],
}

comparison_vars_dict_all = {
    "PRMSRunoff": list(
        set(pywatershed.PRMSRunoff.get_variables()) - {"dprst_vol_thres_open"}
    ),
    "PRMSSoilzone": list(
        set(pywatershed.PRMSSoilzone.get_variables())
        - {  # these variables not output by PRMS
            "soil_zone_max",
            "soil_lower_max",
            "perv_actet_hru",
            "soil_lower_change_hru",
            "soil_rechr_change_hru",
        }
    ),
    "PRMSGroundwater": pywatershed.PRMSGroundwater.get_variables(),
    "PRMSChannel": set(pywatershed.PRMSChannel.get_variables())
    - {"inflow_ts_prev", "outflow_ts"},
}

tol = {
    "PRMSRunoff": 1.0e-8,
    "PRMSRunoffCascadesNoDprst": 1.0e-8,
    "PRMSRunoffNoDprst": 1.0e-8,
    "PRMSSoilzone": 1.0e-8,
    "PRMSSoilzoneCascadesNoDprst": 1.0e-8,
    "PRMSSoilzoneNoDprst": 1.0e-8,
    "PRMSGroundwater": 1.0e-8,
    "PRMSGroundwaterNoDprst": 1.0e-8,
    "PRMSChannel": 5.0e-7,
}


@pytest.fixture(scope="function")
def control(simulation):
    sim_name = simulation["name"]
    config_name = sim_name.split(":")[1]
    if config_name not in test_models.keys():
        pytest.skip(
            f"The configuration is not tested by test_model: {config_name}"
        )
    control = Control.load_prms(
        simulation["control_file"], warn_unused_options=False
    )
    control.options["verbosity"] = 10
    control.options["imbalance_behavior"] = "error"
    control.options["calc_method"] = "numba"
    del control.options["netcdf_output_var_names"]
    return control


@pytest.fixture(scope="function")
def discretization(simulation):
    dis_hru_file = simulation["dir"] / "parameters_dis_hru.nc"
    dis_hru = Parameters.from_netcdf(dis_hru_file, encoding=False)
    # PRMSChannel needs both dis where as it should only need dis_seg
    # and will when we have exchanges
    dis = {
        "dis_hru": dis_hru,
    }

    dis_both_file = simulation["dir"] / "parameters_dis_both.nc"
    if dis_both_file.exists():
        dis_both = Parameters.from_netcdf(dis_both_file, encoding=False)
        dis["dis_both"] = dis_both

    return dis


@pytest.fixture(scope="function", params=invoke_style)
def model_args(simulation, control, discretization, request):
    invoke_style = request.param
    control_key = simulation["name"].split(":")[1]
    process_list = test_models[control_key]

    if invoke_style == "prms":
        # Single params is the backwards compatible way
        param_file = simulation["dir"] / control.options["parameter_file"]
        args = {
            "process_list_or_model_dict": process_list,
            "control": control,
            "parameters": PrmsParameters.load(param_file),
        }

    elif invoke_style == "model_dict":
        # Constructing this model_dict is the new way
        model_dict = discretization
        model_dict["control"] = control
        # could use any names
        model_dict["model_order"] = [
            pp.__name__.lower() for pp in process_list
        ]

        for process in test_models[control_key]:
            proc_name = process.__name__
            proc_name_lower = proc_name.lower()
            model_dict[proc_name_lower] = {}
            proc = model_dict[proc_name_lower]
            proc["class"] = process
            proc_param_file = simulation["dir"] / f"parameters_{proc_name}.nc"
            proc["parameters"] = PrmsParameters.from_netcdf(proc_param_file)
            if proc_name_lower == "PRMSChannel".lower():
                proc["dis"] = "dis_both"
            else:
                proc["dis"] = "dis_hru"

        if verbose:
            pprint(model_dict, sort_dicts=False)

        args = {
            "process_list_or_model_dict": model_dict,
            "control": None,
            "parameters": None,
        }

    elif invoke_style == "model_dict_from_yaml":
        yaml_name = simulation["name"].split(":")[1]
        yaml_file = simulation["dir"] / f"{yaml_name}_model.yaml"
        model_dict = Model.model_dict_from_yaml(yaml_file)

        # Edit this dict from the yaml to use only processes below snow
        class_keys = {
            vv["class"]: kk
            for kk, vv in model_dict.items()
            if isinstance(vv, dict) and "class" in vv.keys()
        }
        to_del = [
            kk for cl, kk in class_keys.items() if cl not in process_list
        ]
        for del_me in to_del:
            _ = model_dict["model_order"].remove(del_me)
            del model_dict[del_me]

        args = {
            "process_list_or_model_dict": model_dict,
            "control": None,
            "parameters": None,
        }

    else:
        msg = "invalid parameter value"
        raise ValueError(msg)

    return args


def test_model(simulation, model_args, tmp_path):
    """Run the full NHM model"""
    tmp_path = pl.Path(tmp_path)
    output_dir = simulation["output_dir"]
    sim_name = simulation["name"]
    config_name = sim_name.split(":")[1]

    # setup input_dir with symlinked prms inputs and outputs
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    for ff in output_dir.resolve().glob("*.nc"):
        shutil.copy(ff, input_dir / ff.name)
    for ff in output_dir.parent.resolve().glob("*.nc"):
        shutil.copy(ff, input_dir / ff.name)

    if model_args["control"] is None:
        control = model_args["process_list_or_model_dict"]["control"]
    else:
        control = model_args["control"]

    control.options["input_dir"] = input_dir
    model_out_dir = tmp_path / "output"
    control.options["netcdf_output_dir"] = model_out_dir

    model = Model(**model_args, write_control=model_out_dir)

    # every process, including subclasses that set self.name before
    # super().__init__(), reports its own class name
    for proc in model.processes.values():
        assert proc.name == proc.__class__.__name__

    # check that control yaml file was written
    control_yaml_file = sorted(model_out_dir.glob("*model_control.yaml"))
    assert len(control_yaml_file) == 1

    # ---------------------------------
    # get the answer data against PRMS5.2.1
    # this is the adhoc set of things to compare, to circumvent fussy issues?

    for vv in ["PRMSRunoff", "PRMSSoilzone", "PRMSGroundwater"]:
        comparison_vars_dict_all[f"{vv}NoDprst"] = comparison_vars_dict_all[vv]

    comparison_vars_dict_all["PRMSRunoffCascadesNoDprst"] = list(
        pywatershed.PRMSRunoffCascadesNoDprst.get_variables()
    )
    comparison_vars_dict_all["PRMSSoilzoneCascadesNoDprst"] = list(
        comparison_vars_dict_all["PRMSSoilzone"]
    ) + [
        "hru_sz_cascadeflow",
        "upslope_dunnianflow",
        "upslope_interflow",
    ]

    comparison_vars_dict = {}

    plomd = model_args["process_list_or_model_dict"]
    config_processes = test_models[config_name]
    if isinstance(plomd, list):
        processes = [pp for pp in plomd if pp in config_processes]
        control = model_args["control"]
        class_key = {
            vv.__class__.__name__: kk for kk, vv in model.processes.items()
        }
    else:
        processes = [
            vv["class"]
            for vv in plomd.values()
            if isinstance(vv, dict) and vv["class"] in config_processes
        ]
        processes = [pp for pp in processes if pp in config_processes]
        control = plomd["control"]
        class_key = {
            vv["class"].__name__: kk
            for kk, vv in plomd.items()
            if isinstance(vv, dict) and "class" in vv.keys()
        }

    for cls in processes:
        key = cls.__name__
        cls_vars = cls.get_variables()
        comparison_vars_dict[key] = {
            vv for vv in comparison_vars_dict_all[key] if vv in cls_vars
        }

    # Read PRMS output into ans for comparison with pywatershed results
    ans = {key: {} for key in comparison_vars_dict.keys()}
    for process_name, var_names in comparison_vars_dict.items():
        for vv in var_names:
            # TODO: this is hacky, improve the design
            if (
                "dprst_flag" in control.options.keys()
                and not control.options["dprst_flag"]
            ):
                if "dprst" in vv:
                    continue

            if vv in ["tmin", "tmax", "prcp"]:
                nc_pth = input_dir.parent / f"{vv}.nc"
            else:
                nc_pth = input_dir / f"{vv}.nc"

            ans[process_name][vv] = adapter_factory(
                nc_pth, variable_name=vv, control=control
            )

    all_success = True
    fail_prms_compare = False
    for istep in range(control.n_times):
        model.advance()
        model.calculate()

        # PRMS5 answers
        # advance the answer, which is being read from a netcdf file
        for process_name, var_names in ans.items():
            for vv in var_names:
                ans[process_name][vv].advance()

        # make a comparison check with answer
        for process_name in ans.keys():
            success = check_timestep_results(
                model.processes[class_key[process_name]],
                istep,
                ans[process_name],
                tol[process_name],
                failfast,
                verbose=verbose,
            )
            if not success:
                fail_prms_compare = True
                all_success = False
                if failfast:
                    assert False, "PRMS comparison failfast"

    # check at the end and error if one or more steps didn't pass
    if not all_success:
        if fail_prms_compare:
            msg = "pywatershed results failed comparison with prms5.2.1"
        else:
            assert False, "this should not be possible"

        raise Exception(msg)

    return


def check_timestep_results(
    storageunit,
    istep,
    ans,
    tol,
    failfast=False,
    verbose=False,
):
    # print(storageunit)
    all_success = True
    active_mask = getattr(storageunit, "_active_hru_mask", None)
    for key in ans.keys():
        # print(key)
        a1 = ans[key].current
        if not isinstance(storageunit[key], np.ndarray):
            a2 = storageunit[key].current
        else:
            a2 = storageunit[key]
        if active_mask is not None and np.shape(a1) == np.shape(active_mask):
            # compare only at active HRUs; pywatershed masks inactive
            # HRUs to nan while PRMS generally reports zeros there.
            a1 = np.asarray(a1)[active_mask]
            a2 = np.asarray(a2)[active_mask]
        success_a = np.isclose(a2, a1, atol=tol, rtol=0.0)
        success_r = np.isclose(a2, a1, atol=0.0, rtol=tol)
        success = success_a | success_r
        if not success.all():
            all_success = False
            diff = a2 - a1
            diffmin = diff.min()
            diffmax = diff.max()
            if True:
                print(f"time step {istep}")
                print(f"output variable {key}")
                print(f"prms   {a1.min()}    {a1.max()}")
                print(f"pywatershed  {a2.min()}    {a2.max()}")
                print(f"diff   {diffmin}  {diffmax}")

                if verbose:
                    idx = np.where(~success)
                    for i in idx:
                        print(
                            f"hru {i} prms {a1[i]} pywatershed {a2[i]} "
                            f"diff {diff[i]}"
                        )
            if failfast:
                raise (ValueError)
        # <
        elif verbose:
            print(f"variable {key} matches")

    return all_success
