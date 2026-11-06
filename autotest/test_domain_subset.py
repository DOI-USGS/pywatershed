import pathlib as pl

import numpy as np
import pyPRMS as pp
import pytest
import xarray as xr

import pywatershed as pws

# TODO: remove test nhm code cheats

full_domain_dir = pl.Path("../test_data/ucb_2yr")
full_control_file_nhm = full_domain_dir / "nhm.control"
full_cbh_dir_nhm = full_domain_dir


full_cbh_nc_file_dict_nhm = {
    kk: full_cbh_dir_nhm / vv
    for kk, vv in {
        "precip_day": "prcp.nc",
        "tmax_day": "tmax.nc",
        "tmin_day": "tmin.nc",
    }.items()
}

use_nhm = False

# TO TEST:
# x PRMS inputs
# x [PRMS CBH files,  NetCDF CBH files]
# x [pywatershed outputs, PRMS outputs]
# x [known, single_seg]
# x [original index, shuffled index]

domain_name = "ucb_2yr"
domain_seg_outlet = {"ucb_2yr": np.int64(44425)}
# fmt: off
domain_known_nhm_ids = np.array(
    [
        86522, 86621, 86343, 86344, 86525, 86622, 86313, 86333, 86270, 86272,
        86274, 86278, 86288, 86290, 86283, 86307, 86261, 86263, 86260, 86262,
        86330, 86337,
    ]
)
# fmt: on
domain_known_nhm_segs = np.array(
    [44409, 44412, 44417, 44418, 44420, 44421, 44422, 44423, 44425, 44430]
)

cbh_types = ("ascii", "netcdf")[1:]  # until implemented
output_types = ["pywatershed"]

# compound test configs
subset_styles = ("known", "single_seg")
ind_order = ["orig", "shuffle"]

sub_ids_segs_types = []
for ss in subset_styles:
    for io in ind_order:
        # there is no way to shuffle single_seg subsetting
        if ss == "single_seg" and io == "shuffle":
            continue
        sub_ids_segs_types += [(ss, io)]


sub_ids_segs_type_ids = [
    f"{tt[0][0:2]}-{tt[1][0:2]}" for tt in sub_ids_segs_types
]


@pytest.fixture(scope="function")
def full_control_file(simulation, tmp_path):
    if simulation["name"] != f"{domain_name}:nhm":
        pytest.skip(f"test_domain_subset only runs for {domain_name}:nhm")

    # To keep the test quick, only test for 30 days. We'll re-write the control
    # file to temp_path but we have to maintain the parameter and CBH file
    # info. (Since we are using pyPRMS, only works for the case of a single
    # param_file.)
    control_file = simulation["control_file"]
    control_parent = control_file.parent.resolve()
    new_control_file = tmp_path / "shortened.control"

    control = pp.ControlFile(
        filename=control_file,
        metadata=pws.constants.pyprms_meta,
        verbose=False,
    )
    cv = control.control_variables
    # edit time
    cv["end_time"].values = cv["start_time"].values + np.timedelta64(30, "D")
    # param_file and data_file paths
    for ff in ["param_file", "data_file"]:
        file_name = pl.Path(cv[ff].values)
        if not file_name.is_absolute():
            cv[ff].values = str(control_parent / file_name)
    # cbh file paths
    for kk, vv in pws.constants.cbh_ctl_var_map.items():
        if kk in cv.keys() and cv[kk].values:
            cbh_file_name = pl.Path(cv[kk].values)
            if not cbh_file_name.is_absolute():
                cv[kk].values = str(control_parent / cbh_file_name)

            if not pl.Path(cv[kk].values).exists():
                control.remove(kk)
                # warn? gosh, not sure
            else:
                print(f"{kk}= {cv[kk].values}")
                pass

    control.write(new_control_file)

    if use_nhm:
        return full_control_file_nhm
    else:
        return new_control_file


@pytest.fixture(scope="function", params=cbh_types)
def full_cbh_nc_file_dict(simulation, request):
    cbh_type = request.param
    if cbh_type == "netcdf":
        # for domains in test_data, the netcdf cbh files are found in
        # simulation["dir"]
        full_cbh_dir = simulation["dir"]
        full_cbh_nc_file_dict = {
            kk: full_cbh_dir / vv
            for kk, vv in {
                "precip_day": "prcp.nc",
                "tmax_day": "tmax.nc",
                "tmin_day": "tmin.nc",
            }.items()
        }
    else:
        full_cbh_nc_file_dict = None

    if use_nhm:
        return full_cbh_nc_file_dict_nhm
    else:
        return full_cbh_nc_file_dict


@pytest.fixture(
    scope="function", params=sub_ids_segs_types, ids=sub_ids_segs_type_ids
)
def sub_ids_segs(simulation, request):
    subset_style = request.param[0]
    ind_order = request.param[1]

    if subset_style == "known":
        nhm_ids = domain_known_nhm_ids
        nhm_segs = domain_known_nhm_segs

        if ind_order == "shuffle":
            np.random.shuffle(nhm_ids)
            np.random.shuffle(nhm_segs)

        sub_ids_segs = {"nhm_ids": nhm_ids, "nhm_segs": nhm_segs}

    else:
        assert subset_style == "single_seg"
        sub_ids_segs = {
            "nhm_ids": None,
            "nhm_segs": np.int64(domain_seg_outlet[domain_name]),
        }

    return sub_ids_segs


@pytest.fixture(scope="function", params=output_types, ids=output_types)
def output_format(simulation, request):
    return request.param


def test_pws_subset_known_ids_segs(
    full_control_file,
    sub_ids_segs,
    full_cbh_nc_file_dict,
    output_format,
    tmp_path,
):
    subdomain = pws.utils.DomainSubset(
        full_control_file=full_control_file,
        sub_nhm_ids=sub_ids_segs["nhm_ids"],
        sub_nhm_segs=sub_ids_segs["nhm_segs"],
        full_cbh_nc_files_dict=full_cbh_nc_file_dict,
        from_seg_calc_check=True,
        output_format=output_format,
    )

    # can test that output_format=None raises error
    sub_domain_dir = tmp_path / "subdomain"
    subdomain.write(write_dir=sub_domain_dir, output_format=output_format)

    full_run_dir = tmp_path / "full_run"
    full_run_dir.mkdir()

    if output_format.lower() == "pywatershed":
        # test NHM
        nhm_proc_list = [
            pws.PRMSSolarGeometry,
            pws.PRMSAtmosphere,
            pws.PRMSCanopy,
            pws.PRMSSnow,
            pws.PRMSRunoff,
            pws.PRMSSoilzone,
            pws.PRMSGroundwater,
            pws.PRMSChannel,
        ]
        # full domain run
        full_dir = full_control_file.parent
        full_ctl = pws.Control.load_prms(
            full_control_file, warn_unused_options=False
        )
        full_param_file = pl.Path(full_ctl.options["parameter_file"])
        if not full_param_file.is_absolute():
            full_param_file = full_dir / str(full_param_file)
        full_params = pws.parameters.PrmsParameters.load(full_param_file)
        input_dirs = np.array(
            [str(dd.parent) for dd in full_cbh_nc_file_dict.values()]
        )
        assert (input_dirs == input_dirs[0]).all()
        full_ctl.options["input_dir"] = str(input_dirs[0])
        full_ctl.options["netcdf_output_dir"] = pl.Path(full_run_dir)
        full_model = pws.Model(
            nhm_proc_list, control=full_ctl, parameters=full_params
        )
        full_model.run()
        # sub domain run
        sub_run_dir = tmp_path / "sub_run"
        sub_ctl_file = list(sub_domain_dir.glob("*.control"))[0]
        sub_ctl = pws.Control.load_prms(
            sub_ctl_file, warn_unused_options=False
        )
        sub_param_file = sub_domain_dir / sub_ctl.options["parameter_file"]
        sub_params = pws.parameters.PrmsParameters.from_netcdf(sub_param_file)
        sub_ctl.options["input_dir"] = sub_domain_dir
        sub_ctl.options["netcdf_output_dir"] = pl.Path(sub_run_dir)
        sub_model = pws.Model(
            nhm_proc_list, control=sub_ctl, parameters=sub_params
        )
        sub_model.run()

        nc_output_files = list(full_run_dir.glob("*.nc"))
        for nc_file in nc_output_files:
            if "budget" in nc_file.name:
                continue
            var_name = nc_file.with_suffix("").name
            file_name = f"{var_name}.nc"
            sub_var = xr.load_dataarray(sub_run_dir / file_name)
            full_var = xr.load_dataarray(full_run_dir / file_name)
            if "nhm_id" in sub_var.dims:
                full_var = full_var.sel(nhm_id=sub_var.nhm_id)
            elif "nhm_seg" in sub_var.dims:
                full_var = full_var.sel(nhm_seg=sub_var.nhm_seg)
            xr.testing.assert_allclose(full_var, sub_var)
            # var_name = full_var.columns[0]
            # mg = sub_var.join(
            #     full_var, how="left", lsuffix="_sub", rsuffix="_full"
            # )
            # if not (mg[f"{var_name}_sub"] == mg[f"{var_name}_full"]).all():
            #     raise ValueError(f"Comparison failed for {var_name=}")

    else:
        raise ValueError("How'd we get here?")
