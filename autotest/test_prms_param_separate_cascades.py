"""separate_domain_params_dis_to_ncdf with a cascade control.

Checks that the per-process parameter files written for the cascade
process classes are complete: every declared parameter is in the process
file or in the dis_hru file written beside it, and every declared
dimension is in the process file even when no parameter is defined on it
(nsegment: the cascade classes declare it, but all their parameters are
on nhru or ncascade).
"""

import pathlib as pl

import pytest

import pywatershed
from pywatershed.base.data_model import open_datasetdict
from pywatershed.utils import separate_domain_params_dis_to_ncdf

cascade_processes = [
    pywatershed.PRMSRunoffCascadesNoDprst,
    pywatershed.PRMSSoilzoneCascadesNoDprst,
]
gw_cascade_processes = [pywatershed.PRMSGroundwaterCascadesNoDprst]


@pytest.fixture(scope="function")
def control(simulation):
    ctl = pywatershed.Control.load_prms(
        simulation["control_file"], warn_unused_options=False
    )
    if not ctl.options.get("cascade_flag", 0):
        pytest.skip("cascade_flag absent or 0")
    del ctl.options["netcdf_output_dir"]
    return ctl


def test_param_sep_cascades(simulation, control, tmp_path):
    prms_param_file = simulation["dir"] / control.options["parameter_file"]
    process_list = list(cascade_processes)
    if control.options.get("cascadegw_flag", 0):
        process_list += gw_cascade_processes
    proc_nc_files = separate_domain_params_dis_to_ncdf(
        prms_param_file,
        simulation["name"],
        pl.Path(tmp_path),
        process_list=process_list,
        control=control,
    )

    # what a Model gives these HRU processes as their discretization
    dis_names = set(open_datasetdict(proc_nc_files["dis_hru"]).variables)
    for proc in process_list:
        file_params = open_datasetdict(proc_nc_files[proc])
        file_names = set(file_params.variables.keys())
        # both loaders must see the same coordinates, including nhm_seg
        # which no parameter uses (only the global coordinates attribute
        # records it)
        file_params_nc4 = open_datasetdict(proc_nc_files[proc], use_xr=False)
        assert set(file_params_nc4.coords.keys()) == set(
            file_params.coords.keys()
        )
        assert "nhm_seg" in file_params.coords.keys()
        # the process file and the dis_hru file together supply every
        # declared parameter
        missing = set(proc.get_parameters()) - file_names - dis_names
        assert not missing, f"{proc.__name__} file lacks {missing}"
        # the cascade parameters are not in the PRMS file; they must be here
        if proc in gw_cascade_processes:
            assert "gwr_route_order" in file_names
            assert "gwr_down" in file_names
        else:
            assert "hru_route_order" in file_names
            assert "hru_down" in file_names
        # every declared dimension is on file, used by a parameter or not
        missing_dims = set(proc.get_dimensions()) - set(file_params.dims)
        assert not missing_dims, f"{proc.__name__} file lacks {missing_dims}"
