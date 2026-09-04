import pytest
from prms_convert_to_netcdf import convert_csv_to_nc, convert_soltab_to_nc
from prms_diagnostic_variables import (
    diagnose_simple_vars_to_nc,
)

import pywatershed as pws

# This map is from PRMS names to PWS names.
# In PRMS seg_depth and seg_width are BOTH parameter and variable names. These
# variables are all claculated with a flow dependence, so renaming of the PRMS
# variables was performed for all 4.
rename_vars = {
    "seg_area": "seg_flow_area",
    "seg_depth": "seg_flow_depth",
    "seg_velocity": "seg_flow_velocity",
    "seg_width": "seg_flow_width",
    "strm_seg_in": "stream_seg_in",
}


@pytest.fixture
def netcdf_file(control_csv_file):
    """Convert CSV files from model output to NetCDF"""
    control_file = control_csv_file[0]
    csv_file = control_csv_file[1]

    var_name = csv_file.stem
    output_name = f"{var_name}.nc"
    rename = None
    if var_name in rename_vars.keys():
        rename = rename_vars[var_name]
        output_name = f"{rename}.nc"

    data_dir = csv_file.parent
    convert_csv_to_nc(var_name, data_dir, rename=rename)

    success = diagnose_simple_vars_to_nc(var_name, data_dir, control_file)

    if not success:
        assert False, "Unable to diagnose {var_name}"

    return output_name


def make_netcdf_files(netcdf_file):
    print(f"Created NetCDF from CSV: {netcdf_file}")


@pytest.fixture()
def soltab_netcdf_file(tmp_path_factory, control_soltab_file):
    """Convert soltab files to NetCDF, one file for each variable"""
    control_file = control_soltab_file[0]
    if "make_cbh_only" in control_file.stem:
        pytest.skip(f"Only generating CBH files with {control_file}")
    soltab_file = control_soltab_file[1]
    domain_dir = soltab_file.parent
    indiv_soltab_files = None

    if not soltab_file.exists():
        indiv_soltab_files = [
            "soltab_sunhrs.csv",
            "soltab_potsw.csv",
            "soltab_horad_potsw.csv",
        ]
        indiv_soltab_files = {
            ff[:-4]: domain_dir / ff for ff in indiv_soltab_files
        }
        all_indiv_files_exist = all(
            [ff.exists() for ff in indiv_soltab_files.values()]
        )
        if not all_indiv_files_exist:
            pytest.skip("No (or insufficient) soltab file(s) found.")

    control = pws.Control.load_prms(control_file, warn_unused_options=False)
    output_dir = control_file.parent / control.options["netcdf_output_dir"]

    if indiv_soltab_files is None:
        convert_soltab_to_nc(
            output_dir, control_file, domain_dir, soltab_file=soltab_file
        )
    else:
        convert_soltab_to_nc(
            output_dir,
            control_file,
            domain_dir,
            soltab_sunhrs_file=indiv_soltab_files["soltab_sunhrs"],
            soltab_potsw_file=indiv_soltab_files["soltab_potsw"],
            soltab_horad_potsw_file=indiv_soltab_files["soltab_horad_potsw"],
        )


def make_soltab_netcdf_files(soltab_netcdf_file):
    print(f"Creating NetCDF files for soltab file {soltab_netcdf_file}")
