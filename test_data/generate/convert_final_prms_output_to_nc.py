import pytest
from prms_diagnostic_variables import (
    diagnose_final_vars_to_nc,
)

import pywatershed as pws

# We separate these files from the other convert_prms_output_to_nc.py files
# because those need to be completely finished before these are run.

# Also, dont run these tests in parallel.


@pytest.fixture(scope="session")
def final_netcdf_file(control_final_file):
    """Create NetCDF files that depend on multiple other NetCDFs"""

    control_file = control_final_file[0]
    if "make_cbh_only" in control_file.stem:
        pytest.skip(f"Only generating CBH files with {control_file}")
    final_file = control_final_file[1]
    domain_dir = control_file.parent
    var_name = final_file.name
    control = pws.Control.load_prms(control_file, warn_unused_options=False)
    output_dir = control_file.parent / control.options["netcdf_output_dir"]

    success = diagnose_final_vars_to_nc(
        var_name, output_dir, control_file, domain_dir
    )

    if not success:
        pytest.skip(
            f"make_final_netcdf_files: unable to diagnose {final_file}"
        )


@pytest.mark.xdist_group(name="finalgroup")
def make_final_netcdf_files(final_netcdf_file):
    print(f"Creating final NetCDF file {final_netcdf_file}")
