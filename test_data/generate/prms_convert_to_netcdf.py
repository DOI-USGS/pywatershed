import pathlib as pl
from typing import Union

import pywatershed as pws
from pywatershed import CsvFile, Soltab

"""This module is intended to aid/make consistent conversions of PRMS files"""

# PRMS output names translated to pywatershed variable names
rename_prms_vars = {"hru_hortn_cascflow": "hru_horton_cascflow"}


def convert_csv_to_nc(
    var_name: str,
    data_dir: pl.Path,
    output_dir: pl.Path = None,
    rename: str = None,
):
    """Convert PRMS CSV files to netcdf.

    Args:
        var_name: str name of the variable to create
        data_dir: where the csv file is found and the netcdf file will be
            written (could add argument to output to a differnt dir)
        output_dir: the directory into which the file is to be written.
        rename: if the input name and output/metadata name differ, this is the
            later.
    """
    if output_dir is None:
        output_dir = data_dir

    csv_path = data_dir / f"{var_name}.csv"

    if rename is None:
        nc_path = output_dir / f"{var_name}.nc"
        CsvFile(csv_path).to_netcdf(nc_path)
    else:
        nc_path = output_dir / f"{rename}.nc"
        CsvFile({rename: csv_path}).to_netcdf(nc_path)

    assert nc_path.exists()

    if var_name in rename_prms_vars.keys():
        # the renaming keeps the original name nc file and produces a new one
        new_var_name = rename_prms_vars[var_name]
        nc_path = output_dir / f"{new_var_name}.nc"
        CsvFile({new_var_name: csv_path}).to_netcdf(nc_path)
        assert nc_path.exists()


def convert_soltab_to_nc(
    output_dir: pl.Path,
    control_file: pl.Path,
    domain_dir: pl.Path,
    soltab_file: Union[pl.Path, None] = None,
    soltab_sunhrs_file: Union[pl.Path, None] = None,
    soltab_potsw_file: Union[pl.Path, None] = None,
    soltab_horad_potsw_file: Union[pl.Path, None] = None,
):
    """Convert soltab files to NetCDF, one file for each variable

    The inputs files are "soltab_debug" specifically written by the PRMS5.2.1
    in the pywatershed repository.

    Args:
        output_dir: pl.Path where the netcdf file output will be written
        control_file: pl.Path the contorl file that generated the output
        domain_dir: defaults to the parent dir of soltab_file, the pl.Path
            where domain files (parameter & control) for the domain can be
            found
        soltab_file: the pl.Path of the  soltab_debug file.
    """
    if domain_dir is None:
        domain_dir = soltab_file.parent

    control = pws.Control.load_prms(control_file, warn_unused_options=False)
    param_file = control_file.parent / control.options["parameter_file"]

    params = pws.parameters.PrmsParameters.load(param_file)
    nhm_ids = params.parameters["nhm_id"]

    if soltab_file is not None:
        soltab = Soltab(soltab_file=soltab_file, nhm_ids=nhm_ids)
    else:
        soltab = Soltab(
            soltab_sunhrs_file=soltab_sunhrs_file,
            soltab_potsw_file=soltab_potsw_file,
            soltab_horad_potsw_file=soltab_horad_potsw_file,
            nhm_ids=nhm_ids,
        )
    soltab.to_netcdf(output_dir=output_dir)
    print(f"Created NetCDF files from soltab file {soltab_file}:")

    for var in soltab.variables:
        nc_path = output_dir / f"{var}.nc"
        assert nc_path.exists()
