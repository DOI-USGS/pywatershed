import pathlib as pl
from warnings import warn

import numpy as np
import xarray as xr

import pywatershed as pws
from pywatershed.constants import dnearzero, nan, nearzero, zero

"""This module is for generating PRMS diagnostic variables"""

# For generating timeseries of previous states
prev_vars_both = {
    "gwres_stor": pws.PRMSGroundwater,
    "hru_intcpstor": pws.PRMSCanopy,
    "freeh2o": pws.PRMSSnow,
    "pk_ice": pws.PRMSSnow,
}

previous_vars = prev_vars_both | {
    "dprst_stor_hru": pws.PRMSRunoff,
    "hru_impervstor": pws.PRMSRunoff,
    "ag_soil_moist": pws.PRMSSoilzoneAg,
    "ag_soil_rechr": pws.PRMSSoilzoneAg,
    "pref_flow_stor": pws.PRMSSoilzone,
    "slow_stor": pws.PRMSSoilzone,
    "soil_lower": pws.PRMSSoilzone,
    "soil_rechr": pws.PRMSSoilzone,
    "ssres_stor": pws.PRMSSoilzone,
}

previous_vars_obs_et = previous_vars | {
    "ag_soil_moist": pws.PRMSSoilzoneAgObsET,
    "ag_soil_rechr": pws.PRMSSoilzoneAgObsET,
    "pref_flow_stor": pws.PRMSSoilzoneAgObsET,
    "slow_stor": pws.PRMSSoilzoneAgObsET,
    "soil_lower": pws.PRMSSoilzoneAgObsET,
    "soil_rechr": pws.PRMSSoilzoneAgObsET,
    "ssres_stor": pws.PRMSSoilzoneAgObsET,
}

previous_vars_no_dprst = prev_vars_both | {
    "hru_impervstor": pws.PRMSRunoffNoDprst,
    "pref_flow_stor": pws.PRMSSoilzoneNoDprst,
    "slow_stor": pws.PRMSSoilzoneNoDprst,
    "soil_lower": pws.PRMSSoilzoneNoDprst,
    "soil_rechr": pws.PRMSSoilzoneNoDprst,
    "ssres_stor": pws.PRMSSoilzoneNoDprst,
}

prev_rename = {
    "gwres_stor": "gwres_stor_old",
    "hru_impervstor": "hru_impervstor_old",
    "hru_intcpstor": "hru_intcpstor_old",
    "dprst_stor_hru": "dprst_stor_hru_old",
}

change_rename = {}


def diagnose_simple_vars_to_nc(
    var_name: str,
    data_dir: pl.Path,
    control_file: pl.Path,
    domain_dir: pl.Path = None,
    output_dir: pl.Path = None,
) -> bool:
    """Diagnose variables from PRMS output with only dependencies on var_name

    These variables are "simple" because they do not have any dependencies
    besides the "var_name" variable which is a PRMS output variable.

    This is not to say that PRMS dosent output diagnostic variables, it
    just dosent output these natively.

    Args:
        var_name: str name of the variable to create
        data_dir: where the netcdf file will be written, also where to look
            for necessary inputs to create the diagnostic variable.
        control_file: the pathlib representation of the control file for
            the run that produced the data.
        domain_dir: defaults to the parent dir of output_dir, this is where
            domain files for the domain can be found
        output_dir: defaults to data_dir unless otherwise specified

    Returns:
        Boolean for success.

    """

    if domain_dir is None:
        domain_dir = data_dir.parent

    if output_dir is None:
        output_dir = data_dir

    nc_path = data_dir / f"{var_name}.nc"
    control = pws.Control.load_prms(
        control_file, warn_unused_options=False, keep_unused_options=True
    )
    param_file = control_file.parent / control.options["parameter_file"]
    if "executable_desc" in control.options.keys():
        exe_desc = control.options["executable_desc"][0].lower()
    else:
        exe_desc = "prms"

    if var_name in previous_vars.keys():
        # the _prev is the desired suffix but PRMS legacy is inconsistent
        current = xr.open_dataarray(nc_path)
        prev = current * nan
        prev[:] = np.roll(current.values, 1, axis=0)

        # the final value (-1) was wrapped to the zeroth position
        # get the initial conditions for the first time by initializing the
        # model This works based on the control file, so could handle restart.
        control.options = control.options | {
            "input_dir": domain_dir / "output",
        }
        params = pws.parameters.PrmsParameters.load(param_file)

        if (
            "iter_aet_flag" in control.options.keys()
            and control.options["iter_aet_flag"]
        ):
            proc_class = previous_vars_obs_et[var_name]
        elif (
            "dprst_flag" in control.options.keys()
            and control.options["dprst_flag"]
        ):
            proc_class = previous_vars[var_name]
        else:
            proc_class = previous_vars_no_dprst[var_name]

        inputs = {}

        for input_name in proc_class.get_inputs():
            # taken from process._initialize_var
            meta = pws.meta.find_variables([input_name])
            dims = [params.dims[vv] for vv in meta[input_name]["dims"]]
            init_type = meta[input_name]["type"]

            if len(dims) == 1:
                inputs[input_name] = np.full(dims, nan, dtype=init_type)
            else:
                inputs[input_name] = pws.base.timeseries.TimeseriesArray(
                    var_name=input_name,
                    control=control,
                    array=np.full(dims, nan, dtype=init_type),
                    time=np.arange(
                        np.datetime64("2017-01-01"),
                        np.datetime64("2017-01-03"),
                    ),
                )

        proc = proc_class(
            control=control,
            discretization=None,
            parameters=params,
            **inputs,
        )
        # these are the init_values
        prev[0, :] = proc[var_name]
        del proc

        change = current - prev

        # write the previous file
        if var_name in prev_rename.keys():
            nc_name = f"{prev_rename[var_name]}.nc"
        else:
            nc_name = f"{var_name}_prev.nc"
        out_nc_path = output_dir / nc_name
        prev.rename(out_nc_path.stem).to_netcdf(out_nc_path)
        assert nc_path.exists()

        # write the change file (skip if validated by mass budget)
        if var_name in change_rename.keys():
            nc_name = f"{change_rename[var_name]}.nc"
        else:
            nc_name = f"{var_name}_change.nc"
        out_nc_path = output_dir / nc_name
        change.rename(out_nc_path.stem).to_netcdf(out_nc_path)
        assert nc_path.exists()

        return nc_path

    if var_name in ["sroff", "ssres_flow", "gwres_flow"]:
        params = pws.parameters.PrmsParameters.load(param_file)
        ds = xr.open_dataset(nc_path)
        ds = ds.rename({var_name: f"{var_name}_vol"})
        if "gsflow" in exe_desc.lower():
            # Zero out tiny values before multiplying by large conversion
            # factor
            ds[f"{var_name}_vol"] = ds[f"{var_name}_vol"].where(
                np.abs(ds[f"{var_name}_vol"]) > 1e-11, 0.0
            )
        ds = ds * params.data_vars["hru_in_to_cf"]
        ds.to_netcdf(output_dir / f"{var_name}_vol.nc")
        ds.close()
    return True


def diagnose_final_vars_to_nc(
    var_name: str,
    data_dir: pl.Path,
    control_file: pl.Path,
    domain_dir: pl.Path = None,
    output_dir: pl.Path = None,
) -> bool:
    """Diagnose variables from multiple PRMS outputs

    In this case "var_name" is not a PRMS output variable, the variable is a
    new variable which requires much/all of the PRMS output to be already
    present in netcdf output format. This is why this is a final diagnostic

    Args:
        var_name: str name of the variable to create, not a PRMS variable.
        output_dir: where the netcdf file will be written, also where to look
            for necessary inputs to create the diagnostic variable.
        domain_dir: defaults to the parent dir of output_dir, this is where
            domain files (parameter & control) for the domain can be found

    Returns:
        Boolean for success.

    """
    if domain_dir is None:
        domain_dir = data_dir.parent

    if output_dir is None:
        output_dir = data_dir

    control = pws.Control.load_prms(
        control_file, warn_unused_options=False, keep_unused_options=True
    )
    param_file = control_file.parent / control.options["parameter_file"]

    if var_name == "through_rain":
        out_file = output_dir / "through_rain.nc"

        data_vars = [
            "net_ppt",
            "pptmix_nopack",
            "snowmelt",
            "pkwater_equiv",
            "pk_ice_change",
            "freeh2o_change",
            "snow_evap",
            "net_snow",
            "net_rain",
        ]

        data = {}
        for vv in data_vars:
            data_file = data_dir / f"{vv}.nc"
            try:
                data[vv] = xr.open_dataarray(data_file)
            except FileNotFoundError:
                warn(f"diagnose_final_vars_to_nc, file not found: {data_file}")
                return False

        cond1 = data["net_ppt"] > zero
        cond2 = data["pptmix_nopack"] != 0
        cond3 = data["snowmelt"] < nearzero
        cond4 = data["pkwater_equiv"] < dnearzero
        cond5 = data["snow_evap"] < nearzero
        cond6 = data["net_snow"] < nearzero
        cond7 = data["snow_evap"] > -1 * (
            data["pk_ice_change"] + data["freeh2o_change"]
        )

        through_rain = data["net_rain"] * zero
        # these are in reverse order
        through_rain[:] = np.where(
            cond1 & cond3 & cond4 & cond6, data["net_rain"], zero
        )
        through_rain[:] = np.where(
            cond1 & cond3 & cond4 & cond5, data["net_ppt"], through_rain
        )
        through_rain[:] = np.where(
            cond1 & cond2, data["net_rain"], through_rain
        )

        # This condition does not exist in PRMS as far as I can tell
        # but is necessary for mass balance
        # This is when it rains on snow (no new snow) and then snow_evap
        # consumes the pack during the timestep.
        through_rain[:] = np.where(
            cond1 & cond6 & cond7,
            zero,
            through_rain,
        )

        through_rain.to_dataset(name="through_rain").to_netcdf(out_file)
        through_rain.close()

        for vv in data_vars:
            data[vv].close()

        assert out_file.exists()

    if var_name == "infil":
        params = pws.parameters.PrmsParameters.load(param_file).parameters
        imperv_frac = params["hru_percent_imperv"]
        if (
            "dprst_flag" in control.options.keys()
            and control.options["dprst_flag"]
        ):
            dprst_frac = params["dprst_frac"]
        else:
            dprst_frac = zero
        perv_frac = 1.0 - imperv_frac - dprst_frac

        # Check if agriculture is active
        soilzone_module = control.options.get("soilzone_module", [None])[0]
        ag_active = soilzone_module == "soilzone_ag"

        ds = xr.open_dataset(data_dir / "infil.nc")
        ds = ds.rename(infil="infil_hru")

        if ag_active:
            # Load infil_ag and ag_frac
            # Check if ag_frac is dynamic
            dyn_ag_frac_flag = control.options.get("dyn_ag_frac_flag", [0])[0]

            if dyn_ag_frac_flag:
                ag_frac_file = (
                    control_file.parent / control.options["ag_frac_dynamic"][0]
                )
                if ag_frac_file.exists():
                    ag_frac = pws.utils.PrmsDynamicParameter.load(
                        ag_frac_file, control=control
                    ).daily_data_array.rename(nhru="nhm_id")
                else:
                    raise FileNotFoundError(
                        f"Dynamic ag_frac file not found: {ag_frac_file}"
                    )
            else:
                # Static ag_frac from parameters
                ag_frac = params["ag_frac"]

            perv_frac = 1.0 - imperv_frac - dprst_frac - ag_frac

            infil_ag_file = data_dir / "infil_ag.nc"
            if not infil_ag_file.exists():
                raise FileNotFoundError(f"File {infil_ag_file} not found")
            else:
                infil_ag = xr.open_dataarray(infil_ag_file)

            # <
            ds["infil_hru"] = ds["infil_hru"] * perv_frac + (
                infil_ag * ag_frac
            )
            infil_ag.close()
        else:
            ds["infil_hru"] = ds["infil_hru"] * perv_frac

        ds.to_netcdf(output_dir / "infil_hru.nc")
        ds.close()

    # The rest of the conversion is on ly for muskingum_mann variables
    if (
        "streamflow_module" not in control.options.keys()
        or control.options["streamflow_module"] != "muskingum_mann"
    ):
        return True

    if var_name == "seg_lateral_inflow":
        data_vars = [
            "sroff_vol",
            "ssres_flow_vol",
            "gwres_flow_vol",
            "seg_outflow",
            "seg_inflow",
        ]
        data = {}
        for vv in data_vars:
            data_file = data_dir / f"{vv}.nc"
            try:
                data[vv] = xr.open_dataarray(data_file)
            except FileNotFoundError:
                warn(f"diagnose_final_vars_to_nc, file not found: {data_file}")
                return False

        control = pws.Control.load_prms(
            control_file, warn_unused_options=False
        )
        s_per_time = control.time_step_seconds
        params = pws.parameters.PrmsParameters.load(param_file)

        ntimes = control.n_times
        nseg = params.dims["nsegment"]
        nhru = params.dims["nhru"]
        hru_segment = params.parameters["hru_segment"] - 1
        to_segment = (params.parameters["tosegment"] - 1).astype("int64")

        outflow_mask = np.full((len(to_segment)), False)
        for iseg in range(nseg):
            if to_segment[iseg] < 0:
                outflow_mask[iseg] = True
                continue

        seg_stor_change = (
            data["seg_inflow"] - data["seg_outflow"]
        ) * s_per_time

        channel_outflow_vol = (
            np.where(outflow_mask, data["seg_outflow"], zero)
        ) * s_per_time

        seg_lateral_inflow = np.zeros((ntimes, nseg))
        channel_sroff_vol = np.zeros((ntimes, nhru))
        channel_ssres_flow_vol = np.zeros((ntimes, nhru))
        channel_gwres_flow_vol = np.zeros((ntimes, nhru))

        for ihru in range(nhru):
            iseg = hru_segment[ihru]
            if iseg < 0:
                # This is bad, selective handling of fluxes is not cool,
                # mass is being discarded in a way that has to be coordinated
                # with other parts of the code.
                # This code shuold be removed evenutally.
                continue

            else:
                channel_sroff_vol[:, ihru] = data["sroff_vol"].values[:, ihru]
                channel_ssres_flow_vol[:, ihru] = data[
                    "ssres_flow_vol"
                ].values[:, ihru]
                channel_gwres_flow_vol[:, ihru] = data[
                    "gwres_flow_vol"
                ].values[:, ihru]

            # cubicfeet to cfs
            lateral_inflow = (
                channel_sroff_vol[:, ihru]
                + channel_ssres_flow_vol[:, ihru]
                + channel_gwres_flow_vol[:, ihru]
            ) / (s_per_time)

            seg_lateral_inflow[:, iseg] += lateral_inflow

        for hvar in [
            "channel_sroff_vol",
            "channel_ssres_flow_vol",
            "channel_gwres_flow_vol",
            "seg_lateral_inflow",
            "seg_stor_change",
            "channel_outflow_vol",
        ]:
            if hvar in [
                "seg_lateral_inflow",
                "seg_stor_change",
                "channel_outflow_vol",
            ]:
                dum = data["seg_outflow"].rename(hvar) * np.nan
            else:
                dum = data["sroff_vol"].rename(hvar) * np.nan

            dum[:, :] = locals()[hvar]
            out_file = data_dir / f"{hvar}.nc"
            dum.to_netcdf(out_file)
            print(out_file)
            assert out_file.exists()

    return True
