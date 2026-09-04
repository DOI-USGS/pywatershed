import numpy as np

import pywatershed

from ..constants import fileish
from ..parameters import PrmsParameters

"""This utility defines how PRMS parameter files are separated

PRMS parameter files are separated to individual processes defined in
pywatershed and into discretization parameter sets. This file is how this
separation is defined and carried out.

"""

# these are prarameters that are provided as scalars which we will
# force expand to their full dimensions
params_expand_scalar = [
    "obsout_segment",
    "seg_humidity",
    "width_m",
    "lat_temp_adj",
    "stream_tave_init",
    "gw_tau",
    "ss_tau",
    "alte",
    "altw",
    "voe",
    "vow",
    "hru_area",
    "hru_percent_imperv",
    "hru_tsta",
    "hru_psta",
    "pref_flow_den",
    "pref_flow_infil_frac",
    "sat_threshold",
    "tmax_allsnow",
    "tmax_allrain_offset",
    "adjmix_rain",
    "dday_slope",
    "dday_intcp",
    "radmax",
    "radadj_intcp",
    "radadj_slope",
    "tmax_adj",
    "tmin_adj",
    "tmax_index",
    "tmax_lapse",
    "tmin_lapse",
    "ppt_rad_adj",
    "jh_coef",
    "transp_beg",
    "transp_end",
    "transp_tmax",
    "srain_intcp",
    "wrain_intcp",
    "snow_intcp",
    "snowinfil_max",
    "melt_force",
    "melt_look",
    "carea_max",
    "smidx_exp",
    "smidx_coef",
    "imperv_stor_max",
    "soil2gw_max",
    "freeh2o_cap",
    "settle_const",
    "hru_deplcrv",
    "cecn_coef",
    "emis_noppt",
    "tstorm_mo",
    "potet_sublim",
    "slowcoef_lin",
    "slowcoef_sq",
    "fastcoef_lin",
    "fastcoef_sq",
    "ssr2gw_exp",
    "rain_cbh_adj",
    "snow_cbh_adj",
    "tmin_cbh_adj",
    "tmax_cbh_adj",
]

expand_hru_to_monthly = [
    "tmax_adj",
    "tmin_adj",
]

var_meta_to_attrs = [
    "default",
    "desc",
    "help",
    "maximum",
    "minimum",
    "units",
]

dis_hru_vars = [
    "hru_area",
    "hru_aspect",
    "hru_elev",
    "hru_in_to_cf",
    "hru_lat",
    "hru_lon",
    "hru_slope",
    "hru_type",
    "nhm_id",
]

dis_seg_vars = [
    "lat_temp_adj",
    "nhm_seg",
    "poi_gage_segment",
    "seg_cum_area",
    "seg_depth",
    "seg_elev",
    "seg_lat",
    "seg_length",
    "seg_slope",
    "seg_width",
    "segment_type",
    "tosegment",
    "tosegment_nhm",
]


def shape(val):
    if isinstance(val, np.ndarray):
        return val.shape
    elif isinstance(val, (int, float)):
        return 1
    else:
        return len(val)


def separate_domain_params_dis_to_ncdf(
    prms_param_file: fileish,
    domain_name: str,
    out_dir: fileish,
    process_list: list = None,
    use_xr=True,
) -> dict:
    """Separate PRMS parameters into discretizations and individual processes

    This separates the PRMS parameter file into two discretization files
    (dis_hru and dis_seg) and parameter files for individual processes defined
    in pywatershed.

    Args:
        prms_param_file: the native PRMS parameter file to separate
        domain_name: string name to include in the output files, None includes
            no additional string.
        out_dir: the directory to which output files will be writen
        process_list: optional, the list of process classes desired for
            individual output files. If not specified, all process classes
            will be assumed.

    Returns:
        A dictionary of `process_class: file` and `dis_name: file` pairs
        corresponding to the files written for (requested) processes in
        process_list.

    """
    nhm_processes = [
        pywatershed.PRMSSolarGeometry,
        pywatershed.PRMSAtmosphere,
        pywatershed.PRMSCanopy,
        pywatershed.PRMSSnow,
        pywatershed.PRMSRunoff,
        pywatershed.PRMSRunoffAg,
        pywatershed.PRMSRunoffCascadesNoDprst,
        pywatershed.PRMSRunoffNoDprst,
        pywatershed.PRMSSoilzone,
        pywatershed.PRMSSoilzoneAg,
        pywatershed.PRMSSoilzoneCascadesNoDprst,
        pywatershed.PRMSSoilzoneNoDprst,
        pywatershed.PRMSGroundwater,
        pywatershed.PRMSGroundwaterNoDprst,
        pywatershed.PRMSChannel,
        pywatershed.PRMSHydraulicGeometryFull,
        pywatershed.PRMSHydraulicGeometryWidthOnly,
        pywatershed.PRMSStreamShadeDynamic,
        pywatershed.PRMSStreamTemp,
        pywatershed.PRMSStreamTempHumidityCBH,
    ]
    if process_list is None:
        process_list = nhm_processes

    prms_parameters = PrmsParameters.load(prms_param_file)

    written_files = {}

    if domain_name is None:
        domain_name = ""
    else:
        domain_name = f"{domain_name}_"

    for proc in process_list:
        dis_param_names = set(dis_hru_vars + dis_seg_vars)
        proc_param_names = set(proc.get_parameters())
        proc_params_no_dis_names = proc_param_names.difference(dis_param_names)
        proc_params = prms_parameters.subset(proc_params_no_dis_names)
        print(proc, proc_params_no_dis_names)
        nc_out_file = out_dir / f"parameters_{domain_name}{proc.__name__}.nc"
        proc_params.to_netcdf(nc_out_file, use_xr=use_xr)
        written_files[proc] = nc_out_file

    dis_dict = {"dis_hru": dis_hru_vars, "dis_seg": dis_seg_vars}
    for dis_name, dis_var_names in dis_dict.items():
        dis_params = prms_parameters.subset(dis_var_names)
        nc_out_file = out_dir / f"parameters_{domain_name}{dis_name}.nc"
        dis_params.to_netcdf(nc_out_file, use_xr=use_xr)
        written_files[dis_name] = nc_out_file

    # dis_both is both combined since PRMSchannel has to know about hrus & segs
    # and it seems each process should only know about a single dis_both
    # when we introduce exchanges, we can just use dis_seg for channel.

    dis_both = PrmsParameters.merge(
        prms_parameters.subset(dis_dict["dis_hru"]),
        prms_parameters.subset(dis_dict["dis_seg"]),
    )
    nc_out_file = out_dir / f"parameters_{domain_name}dis_both.nc"
    dis_both.to_netcdf(nc_out_file, use_xr=use_xr)
    written_files["dis_both"] = nc_out_file

    return written_files
