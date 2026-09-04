import os
import pathlib as pl
from enum import Enum
from typing import Union

import numpy as np
import pyPRMS as pp

# Environment variables
numba_num_threads = os.getenv("NUMBA_NUM_THREADS")
if numba_num_threads is None:
    numba_num_threads = 0
else:
    numba_num_threads = int(numba_num_threads)

# Typing constants
fileish = Union[str, pl.Path]
listish = Union[str, list, tuple]  # Todo deprecate to Typing.Iterable

# PRMS6 Constants module:
# https://github.com/nhm-usgs/prms/blob/6.0.0_dev/src/prmslib/misc/m_constants.f90

__pywatershed_root__ = pl.Path(__file__).parent

zero = np.zeros([1])[0]
one = np.ones([1])[0]
nan = np.nan
# the unit is required: NumPy >= 2.5 deprecates the generic ("NaT") form.
# ns matches the only use, starfit_parameters.py, which casts to "<M8[ns]".
nat = np.datetime64("NaT", "ns")


def nan1d():
    return np.zeros(1) * nan


epsilon = np.finfo(zero).eps
# https://en.wikipedia.org/wiki/Machine_epsilon
# use values slightly larger than the informal definition
epsilon64 = 2.23e-16  # epsilon
epsilon32 = 1.20e-07  # np.finfo(zero.astype("float32")).eps

# These are PRMS conventions, should not be used elsewhere
nearzero = 1.0e-6
dnearzero = epsilon64
closezero = epsilon32

fill_value_f4 = 9.96921e36

# work in progress...
# Default netcdf _FillValue by dtype (used by dd_to_nc4_ds). Ints have no
# default: an int _FillValue makes xarray promote the variable to float on
# read and turns legitimate sentinel values (e.g. -9999) into NaN.
fill_values_dict = {
    np.dtype("float64"): np.nan,
    np.dtype("float32"): np.nan,
    np.dtype("float16"): np.nan,
    np.dtype("int64"): None,
    np.dtype("int32"): None,
    np.dtype("int16"): None,
    np.dtype("int8"): None,
    np.dtype("bool"): None,
}

# In-memory fill values for masking inactive HRUs (HruMixin). Not a netcdf
# encoding default.
mask_fill_values_dict = {
    **fill_values_dict,
    np.dtype("int64"): -9999,
    np.dtype("int32"): -9999,
    np.dtype("int16"): -9999,
    np.dtype("int8"): -9999,
    np.dtype("bool"): False,
}

np_type_to_netcdf_type_dict = {
    np.dtype("float64"): "f8",
    np.dtype("float32"): "f4",
    np.dtype("int64"): "i8",
    np.dtype("int32"): "i4",
    np.dtype("int16"): "i2",
    np.dtype("int8"): "i1",
    np.dtype("uint64"): "u8",
    np.dtype("uint32"): "u4",
    np.dtype("uint16"): "u2",
    np.dtype("uint8"): "u1",
    np.dtype("bool"): None,
}

# Map variable type strings from metadata YAML to numpy dtypes
var_type_to_numpy_type = {
    "float64": np.float64,
    "float32": np.float32,
    "int64": np.int64,
    "int32": np.int32,
    "int16": np.int16,
    "int8": np.int8,
    "bool": np.bool_,
}

inch2cm = 2.54
ft2_per_acre = 43560.0
inches_per_foot = 12.0
cms_to_cfs = 35.314666721489
cfs_to_cms = 1 / cms_to_cfs
cm_to_cf = cms_to_cfs
cf_to_cm = cfs_to_cms
cubic_ft_per_acre_in = ft2_per_acre / inches_per_foot


ndoy = 366
nmonth = 12

INACTIVE = 0
ACTIVE = 1


class HruType(Enum):
    """HRU Type
    INACTIVE = 0
    LAND = 1
    LAKE = 2
    SWALE = 3
    GLACIER = 4
    """

    INACTIVE = 0
    LAND = 1
    LAKE = 2
    SWALE = 3
    GLACIER = 4


class CovType(Enum):
    BARESOIL = 0
    GRASSES = 1
    SHRUBS = 2
    TREES = 3
    CONIFEROUS = 4


class SoilType(Enum):
    SAND = 1
    LOAM = 2
    CLAY = 3


class ETType(Enum):
    ET_DEFAULT = 1
    EVAP_ONLY = 2
    EVAP_PLUS_TRANSP = 3


class SegmentType(Enum):
    SEGMENT = 0
    HEADWATER = 1
    LAKE = 2
    REPLACEINFLOW = 3
    INBOUNDNHM = 4
    OUTBOUNDNHM = 5
    INBOUNDREGION = 6
    OUTBOUNDREGION = 7
    OUTBOUNDOCEAN = 8
    SINK = 9
    INBOUNDGREATLAKES = 10
    OUTBOUNDGREATLAKES = 11


# PRMS naming conventions

# The keys/names for CBH variables in the control files dont match the internal
# variable names. It's vague to me what these are in the headers of the CBH
# files, it's also a bit vague in the tables.
# One place it is partially documented in pyPRMS
# https://github.com/DOI-USGS/pyPRMS/blob/49cbb8cd46b6760b1be67c106b8074688abaab39/tests/func/test_Control/ctl_metadata_default.csv#L96
cbh_ctl_var_map = {
    "albedo_day": "albedo_hru",
    "cloud_cover_day": "cloud_cover_cbh",
    "humidity_day": "humidity_hru",
    "potet_day": "potet",
    "precip_day": "prcp",
    "swrad_day": "swrad",
    "tmax_day": "tmax",
    "tmin_day": "tmin",
    "transp_day": "transp_on",
    "windspeed_day": "windspeed_hru",
    "AET_cbh_file": "aet_observed",
    "PET_cbh_file": "pet_observed",  # not used in PWS but required for subsetting domains  # noqa: E501
    "rhavg": "rhavg",
}

# pyPRMS metadata - consolidated in one place
pyprms_meta = pp.MetaData(verbose=False).metadata
