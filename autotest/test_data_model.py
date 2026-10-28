import pathlib as pl
from copy import deepcopy

import netCDF4 as nc4
import numpy as np
import pandas as pd
import pytest
import xarray as xr

from pywatershed.base import data_model as dm
from pywatershed.base.data_model import DatasetDict


@pytest.fixture
def nc_file(simulation):
    return simulation["dir"] / "prcp.nc"


# TODO: add more attributes to variables
def mk_ds0():
    # a simple dataset adapted from xarray Dataset documentation
    np.random.seed(0)
    temperature = 15 + 8 * np.random.randn(2, 2, 3)
    precipitation = 10 * np.random.rand(2, 2, 3)
    precip_occurence = precipitation > 0.0
    precip_int = precipitation.astype(np.int32)
    lon = [[-99.83, -99.32], [-99.79, -99.23]]
    lat = [[42.25, 42.21], [42.63, 42.59]]
    time = pd.date_range("2014-09-06", periods=3).values
    reference_time = pd.Timestamp("2014-09-05")

    ds = xr.Dataset(
        data_vars=dict(
            temperature=(["x", "y", "time"], temperature),
            precipitation=(["x", "y", "time"], precipitation),
            precip_occurence=(["x", "y", "time"], precip_occurence),
            precip_int=(["x", "y", "time"], precip_int),
        ),
        coords=dict(
            lon=(["x", "y"], lon),
            lat=(["x", "y"], lat),
            time=time,
            reference_time=reference_time,
        ),
        attrs=dict(description="Weather related data."),
    )

    return ds


ds0 = mk_ds0()
xr_dd0 = ds0.to_dict(data="array", encoding=True)
dd0 = DatasetDict.from_ds(ds0)


def assert_encodings_identical(ds1, ds2):
    assert sorted(ds1.variables) == sorted(ds2.variables)
    for vv in ds1.variables:
        np.testing.assert_equal(ds1[vv].encoding, ds2[vv].encoding)


def coord_atts_sort(dd):
    # from xarray, coordinate order is not ordered AFAIK, make it alphabetical
    # cause that's what we are doing
    for ev in dd.encoding.values():
        for kk, vv in ev.items():
            if kk == "coordinates":
                ev[kk] = " ".join(sorted(vv.split(" ")))

    return dd


def del_encodings(dd, keys):
    for del_key in keys:
        for kk, vv in dd.encoding.items():
            if del_key in vv.keys():
                del vv[del_key]

    return dd


# helpers and data above
# -------------------------------------------------------
# tests below


def test_to_dd_identical(nc_file):
    dd_xr = dm.xr_ds_to_dd(nc_file)  # pass the file
    dd_nc4 = dm.nc4_ds_to_dd(nc_file)

    _ = dd_xr.pop("encoding")
    _ = dd_nc4.pop("encoding")

    np.testing.assert_equal(dd_xr, dd_nc4)


def test_xr_dd_xr(nc_file, tmp_path):
    tmp_path = pl.Path(tmp_path)

    # start with ds loaded: ds -> dd -> ds
    ds1 = xr.open_dataset(nc_file)
    ds2 = dm.dd_to_xr_ds(dm.xr_ds_to_dd(ds1))  # round trip from ds
    xr.testing.assert_identical(ds1, ds2)
    assert_encodings_identical(ds1, ds2)

    # start with ds as file: file -> ds -> dd -> ds
    ds3 = dm.dd_to_xr_ds(dm.xr_ds_to_dd(nc_file))  # round trip from file
    xr.testing.assert_identical(ds1, ds3)
    assert_encodings_identical(ds1, ds3)

    # start with a dd
    # round trip through a xr.dict: dd -> xr.dict -> dd
    dd1 = dm.xr_dd_to_dd(ds1.to_dict(data="array", encoding=True))
    dd2 = dm.xr_dd_to_dd(dm.dd_to_xr_dd(dd1))
    np.testing.assert_equal(dd1, dd2)

    # roundtrip a dd though xr.ds: dd -> ds -> dd
    dd3 = dm.xr_ds_to_dd(dm.dd_to_xr_ds(dd1))
    np.testing.assert_equal(dd1, dd3)

    # roundtrip a dd though netcdf file: dd -> ds -> netcdf file -> ds -> dd
    ff = tmp_path / "ds.nc"
    _ = dm.dd_to_xr_ds(dd1).to_netcdf(ff)
    dd4 = dm.xr_ds_to_dd(xr.open_dataset(ff))
    # The source file is obviously different
    for kk in dd1["encoding"].keys():
        del dd1["encoding"][kk]["source"]
        del dd4["encoding"][kk]["source"]

    return


def test_nc4_dd_nc4(nc_file, tmp_path):
    # pass the file and round trip it to file, compare via xarry
    out_file = pl.Path(tmp_path / "rt_1.nc")
    _ = dm.dd_to_nc4_ds(dm.nc4_ds_to_dd(nc_file), out_file)
    ds1 = xr.open_dataset(nc_file)
    ds2 = xr.open_dataset(out_file)
    xr.testing.assert_identical(ds1, ds2)

    # the source file was not the same
    # coordinates encoded on the coordinate variables in netcdf
    for kk in ["source", "coordinates"]:
        for vv in sorted(ds1.variables):
            for dd in [ds1, ds2]:
                if kk in dd[vv].encoding.keys():
                    del dd[vv].encoding[kk]

    assert_encodings_identical(ds1, ds2)

    # test passing a nc4.Dataset, cant not use_xr_encodings
    # round trip via file and compare in xarray
    out_file = pl.Path(tmp_path / "rt_2.nc")
    nc_file_ds = nc4.Dataset(nc_file)
    _ = dm.dd_to_nc4_ds(
        dm.nc4_ds_to_dd(nc_file_ds, use_xr_enc=False), out_file
    )
    ds1 = xr.open_dataset(nc_file)
    ds2 = xr.open_dataset(out_file)
    xr.testing.assert_identical(ds1, ds2)

    # nc4 encoding defaults dont match the source since we didnt read them
    # with xarray
    for kk in [
        "source",
        "zlib",
        "complevel",
        "shuffle",
        "contiguous",
        "chunksizes",
        "coordinates",
        "preferred_chunks",
    ]:
        for vv in sorted(ds1.variables):
            for dd in [ds1, ds2]:
                if kk in dd[vv].encoding.keys():
                    del dd[vv].encoding[kk]

    assert_encodings_identical(ds1, ds2)

    return


def dd_basics():
    # init
    # properties and accessors
    #  - return dicts are still editable, which is dangerous?
    # repr
    # str
    pass


def dd_spatial_coord_names():
    pass


@pytest.mark.domainless
def test_dd_netcdf(tmp_path):
    # xarray write, xarray read
    # ds0 -> dd0 -xr-> file -xr-> ds1(compare ds0) -> dd1(compare dd0)
    tmp_path = pl.Path(tmp_path)
    dd0 = DatasetDict.from_ds(ds0)  # from xarray ds
    ds1_file = tmp_path / "ds1_xr.nc"
    dd0.to_netcdf(ds1_file, use_xr=True)  # to file via xr
    ds1 = xr.open_dataset(ds1_file)
    xr.testing.assert_identical(ds1, ds0)  # ds -> file(via xr) -> ds
    dd1 = DatasetDict.from_ds(ds1)
    # there are no encodings on dd0 because it was never encoded
    dd1_noenc = deepcopy(dd1)
    for kk in dd1_noenc.encoding.keys():
        dd1_noenc.encoding[kk] = {}
    np.testing.assert_equal(dd1_noenc.data, dd0.data)
    del ds1  # close the file

    # nc4 write, xarray read
    # ds0 -> dd0 -nc4-> file -xr-> ds2(compare to ds0) -> dd2(compare)
    ds2_file = tmp_path / "ds2_nc4.nc"
    dd0.to_netcdf(ds2_file, use_xr=False)  # to file via nc4
    ds2 = xr.open_dataset(ds2_file)
    xr.testing.assert_identical(ds2, ds0)  # ds -> file(via nc4) -> ds
    dd2 = DatasetDict.from_ds(ds2)
    # from xarray, coordinate order is not ordered AFAIK, make it alphabetical
    dd1 = coord_atts_sort(dd1)
    # TODO: The _FillValue thing is all over the place, sort it out
    dd1 = del_encodings(dd1, ["source", "_FillValue"])
    dd2 = del_encodings(dd2, ["source", "_FillValue"])
    np.testing.assert_equal(dd2.data, dd1.data)
    del ds2

    # read using nc4
    # apparently netcdf4 doesnt do any coordinate level organization
    # so we'll have to implement this.
    dd3 = DatasetDict.from_netcdf(ds1_file, use_xr=False, encoding=True)
    dd3 = coord_atts_sort(dd3)
    dd3 = del_encodings(dd3, ["source", "_FillValue"])
    np.testing.assert_equal(dd3.data, dd1.data)

    return


@pytest.mark.domainless
def test_dd_subset_merge(tmp_path):
    sub1_keys = ["temperature"]
    sub2_keys = [kk for kk in dd0.data_vars.keys() if "precip" in kk]
    sub1 = dd0.subset(sub1_keys, keep_global=True)
    sub2 = dd0.subset(sub2_keys, keep_global=True)
    sub3 = dd0.subset("lon")

    dd_merge = DatasetDict.merge(sub1, sub2)
    np.testing.assert_equal(dd_merge.data, dd0.data)

    dd_merge = DatasetDict.merge(sub1, sub2, sub3)
    np.testing.assert_equal(dd_merge.data, dd0.data)

    # this passes because of refs
    sub3.coords["lon"][:] = sub3.coords["lon"] - 25.0
    dd_merge = DatasetDict.merge(sub1, sub2, sub3)
    np.testing.assert_equal(dd_merge.data, dd0.data)

    sub3 = dd0.subset("lon", copy=True)
    sub3.coords["lon"][:] = sub3.coords["lon"] - 25.0
    with pytest.raises(ValueError):
        dd_merge = DatasetDict.merge(sub1, sub2, sub3)

    return


@pytest.mark.domainless
def test_merge_dicts():
    dd_0 = {"a": {"aa": 123, "dims": {0: "zero", 1: "one"}}}
    dd_1 = {"a": {"aa": 123, "dims": {0: "dd_1"}}}
    dd_2 = {"a": {"aa": 123, "dims": {0: "dd_2"}}}

    with pytest.raises(ValueError):
        result = dm._merge_dicts([dd_0, dd_1])

    with pytest.raises(ValueError):
        result = dm._merge_dicts([dd_0, dd_1], conflicts="error")

    with pytest.warns(UserWarning):
        result = dm._merge_dicts([dd_0, dd_1], conflicts="warn")

    result = dm._merge_dicts([dd_0, dd_1], conflicts="left")
    assert set(result["a"].keys()) == set(dd_0["a"].keys())
    assert result == dd_0

    result = dm._merge_dicts([dd_0, dd_1, dd_2], conflicts="left")
    assert set(result["a"].keys()) == set(dd_0["a"].keys())
    assert result == dd_0

    result = dm._merge_dicts([dd_2, dd_1, dd_0], conflicts="left")
    assert set(result["a"].keys()) == set(dd_2["a"].keys())
    assert result == {"a": {"aa": 123, "dims": {0: "dd_2", 1: "one"}}}

    result = dm._merge_dicts([dd_1, dd_2, dd_0], conflicts="left")
    assert set(result["a"].keys()) == set(dd_0["a"].keys())
    assert result == {"a": {"aa": 123, "dims": {0: "dd_1", 1: "one"}}}

    return
