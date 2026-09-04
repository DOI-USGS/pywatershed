import warnings
from copy import deepcopy
from typing import Iterable, Literal

import cftime
import netCDF4 as nc4
import numpy as np
import xarray as xr

from ..constants import fileish, fill_values_dict, np_type_to_netcdf_type_dict
from .accessor import Accessor

# This file defines the data model for pywatershed. It is called a
# "dataset_dict" and has a invertible mapping with non-hierarchical netcdf
# or xarray datasets.

# Show a basic example of how an xr_dd is changed to a dd.
# What is the difference


# TODO: what about hierarchical/groups in netcdf files?


# This is what a dataset_dict looks like. Metadata for coord and data_vars
# is found in metadata.
# These must always be deepcopied.
template_dd = {
    "dims": {},
    "coords": {},
    "data_vars": {},
    "metadata": {},
    "encoding": {},
}

template_xr_dd = {
    "attrs": {},
    "dims": {},
    "coords": {},
    "data_vars": {},
    "encoding": {},
}


# Note: Methods do not deep copy by default, but not all references may be
# preserved. use with caution and test.


class DatasetDict(Accessor):
    """DatasetDict: a data model following NetCDF-like conventions

    This is the core class in the data model adopted by pywatershed.

    The DatasetDict handles dimensions, coordinates, data, and metadata in
    a way like `NetCDF <https://www.unidata.ucar.edu/software/netcdf/>`__ and
    `xarray <https://docs.xarray.dev/en/stable/>`__  and provides invertible
    mappings between both the
    `netCDF4 <https://unidata.github.io/netcdf4-python/>`__ and
    `xarray <https://docs.xarray.dev/en/stable/>`__ Python packages.

    Where metadata is typically stored on a variable in NetCDF and in xarray,
    a DatasetDict maintains metadata in dictionary collocated with coordinate
    and data variables. The data model is a DatasetDict with dims, coords,
    data_vars, and metadata keys. The dims track the length of each dimension.
    The coordinates are the discrete locations along dimensions or sets of
    dimensions. The data_vars contain the data located on dims and
    coordinates. The metadata describes the relationship between both coords
    and data_vars and their dims. Together the coords and data_vars are the
    variables of the DatasetDict. All keys in the variables must be present
    in the metadata dictionary and each ke contains two more keys: dims and
    attrs. The dims is a tuple of the variable's dimensions and attrs are
    more general attributes.

    When a NetCDF file is read from disk, it has encoding properties that may
    come along. Alternatively, encodings may be specified before writing to
    file.

    Args:
        dims: A dictionary of pairs of `dim_names: dim_len` where `dim_len` is
            an integer value.
        coords: A dictionary of pairs of `coord_names: coord_data` where
            `coord_data` is an np.ndarray.
        data_vars: A dictionary of pairs of `var_names: var_data` where
            `coord_data` is an np.ndarray.
        metadata: For all names in `coords` and `data_vars`, metadata entries
            with the required fields:

            - dims: tuple of names in dim,
            - attrs: dictionary whose values may be strings, ints, floats

            The metadata argument may also contain a special `global` key
            paired with a dictionary of global metadata of arbitrary name and
            values of string, integer, or float types.
        encoding: (to document)
        validate: A bool that defaults to True, enforcing the consistency
            of the supplied dictionaries


    See Also
    --------
    pywatershed.Parameters


    Examples
    ---------

    ..
        # This code is commented, copy and paste in to python, then paste the
        # output below to keep it clean
        from pprint import pprint
        import pywatershed as pws
        import numpy as np
        coords = {
            'time': np.arange(
                '2005-02-01', '2005-02-03', dtype='datetime64[D]'
            ),
            'space': np.arange(3)
        }
        dims = {'ntime': len(coords['time']), 'nspace': len(coords['space'])}
        data = {'precip': 10 * np.random.rand(dims['ntime'], dims['nspace'])}
        metadata = {
            "time": {"dims": ("ntime",), "attrs": {"description": "days"}},
            "space": {
                "dims": ("nspace",),
                "attrs": {"description": "points of interest"},
            },
            "precip": {
                "dims": (
                    "ntime",
                    "nspace",
                ),
                "attrs": {
                    "description": "precipitation rate of all phases",
                    "units": "mm/day",
                },
            },
        }
        dd = pws.base.DatasetDict(
            dims=dims, coords=coords, data_vars=data, metadata=metadata
        )
        dd.dims.keys()
        dd.variables.keys()
        ds = dd.to_xr_ds()
        print(ds)


    >>> from pprint import pprint
    >>> import pywatershed as pws
    >>> import numpy as np
    >>> coords = {
    ...     "time": np.arange(
    ...         "2005-02-01", "2005-02-03", dtype="datetime64[D]"
    ...     ),
    ...     "space": np.arange(3),
    ... }
    >>> dims = {"ntime": len(coords["time"]), "nspace": len(coords["space"])}
    >>> data = {"precip": 10 * np.random.rand(dims["ntime"], dims["nspace"])}
    >>> metadata = {
    ...     "time": {"dims": ("ntime",), "attrs": {"description": "days"}},
    ...     "space": {
    ...         "dims": ("nspace",),
    ...         "attrs": {"description": "points of interest"},
    ...     },
    ...     "precip": {
    ...         "dims": (
    ...             "ntime",
    ...             "nspace",
    ...         ),
    ...         "attrs": {
    ...             "description": "precipitation rate of all phases",
    ...             "units": "mm/day",
    ...         },
    ...     },
    ... }
    >>> dd = pws.base.DatasetDict(
    ...     dims=dims, coords=coords, data_vars=data, metadata=metadata
    ... )
    >>> dd.dims.keys()
    dict_keys(['ntime', 'nspace'])
    >>> dd.variables.keys()
    dict_keys(['time', 'space', 'precip'])
    >>> ds = dd.to_xr_ds()
    >>> print(ds)
    <xarray.Dataset>
    Dimensions:  (ntime: 2, nspace: 3)
    Coordinates:
        time     (ntime) datetime64[ns] 2005-02-01 2005-02-02
        space    (nspace) int64 0 1 2
    Dimensions without coordinates: ntime, nspace
    Data variables:
        precip   (ntime, nspace) float64 8.835 5.667 9.593 7.239 3.92 0.4195

    """

    def __init__(
        self,
        dims: dict[int] = None,
        coords: dict = None,
        data_vars: dict = None,
        metadata: dict = None,
        encoding: dict = None,
        validate: bool = True,
    ) -> None:
        if dims is None:
            dims = {}
        if coords is None:
            coords = {}
        if data_vars is None:
            data_vars = {}
        if metadata is None:
            metadata = {}
        if encoding is None:
            encoding = {}

        self._data_vars = data_vars
        self._dims = dims
        self._coords = coords
        self._metadata = metadata
        self._encoding = encoding

        for cat_data in ["_data_vars", "_coords"]:
            for kk, vv in self[cat_data].items():
                if not isinstance(vv, np.ndarray):
                    msg = f"Coercing {cat_data[1:]}: {kk} to an np.ndarray."
                    warnings.warn(msg)
                    self[cat_data][kk] = np.array(vv)

        if "global" not in self._metadata.keys():
            self._metadata["global"] = {}

        if validate:
            self.validate()

        return

    @property
    def dims(self, copy=False) -> dict:
        """Return the dimensions"""
        if copy:
            return deepcopy(self._dims)
        return self._dims

    @property
    def coords(self, copy=False) -> dict:
        """Return the coordinates"""
        if copy:
            return deepcopy(self._coords)
        return self._coords

    @property
    def data_vars(self, copy=False) -> dict:
        """Return the data_vars."""
        if copy:
            return deepcopy(self._data_vars)
        return self._data_vars

    @property
    def variables(self, copy=False) -> dict:
        """Return coords and data_vars together"""
        vars = {**self._coords, **self._data_vars}
        if copy:
            vars = deepcopy(vars)
        return vars

    @property
    def metadata(self, copy=False) -> dict:
        """Return the metadata"""
        if copy:
            return deepcopy(self._metadata)
        return self._metadata

    @property
    def encoding(self, copy=False) -> dict:
        """Return the encoding"""
        if copy:
            return deepcopy(self._encoding)
        return self._encoding

    @property
    def data(self, copy=False) -> dict:
        """Return a dict of dicts: dims, coords, data_vars, metadata, encoding

        Args:
            copy: boolean if a deepcopy is desired

        Returns:
            A dict of dicts containing all the data
        """
        data = {
            "dims": self._dims,
            "coords": self._coords,
            "data_vars": self._data_vars,
            "metadata": self._metadata,
            "encoding": self._encoding,
        }
        if copy:
            return deepcopy(data)
        return data

    def _keys(self) -> list:
        return ["dims", "coords", "data_vars", "metadata", "encoding"]

    # are __repr__ and __str__ better than default?
    # def __repr__(self):
    #     return pprint.pformat(
    #         {
    #             "dims": self.dims,
    #             "coords": self.coords,
    #             "data_vars": self.data_vars,
    #             "metadata": self.metadata,
    #             "encoding": self.encoding,
    #         }
    #     )

    # def __str__(self):
    #     return

    @classmethod
    def from_dict(cls, dict_in, copy=False):
        """Return this class from a passed dictionary.

        Parameters
        ----------
        dict_in : dict
            A dictionary from which to create an instance of this class
        copy : bool, optional
            If True, the passed dictionary will be deep copied. Default is
            False.

        Returns
        -------
        DatasetDict
            An object of this class.
        """
        if copy:
            return cls(**deepcopy(dict_in))
        return cls(**dict_in)

    @property
    def spatial_coord_names(self) -> dict:
        """Return the spatial coordinate names.
        Args:
           None
        Returns:
           Dictionary of spatial coordinates with names.
        """
        attrs = self._metadata["global"]
        return {kk: vv for kk, vv in attrs.items() if "spatial" in kk}

    @classmethod
    def from_ds(cls, ds):
        """Get this class from a dataset (nc4 or xarray)."""
        # detect typ as xr or nc4
        if isinstance(ds, xr.Dataset):
            return cls(**xr_ds_to_dd(ds))
        elif isinstance(ds, nc4.Dataset):
            return cls(**nc4_ds_to_dd(ds))
        else:
            raise ValueError("Passed dataset neither from xarray nor netCDF4")

    @classmethod
    def from_netcdf(
        cls, nc_file: fileish, use_xr: bool = False, encoding=False
    ) -> "DatasetDict":
        """Load this class from a netcdf file."""
        # handle more than one file?
        if use_xr:
            return cls(**xr_ds_to_dd(nc_file, encoding=encoding))
        else:
            return cls(**nc4_ds_to_dd(nc_file, use_xr_enc=encoding))

    def to_xr_ds(self) -> xr.Dataset:
        """Export to an xarray Dataset"""
        return dd_to_xr_ds(self.data)

    def to_xr_dd(self) -> dict:
        """Export to an xarray DatasetDict (xr.Dataset.to_dict())."""
        return dd_to_xr_dd(self.data)

    def to_nc4_ds(self, filename) -> None:
        """Export to a netcdf file via netcdf4"""
        return dd_to_nc4_ds(self.data, filename)

    def to_netcdf(self, filename, use_xr=False) -> None:
        """Write parameters to a netcdf file"""
        if use_xr:
            self.to_xr_ds().to_netcdf(filename)
        else:
            self.to_nc4_ds(filename)
        return

    def rename_dim(self, name_maps: dict, in_place: bool = True):
        """Rename dimensions."""
        if not in_place:
            raise NotImplementedError
        for old_name, new_name in name_maps.items():
            self.dims[new_name] = self.dims.pop(old_name)
            for mk, mv in self.metadata.items():
                if "dims" in mv.keys() and old_name in mv["dims"]:
                    dim_list = list(mv["dims"])
                    dim_list[dim_list.index(old_name)] = new_name
                    mv["dims"] = tuple(dim_list)

        self.validate()
        return

    def rename_var(self, name_maps: dict, in_place=True):
        """Rename variables."""
        if not in_place:
            raise NotImplementedError
        for old_name, new_name in name_maps.items():
            for cv in ["coords", "data_vars"]:
                if old_name in self[cv].keys():
                    self[cv][new_name] = self[cv].pop(old_name)
                    for aa in ["metadata", "encoding"]:
                        self[aa][new_name] = self[aa].pop(old_name)

        self.validate()
        return

    def drop_var(self, var_names):
        """Drop variables"""
        if not isinstance(var_names, list):
            var_names = [var_names]
        for vv in var_names:
            for cv in ["coords", "data_vars"]:
                if vv in self[cv].keys():
                    del self[cv][vv]
                    del self.metadata[vv]
                    del self.encoding[vv]

        # todo: can any coords be dropped?

        # can any dims be dropped?
        coord_dims = self._get_var_dims(list(self.variables.keys()))
        coord_dims = set([ii for cc in coord_dims.values() for ii in cc])
        dims_rm = set(self.dims.keys()).difference(coord_dims)
        for dd in dims_rm:
            del self.dims[dd]
        return

    # TODO: add_var

    def _get_var_dims(self, var_names, data=False):
        """Get the dims of variables"""
        # dims will never return a dict or numpy array
        result = {}
        if not isinstance(var_names, list):
            var_names = [var_names]
        for vv in var_names:
            dim_names = self._metadata[vv]["dims"]
            if not data:
                result[vv] = dim_names
            else:
                dim_data = {
                    kk: vv for kk, vv in self._dims.items() if kk in dim_names
                }
                result[vv] = dim_data

        return result

    def _get_dim_coords(self, dim_list, data=False, copy=False):
        """Given a set of dimensions, get the corresponding coords"""
        # if all of a coords dims are in supplied dims, take the coordinate
        coords_out = []
        coord_dims = self._get_var_dims(list(self._coords.keys()))
        for c_name, c_dims in coord_dims.items():
            # if c_dims is empty (a scalar coord), it automatically goes
            if len(set(c_dims) - set(dim_list)) == 0:
                coords_out += [c_name]
        if not data:
            return coords_out

        coord_data = {
            ck: cv for ck, cv in self._coords.items() if ck in coords_out
        }

        if copy:
            coord_data = deepcopy(coord_data)
        return coord_data

    def subset(
        self,
        keys: Iterable,
        copy: bool = False,
        keep_global: bool = False,
        keep_global_metadata: bool = None,
        keep_global_encoding: bool = None,
        strict: bool = False,
        keep_dims: list = None,
    ) -> "DatasetDict":
        """Subset a DatasetDict to keys in data_vars or coordinates

        Args:
            keys: Iterable to subset on
            copy: bool to copy the input or edit it
            keep_global: bool that sets both keep_global_metadata and
                keep_global_encoding
            keep_global_metadata: bool retain the global metadata in the subset
            keep_global_encoding: bool retain the global encoding in the subset
            keep_dims: list of dimension names to retain in the subset even
                if not required by the subset variables

        Returns:
          A subset Parameter object on the passed keys.

        """
        # Instantiate the DatasetDict at end as deepcopy will be used
        # on the constructed subset dict (if requested)

        if not isinstance(keys, Iterable) or isinstance(keys, str):
            keys = [keys]

        for kk in keys:
            if kk not in self.variables.keys():
                if not strict:
                    continue
                msg = f"key '{kk}' not in this {type(self).__name__} object"
                raise KeyError(msg)

        if keep_global_metadata is None:
            keep_global_metadata = keep_global
        if keep_global_encoding is None:
            keep_global_encoding = keep_global

        subset = deepcopy(template_dd)

        subset["metadata"]["global"] = {}
        subset["encoding"]["global"] = {}
        if keep_global_metadata:
            subset["metadata"]["global"] = self.metadata["global"]
        if keep_global_encoding:
            subset["encoding"]["global"] = self.encoding["global"]

        for vv in self.variables.keys():
            if vv not in keys:
                continue

            is_coord = vv in self._coords.keys()
            if is_coord:
                subset["coords"][vv] = self._coords[vv]
            else:
                subset["data_vars"][vv] = self._data_vars[vv]

            subset["metadata"][vv] = self._metadata[vv]
            if vv in subset["encoding"].keys():
                subset["encoding"][vv] = self._encoding[vv]

            var_dim_data = self._get_var_dims(vv, data=True)[vv]
            # dims
            for dd in var_dim_data:
                if dd not in subset["dims"].keys():  # faster?
                    subset["dims"][dd] = var_dim_data[dd]

            if not is_coord:
                # build coords from variables
                var_coord_data = self._get_dim_coords(
                    list(var_dim_data.keys()), data=True, copy=copy
                )
                for ck, cv in var_coord_data.items():
                    if ck not in subset["coords"].keys():
                        subset["coords"][ck] = cv

        if keep_dims is not None:
            for dd in keep_dims:
                if dd not in subset["dims"].keys() and dd in self.dims.keys():
                    subset["dims"][dd] = self.dims[dd]

        # build metadata and encoding from coords and data_vars
        for cv in ["coords", "data_vars"]:
            for cc in subset[cv].keys():
                for aa in ["metadata", "encoding"]:
                    if cc in subset[aa].keys():
                        continue
                    if aa == "encoding" and cc not in self[aa].keys():
                        continue
                    subset[aa][cc] = self[aa][cc]

        # result = DatasetDict.from_dict(subset, copy=copy)
        # If this is in-place, then cant be a classmethod
        result = type(self).from_dict(subset, copy=copy)
        return result

    def subset_on_coord(
        self,
        coord_name: str,
        where: np.ndarray,
    ) -> None:
        """Subset DatasetDict to a np.where along a named coordinate in-place

        Args:
            coord_name: string name of a coordinate
            where: the result of an np.where along that coordinate (or likewise
                constructed)

        Returns:
            None
        """
        # only doing it in place for now
        # should we reall roll our own?

        # TODO: should almost work for 2+D? just linearizes np.where
        #       except that dims should be droped with >1

        if len(where) > 1:
            raise NotImplementedError("at least not tested")

        # add the where to the data_vars
        wh_data_name = "subset_inds"
        if wh_data_name in self.data_vars.keys():
            raise NotImplementedError("more work needed to subset twice")

        # what are the dim names of of the coord
        coord_dims = self.metadata[coord_name]["dims"]

        # new dim for the numer of dims in the where
        subset_dims_dim = wh_data_name + "_n_dims"
        self.dims[subset_dims_dim] = len(where)

        # new var
        self.data_vars[wh_data_name] = np.array(where).transpose()
        self.metadata[wh_data_name] = {
            "dims": coord_dims + (subset_dims_dim,),
            "attrs": {
                "description": (
                    "Zero-based indices used to subset original data "
                    f"coordinate '{coord_name}'"
                ),
            },
        }
        self.encoding[wh_data_name] = {}

        # have to edit the dim AND all variables with this dim
        for ii, dd in enumerate(coord_dims):
            self.dims[dd] = len(where[ii])
        dim_where = dict(zip(coord_dims, where))
        for vk, vv in self.variables.items():
            if vk == wh_data_name:
                continue
            var_dims = self.metadata[vk]["dims"]
            var_wh = tuple(
                [dim_where[dd] for dd in var_dims if dd in dim_where.keys()]
            )
            if vk in self.coords.keys():
                self["coords"][vk] = self.variables[vk][var_wh]
            else:
                self["data_vars"][vk] = self.variables[vk][var_wh]
            self.metadata[vk]["attrs"]["subset_on_coord"] = coord_name
            self.metadata[vk]["attrs"]["subset_inds_on_orig"] = wh_data_name

        return

    def validate(self) -> None:
        """Check that a DatasetDict is internally consistent.

        Returns:
            None
        """

        # required keys
        assert sorted(self.data.keys()) == sorted(
            ["dims", "coords", "data_vars", "metadata", "encoding"]
        )

        # can not have same names in coords and data_vars
        common_keys = set(self.coords.keys()).intersection(
            set(self.data_vars.keys())
        )
        assert len(common_keys) == 0

        # metadata and encoding keys against variable keys
        var_keys = set(self.variables.keys())
        # all meta keys have to be in var keys
        meta_keys = set(self.metadata.keys())
        assert meta_keys == var_keys.union(set(["global"]))
        # all enc_keys besides global have to be in var_keys
        enc_keys = set(self.encoding.keys())
        enc_keys = enc_keys.difference(set(["global"]))
        assert enc_keys.intersection(var_keys) == enc_keys

        # check all vars dims exist
        var_dims = [self.metadata[kk]["dims"] for kk in self.variables.keys()]
        dims = set([dim for dims in var_dims for dim in dims])
        for dd in dims:
            assert dd in self.dims.keys()

        # TODO: check dims lens equal data in the coordinates with those dims

        return

    @classmethod
    def merge(cls, *dd_list, copy=True, del_global_src=True):
        """Merge a list of this class in to a single instance

        Args:
            dd_list: a list of object of this class
            copy: boolean if a deep copy of inputs is desired
            del_global_src: boolean to delete encodings' global source
                attribute prior to merging (as these often conflict)

        Returns:
            An object of this class.
        """
        if del_global_src or copy:
            dd_list = [deepcopy(dd.data) for dd in dd_list]
            if del_global_src:
                for dd in dd_list:
                    if "global" not in dd["encoding"]:
                        continue
                    if ("global" in dd["encoding"].keys()) and (
                        "source" in dd["encoding"]["global"]
                    ):
                        del dd["encoding"]["global"]["source"]
            # <<<
            merged_dict = _merge_dicts(dd_list)
        else:
            merged_dict = _merge_dicts([deepcopy(dd.data) for dd in dd_list])

        if copy:
            merged_dict = deepcopy(merged_dict)
        return cls.from_dict(merged_dict)


# DatasetDict
# ---------------------------
# module scope functions


def _is_equal(aa, bb):
    # How sketchy is this? (honest question)
    try:
        np.testing.assert_equal(aa, bb)
        return True
    except:  # noqa
        return False


def _merge_dicts(
    dict_list: list[dict],
    conflicts: Literal["left", "warn", "error"] = "error",
):
    if not isinstance(dict_list, list):
        raise ValueError("argument 'dict_list' is not a list")
    merged = {}
    for dd in dict_list:
        for key, value in dd.items():
            if key not in merged:
                merged[key] = value
            elif isinstance(value, dict) and isinstance(merged[key], dict):
                merged[key] = _merge_dicts(
                    [value, merged[key]], conflicts=conflicts
                )
            elif _is_equal(value, merged[key]):
                pass
            else:
                msg = (
                    f"Duplicate key '{key}' with non-identical data:\n"
                    f"    L={merged[key]}\n"
                    f"    R={value}\n"
                )
                if conflicts == "error":
                    raise ValueError(msg)
                elif conflicts == "warn":
                    warnings.warn(msg)
                elif conflicts == "left":
                    pass
                else:
                    raise ValueError(
                        f"Argument 'conflicts' can not be '{conflicts}'"
                    )

    return merged


def xr_ds_to_dd(file_or_ds, schema_only=False, encoding=True) -> dict:
    """Xarray dataset to a pywatershed dataset dict

    The pyws data model moves metadata off the variables to a separate
    var_metadata dictionary with the same keys found in the union of
    the keys of coords and data_vars.
    """
    if not isinstance(file_or_ds, xr.Dataset):
        xr_ds = xr.load_dataset(file_or_ds)
    else:
        xr_ds = file_or_ds

    if schema_only:
        data_arg = False
    else:
        data_arg = "array"

    dd = xr_ds.to_dict(data=data_arg, encoding=encoding)

    # before = xr_ds.time.values.dtype
    # after = dd["coords"]["time"]["data"].dtype
    # assert before == after

    dd = xr_dd_to_dd(dd)

    return dd


def xr_dd_to_dd(xr_dd: dict) -> dict:
    dd = deepcopy(xr_dd)
    # asdf

    # Move the global encoding to a global key of itself
    dd["encoding"] = {"global": dd.get("encoding", {})}

    # rename data to metadata for var and coord
    var_metadata = dd.pop("data_vars")
    coord_metadata = dd.pop("coords")

    # create empty data dicts and move the data out of the metadata
    # and move the encoding to encoding[var]
    dd["data_vars"] = {}
    for key, val in var_metadata.items():
        dd["data_vars"][key] = val.pop("data")
        if "encoding" in val.keys():
            dd["encoding"][key] = val.pop("encoding")
        else:
            dd["encoding"][key] = {}

    dd["coords"] = {}
    for key, val in coord_metadata.items():
        dd["coords"][key] = val.pop("data")
        if "encoding" in val.keys():
            dd["encoding"][key] = val.pop("encoding")
        else:
            dd["encoding"][key] = {}

    dd["metadata"] = {**coord_metadata, **var_metadata}
    dd["metadata"]["global"] = dd.pop("attrs")

    return dd


def dd_to_xr_dd(dd: dict) -> dict:
    dd = deepcopy(dd)

    # remove metadata and encoding from the dict/model
    meta = dd.pop("metadata")
    encoding = dd.pop("encoding")

    # loop over meta data, putting it back on the data_vars and/or coords
    # and moving the data to "data"
    dd["attrs"] = meta.pop("global", {})
    dd["encoding"] = encoding.pop("global", {})

    for key, val in meta.items():
        # coordinate or variable?
        cv = None
        if key in dd["data_vars"].keys():
            cv = "data_vars"
        elif key in dd["coords"].keys():
            cv = "coords"

        data_vals = dd[cv][key]

        if np.issubdtype(data_vals.dtype, np.datetime64):
            # conversion to datetime64[ns] silences xr warnings
            # but should be able to be removed in the future
            data_vals = data_vals.astype("datetime64[ns]")

        key_enc = {}
        if key in encoding.keys():
            key_enc = encoding[key]

        dd[cv][key] = {
            **val,
            "data": data_vals,
            "encoding": key_enc,
        }

    return dd


def dd_to_xr_ds(dd: dict) -> xr.Dataset:
    """pywatershed dataset dict to xarray dataset

    The pyws data model moves metadata off the variables to a separate
    var_metadata dictionary with the same keys found in the union of
    the keys of coords and data_vars. This maps the metadata back to the
    variables.
    """
    return xr.Dataset.from_dict(dd_to_xr_dd(dd))


def _nc4_var_to_datetime64(var, attrs, encoding):
    """netCDF4 conversion of time to numpy.datetime64 based on metadata."""
    if not (hasattr(var, "units") and "since" in var.units):
        return var[:], attrs, encoding

    # Check if the variable has a calendar attribute
    if hasattr(var, "calendar"):
        time_data = nc4.num2date(
            var[:], var.units, var.calendar, only_use_cftime_datetimes=False
        )
    else:
        time_data = nc4.num2date(var[:], var.units)

    if isinstance(time_data, cftime.real_datetime):
        time_data = np.datetime64(time_data)
    else:
        time_data = time_data.filled().astype("datetime64[us]")

    for aa in ["calendar", "units"]:
        if aa in attrs.keys():
            encoding[aa] = attrs.pop(aa)

    return time_data, attrs, encoding


def _datetime64_to_nc4_var(var, units, calendar):
    """vectorized conversion of numpy.datetime64 to a netcdf4 variable."""
    # Based on what xarray does
    # https://github.com/pydata/xarray/blob/
    #   a1f5245a48146bd8fc5bdb07ef8ae6077d6e511c/xarray/coding/times.py#L687
    from xarray.coding.times import decode_cf_datetime

    # This can take encoding info but we are using defaults.
    # (data, units, calendar) = decode_cf_datetime(var, units, calendar)
    # return {"data": data, "units": units, "calendar": calendar}

    data = decode_cf_datetime(var, units, calendar)
    return {"data": data}


def nc4_ds_to_xr_dd(file_or_ds, xr_enc: dict = None) -> dict:
    """Convert a netCDF4 dataset to and xarray dataset dictionary"""

    if not isinstance(file_or_ds, nc4.Dataset):
        ds = nc4.Dataset(file_or_ds, "r")
    else:
        ds = file_or_ds

    # An empty xr_dd dictionary to hold the data
    xr_dd = deepcopy(template_xr_dd)

    # xr_dd["attrs"] = nc_file.__dict__  # ugly
    for attrname in ds.ncattrs():
        xr_dd["attrs"][attrname] = ds.getncattr(attrname)

    for dimname, dim in ds.dimensions.items():
        xr_dd["dims"][dimname] = len(dim)

    for varname, var in ds.variables.items():
        # _Encoding is used for string encoding in nc4
        var_encoding = {}
        if "_Encoding" in var.__dict__.keys():
            var_encoding["_Encoding"] = var._Encoding

        data_dict = {"dims": var.dimensions}

        var_attrs = {}
        for attrname in var.ncattrs():
            var_attrs[attrname] = var.getncattr(attrname)

        var_data, var_attrs, var_encoding = _nc4_var_to_datetime64(
            var,
            var_attrs,
            var_encoding,
        )

        if isinstance(var_data, np.ma.core.MaskedArray):
            var_data = var_data.data

        for aa in ["_FillValue"]:
            if aa in var_attrs.keys():
                var_encoding[aa] = var_attrs.pop(aa)

        data_dict["data"] = var_data
        data_dict["attrs"] = var_attrs
        data_dict["encoding"] = var_encoding

        if varname in ds.dimensions:
            xr_dd["coords"][varname] = data_dict
        else:
            xr_dd["data_vars"][varname] = data_dict

    ds.close()

    # have to promote coord variables here?
    all_coords = [
        vv["attrs"].pop("coordinates")
        for vv in xr_dd["data_vars"].values()
        if "coordinates" in vv["attrs"].keys()
    ]
    all_coords = sorted(set(" ".join(all_coords).split(" ")))
    for cc in all_coords:
        if cc == "":
            continue
        xr_dd["coords"][cc] = xr_dd["data_vars"].pop(cc)

    # handle bools
    for meta in [xr_dd["coords"], xr_dd["data_vars"]]:
        for kk, vv in meta.items():
            if "dtype" in vv["attrs"].keys():
                dtype = vv["attrs"]["dtype"]
                if dtype == "bool":
                    vv["data"] = vv["data"].astype("bool")
                    _ = vv["attrs"].pop("dtype")

    # bring in the encoding information using xarray (cheating?)
    if xr_enc:
        xr_dd["encoding"] = {**xr_dd["encoding"], **xr_enc.pop("global")}
        for cc in xr_dd["coords"].keys():
            if cc in xr_enc.keys():
                xr_dd["coords"][cc]["encoding"] = {
                    **xr_dd["coords"][cc]["encoding"],
                    **xr_enc.pop(cc),
                }
        for vv in xr_dd["data_vars"].keys():
            if vv in xr_enc.keys():
                xr_dd["data_vars"][vv]["encoding"] = {
                    **xr_dd["data_vars"][vv]["encoding"],
                    **xr_enc.pop(vv),
                }

    return xr_dd


def _get_xr_encoding(nc_file) -> dict:
    ds = xr.load_dataset(nc_file)
    encoding = {}
    encoding["global"] = ds.encoding
    for vv in ds.variables:
        encoding[vv] = ds[vv].encoding

    return encoding


def nc4_ds_to_dd(
    nc4_file_ds, subset: np.ndarray = None, use_xr_enc=True
) -> dict:
    """netCDF4 dataset to a pywatershed dataset dict."""
    xr_enc = None
    if not isinstance(nc4_file_ds, nc4.Dataset):
        if use_xr_enc:
            xr_enc = _get_xr_encoding(nc4_file_ds)
        nc4_file_ds = nc4.Dataset(nc4_file_ds)
    else:
        if use_xr_enc:
            raise ValueError(
                "Pass a file and not an nc4.Dataset to use_xr_enc argument"
            )

    xr_dd = nc4_ds_to_xr_dd(nc4_file_ds, xr_enc=xr_enc)
    dd = xr_dd_to_dd(xr_dd)
    return dd


def dd_to_nc4_ds(dd, nc_file):
    """nc4_ds is a bit of a misnomer since it's on disk, dd_to_nc4"""
    # import cftime
    from xarray.coding.times import encode_cf_datetime

    dd = deepcopy(dd)

    # work from xrarray's dict representation of a dataset
    xr_dd = dd_to_xr_dd(dd)
    del dd

    # create a new netCDF4 file
    with nc4.Dataset(nc_file, "w") as ds:
        ds.set_fill_on()

        for key, value in xr_dd["attrs"].items():
            setattr(ds, key, value)

        for dim, size in xr_dd["dims"].items():
            ds.createDimension(dim, size)

        for coord_name, values in xr_dd["coords"].items():
            enc = values["encoding"]
            # handle time encoding
            if np.issubdtype(values["data"].dtype, np.datetime64):
                dates_enc, units, calendar = encode_cf_datetime(
                    values["data"].astype("datetime64[ns]"),
                    enc.get("units", None),
                    enc.get("calendar", None),
                )

                # TODO: what if .astype("datetime64[ns]") fails?
                # this would be a manual way that's looped and that
                # dosent guess at units and calendar
                # cf_dates = cftime.date2num(
                #     dates,
                #     units=enc.get("units"),
                #     calendar=enc.get("calendar"),
                # )

                values["data"] = dates_enc
                values["attrs"]["units"] = units
                values["attrs"]["calendar"] = calendar
                for kk in ["units", "calendar"]:
                    if kk in enc.keys():
                        del enc[kk]

            var_type = values["attrs"].get("type", values["data"].dtype)
            var = ds.createVariable(
                coord_name,
                var_type,
                dimensions=xr_dd["coords"][coord_name]["dims"],
                fill_value=enc.get("_FillValue", None),
                # This is not complete. Defaults from nc4
                zlib=enc.get("zlib", False),
                complevel=enc.get("complevel", 4),
                shuffle=enc.get("shuffle", True),
                contiguous=enc.get("contiguous", False),
                fletcher32=enc.get("fletcher32", False),
                chunksizes=enc.get("chunksizes", None),
            )
            var[:] = values["data"]
            var.setncatts(values["attrs"])

        for var_name, values in xr_dd["data_vars"].items():
            enc = values["encoding"]
            is_bool = values["data"].dtype == "bool"
            np_type = values["data"].dtype
            nc_type = np_type_to_netcdf_type_dict[np_type]
            var_type = values["attrs"].get("type", nc_type)
            if is_bool:
                var_type = "i1"
            default_fill = fill_values_dict[np_type]
            var = ds.createVariable(
                var_name,
                var_type,
                dimensions=values["dims"],
                fill_value=enc.get("_FillValue", default_fill),
                # This is not complete. Defaults from nc4
                zlib=enc.get("zlib", False),
                complevel=enc.get("complevel", 4),
                shuffle=enc.get("shuffle", True),
                contiguous=enc.get("contiguous", False),
                fletcher32=enc.get("fletcher32", False),
                chunksizes=enc.get("chunksizes", None),
            )

            if is_bool:
                var[:] = values["data"].astype("int8")
            else:
                var[:] = values["data"]
            var.setncatts(values["attrs"])
            # solve coords for this var: any coord which has any of its dims
            coords = []
            # these rules are a bit opaque
            coords_just_cuz = ["reference_time"]
            coords_no_way = ["time"]
            for c_name, c_val in xr_dd["coords"].items():
                if c_name in coords_no_way:
                    continue
                common = set(values["dims"]).intersection(set(c_val["dims"]))
                if len(common) or c_name in coords_just_cuz:
                    coords += [c_name]
            if len(coords):
                var.coordinates = " ".join(sorted(coords))
            if is_bool:
                var.setncattr("dtype", "bool")

    return


def open_datasetdict(nc_file: fileish, use_xr=True):
    """Convenience method for opening a DatasetDict.

    Args:
      nc_file: the file containing the DatasetDict.
      use_xr: Use xarray or NetCDF4 for opening the NetCDF file?
    """
    return DatasetDict.from_netcdf(nc_file, use_xr=use_xr)
