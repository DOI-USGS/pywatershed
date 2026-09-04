import pathlib as pl
from copy import deepcopy
from types import MappingProxyType
from typing import Union

import numpy as np
import xarray as xr

from .data_model import DatasetDict, dd_to_nc4_ds, dd_to_xr_ds

# MappingProxyType used as per
# https://adamj.eu/tech/2022/01/05/how-to-make-immutable-dict-in-python/


class Parameters(DatasetDict):
    """Parameter base class

    This is a subclass of :func:`~pywatershed.base.DatasetDict` where data are
    read-only by design. Parameters has all the same methods as DatasetDict
    plus several new ones that map to DatasetDict as follows:

    * parameters: dd.variables (dd.coords + dd.data)
    * get_param_values: get values from dd.variables
    * get_dim_values: get values from dd.dims

    Parameters
    ----------
    dims : dict
        A dictionary of pairs of `dim_names: dim_len` where `dim_len` is
        an integer value.
    coords : dict
        A dictionary of pairs of `coord_names: coord_data` where
        `coord_data` is an np.ndarray.
    data_vars : dict
        A dictionary of pairs of `var_names: var_data` where
        `coord_data` is an np.ndarray.
    metadata : dict
        For all names in `coords` and `data_vars`, metadata entries
        with the required fields:

        - dims: tuple of names in dim,
        - attrs: dictionary whose values may be strings, ints, floats

        The metadata argument may also contain a special `global` key
        paired with a dictionary of global metadata of arbitrary name and
        values of string, integer, or float types.

    encoding : dict, optional
        The encoding attributes to/from file when reading/writing.
    validate : bool, optional
        If True (default), enforces the consistency of the supplied dictionaries.

    See Also
    --------
    pywatershed.base.DatasetDict

    Examples
    --------

    See :func:`~pywatershed.base.DatasetDict` for more examples.

    ..

    >>> from pprint import pprint
    >>> import numpy as np
    >>> import pywatershed as pws
    >>> nreach = 3
    >>> params = pws.parameters.PrmsParameters(
    ...     dims={
    ...         "nsegment": nreach,
    ...     },
    ...     coords={
    ...         "nsegment": np.array(range(nreach)),
    ...     },
    ...     data_vars={
    ...         "tosegment": np.array(
    ...             [2, 3, 0]
    ...         ),  # one-based index, 0 is outflow
    ...         "seg_length": np.ones(nreach) * 1.0e3,
    ...     },
    ...     metadata={
    ...         "nsegment": {"dims": ["nsegment"]},
    ...         "tosegment": {"dims": ["nsegment"]},
    ...         "seg_length": {"dims": ["nsegment"]},
    ...     },
    ...     validate=True,
    ... )
    >>> params
    <pywatershed.parameters.prms_parameters.PrmsParameters object at 0x105781390>
    >>> params.dims
    mappingproxy({'nsegment': 3})
    >>> params.coords
    mappingproxy({'nsegment': array([0, 1, 2])})
    >>> pprint(params.data)
    {'coords': mappingproxy({'nsegment': array([0, 1, 2])}),
     'data_vars': mappingproxy({'seg_length': array([1000., 1000., 1000.]),
                                'tosegment': array([2, 3, 0])}),
     'dims': mappingproxy({'nsegment': 3}),
     'encoding': mappingproxy({}),
     'metadata': mappingproxy({'global': mappingproxy({}),
                               'nsegment': mappingproxy({'dims': ['nsegment']}),
                               'seg_length': mappingproxy({'dims': ['nsegment']}),
                               'tosegment': mappingproxy({'dims': ['nsegment']})})}
    >>> pprint(params.metadata)
    mappingproxy({'global': mappingproxy({}),
                  'nsegment': mappingproxy({'dims': ['nsegment']}),
                  'seg_length': mappingproxy({'dims': ['nsegment']}),
                  'tosegment': mappingproxy({'dims': ['nsegment']})})
    >>> xrds = params.to_xr_ds()
    >>> xrds
    <xarray.Dataset>
    Dimensions:     (nsegment: 3)
    Coordinates:
      * nsegment    (nsegment) int64 0 1 2
    Data variables:
        tosegment   (nsegment) int64 2 3 0
        seg_length  (nsegment) float64 1e+03 1e+03 1e+03

    """  # noqa: E501

    def __init__(
        self,
        dims: dict = None,
        coords: dict = None,
        data_vars: dict = None,
        metadata: dict = None,
        encoding: dict = None,
        validate: bool = True,
        copy: bool = True,
    ) -> None:
        if copy:
            dims = deepcopy(_set_dict_read_write(dims))
            coords = deepcopy(_set_dict_read_write(coords))
            data_vars = deepcopy(_set_dict_read_write(data_vars))
            metadata = deepcopy(_set_dict_read_write(metadata))
            encoding = deepcopy(_set_dict_read_write(encoding))

        super().__init__(
            dims=dims,
            coords=coords,
            data_vars=data_vars,
            metadata=metadata,
            encoding=encoding,
            validate=validate,
        )

        for kk in self._keys():
            self[f"_{kk}"] = _set_dict_read_only(self[kk])

        return

    @property
    def dimensions(self) -> dict:
        """Get the dimensions from the parameters

        Returns:
            dimensions in the PRMS parameter dictionary

        """
        dimensions = {}
        for key, value in self.dims.items():
            if isinstance(value, int):
                dimensions[key] = value
        return dimensions

    @property
    def parameters(self) -> dict:
        return self.variables

    def get_param_values(
        self,
        keys: Union[list, str] = None,
    ) -> Union[dict, np.ndarray]:
        """Get the values of the parameters (coords or data_vars) by keys

        Also see:
            subset() method is a Parameter object is desired.
        """
        if not isinstance(keys, list):
            return self.variables[keys]
        else:
            return {kk: self.variables[kk] for kk in keys}

    def get_dim_values(
        self,
        keys: Union[list, tuple, str] = None,
    ) -> Union[dict, np.ndarray]:
        """Get the values of the dimensions by keys."""
        if not isinstance(keys, (list, tuple)):
            return self.dims[keys]
        else:
            return {kk: self.dims[kk] for kk in keys}

    @classmethod
    def from_dataset_dict(cls, dataset_dict: DatasetDict) -> "Parameters":
        """A Parameters object from a DatasetDict.

        Args:
          dataset_dict: a DatasetDict to transform into parameters.
        """
        return cls(**dataset_dict.data)

    def to_xr_ds(self) -> xr.Dataset:
        """Export Parameters to an xarray dataset"""
        return dd_to_xr_ds(_set_dict_read_write(self.data))

    def to_nc4_ds(self, filename: Union[str, pl.Path]) -> None:
        """Export Parameters to a netcdf4 dataset

        Args:
            filename: a file to write to as nc4 is not in memory
        """
        dd_to_nc4_ds(_set_dict_read_write(self.data), filename)
        return

    def to_dd(self, copy=True) -> DatasetDict:
        """Export Parameters to a DatasetDict (for editing).

        Parameters can NOT be edited, but DatasetDicts can.
        To convert back ``pws.Parameters(**dataset_dict.data)``.

        Args:
            copy: return a copy or a reference?
        """
        return DatasetDict.from_dict(
            _set_dict_read_write(self.data), copy=copy
        )

    @classmethod
    def merge(cls, *args, copy=True, del_global_src=True):
        """Merge Parameter classes

        Args:
            *args: several Parameters objects as individual objects.
            copy: bool if the args should be copied?
            del_golbal_src: bool delete the file source attribute to avoid
                meaningless merge conflicts?
        """
        dd_list = [
            DatasetDict.from_dict(_set_dict_read_write(pp.data)) for pp in args
        ]
        merged = super().merge(*dd_list, copy=copy, del_global_src=True)
        return merged


# TODO: test that these dont modify in place
def _set_dict_read_write(mp: MappingProxyType):
    if mp is None:
        mp = {}
    dd = mp | {}
    for kk, vv in dd.items():
        if isinstance(vv, (dict, MappingProxyType)):
            dd[kk] = _set_dict_read_write(vv)
        elif isinstance(vv, np.ndarray):
            # copy is sufficient to make writeable
            # https://numpy.org/doc/stable/reference/generated/numpy.copy.html
            dd[kk] = dd[kk].copy()

    return dd


def _set_dict_read_only(dd: dict):
    for kk, vv in dd.items():
        if isinstance(vv, dict):
            _set_dict_read_only(vv)
            dd[kk] = MappingProxyType(vv)
        elif isinstance(vv, np.ndarray):
            vv.flags.writeable = False

    return MappingProxyType(dd)
