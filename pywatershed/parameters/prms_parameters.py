import json
from copy import deepcopy
from typing import Union
from warnings import warn

import numpy as np

from ..base import meta
from ..base.parameters import Parameters
from ..constants import fileish, ft2_per_acre, inches_per_foot, ndoy
from ..utils.prms5_file_util import PrmsFile

# TODO:
# PRMS uses "ndays"for the number of days in "year" defined as 366.
# I have changed this for pywatershed because it is ambiguous and have
# adopted ndoy instead, which is somewhat better but not perfect.
# I ndays is in a parameter file, it may cause errors that we should
# probably trap.

prms_dim_names = (
    "ncascade",
    "ncascdgw",
    "ndepl",
    "ndeplval",
    "ndoy",
    "ngw",
    "nhru",
    "nlake",
    "nlake_hrus",
    "nmonth",
    "nobs",
    "npoigages",
    "nrain",
    "nsegment",
    "nssr",
    "ntemp",
    "scalar",
)


class JSONParameterEncoder(json.JSONEncoder):
    """
    Simple encoder to cast numpy objects to json-friendly formats
    """

    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(JSONParameterEncoder, self).default(obj)


def _json_load(json_filename):
    with open(json_filename) as ff:
        pars = json.load(ff)

    # need to convert lists to numpy arrays
    for k, v in pars.items():
        if isinstance(v, list):
            pars[k] = np.array(v)
    return pars


class PrmsParameters(Parameters):
    """A parameter class with methods for native PRMS files.

    See Also
    --------
    pywatershed.Parameters

    """

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

        super().__init__(
            dims=dims,
            coords=coords,
            data_vars=data_vars,
            metadata=metadata,
            encoding=encoding,
            validate=validate,
            copy=copy,
        )

        return

    def parameters_to_json(self, json_filename) -> None:
        """write the parameters dictionary out to a json file"""
        with open(json_filename, "w") as ff:
            json.dump(
                {**self.dims, **self.parameters},
                ff,
                indent=4,
                cls=JSONParameterEncoder,
            )
        return None

    @staticmethod
    def _from_dict(param_dict: dict) -> "PrmsParameters":
        """Load parameters from a dictionary of just parameters"""
        return PrmsParameters._after_load(param_dict)

    @staticmethod
    def load_from_json(json_filename: fileish) -> "PrmsParameters":
        """Load parameters from a json file.
        Args:
          json_filename: json file path
        """
        pars = _json_load(json_filename)
        params = PrmsParameters._process_file_input(pars)

        return params

    @staticmethod
    def load(
        parameter_files: Union[
            fileish,
            list,
        ],
    ) -> "PrmsParameters":
        """Load parameters from PRMS parameter files

        Args:
            parameter_files: parameter file path(s). Files beyond the first
            are "addenda" inheriting the dimensions of the first file.

        Returns:
            PrmsParameters: full PRMS parameter dictionary

        """
        if isinstance(parameter_files, fileish):
            data = PrmsFile(parameter_files, "parameter").get_data()
            data = data["parameter"]["parameters"]

        elif isinstance(parameter_files, (list, np.ndarray)):
            print(
                "Loading and merging the parameter files (order matters):\n"
                f"{parameter_files}"
            )
            param_list = []
            dims_from_previous_file = {}
            for pf in parameter_files:
                data = PrmsFile(
                    pf,
                    "parameter",
                    dims_from_previous_file=dims_from_previous_file,
                ).get_data()
                dims_from_previous_file = data["parameter"][
                    "parameter_dimensions"
                ]
                param_list += [data["parameter"]["parameters"]]

            data = {}
            for dd in param_list:
                dd_keys_in_data = set(dd.keys()).intersection(data.keys())
                for ddk in dd_keys_in_data:
                    if np.testing.assert_equal(dd[ddk], data[ddk]):
                        msg = (
                            f"Multiple copies of parameter {ddk} found, "
                            "keeping the last one found in the "
                            "parameter_files."
                        )
                        warn(msg)
                data.update(dd)

        else:
            raise ValueError(f"Unacceptable type: {parameter_files=}")

        # <
        params = PrmsParameters._process_file_input(data)

        return params

    def to_netcdf(self, filename, use_xr=False) -> None:
        """Write PrmsParameters to a netcdf file"""
        self.data_vars
        if use_xr:
            self.to_xr_ds().to_netcdf(filename)
        else:
            self.to_nc4_ds(filename)
        return

    @staticmethod
    def _process_file_input(
        parameter_dict: dict,
        parameter_dimensions_dict: dict = None,
    ) -> "PrmsParameters":
        # move dims from params to dims
        if parameter_dimensions_dict is None:
            parameter_dimensions_dict = {}

        dims = {}
        for dd in prms_dim_names:
            if dd in parameter_dict.keys():
                dims[dd] = parameter_dict.pop(dd)
                if dd in parameter_dimensions_dict.keys():
                    _ = parameter_dimensions_dict.pop(dd)

        # add implied dimensions
        dims["ndoy"] = ndoy
        dims["nmonth"] = 12
        if "ndays" in parameter_dict.keys():
            del parameter_dict["ndays"]

        # build dimension metadata from data
        if len(parameter_dimensions_dict) == 0:
            for key, value in parameter_dict.items():
                if key == "ndays":
                    continue
                # errors in the next line, see prms_dim_names at top
                param_dim_names = meta.get_params(key)[key]["dims"]
                parameter_dimensions_dict[key] = {"dims": param_dim_names}

                common_params = set(param_dim_names) & set(dims)
                if not len(common_params):
                    parameter_dimensions_dict[key] = {
                        "dims": tuple(["unknown"])
                    }
                    continue

                param_dims = {kk: dims[kk] for kk in common_params}

                if isinstance(value, int):
                    parameter_dimensions_dict[key] = None
                elif isinstance(value, np.ndarray):
                    shape = value.shape
                    temp_dims = []
                    for isize in shape:
                        found_dim = False

                        for dim_key, dim_value in param_dims.items():
                            if dim_value == isize:
                                found_dim = True
                                temp_dims.append(dim_key)
                                break

                        if isize == 1 and not found_dim:
                            found_dim = True
                            temp_dims.append("scalar")

                        if not found_dim:
                            raise ValueError(
                                "Unable to identify dimension name for"
                                f"parameter {key} with length {isize}"
                            )

                parameter_dimensions_dict[key] = {"dims": tuple(temp_dims)}

        # build coords, only some dims have coords that are not indexes
        coords = {}
        # edge case of dims that have only index or implied coords
        # this is justified, as this data should probably be in the parameter
        # files. netcdf neccisitates this to even carry the dimension data.
        # alternative would be to have a scalar, but in fact this is a
        # dimension of the data.
        coords["doy"] = np.arange(dims["ndoy"], dtype="int32") + 1

        for cc in ["nhm_id", "nhm_seg", "poi_gage_id", "doy"]:
            coord_to_dim = {
                "nhm_id": "nhru",
                "nhm_seg": "nsegment",
                "poi_gage_id": "npoigages",
                "doy": "ndoy",
            }
            dim_name = coord_to_dim[cc]
            if dim_name not in dims:
                continue

            dim_val = dims[dim_name]

            if cc in parameter_dict.keys():
                coords[cc] = parameter_dict.pop(cc)
            else:
                coord_to_dim = {"nhm_id": "nhru", "nhm_seg": "nsegment"}
                coords[cc] = np.arange(1, dim_val + 1)

            parameter_dimensions_dict[cc] = {"dims": (dim_name,)}

        # Derived parameter converting inches to cubic feet on hrus.
        parameter_dict["hru_in_to_cf"] = (
            parameter_dict["hru_area"] * ft2_per_acre / (inches_per_foot)
        )
        parameter_dimensions_dict["hru_in_to_cf"] = {"dims": ("nhru",)}

        for key, value in parameter_dimensions_dict.items():
            key_meta = deepcopy(meta.get_params(key)[key])
            _ = key_meta.pop("dims")
            parameter_dimensions_dict[key]["attrs"] = key_meta

        parameter_dimensions_dict["global"] = {
            "Description": "Parameter data for PRMS"
        }

        prms_params = PrmsParameters(
            dims=dims,
            coords=coords,
            data_vars=parameter_dict,
            metadata=parameter_dimensions_dict,
            validate=True,
        )

        return prms_params
