import pathlib as pl

import numpy as np
import pytest
from utils import assert_or_print

from pywatershed import Control, Parameters, PRMSCanopy
from pywatershed.parameters import PrmsParameters
from pywatershed.utils.prms5_file_util import expand_scalar_to_dims

test_ans = {
    "drb_2yr": {
        "ndepl": 5,
        "ndeplval": 55,
        "ndoy": 366,
        "ngw": 765,
        "nhru": 765,
        "nmonth": 12,
        "nobs": 168,
        "npoigages": 168,
        "nsegment": 456,
        "nssr": 765,
        "scalar": 1,
    },
    "hru_1": {
        "nhru": 1,
        "nsegment": 1,
        "nssr": 1,
        "ngw": 1,
        "nobs": 1,
        "ndeplval": 11,
        "ndepl": 1,
        "nmonth": 12,
        "scalar": 1,
        "ndoy": 366,
    },
    "ucb_2yr": {
        "nhru": 3851,
        "nsegment": 1942,
        "nssr": 3851,
        "ngw": 3851,
        "npoigages": 347,
        "nobs": 347,
        "ndeplval": 55,
        "ndepl": 5,
        "nmonth": 12,
        "scalar": 1,
        "ndoy": 366,
    },
    "sagehen_5yr": {
        "nhru": 128,
        "nsegment": 15,
        "ncascade": 317,
        "ncascdgw": 317,
        "nssr": 128,
        "ngw": 128,
        "nobs": 1,
        "ndeplval": 22,
        "ndepl": 2,
        "nmonth": 12,
        "scalar": 1,
        "ndoy": 366,
    },
}


@pytest.mark.domainless
def test_parameter_init():
    # TODO: this test should be moved to tests for the Parameters class
    parameters = {
        "nhru": 2,
        "abc": np.zeros(2, dtype=float),
        "nmonth": 12,
        "nsegment": 10,
        "xyz": np.arange(12 * 10, dtype=float).reshape(12, 10),
    }
    param_obj = Parameters(data_vars=parameters, validate=False)

    answers = {"nhru": 2, "nmonth": 12, "nsegment": 10}

    results = {
        key: val
        for key, val in param_obj.parameters.items()
        if key in answers.keys()
    }
    assert_or_print(results, answers)

    print("success initializing Parameters object")


def test_expand_scalar_to_dims():
    # A 12-HRU domain: per-HRU and monthly arrays have the same length, so
    # the declared dimension name must decide how to expand.
    nhru = 12
    param_dict = {
        "nhru": nhru,
        "nmonth": 12,
        # scalar -> (nmonth, nhru)
        "cecn_coef": np.array([5.0]),
        # monthly -> (nmonth, nhru)
        "rain_cbh_adj": np.arange(12, dtype=float),
        # per-HRU, in expand_hru_to_monthly -> (nmonth, nhru)
        "tmax_adj": np.arange(nhru, dtype=float),
        # per-HRU, not expandable to monthly -> passed through unchanged
        "snow_cbh_adj": np.arange(nhru, dtype=float),
        # already full shape -> unchanged
        "hru_area": np.arange(nhru, dtype=float),
    }
    param_dim_dict = {
        "nhru": None,
        "nmonth": None,
        "cecn_coef": ("scalar",),
        "rain_cbh_adj": ("nmonth",),
        "tmax_adj": ("nhru",),
        "snow_cbh_adj": ("nhru",),
        "hru_area": ("nhru",),
    }
    result, result_dims = expand_scalar_to_dims(
        dict(param_dict), dict(param_dim_dict)
    )

    assert result["cecn_coef"].shape == (12, nhru)
    assert (result["cecn_coef"] == 5.0).all()

    assert result["rain_cbh_adj"].shape == (12, nhru)
    # month varies along axis 0, constant across HRUs
    assert (result["rain_cbh_adj"][:, 0] == np.arange(12)).all()
    assert (result["rain_cbh_adj"][3, :] == 3.0).all()

    assert result["tmax_adj"].shape == (12, nhru)
    # HRU varies along axis 1, constant across months
    assert (result["tmax_adj"][0, :] == np.arange(nhru)).all()
    assert (result["tmax_adj"][:, 3] == 3.0).all()

    # unhandled shape: no error, data untouched
    assert result["snow_cbh_adj"].shape == (nhru,)
    assert (result["snow_cbh_adj"] == param_dict["snow_cbh_adj"]).all()

    assert result["hru_area"].shape == (nhru,)

    # dims are reported from metadata in every case
    for name in ["cecn_coef", "rain_cbh_adj", "tmax_adj", "snow_cbh_adj"]:
        assert tuple(result_dims[name]) == ("nmonth", "nhru")
    assert tuple(result_dims["hru_area"]) == ("nhru",)


def test_parameter_read(simulation):
    ctl = Control.load_prms(
        simulation["control_file"], warn_unused_options=False
    )
    parameter_file = simulation["dir"] / ctl.options["parameter_file"]
    print(f"parsing...'{parameter_file}'")

    parameters = PrmsParameters.load(parameter_file)

    # check dimensions
    # this only depends on domain and not simulation AFAICT
    test_domain = simulation["name"].split(":")[0]
    answers = test_ans[test_domain]
    results = parameters.dims
    assert_or_print(results, answers, print_ans=simulation["print_ans"])

    print(f"success parsing...'{parameter_file}'")
    return


def test_parameter_canopy_subset(simulation):
    ctl = Control.load_prms(
        simulation["control_file"], warn_unused_options=False
    )
    parameter_file = simulation["dir"] / ctl.options["parameter_file"]
    print(f"parsing...'{parameter_file}'")
    parameters = PrmsParameters.load(parameter_file)

    canopy_params = PRMSCanopy.get_parameters()
    canopy_subset = parameters.subset(canopy_params)

    canopy_params_2 = tuple([*canopy_params, "bar"])
    with pytest.raises(KeyError):
        canopy_subset = parameters.subset(canopy_params_2, strict=True)

    for key in canopy_params:
        if key != "unknown":
            assert canopy_subset.parameters[key] is not None, (
                f"'{key}' parameter should not be None"
            )

    print(f"success parsing...'{parameter_file}'")


def test_parameter_access(simulation):
    ctl = Control.load_prms(
        simulation["control_file"], warn_unused_options=False
    )
    parameter_file = simulation["dir"] / ctl.options["parameter_file"]
    print(f"parsing...'{parameter_file}'")

    parameters = PrmsParameters.load(parameter_file)

    assert parameters.parameters["srain_intcp"] is not None, (
        "'srain_intcp' should not return None"
    )

    with pytest.raises(KeyError):
        _ = parameters.parameters["unknown"]

    # can not delete parameters this way so will not raise a keyerror
    del parameters.parameters["srain_intcp"]
    _ = parameters.parameters["srain_intcp"]


def test_parameter_json(simulation, tmp_path):
    # read a myparams.param file to a parameter object,
    # write it to json,
    # read the json back in
    # check that the read json gives an identical parameter object.
    ctl = Control.load_prms(
        simulation["control_file"], warn_unused_options=False
    )
    parameter_file = simulation["dir"] / ctl.options["parameter_file"]
    parameters = PrmsParameters.load(parameter_file)

    json_file = pl.Path(tmp_path) / "params.json"
    parameters.parameters_to_json(json_file)
    assert json_file.exists()

    params_from_json = PrmsParameters.load_from_json(json_file)
    assert parameters.data.keys() == params_from_json.data.keys()
    for kk in parameters.data.keys():
        np.testing.assert_equal(
            dict(parameters.data[kk]), dict(params_from_json.data[kk])
        )
    return
