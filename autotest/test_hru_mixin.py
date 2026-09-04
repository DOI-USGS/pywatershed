import numpy as np
import pytest

from pywatershed.base.hru_mixin import HruMixin
from pywatershed.base.process import Process
from pywatershed.base.timeseries import TimeseriesArray
from pywatershed.constants import HruType
from pywatershed.parameters import Parameters

# These tests are domainless. Constructing a real Process requires generated
# domain data, forcings and a Control, so a minimal Process subclass is used
# which never calls Process.__init__ but does exercise the real
# Process._set_params, HruMixin._set_active_hrus and
# HruMixin._mask_inactive_hrus code.

INACTIVE = HruType.INACTIVE.value

ACTIVE_HRU_KEYS = ("active_hru_mask", "wh_active_hrus", "nactive_hrus")


class _HruProcess(Process, HruMixin):
    """A Process using HruMixin, base order as in the real consumers."""

    def __init__(self):
        pass

    @staticmethod
    def get_dimensions() -> tuple:
        return ("nhru",)

    @staticmethod
    def get_parameters() -> tuple:
        return ("hru_type",)

    @staticmethod
    def get_variables() -> tuple:
        return ("soil_moist", "hru_ppt", "seg_outflow")


def make_parameters(hru_type: np.ndarray, supplied: dict = None) -> Parameters:
    """Minimal Parameters, optionally supplying the active-HRU quantities.

    The active-HRU quantities are never parameters of any Process. They can be
    supplied here to show that they are ignored.
    """
    nhru = len(hru_type)
    dims = {"nhru": nhru}
    data_vars = {"hru_type": hru_type}
    metadata = {"nhru": {"dims": ["nhru"]}, "hru_type": {"dims": ["nhru"]}}

    if supplied is not None:
        if "active_hru_mask" in supplied:
            data_vars["active_hru_mask"] = supplied["active_hru_mask"]
            metadata["active_hru_mask"] = {"dims": ["nhru"]}
        if "wh_active_hrus" in supplied:
            wh = supplied["wh_active_hrus"]
            dims["nactive_hru"] = len(wh)
            data_vars["wh_active_hrus"] = wh
            metadata["wh_active_hrus"] = {"dims": ["nactive_hru"]}
        if "nactive_hrus" in supplied:
            dims["scalar"] = 1
            data_vars["nactive_hrus"] = np.array(
                [supplied["nactive_hrus"]], dtype="int64"
            )
            metadata["nactive_hrus"] = {"dims": ["scalar"]}

    return Parameters(
        dims=dims,
        coords={"nhru": np.arange(nhru)},
        data_vars=data_vars,
        metadata=metadata,
        validate=True,
    )


def make_process(hru_type, supplied=None, keep_supplied=False):
    """Build the stub process on the given parameters.

    If keep_supplied, the un-subset Parameters object is put on the process,
    so _set_active_hrus is exercised with the supplied values visible to it.
    """
    params = make_parameters(hru_type, supplied)
    proc = _HruProcess()
    proc._set_params(params, None)
    if keep_supplied:
        proc._params = params
    return proc


def set_variables(proc, nhru, ntime=3):
    """Put known values on the stub's variables."""
    proc.soil_moist = np.arange(nhru, dtype="float64") + 1.0
    proc.hru_ppt = TimeseriesArray(
        control=None,
        var_name="hru_ppt",
        array=np.tile(np.arange(nhru, dtype="float64") + 1.0, (ntime, 1)),
        time=np.arange(ntime),
    )
    # not an nhru variable, must never be masked
    proc.seg_outflow = np.arange(4, dtype="float64") + 1.0
    return


# -------------------------------------------------------------------
# The active-HRU quantities are always derived from hru_type
# -------------------------------------------------------------------


@pytest.mark.domainless
def test_set_active_hrus_computes_from_hru_type():
    hru_type = np.array([1, 1, INACTIVE, 1, INACTIVE], dtype="int32")
    proc = make_process(hru_type)
    proc._set_active_hrus()
    assert (proc._active_hru_mask == (hru_type != INACTIVE)).all()
    assert proc._active_hru_mask.dtype == bool
    assert (proc._wh_active_hrus == np.array([0, 1, 3])).all()
    assert proc._nactive_hrus == 3
    assert isinstance(proc._nactive_hrus, int)


@pytest.mark.domainless
def test_active_hru_quantities_are_not_parameters():
    """The active-HRU quantities never enter the Parameters on the process.

    Supplying them does not make them parameters; hru_type does not become
    optional either.
    """
    hru_type = np.array([1, 1, INACTIVE, 1], dtype="int32")
    supplied = {
        "active_hru_mask": hru_type != INACTIVE,
        "wh_active_hrus": np.array([0, 1, 3]),
        "nactive_hrus": 3,
    }
    for supply in (None, supplied):
        proc = make_process(hru_type, supply)
        assert set(proc._params.parameters.keys()) == {"nhru", "hru_type"}
        for kk in ACTIVE_HRU_KEYS:
            assert kk not in proc._params.parameters.keys()


@pytest.mark.domainless
def test_set_active_hrus_ignores_supplied_values():
    """Supplied values disagreeing with hru_type are ignored, by design.

    hru_type is the single source of truth. A supplied mask that contradicts
    it is a contradiction, not a feature, so recomputation wins. This is
    pinned so it does not get "fixed" back.
    """
    hru_type = np.array([1, 1, 1, 1, 1], dtype="int32")  # all active
    supplied_mask = np.array([True, False, True, False, True])
    supplied = {
        "active_hru_mask": supplied_mask,
        "wh_active_hrus": np.array([0, 2, 4]),
        "nactive_hrus": 3,
    }
    # keep_supplied puts all three keys in front of _set_active_hrus, which
    # the Process parameter subsetting would otherwise drop
    proc = make_process(hru_type, supplied, keep_supplied=True)
    for kk in ACTIVE_HRU_KEYS:
        assert kk in proc._params.parameters.keys()

    proc._set_active_hrus()

    assert (proc._active_hru_mask == (hru_type != INACTIVE)).all()
    assert proc._active_hru_mask.all()
    assert (proc._wh_active_hrus == np.arange(len(hru_type))).all()
    assert proc._nactive_hrus == len(hru_type)
    assert isinstance(proc._nactive_hrus, int)

    # and specifically not the supplied values
    assert not (proc._active_hru_mask == supplied_mask).all()


@pytest.mark.domainless
def test_mask_inactive_hrus_ignores_supplied_mask():
    """Masking follows hru_type, not a contradicting supplied mask."""
    hru_type = np.array([1, 1, INACTIVE, 1, INACTIVE], dtype="int32")
    nhru = len(hru_type)
    supplied_mask = np.array([True, False, True, False, True])
    supplied = {
        "active_hru_mask": supplied_mask,
        "wh_active_hrus": np.array([0, 2, 4]),
        "nactive_hrus": 3,
    }
    proc = make_process(hru_type, supplied, keep_supplied=True)
    proc._set_active_hrus()
    set_variables(proc, nhru)

    proc._mask_inactive_hrus()

    active = hru_type != INACTIVE
    assert np.isnan(proc.soil_moist[~active]).all()
    assert not np.isnan(proc.soil_moist[active]).any()


@pytest.mark.domainless
def test_missing_required_param_still_raises():
    params = Parameters(
        dims={"nhru": 2},
        coords={"nhru": np.arange(2)},
        data_vars={"hru_area": np.ones(2)},
        metadata={"nhru": {"dims": ["nhru"]}, "hru_area": {"dims": ["nhru"]}},
        validate=True,
    )
    proc = _HruProcess()
    with pytest.raises(ValueError, match="required parameters"):
        proc._set_params(params, None)


# -------------------------------------------------------------------
# FIX 2: the early return happens when there is nothing to mask
# -------------------------------------------------------------------


@pytest.mark.domainless
def test_mask_inactive_hrus_all_active_early_out():
    hru_type = np.array([1, 1, 1, 1], dtype="int32")
    nhru = len(hru_type)
    proc = make_process(hru_type)
    proc._set_active_hrus()
    set_variables(proc, nhru)
    before_soil_moist = proc.soil_moist.copy()
    before_hru_ppt = proc.hru_ppt.data.copy()
    before_seg_outflow = proc.seg_outflow.copy()

    proc._mask_inactive_hrus()

    assert (proc.soil_moist == before_soil_moist).all()
    assert (proc.hru_ppt.data == before_hru_ppt).all()
    assert (proc.seg_outflow == before_seg_outflow).all()


@pytest.mark.domainless
def test_mask_inactive_hrus_all_inactive():
    """All inactive is fully masked; before the fix it was not masked at all"""
    hru_type = np.array([INACTIVE] * 4, dtype="int32")
    nhru = len(hru_type)
    proc = make_process(hru_type)
    proc._set_active_hrus()
    assert proc._nactive_hrus == 0
    set_variables(proc, nhru)
    before_seg_outflow = proc.seg_outflow.copy()

    proc._mask_inactive_hrus()

    assert np.isnan(proc.soil_moist).all()
    assert np.isnan(proc.hru_ppt.data).all()
    # non-nhru variables are untouched
    assert (proc.seg_outflow == before_seg_outflow).all()


@pytest.mark.domainless
def test_mask_inactive_hrus_some_inactive():
    hru_type = np.array([1, 1, INACTIVE, 1, INACTIVE], dtype="int32")
    nhru = len(hru_type)
    active = hru_type != INACTIVE
    proc = make_process(hru_type)
    proc._set_active_hrus()
    set_variables(proc, nhru)
    before_soil_moist = proc.soil_moist.copy()
    before_hru_ppt = proc.hru_ppt.data.copy()

    proc._mask_inactive_hrus()

    assert np.isnan(proc.soil_moist[~active]).all()
    assert (proc.soil_moist[active] == before_soil_moist[active]).all()
    assert np.isnan(proc.hru_ppt.data[:, ~active]).all()
    assert (proc.hru_ppt.data[:, active] == before_hru_ppt[:, active]).all()
    # non-nhru variables are untouched
    assert (proc.seg_outflow == np.arange(4, dtype="float64") + 1.0).all()
