import numpy as np
import xarray as xr

from ..constants import HruType
from ..parameters import Parameters


def preprocess_gridded_params(parameters: Parameters) -> Parameters:
    """Preprocess gridded parameters for active HRU masks, indices, and counts.

    Args:
        parameters (Parameters): The parameters to preprocess.

    Returns:
        Parameters: The preprocessed Parameters object
    """
    new_params = parameters.to_xr_ds()

    active_results = get_active_hru_params(parameters.parameters["hru_type"])
    # Explicit dimensions are required, as in preprocess_cascades.py. The
    # mask is on nhru, the indices are on their own dimension, and the count
    # is a scalar (as active_hrus is in preprocess_cascades.py).
    new_params["active_hru_mask"] = xr.Variable(
        "nhru", active_results["active_hru_mask"]
    )
    new_params["wh_active_hrus"] = xr.Variable(
        "nactive_hru", active_results["wh_active_hrus"]
    )
    new_params["nactive_hrus"] = xr.Variable(
        "scalar", np.array([active_results["nactive_hrus"]], dtype="int64")
    )

    ret_params = Parameters.from_ds(new_params)

    return ret_params


def get_active_hru_params(hru_type: np.ndarray) -> dict:
    """Get active HRU parameters.

    Args:
        hru_type (np.ndarray): The HRU type array.

    Returns:
        dict: A dictionary containing the active HRU mask, indices, and count.
            The indices are a 1-D integer index array into the nhru dimension.
    """
    active_hru_mask = hru_type != HruType.INACTIVE.value
    wh_active_hrus = np.where(active_hru_mask)[0]
    nactive_hrus = len(wh_active_hrus)

    return {
        "active_hru_mask": active_hru_mask,
        "wh_active_hrus": wh_active_hrus,
        "nactive_hrus": nactive_hrus,
    }
