from pywatershed.base.timeseries import TimeseriesArray

from ..base import meta
from ..constants import mask_fill_values_dict
from ..utils.preprocess_gridded import get_active_hru_params


class HruMixin:
    """Mixin for HRU functionalities."""

    def _set_active_hrus(self) -> None:
        """Set _active_hru_mask, _wh_active_hrus, and _nactive_hrus.

        All three are derived from the hru_type parameter, via
        :func:`~pywatershed.utils.preprocess_gridded.get_active_hru_params`,
        every time this method is called. They are set on self as private,
        derived quantities; they are neither read from nor tracked in the
        Parameters object. Values of the same names present in a Parameters
        object are ignored: hru_type is the single source of truth for which
        HRUs are active.

        Returns:
            None
        """
        result = get_active_hru_params(self._params.parameters["hru_type"])
        for kk in ("active_hru_mask", "wh_active_hrus", "nactive_hrus"):
            self[f"_{kk}"] = result[kk]

        return

    def _mask_inactive_hrus(self) -> None:
        """Set all variables to missing values outside of _active_hru_mask."""
        if self._active_hru_mask.all():
            # nothing to mask
            return

        for var_name in self.get_variables():
            var_dims = list(meta.get_dimensions(var_name).values())[0]
            if "nhru" not in var_dims:
                continue

            var = self[var_name]
            if isinstance(var, TimeseriesArray):
                # data are (ntimes, nhru)
                var.data[:, ~self._active_hru_mask] = mask_fill_values_dict[
                    var.data.dtype
                ]
            else:
                axis = var_dims.index("nhru")
                index = [slice(None)] * var.ndim
                index[axis] = ~self._active_hru_mask
                var[tuple(index)] = mask_fill_values_dict[var.dtype]

        return
