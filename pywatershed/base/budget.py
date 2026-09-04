import pathlib as pl
from copy import deepcopy
from typing import Literal, Union
from warnings import warn

import netCDF4 as nc4
import numpy as np

from pywatershed.base.control import Control

from ..constants import zero
from ..utils.netcdf_utils import NetCdfWrite, suppress_netcdf4_shape_warning
from .accessor import Accessor
from .parameters import Parameters

# Todo
# * documentation
# * get time_units / time_step from control
# * decide on the fate of "from storage unit"
# * terminology: "balance" just means "storage change"
# * terminology: what is the term for the budget dosent close/zero?


class Budget(Accessor):
    """Budget class for mass and energy conservation.

    ``active_mask`` restricts the balance check to active HRUs: a boolean
    array over ``nhru``, or ``False`` (the default) to check every HRU.

    Currently no energy budget has been implmenented, todo.
    """

    def __init__(
        self,
        control: Control,
        inputs: Union[list, dict],
        outputs: Union[list, dict],
        storage_changes: Union[list, dict],
        exchanges: Union[list, dict] = None,
        init_accumulations: dict = None,
        accum_start_time: np.datetime64 = None,
        units: dict = None,
        time_unit: str = "D",  # TODO: get this from control
        description: str = None,
        rtol: float = 1e-5,
        atol: float = 1e-5,
        basis: Literal["unit", "global"] = "unit",
        imbalance_fatal: bool = False,
        ignore_nans: bool = False,
        active_mask: Union[bool, np.ndarray] = False,
        unit_desc: str = "",
        verbose: bool = True,
    ):
        self.name = "Budget"
        self.control = control
        self.inputs = self.init_component(inputs)
        self.outputs = self.init_component(outputs)
        self.storage_changes = self.init_component(storage_changes)

        # Handle optional exchanges component (for bi-directional fluxes)
        if exchanges is None:
            self.exchanges = {}
            self._has_exchanges = False
        else:
            self.exchanges = self.init_component(exchanges)
            self._has_exchanges = True
        self.units = units
        self.time_unit = time_unit
        self.description = description
        self.rtol = rtol
        self.atol = atol
        self.imbalance_fatal = imbalance_fatal
        self._ignore_nans = ignore_nans
        # active_mask: False or a boolean np.ndarray where False indicates
        # inactive locations excluded from balance checks and global sums
        # (e.g. inactive HRUs whose variables are masked to nan).
        self.active_mask = active_mask
        self._unit_desc = unit_desc
        if self._unit_desc != "":
            self._unit_desc = f" ({self._unit_desc})"
        else:
            self._unit_desc = " "

        self.verbose = verbose
        self.basis = basis

        self._output_netcdf = False
        self._inputs_sum = None
        self._outputs_sum = None
        self._storage_changes_sum = None
        self._exchanges_sum = None
        self._balance = None
        self._accumulations = None
        self._accumulations_sum = None
        self._zero_sum = None

        self._time = self.control.current_time
        self._itime_step = self.control.itime_step

        # metadata
        all_vars = [list(self[cc].keys()) for cc in self.get_components()]
        all_vars = [x for xs in all_vars for x in xs]
        self.meta = self.control.meta.get_vars(all_vars)
        if len(self.meta) == len(all_vars):
            if self.units is None:
                all_units = [val["units"] for val in self.meta.values()]
            else:
                all_units = list(self.units.values())
            # check consistent units
            if not (np.array(all_units) == all_units[0]).all():
                msg = "Units not consistent over all terms"
                raise ValueError(msg)
            self.units = all_units[0]
        else:
            all_vars_0 = [vv[0] == "_" for vv in all_vars]
            all_vars_private = all(all_vars_0)
            if not all_vars_private or verbose:
                msg = (
                    f"Metadata unavailable for some Budget terms in {all_vars}"
                )
                warn(msg)
            self.units = None

        # generate metadata for derived output variables
        self.output_vars_desc = {
            "inputs_sum": f"Sum of input fluxes ({self.basis})",
            "outputs_sum": f"Sum of output fluxes ({self.basis})",
            "storage_changes_sum": f"Sum of storage changes ({self.basis})",
            # "balance":
            #     f"Balance of fluxes and storage changes ({self.basis})",
        }

        sum_keys = list(self.output_vars_desc.keys())
        for kk in sum_keys:
            terms = self.terms[kk[0:-4]]
            if terms is None or not len(terms):
                del self.output_vars_desc[kk]

        for kk, desc in self.output_vars_desc.items():
            terms = self.terms[kk[0:-4]]
            term = terms[0]
            if term in self.meta.keys():
                term_meta = self.meta[term]
                self.meta[kk] = deepcopy(term_meta)
                self.meta[kk]["desc"] = desc
                self.meta[kk]["units"] = self.units
            elif verbose:
                msg = f"budget term {term} not available in metadata"
                warn(msg)
            # if (var == "balance") and (self.basis == "global"):
            #    self.meta[var]["dimensions"] = {0: "one"}

        self.set_initial_accumulations(init_accumulations, accum_start_time)
        return

    @staticmethod
    def init_component(data: Union[list, dict]) -> dict:
        if isinstance(data, dict):
            return data
        else:
            return {dd: None for dd in data}

    def set(self, data: dict):
        """Set the data on the components after initialization
        Args:
            data: a dict of dicts with top level optional keys:
            [inputs, outputs, storage_changes].
            Each of those is a dict with var: np.ndarray, eg.
            data = {'inputs': {'var': np.ndarray}}
        """
        for comp_name, comp_dict in data.items():
            for var_name, var_data in comp_dict.items():
                if self[comp_name][var_name] is not None:
                    msg = (
                        f"Component '{comp_name}' variable '{var_name}'"
                        f"has already been set and should not be reset."
                    )
                    raise ValueError(msg)
                elif var_name not in self[comp_name].keys():
                    msg = (
                        f"Component '{comp_name}' has no variable '{var_name}'"
                    )
                    raise KeyError(msg)
                else:
                    self[comp_name][var_name] = var_data

    @classmethod
    def from_storage_unit(cls, storage_unit, quantity="mass", **kwargs):
        # Get budget terms based on quantity
        if quantity == "mass":
            budget_terms = storage_unit.get_mass_budget_terms()
        elif quantity == "energy":
            budget_terms = storage_unit.get_energy_budget_terms()
        else:
            raise ValueError(f"Unknown quantity: {quantity}")

        for component in budget_terms.keys():
            kwargs[component] = {}
            for var in budget_terms[component]:
                kwargs[component][var] = storage_unit[var]

        return Budget(storage_unit.control, **kwargs)

    @staticmethod
    def get_meta_keys():
        """Return a tuple of the metadata keys used by Budget."""
        return ("desc", "modules", "var_category", "units")

    @staticmethod
    def get_components(has_exchanges=False):
        if has_exchanges:
            return ("inputs", "exchanges", "outputs", "storage_changes")
        return ("inputs", "outputs", "storage_changes")

    @property
    def components(self):
        return self.get_components(self._has_exchanges)

    @property
    def inputs_sum(self):
        return self._inputs_sum

    @property
    def outputs_sum(self):
        return self._outputs_sum

    @property
    def storage_changes_sum(self):
        return self._storage_changes_sum

    @property
    def exchanges_sum(self):
        return self._exchanges_sum

    @property
    def terms(self):
        return {comp: list(self[comp].keys()) for comp in self.components}

    def set_initial_accumulations(self, init_accumulations, accum_start_time):
        self._itime_accumulated = self._itime_step  # -1
        self._time_accumulated = self._time  # None
        self._accumulations = {}
        self._accumulations_sum = {}

        # init to zero
        self.reset_accumulations()

        if init_accumulations is None:
            self._accum_start_time = self.control.init_time
            return

        self._accum_start_time = accum_start_time
        for component in self.components:
            if component not in init_accumulations.keys():
                continue
            for var in self[component].keys():
                if var in init_accumulations[component].keys():
                    self._accumulations[component][var] = init_accumulations[
                        component
                    ][var]

        self._sum_component_accumulations()
        return

    def reset_accumulations(self):
        self._accum_start_time = self._time_accumulated
        for component in self.components:
            self._accumulations[component] = {}
            for var in self[component].keys():
                self._accumulations[component][var] = zero
        self._sum_component_accumulations()
        return

    def advance(self):
        """Advance time (taken from storageUnit)"""
        if self._itime_step >= self.control.itime_step:
            if self.verbose:
                msg = (
                    f"{self.name} did not advance because it is "
                    f"not behind control time"
                )
                print(msg)
            return

        self._itime_step = self.control.itime_step
        self._time = self.control.current_time

    def calculate(self):
        """Accumulate for the timestep."""
        if self._itime_accumulated >= self._itime_step:
            raise ValueError("Can not accumulate twice per timestep")

        self._inputs_sum = self._sum_inputs()
        self._outputs_sum = self._sum_outputs()
        self._storage_changes_sum = self._sum_storage_changes()
        if self._has_exchanges:
            self._exchanges_sum = self._sum_exchanges()

        # accumulate
        for component in self.components:
            for var in self[component].keys():
                self._accumulations[component][var] += self[component][
                    var
                ] * self.control.time_step.astype(
                    f"timedelta64[{self.time_unit}]"
                ).astype(int)

        self._sum_component_accumulations()

        # check balance
        if self.basis == "unit":
            self._balance = self._calc_unit_balance()
        elif self.basis == "global":
            self._balance = self._calc_global_balance()

        self._itime_accumulated = self._itime_step
        self._time_accumulated = self._time
        return

    def _sum_component_accumulations(self):
        # sum the individual component accumulations
        for component in self.components:
            self._accumulations_sum[component] = None
            for var in self[component].keys():
                if self._accumulations_sum[component] is None:
                    if self.basis == "unit":
                        self._accumulations_sum[component] = (
                            self._accumulations[component][var].copy()
                        )
                    elif self.basis == "global":
                        self._accumulations_sum[component] = (
                            self._accumulations[component][var].copy().sum()
                        )
                else:
                    if self.basis == "unit":
                        self._accumulations_sum[component] += (
                            self._accumulations[component][var]
                        )
                    elif self.basis == "global":
                        self._accumulations_sum[component] += (
                            self._accumulations[component][var].sum()
                        )

        return

    @property
    def accumulations(self):
        return self._accumulations

    def _sum(self, attr):
        """Sum over the individual terms in a budget component."""
        if self.basis == "unit":
            vals = [val for val in self[attr].values()]
            the_sum = sum(vals)
        elif self.basis == "global":
            # in global case, the variable dims dont need to match, collapse
            # to a scalar
            if self.active_mask is not False:
                vals = [
                    np.sum(val[self.active_mask])
                    for val in self[attr].values()
                ]
            else:
                vals = [sum(val) for val in self[attr].values()]
            the_sum = sum(vals)
        else:
            raise ValueError(f"self.basis '{self.basis}' is invalid")
        return the_sum

    def _sum_inputs(self):
        return self._sum("inputs")

    def _sum_outputs(self):
        return self._sum("outputs")

    def _sum_storage_changes(self):
        return self._sum("storage_changes")

    def _sum_exchanges(self):
        return self._sum("exchanges")

    def _calc_unit_balance(self):
        self._zero_sum = True

        # compare
        # lhs ?=? rhs
        # i + e ?=? o + ds  (e is optional term)
        # so that relative errors are not compared to
        rhs = self._outputs_sum + self._storage_changes_sum
        # LHS depends on if we have exchanges or not
        if self._has_exchanges:
            unit_balance = (
                self._inputs_sum + self._exchanges_sum - self._outputs_sum
            )
            lhs = self._inputs_sum + self._exchanges_sum
        else:
            unit_balance = self._inputs_sum - self._outputs_sum
            lhs = self._inputs_sum

        if self.active_mask is not False:
            # only check the balance at active locations; inactive locations
            # are masked to nan and excluded.
            lhs = lhs[self.active_mask]
            rhs = rhs[self.active_mask]

        # zero when ds is zero
        if not np.allclose(
            lhs,
            rhs,
            rtol=self.rtol,
            atol=self.atol,
            equal_nan=self._ignore_nans,
        ):
            self._zero_sum = False

            if self._ignore_nans:
                actual_nan = np.where(np.isnan(lhs), True, False)
                desired_nan = np.where(np.isnan(rhs), True, False)
                msg = "The nan values are not the same for the two arrays"
                assert (actual_nan == desired_nan).all(), msg

            abs_diff = abs(lhs - rhs)
            with np.errstate(divide="ignore", invalid="ignore"):
                rel_abs_diff = abs(abs_diff / rhs)

            abs_close = abs_diff < self.atol
            rel_close = rel_abs_diff < self.rtol
            rel_close = np.where(np.isnan(rel_close), False, rel_close)

            close = abs_close | rel_close
            if self._ignore_nans:
                close = np.where(np.isnan(abs_diff), True, close)

            wh_not_close = np.where(~close)[0]
            if self.active_mask is not False:
                # map compressed positions back to full HRU indices
                wh_not_close = np.flatnonzero(self.active_mask)[wh_not_close]

            msg = (
                "The flux unit balance not equal to the change in unit "
                f"storage at time {self.control.current_time} and at the "
                f"following locations for {self.description}: {wh_not_close}"
            )

            if self.imbalance_fatal:
                raise ValueError(msg)
            else:
                warn(msg, UserWarning)

        # <<
        return unit_balance

    def _calc_global_balance(self):
        global_balance = self._inputs_sum - self._outputs_sum
        self._zero_sum = True
        # compare i ?=? o + ds so that relative errors are not compared to
        # zero when ds is zero
        if not np.allclose(
            self._inputs_sum,
            self._outputs_sum + self._storage_changes_sum,
            rtol=self.rtol,
            atol=self.atol,
        ):
            self._zero_sum = False
            msg = (
                "The global flux balance not equal to the change in global "
                f"storage: {self.description}"
            )
            if self.verbose:
                aerr = self._inputs_sum - (
                    self._outputs_sum + self._storage_changes_sum
                )
                rerr = aerr / self._inputs_sum
                msg += f"\n{self.control.current_time}: {aerr=}, {rerr=}"

            if self.imbalance_fatal:
                raise ValueError(msg)
            else:
                warn(msg, UserWarning)

        return global_balance

    @property
    def balance(self):
        return self._balance

    def __repr__(self):
        """Budget string representation"""
        if self._itime_step == -1:
            msg = (
                f"Budget (units: {self.units}) of {self.description} "
                f"only initialized. Check back later."
            )
            return msg

        # Determine which components to display
        components_to_display = []
        if len(self.inputs) > 0:
            components_to_display.append(("inputs", self.inputs))
        if self._has_exchanges and len(self.exchanges) > 0:
            components_to_display.append(("exchanges", self.exchanges))
        if len(self.outputs) > 0:
            components_to_display.append(("outputs", self.outputs))
        if len(self.storage_changes) > 0:
            components_to_display.append(
                ("storage_changes", self.storage_changes)
            )

        # Calculate column widths
        col_widths = {}
        col_extra = 12  # ': ' + scientific notation space
        for comp_name, comp_dict in components_to_display:
            max_key_len = (
                max([len(k) for k in comp_dict.keys()]) if comp_dict else 5
            )
            col_widths[comp_name] = max_key_len + col_extra

        indent = " " * 9
        col_sep = "    "
        total_width = (
            sum(col_widths.values())
            + len(col_sep) * (len(components_to_display) - 1)
            + 9
        )

        # Build output
        summary = ["*-" * int(total_width / 2)]

        # Header
        if self.basis == "unit":
            summary += [
                f"Individual spatial unit budget for {self.description} "
                f"(units: {self.units}).\n"
                "Budget is checked on each spatial unit. This is summary shows"
                "\nspatial sums for the entire model domain.\n"
            ]
        elif self.basis == "global":
            summary += [
                f"Global budget of {self.description} (units: {self.units}).\n"
                "Budget is checked on full domain: spatially summed fluxes\n"
                "and storage changes are checked for balance."
            ]

        summary += [f"@ time: {self._time} (itime_step: {self._itime_step})"]
        summary += ["", "This timestep:"]

        # Column headers
        header_line = indent
        for comp_name, _ in components_to_display:
            display_name = comp_name.replace("_", " ")
            header_line += display_name.ljust(col_widths[comp_name]) + col_sep
        summary += [header_line]

        # Separator
        sep_line = indent
        for comp_name, _ in components_to_display:
            sep_line += "-" * col_widths[comp_name] + col_sep
        summary += [sep_line]

        # Data rows
        max_rows = max(
            [len(comp_dict) for _, comp_dict in components_to_display]
        )
        for row_idx in range(max_rows):
            line = indent
            for comp_name, comp_dict in components_to_display:
                keys = list(comp_dict.keys())
                vals = list(comp_dict.values())
                if row_idx < len(keys):
                    key = keys[row_idx]
                    val_sum = vals[row_idx].sum()
                    # Format value
                    precision = col_widths[comp_name] - len(key) - 8
                    if val_sum < 0:
                        precision -= 1
                    val_str = np.format_float_scientific(
                        val_sum, precision=precision
                    )
                    line += (
                        f"{key}: {val_str}".ljust(col_widths[comp_name])
                        + col_sep
                    )
                else:
                    line += " " * col_widths[comp_name] + col_sep
            summary += [line]

        # Balance line
        summary += [
            indent
            + "-"
            * (
                sum(col_widths.values())
                + len(col_sep) * (len(components_to_display) - 1)
            )
        ]

        eq_op = "=" if self._zero_sum else "!=!"
        balance_line = "Balance: "

        # Build balance equation based on presence of exchanges
        balance_parts = []
        for comp_name, _ in components_to_display:
            comp_sum = getattr(self, f"_{comp_name}_sum")
            if comp_sum is None:
                comp_sum = np.float64(0.0)
            total = comp_sum.sum() if hasattr(comp_sum, "sum") else comp_sum

            # Determine operator
            if comp_name == "inputs":
                op = ""
            elif comp_name == "exchanges":
                op = "+"
            elif comp_name == "outputs":
                op = "-"
            elif comp_name == "storage_changes":
                # For exchanges: storage_changes gets subtracted like outputs
                # For no exchanges: storage_changes gets the eq_op (= or !=!)
                op = "-" if self._has_exchanges else eq_op
            else:
                op = "+"

            # Format value
            key_width = max(
                [len(k) for k in list(components_to_display[0][1].keys())]
            )
            precision = col_widths[comp_name] - key_width - 8
            if total < 0:
                precision -= 1
            val_str = np.format_float_scientific(
                total, precision=max(1, precision)
            )

            balance_parts.append(
                f"{op} ".ljust(key_width + 2) + val_str.rjust(10)
            )

        # Calculate residual
        if self._has_exchanges:
            residual = (
                (
                    self._inputs_sum.sum()
                    if hasattr(self._inputs_sum, "sum")
                    else self._inputs_sum
                )
                + (
                    self._exchanges_sum.sum()
                    if hasattr(self._exchanges_sum, "sum")
                    else self._exchanges_sum
                )
                - (
                    self._outputs_sum.sum()
                    if hasattr(self._outputs_sum, "sum")
                    else self._outputs_sum
                )
                - (
                    self._storage_changes_sum.sum()
                    if hasattr(self._storage_changes_sum, "sum")
                    else (
                        self._storage_changes_sum
                        if self._storage_changes_sum is not None
                        else 0.0
                    )
                )
            )
        else:
            residual = (
                (
                    self._inputs_sum.sum()
                    if hasattr(self._inputs_sum, "sum")
                    else self._inputs_sum
                )
                - (
                    self._outputs_sum.sum()
                    if hasattr(self._outputs_sum, "sum")
                    else self._outputs_sum
                )
                - (
                    self._storage_changes_sum.sum()
                    if hasattr(self._storage_changes_sum, "sum")
                    else (
                        self._storage_changes_sum
                        if self._storage_changes_sum is not None
                        else 0.0
                    )
                )
            )

        # Only show residual column if there are exchanges
        if self._has_exchanges:
            residual_str = np.format_float_scientific(residual, precision=1)
            residual_op = "=" if self._zero_sum else "!=!"
            summary += [
                balance_line
                + col_sep.join(balance_parts)
                + col_sep
                + f"{residual_op} {residual_str}".rjust(12)
            ]
        else:
            summary += [balance_line + col_sep.join(balance_parts)]

        # Accumulations
        summary += ["", f"Accumulations  (since {self._accum_start_time}):"]

        # Accumulation header
        header_line = indent
        for comp_name, _ in components_to_display:
            display_name = comp_name.replace("_", " ")
            header_line += display_name.ljust(col_widths[comp_name]) + col_sep
        summary += [header_line, sep_line]

        # Accumulation data
        for row_idx in range(max_rows):
            line = indent
            for comp_name, comp_dict in components_to_display:
                keys = list(comp_dict.keys())
                if row_idx < len(keys):
                    key = keys[row_idx]
                    acc_val_sum = self._accumulations[comp_name][key].sum()
                    precision = col_widths[comp_name] - len(key) - 8
                    if acc_val_sum < 0:
                        precision -= 1
                    val_str = np.format_float_scientific(
                        acc_val_sum, precision=precision
                    )
                    line += (
                        f"{key}: {val_str}".ljust(col_widths[comp_name])
                        + col_sep
                    )
                else:
                    line += " " * col_widths[comp_name] + col_sep
            summary += [line]

        # Accumulation balance
        summary += [
            indent
            + "-"
            * (
                sum(col_widths.values())
                + len(col_sep) * (len(components_to_display) - 1)
            )
        ]

        balance_line = "Balance: "
        balance_parts = []
        for comp_name, _ in components_to_display:
            acc_sum = self._accumulations_sum[comp_name]
            if acc_sum is None:
                acc_sum = np.float64(0.0)
            total = acc_sum.sum() if hasattr(acc_sum, "sum") else acc_sum

            if comp_name == "inputs":
                op = ""
            elif comp_name == "exchanges":
                op = "+"
            elif comp_name == "outputs":
                op = "-"
            elif comp_name == "storage_changes":
                op = "-" if self._has_exchanges else eq_op
            else:
                op = "+"

            key_width = max(
                [len(k) for k in list(components_to_display[0][1].keys())]
            )
            precision = col_widths[comp_name] - key_width - 8
            if total < 0:
                precision -= 1
            val_str = np.format_float_scientific(
                total, precision=max(1, precision)
            )

            balance_parts.append(
                f"{op} ".ljust(key_width + 2) + val_str.rjust(10)
            )

        # Calculate accumulated residual
        if self._has_exchanges:
            acc_residual = (
                (
                    self._accumulations_sum["inputs"].sum()
                    if hasattr(self._accumulations_sum["inputs"], "sum")
                    else self._accumulations_sum["inputs"]
                )
                + (
                    self._accumulations_sum["exchanges"].sum()
                    if hasattr(self._accumulations_sum["exchanges"], "sum")
                    else self._accumulations_sum["exchanges"]
                )
                - (
                    self._accumulations_sum["outputs"].sum()
                    if hasattr(self._accumulations_sum["outputs"], "sum")
                    else self._accumulations_sum["outputs"]
                )
                - (
                    self._accumulations_sum["storage_changes"].sum()
                    if hasattr(
                        self._accumulations_sum["storage_changes"], "sum"
                    )
                    else (
                        self._accumulations_sum["storage_changes"]
                        if self._accumulations_sum["storage_changes"]
                        is not None
                        else 0.0
                    )
                )
            )
        else:
            acc_residual = (
                (
                    self._accumulations_sum["inputs"].sum()
                    if hasattr(self._accumulations_sum["inputs"], "sum")
                    else self._accumulations_sum["inputs"]
                )
                - (
                    self._accumulations_sum["outputs"].sum()
                    if hasattr(self._accumulations_sum["outputs"], "sum")
                    else self._accumulations_sum["outputs"]
                )
                - (
                    self._accumulations_sum["storage_changes"].sum()
                    if hasattr(
                        self._accumulations_sum["storage_changes"], "sum"
                    )
                    else (
                        self._accumulations_sum["storage_changes"]
                        if self._accumulations_sum["storage_changes"]
                        is not None
                        else 0.0
                    )
                )
            )

        # Only show residual column if there are exchanges
        if self._has_exchanges:
            acc_residual_str = np.format_float_scientific(
                acc_residual, precision=1
            )
            summary += [
                balance_line
                + col_sep.join(balance_parts)
                + col_sep
                + f"{residual_op} {acc_residual_str}".rjust(12),
                "",
            ]
        else:
            summary += [balance_line + col_sep.join(balance_parts), ""]

        return "\n".join(summary)

    def output(self) -> None:
        """Output to previously initialized output types.

        Returns:
            None

        """
        if self._output_netcdf:
            self.__output_netcdf()
        return

    def initialize_netcdf(
        self,
        params: Parameters,
        output_dir: str,
        extra_coords: dict = None,
        write_sum_vars: Union[list, bool] = True,
        write_individual_vars: bool = False,
    ) -> None:
        """Initialize NetCDF output

        Args:
            output_dir: directory for NetCDF file

        Returns:
            None

        """
        self._output_netcdf = True
        if extra_coords is None:
            extra_coords = {}
        # make working directory
        output_dir = pl.Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        nc_path = pl.Path(output_dir) / f"{self.description}_budget.nc"

        # Construct a dictionary of {term: var}. If the variables are not
        # in a term their term is None
        if write_sum_vars is True:
            nc_out_vars = list(self.output_vars_desc.keys())
        elif isinstance(write_sum_vars, list):
            nc_out_vars = [
                var
                for var in self.output_vars_desc.keys()
                if var in write_sum_vars
            ]
        elif (write_sum_vars is False) or (write_sum_vars is None):
            nc_out_vars = []
        else:
            msg = "Unexpected value of write_sum_vars: {write_sum_vars}"
            raise ValueError(msg)

        self._netcdf_output_var_dict = {}
        if len(nc_out_vars):
            self._netcdf_output_var_dict = {None: nc_out_vars}
        if write_individual_vars:
            self._netcdf_output_var_dict = {
                None: nc_out_vars,
                **self.terms,
            }

        if len(self._netcdf_output_var_dict) == 0:
            msg = (
                f"Budget for {self.description} has no requested output, "
                "setting self._output_netcdf = False"
            )
            warn(msg)
            self._output_netcdf = False
            return

        global_attrs = {
            "Description": (
                f"pywatershed ({self.basis}) budget for {self.description}"
            ),
            "Budget basis": f"{self.basis} (unit or global)",
        }
        for key in self.terms.keys():
            global_attrs[key] = "[" + ", ".join(self.terms[key]) + "]"

        if self.basis == "unit":
            coordinates = params.coords
            meta = self.meta
        else:
            coordinates = {"one": 0}
            meta = deepcopy(self.meta)
            for kk, vv in meta.items():
                meta[kk]["dims"] = ("one",)

        self._netcdf = NetCdfWrite(
            nc_path,
            coordinates,
            self._netcdf_output_var_dict,
            meta,
            extra_coords=extra_coords,
            global_attrs=global_attrs,
        )

        # todo jlm: put terms in to metadata
        return

    def __output_netcdf(self) -> None:
        """Output variable data for a time step

        Returns:
            None

        """
        if self._output_netcdf:
            self._netcdf.time[self.control.itime_step] = nc4.date2num(
                self.control.current_datetime, self._netcdf.time.units
            )
            for nc_group, group_vars in self._netcdf_output_var_dict.items():
                for nc_var in group_vars:
                    if nc_group is None:
                        var_path = nc_var
                        value = self[nc_var]
                    else:
                        var_path = f"{nc_group}/{nc_var}"
                        value = self[nc_group][nc_var]

                    with suppress_netcdf4_shape_warning():
                        self._netcdf.dataset[var_path][
                            self.control.itime_step, :
                        ] = value

        return

    def _finalize_netcdf(self) -> None:
        if self._output_netcdf:
            self._netcdf.close()

        return
