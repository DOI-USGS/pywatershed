import pathlib as pl
from typing import Literal, Union
from warnings import warn

import numpy as np

from ..base import meta
from ..base.adapter import Adapter
from ..base.budget import Budget
from ..parameters import Parameters
from .control import Control
from .process import Process

# Deprecation message for budget property
_BUDGET_DEPRECATION_MSG = (
    "The 'budget' property is deprecated and will be removed in the next "
    "major release. Please use 'mass_budget' or 'energy_budget' explicitly."
)


class ConservativeProcess(Process):
    """Base class for representation of conservative physical processes.

    ConservativeProcess is a base class for mass and energy conservation which
    extends the :func:`~pywatershed.base.Process` class with budgets for
    mass and/or energy. Please see :func:`~pywatershed.base.Process`
    for many details on the design of this parent class. In ConservativeProcess
    both mass and energy conservation can be tracked. Budgets can optionally be
    established for mass and/or energy and these can be enforced or
    simply diagnosed with the model run.

    Conventions are adopted through the use of the following
    properties/methods:

    mass_budget_terms/get_mass_budget_terms():
        These terms must all in in the same units across all components of
        the budget (inputs, outputs, storage_changes). Diagnostic variables
        should not appear in the budget terms, only prognostic variables
        should.

    _calculate():
        This method is to be overridden by the subclass. Near the end of
        the method, the subclass should calculate its changes in mass and
        energy storage in an obvious way. As commented for
        mass_budget_terms, storage changes should only be tracked for
        prognostic variables. (For example is snow_water_equiv = snow_ice +
        snow_liquid, then storage changes for snow_ice and snow_liquid
        should be tracked and not for snow_water_equiv).

    See Also
    --------
    pywatershed.base.Process
    pywatershed.base.Budget

    Args
    ----
    control:
        A Control object
    discretization:
        A discretization object
    parameters:
        The parameters for this object
    imbalance_behavior: one of ["defer", None, "warn", "error"] with
        "defer" being the default and defering to
        control.options["imbalance_behavior"] when available. When
        control.options["imbalance_behavior"] is not avaiable,
        imbalance_behavior is set to "warn".
    metadata_patches:
        Override static metadata for any public parameter or variable --
        experimental.
    metadata_patch_conflicts:
        How to handle metadata_patches conflicts. Experimental.
    restart_read:
        May be boolean or a Pathlib.Path. If False, control.options
        will be examined for this key. If True, the working
        directory is searched for restart files. If a Pathlib.Path, this
        specifies an alternative directory to search for restart files.
        Files searched for are of the pattern YYYY-mm-dd-varname.nc where the
        date is the control.init_time. The timestamp on the file is the valid
        time of the states in the file with the exception of processes with
        sub-daily timesteps. For example, the outflow_ts variable of
        PRMSChannel is instantaneous and valid at the 23rd hour of the
        timestampped day whereas its variable seg_outflow is the daily averge
        value over the timestampped day.
    restart_write:
        As for restart_read but for writing. The directory in either
        case will be attempted to be created if it does not exist.
    restart_write_freq:
        If False, then control.options is examined for this key. The follwing
        values set the frequency of restart output with "y" for yearly, "m"
        for monthly, "d" for daily, or "f" for final. "Final" means that
        restart files are written with the states at control.end_time to files
        timestampped with control.end_time. Yearly and monthly restart options
        write files with timestamps on the last day of each year or month
        during the run. If daily, restarts are written every day. If
        restart_write is not False and restart_write_freq is False, the default
        of "f" is used.

    """

    def __init__(
        self,
        control: Control,
        discretization: Parameters,
        parameters: Parameters,
        imbalance_behavior: Literal["defer", None, "warn", "error"] = "defer",
        input_aliases: dict = None,
        metadata_patches: dict[dict] = None,
        metadata_patch_conflicts: Literal["left", "warn", "error"] = "error",
        restart_read: Union[pl.Path, bool] = False,
        restart_write: Union[pl.Path, bool] = False,
        restart_write_freq: Literal["y", "m", "d", "f", False] = False,
    ):
        super().__init__(
            control=control,
            discretization=discretization,
            parameters=parameters,
            input_aliases=input_aliases,
            metadata_patches=metadata_patches,
            metadata_patch_conflicts=metadata_patch_conflicts,
            restart_read=restart_read,
            restart_write=restart_write,
            restart_write_freq=restart_write_freq,
        )

        if not hasattr(self, "name"):
            self.name = "ConservativeProcess"

        # Initialize budget attributes
        self._mass_budget = None
        self._energy_budget = None

        return

    def output(self) -> None:
        super().output()
        if self._mass_budget is not None:
            self._mass_budget.output()
        if self._energy_budget is not None:
            self._energy_budget.output()
        return

    def finalize(self) -> None:
        super().finalize()
        if self._mass_budget is not None:
            self._mass_budget._finalize_netcdf()
        if self._energy_budget is not None:
            self._energy_budget._finalize_netcdf()
        return

    @classmethod
    def get_mass_budget_terms(cls) -> dict:
        """Get a dictionary of variable names for mass budget terms."""
        # TODO: this is nice in theory but this is probably better as a
        # staticmethod from the POV of helping users. It could be nice to have
        # this code to check that the staticmethod definition matches the
        # metadata.
        mass_budget_terms = {
            "inputs": list(
                meta.filter_vars(
                    cls.get_inputs(), "var_category", "mass flux"
                ).keys()
            ),
            "outputs": list(
                meta.filter_vars(
                    cls.get_variables(), "var_category", "mass flux"
                ).keys()
            ),
            "storage_changes": list(
                meta.filter_vars(
                    cls.get_variables(), "var_category", "mass storage change"
                ).keys()
            ),
        }
        return mass_budget_terms

    @classmethod
    def get_energy_budget_terms(cls) -> dict:
        """Get a dictionary of variable names for energy budget terms."""
        # Default implementation returns empty - subclasses override
        energy_budget_terms = {
            "inputs": [],
            "outputs": [],
            "storage_changes": [],
        }
        return energy_budget_terms

    @property
    def mass_budget_terms(self) -> dict:
        """A dictionary of variable names for the mass budget terms."""
        return self.get_mass_budget_terms()

    @property
    def energy_budget_terms(self) -> dict:
        """A dictionary of variable names for the energy budget terms."""
        return self.get_energy_budget_terms()

    @property
    def mass_budget(self):
        """The mass budget for this process, if enabled."""
        return self._mass_budget

    @property
    def energy_budget(self):
        """The energy budget for this process, if enabled."""
        return self._energy_budget

    @property
    def budget(self):
        """Legacy property for backward compatibility - returns mass budget.

        .. deprecated::
            The 'budget' property is deprecated. Use 'mass_budget' instead.
        """
        warn(_BUDGET_DEPRECATION_MSG, DeprecationWarning, stacklevel=2)
        return self._mass_budget

    @classmethod
    def description(cls) -> dict:
        """A dictionary description of this Process.

        Returns:
            All metadata for all variables in inputs, variables, parameters,
            mass_budget_terms, and energy_budget_terms for this Process.
        """
        desc = super().description()
        desc = desc | {
            "mass_budget_terms": cls.get_mass_budget_terms(),
            "energy_budget_terms": cls.get_energy_budget_terms(),
        }
        return desc

    def set_input_to_adapter(self, input_variable_name: str, adapter: Adapter):
        super().set_input_to_adapter(
            input_variable_name=input_variable_name, adapter=adapter
        )
        # Notes from the super()
        # can NOT use [:] on the LHS as we are relying on pointers between
        # boxes. [:] on the LHS here means it's not a pointer and then
        # requires that the calculation of the input happens before the
        # advance of this storage unit. But that gives the incorrect budget
        # for et.

        # Using a pointer between boxes means that the same pointer has to
        # be used for the budget, so there's no way to have a preestablished
        # pointer between Process and its budget. So this stuff...
        for budget in [self._mass_budget, self._energy_budget]:
            if budget is not None:
                for comp in budget.components:
                    if input_variable_name in budget[comp].keys():
                        # can not use [:] on the LHS?
                        budget[comp][input_variable_name] = self[
                            input_variable_name
                        ]

        return

    def _set_budget(
        self,
        basis: str = None,
        quantity: Literal["mass", "energy"] = "mass",
        ignore_nans: bool = False,
        active_mask: Union[bool, np.ndarray] = False,
        unit_desc: str = "",
    ):
        """Set up budget(s) for this process.

        Args:
            basis: "unit" or "global"
            quantity: Quantity to budget: "mass" or "energy"
            ignore_nans: Ignore NaN values in budget calculations
            active_mask: False or a boolean np.ndarray masking active
                locations (e.g. active HRUs); inactive locations are
                excluded from balance checks.
            unit_desc: Description of units for budget output
        """
        if basis is None:
            basis = "unit"

        if self._imbalance_behavior == "defer":
            if "imbalance_behavior" in self.control.options.keys():
                self._imbalance_behavior = self.control.options[
                    "imbalance_behavior"
                ]
            else:
                self._imbalance_behavior = "warn"

        if self._imbalance_behavior is None:
            self._mass_budget = None
            self._energy_budget = None
        elif self._imbalance_behavior in ["error", "warn"]:
            # Create mass budget if requested
            if quantity == "mass":
                units = {}
                for cc, vv_list in self.get_mass_budget_terms().items():
                    for vv in vv_list:
                        units[vv] = self.meta[vv]["units"]

                self._mass_budget = Budget.from_storage_unit(
                    self,
                    time_unit="D",
                    description=f"{self.name}_mass",
                    imbalance_fatal=(self._imbalance_behavior == "error"),
                    basis=basis,
                    ignore_nans=ignore_nans,
                    active_mask=active_mask,
                    units=units,
                    unit_desc=unit_desc,
                    quantity="mass",
                )

            # Create energy budget if requested
            if quantity == "energy":
                units = {}
                for cc, vv_list in self.get_energy_budget_terms().items():
                    for vv in vv_list:
                        units[vv] = self.meta[vv]["units"]

                self._energy_budget = Budget.from_storage_unit(
                    self,
                    time_unit="D",
                    description=f"{self.name}_energy",
                    imbalance_fatal=(self._imbalance_behavior == "error"),
                    basis=basis,
                    ignore_nans=ignore_nans,
                    units=units,
                    unit_desc=unit_desc,
                    quantity="energy",
                )
        else:
            raise ValueError(f"Illegal behavior: {self._imbalance_behavior}")

        return

    def calculate(self, time_length: float, **kwargs) -> None:
        super().calculate(time_length=time_length)

        # move to a timestep finalization method at some future date.
        if self._mass_budget is not None:
            self._mass_budget.advance()
            self._mass_budget.calculate()

        if self._energy_budget is not None:
            self._energy_budget.advance()
            self._energy_budget.calculate()

        return

    def initialize_netcdf(
        self,
        output_dir: [str, pl.Path] = None,
        separate_files: bool = None,
        budget_args: dict = None,
        output_vars: list = None,
        extra_coords: dict = None,
        addtl_output_vars: list = None,
    ) -> None:
        if self._netcdf_initialized:
            msg = (
                f"{self.name} class previously initialized netcdf output "
                f"in {self._netcdf_output_dir}"
            )
            warn(msg)
            return

        super().initialize_netcdf(
            output_dir=output_dir,
            separate_files=separate_files,
            output_vars=output_vars,
            extra_coords=extra_coords,
            addtl_output_vars=addtl_output_vars,
        )

        if budget_args is None:
            budget_args = {}

        if self._mass_budget is not None:
            mass_budget_args = budget_args.copy()
            mass_budget_args["output_dir"] = self._netcdf_output_dir
            mass_budget_args["params"] = self._params
            self._mass_budget.initialize_netcdf(**mass_budget_args)

        if self._energy_budget is not None:
            energy_budget_args = budget_args.copy()
            energy_budget_args["output_dir"] = self._netcdf_output_dir
            energy_budget_args["params"] = self._params
            self._energy_budget.initialize_netcdf(**energy_budget_args)

        return

    def _finalize_netcdf(self) -> None:
        """Finalize NetCDF output to disk.

        Returns:
            None
        """
        super()._finalize_netcdf()

        if self._mass_budget is not None and self._netcdf_initialized:
            self._mass_budget._finalize_netcdf()

        if self._energy_budget is not None and self._netcdf_initialized:
            self._energy_budget._finalize_netcdf()

        return
