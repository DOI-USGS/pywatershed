import inspect
import os
import pathlib as pl
from typing import Iterable, Literal, Union
from warnings import warn

import numpy as np

from ..base import meta
from ..base.adapter import Adapter, adapter_factory
from ..base.data_model import _merge_dicts
from ..base.timeseries import TimeseriesArray
from ..parameters import Parameters
from ..utils.netcdf_utils import NetCdfWrite
from .accessor import Accessor
from .control import Control


class Process(Accessor):
    """Base class for physical process representation.

    The  class aims to describe itself through its staticmethods and
    properties.

    Conventions are adopted through the use of the following
    properties/methods:

    inputs/get_inputs():
        List the names of variables required from external sources.
        Still working on conventions if these are to be modified.
        For an input to be successfully inicluded,
        that variable must be defined in the metadata
        (pywatershed/static/metadata/variables.yaml).
        Efforts should be made to not use diagnostic variables as input
        as much as possible.
    variables/get_variables():
        List the names of internal public variables. If not necessary,
        to be public, variables should be made private with a single,
        leading underscore and not maintained in this list. For an input
        to be successfully inicluded, that variable must be defined in the
        metadata (pywatershed/static/metadata/variables.yaml).
        Efforts should be made not to track diagnostic variables in this
        public variable set, as much as possible.
    parameters/get_parameters():
        List the names of parameters used by the subclass.
    description:
        Return a dictionary of with the process subclass name and its
        metadata for all variables for each of inputs, variables, and
        parameters.
    get_init_values:
        Return a dictionary of initialization values for variables.
        Note that these may be overridden by subclass initialization
        routines (e.g. using parameters) or by restart values. So these
        are not "initial values", they are initialization values that
        are set when the variable is declared from metadata in
        _initialize_var(). Initization values should be nan as much as
        possible.
    _advance_variables():
        This advance should exactly specify the prognostic variables in
        setting previous values to current values. When/if necessary to
        keep previous diagnostic variables, those must not appear here but
        in _calculate().
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
    pywatershed.base.ConservativeProcess

    Args
    ----
    control:
        A Control object
    discretization:
        A discretization object
    parameters:
        The parameters for this object
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
        input_aliases: dict = None,
        metadata_patches: dict[dict] = None,
        metadata_patch_conflicts: Literal["left", "warn", "error"] = "error",
        restart_read: Union[pl.Path, bool] = False,
        restart_write: Union[pl.Path, bool] = False,
        restart_write_freq: Literal["y", "m", "d", "f", False] = False,
    ):
        if not hasattr(self, "name"):
            self.name = "Process"

        # Maps internal input variable names to the variable name in the
        # source file. E.g. {"humidity_hru": "rhavg"} means the input
        # known internally as humidity_hru is stored as "rhavg" in the
        # netCDF file.  Can be supplied at Process or Model level.
        self._input_aliases = input_aliases or {}
        self.control = control

        self._set_params(parameters, discretization)

        # netcdf output variables
        self._netcdf_initialized = False

        self._itime_step = -1

        # TODO metadata patching.
        self._set_metadata()
        if metadata_patches is not None:
            self._patch_metadata(
                metadata_patches,
                conflicts=metadata_patch_conflicts,
            )

        # Below, can remove the condition checking in locals().keys() when all
        # processes have the opt.

        # For these options not passed, look in control.option. That is, if
        # the option is passed specifically, it is used.
        # This cant be done by setting locals()[] unfortunately.
        if "restart_read" in locals().keys() and restart_read is False:
            if "restart_read" in self.control.options.keys():
                restart_read = self.control.options["restart_read"]
        if "restart_write" in locals().keys() and restart_write is False:
            if "restart_write" in self.control.options.keys():
                restart_write = self.control.options["restart_write"]
        if (
            "restart_write_freq" in locals().keys()
            and restart_write_freq is False
        ):
            if "restart_write_freq" in self.control.options.keys():
                restart_write_freq = self.control.options["restart_write_freq"]

        if "restart_read" in locals().keys() and restart_read is not False:
            if restart_read is True:
                restart_path = pl.Path(".")
            else:
                restart_path = pl.Path(restart_read)
            # <
            self._restart_read = restart_path
        else:
            self._restart_read = False

        if "restart_write" in locals().keys() and restart_write is not False:
            if restart_write is True:
                restart_path = pl.Path(".")
            else:
                restart_path = pl.Path(restart_write)
            # <
            if not restart_path.exists():
                restart_path.mkdir(parents=True)
            self._restart_write = restart_path

            if restart_write_freq is False:
                restart_write_freq = "f"
            restart_write_freq_xform = {
                "y": "%j",
                "m": "%d",
                "d": "%H",
                "f": "f",
            }
            self._restart_write_strf_code = restart_write_freq_xform[
                restart_write_freq
            ]

        else:
            self._restart_write = False

        # <
        self._initialize_self_variables()
        self._set_initial_conditions()
        if self._restart_read:
            self._restart_from_file()
        # self._init_diagnostic_vars()

        return None

    def output(self) -> None:
        """Output data to previously initialized output types.
        Returns:
            None
        """
        if self._netcdf_initialized:
            if self._verbose:
                print(f"writing output for: {self.name}")
            self._output_netcdf()

        self._output_restart()

        return

    def finalize(self) -> None:
        """Finalize the Process, output methods, and close input adapters.

        Returns:
            None
        """
        if self._verbose:
            print(f"finalizing: {self.name}")

        # Close input adapters to release file handles
        for adapter in self._input_variables_dict.values():
            if hasattr(adapter, "close"):
                adapter.close()
            elif hasattr(adapter, "_nc_read") and hasattr(
                adapter._nc_read, "close"
            ):
                adapter._nc_read.close()

        self._finalize_netcdf()
        return

    @staticmethod
    def get_dimensions() -> tuple:
        """Get a tuple of dimension names for this Process."""
        raise NotImplementedError("This must be implemented")

    @staticmethod
    def get_parameters() -> tuple:
        """Get a tuple of parameter names for this Process."""
        raise NotImplementedError("This must be implemented")

    @staticmethod
    def get_inputs() -> tuple:
        """Get a tuple of input variable names for this Process."""
        raise NotImplementedError("This must be implemented")

    @classmethod
    def get_variables(cls) -> tuple:
        """Get a tuple of (public) variable names for this Process."""
        return list(cls.get_init_values().keys())

    @classmethod
    def description(cls) -> dict:
        """A dictionary description of this Process.

        Returns:
            All metadata for all variables in inputs, variables,
            and parameters."""
        return {
            "class_name": cls.__name__,
            "inputs": meta.get_vars(cls.get_inputs()),
            "variables": meta.get_vars(cls.get_variables()),
            "parameters": meta.get_params(cls.get_parameters()),
        }

    @staticmethod
    def get_restart_variables() -> list:
        """A list of restart varible names."""
        raise NotImplementedError("This must be implemented")

    @staticmethod
    def get_init_values() -> dict:
        """Get a dictionary of initialization values for each public
        variable."""
        raise NotImplementedError("This must be implemented")

    @property
    def dimensions(self) -> tuple:
        """A tuple of parameter names."""
        return self.get_dimensions()

    @property
    def parameters(self) -> tuple:
        """A tuple of parameter names."""
        return self.get_parameters()

    @property
    def inputs(self) -> tuple:
        """A tuple of input variable names."""
        return self.get_inputs()

    @property
    def variables(self) -> tuple:
        """A tuple of public variable names."""
        return self.get_variables()

    @property
    def restart_variables(self) -> dict:
        """A dict of restart variable names mapping current: previous."""
        return self.get_restart_variables()

    @property
    def init_values(self) -> dict:
        """A dictionary of initial values for each public variable."""
        return self.get_init_values()

    def _set_params(self, parameters, discretization):
        if hasattr(self, "_params"):
            return
        param_keys = set(parameters.variables.keys())
        missing_params = set(self.parameters).difference(param_keys)
        if missing_params:
            if discretization is not None:
                dis_keys = set(discretization.variables.keys())
                missing_params = missing_params - dis_keys
                all_missing_in_dis = (
                    missing_params.intersection(dis_keys) == missing_params
                )
            else:
                all_missing_in_dis = False

            if not all_missing_in_dis:
                raise ValueError(
                    "The following required parameters were not found in the "
                    f"parameter file: {missing_params}"
                )

            self._params = type(parameters).merge(parameters, discretization)
        else:
            self._params = parameters.subset(
                self.parameters, keep_dims=self.dimensions
            )

    def _initialize_self_variables(self):
        # dims
        for name in self.dimensions:
            if name == "ntime":
                setattr(self, name, self.control.n_times)
            else:
                setattr(self, name, self._params.dims[name])

        # parameters
        for name in self.parameters:
            setattr(self, name, self._params.get_param_values(name))

        # inputs
        for name in self.inputs:
            # dims of internal variables never have time, so they are spatial
            spatial_dims = self._params.get_dim_values(
                list(meta.find_variables(name)[name]["dims"])
            )
            spatial_dims = tuple(spatial_dims.values())
            setattr(self, name, np.zeros(spatial_dims, dtype=float) + np.nan)

        # variables
        for name in self.variables:
            self._initialize_var(name)

        return

    def _initialize_var(self, var_name: str, flt_to_dbl: bool = True):
        """Initialize a variable using get_init_values and metadata.

        Initialized variables can be for single time or they can be a timeries
        array object if they have a time dimension in metadata.
        """
        init_vals = self.get_init_values()
        if var_name not in init_vals.keys():
            if self._verbose:
                warn(
                    f"{var_name} not initialized (no initial value specified)"
                )
            return

        dims = [self[vv] for vv in self.meta[var_name]["dims"]]
        init_type = self.meta[var_name]["type"]

        if len(dims) == 1:
            self[var_name] = np.full(
                dims, init_vals[var_name], dtype=init_type
            )
        else:
            self[var_name] = TimeseriesArray(
                var_name=var_name,
                control=self.control,
                array=np.full(dims, init_vals[var_name], dtype=init_type),
                time=self._time,
            )
        return

    def _set_initial_conditions(self) -> None:
        """Set initial conditions for variables not in get_init_values"""
        raise Exception("This must be implemented")

    def _advance_variables(self) -> None:
        """Advance prognostic variables."""
        raise Exception("This must be implemented.")

    def _advance_inputs(self):
        for key, value in self._input_variables_dict.items():
            value.advance()
            self[key][:] = value.current

        return

    def _set_inputs(self, args):
        # An argument naming a model variable that is not one of this
        # class's inputs would otherwise be silently ignored.
        dropped = [
            kk
            for kk, vv in args.items()
            if vv is not None
            and kk in meta.variables
            and kk not in self.inputs
        ]
        if dropped:
            raise ValueError(
                f"{type(self).__name__} does not take {dropped} as "
                "input(s); they are not in its get_inputs()."
            )

        self._input_variables_dict = {}
        for ii in self.inputs:
            if args[ii] is None:
                # This should need no warning, just downstream consequences
                continue

            ii_dims = self.control.meta.get_dimensions(ii)[ii]
            # This accomodates Timeseries like objects that need to init
            # both full rank and reduced rank versions of their data
            # this is pretty adhoc
            check_list = ["time", "doy"]
            if len([mm for mm in check_list if mm in ii_dims[0]]):
                ii_dims = ii_dims[1:]

            file_var_name = self._input_aliases.get(ii, ii)
            self._input_variables_dict[ii] = adapter_factory(
                args[ii],
                variable_name=file_var_name,
                control=args["control"],
            )
            if self._input_variables_dict[ii]:
                self[ii] = self._input_variables_dict[ii].current

        return

    def _restart_from_file(self):
        from xarray import load_dataarray

        init_strftime = self.control.init_time.item().strftime("%Y-%m-%d")
        for vv in self.restart_variables:
            rst_file = self._restart_read / f"{init_strftime}-{vv}.nc"
            print(f"Restarting from file: {rst_file}")
            # Use decode_timedelta=False to prevent xarray from converting
            # float data with time-like units (e.g., "days") to timedelta64
            data = load_dataarray(rst_file, decode_timedelta=False).values
            if isinstance(self[vv], TimeseriesArray):
                self[vv].data[0, :] = data
            else:
                self[vv][:] = data

        return

    def _set_options(self, init_locals):
        """Set options on self if supplied on init, else take
        from control"""
        # some self and Process introspection reveals the option names
        init_arg_names = set(
            inspect.signature(self.__init__).parameters.keys()
        )
        process_init_arg_names = set(
            inspect.signature(Process.__init__).parameters.keys()
        )
        inputs_arg_names = set(self.inputs)

        non_option_args = process_init_arg_names.union(inputs_arg_names)

        # all process options should be set in the init
        # process_options = {
        #     "restart_read",
        #     "restart_write",
        #     "restart_write_freq",
        # }
        # option_names = (
        #     init_arg_names.difference(non_option_args) | process_options
        # )
        option_names = init_arg_names.difference(non_option_args)

        for opt in option_names:
            if opt in init_locals.keys() and init_locals[opt] is not None:
                self[f"_{opt}"] = init_locals[opt]
            elif opt in self.control.options.keys():
                self[f"_{opt}"] = self.control.options[opt]
            else:
                self[f"_{opt}"] = None

        return

    def set_input_to_adapter(self, input_variable_name: str, adapter: Adapter):
        """Set input variables to adapter.current and manage the adapter.

        TODO: make this private?

        Args:
            input_variable_name: key of input variable
            adapter: the Adapter for the input.
        """
        self._input_variables_dict[input_variable_name] = adapter
        # can NOT use [:] on the LHS as we are relying on pointers between
        # boxes. [:] on the LHS here means it's not a pointer and then
        # requires that the calculation of the input happens before the
        # advance of this process.
        self[input_variable_name] = adapter.current
        return

    def advance(self):
        """
        Advance the Process in time.

        Returns:
            None
        """

        if self._itime_step >= self.control.itime_step:
            if self._verbose:
                msg = (
                    f"{self.name} did not advance because "
                    f"it is not behind control time"
                )
                print(msg)  # can/howto make warn flush in real time?
            return

        if self._verbose:
            print(f"advancing: {self.name}")

        self._advance_variables()
        self._advance_inputs()
        self._itime_step += 1
        return

    def _calculate(self) -> None:
        raise NotImplementedError("This must be implemented")

    def calculate(self, time_length: float, **kwargs) -> None:
        """Calculate Process terms for a time step

        Args:
            simulation_time: current simulation time

        Returns:
            None
        """
        if self._verbose:
            print(f"calculating: {self.name}")

        # self._calculate must be implemented by the subclass
        self._calculate(time_length, *kwargs)

        return

    def _set_metadata(self):
        """Set metadata on self for self's inputs, parameters, and variables"""
        meta_keys = (*self.variables, *self.inputs, *self.parameters)
        msg = (
            "Duplicate varible names amongst self's variables, "
            "inputs, and parameters"
        )
        assert len(meta_keys) == len(self.variables) + len(self.inputs) + len(
            self.parameters
        ), msg

        self.meta = self.control.meta.find_variables(meta_keys)
        if "global" not in self.meta.keys():
            self.meta["global"] = {}
        return

    def _patch_metadata(
        self, patches, conflicts: Literal["left", "warn", "error"] = "error"
    ):
        patch_meta_on_self = {
            kk: vv for kk, vv in patches.items() if kk in self.meta.keys()
        }
        self.meta = _merge_dicts(
            [self.meta, patch_meta_on_self], conflicts=conflicts
        )
        return

    def output_to_csv(self, pth):
        """
        Save each output variable to separate csv file in specified path

        """
        output_data = self.get_output_dataframes()
        for key in output_data:
            df = output_data[key]
            fname = os.path.join(pth, f"{key}.csv")
            df.to_csv(fname)
        return

    def initialize_netcdf(
        self,
        output_dir: [str, pl.Path] = None,
        separate_files: bool = None,
        output_vars: list = None,
        extra_coords: dict = None,
        addtl_output_vars: list = None,
        **kwargs,
    ) -> None:
        """Initialize NetCDF output.

        Args:
            output_dir: base directory path or NetCDF file path if
                separate_files is True
            separate_files: boolean indicating if storage component output
                variables should be written to a separate file for each
                variable
            output_vars: list of variable names to output.

        Returns:
            None

        """
        if self._netcdf_initialized:
            msg = (
                f"{self.name} class previously initialized netcdf output "
                f"in {self._netcdf_output_dir}"
            )
            warn(msg)
            return

        if self._verbose:
            print(f"initializing netcdf output for: {self.name}")

        (
            output_dir,
            output_vars,
            separate_files,
        ) = self._reconcile_nc_args_w_control_opts(
            output_dir, output_vars, separate_files
        )

        # apply defaults if necessary
        if output_dir is None:
            msg = (
                "An output directory is required to be specified for netcdf"
                "initialization."
            )
            raise ValueError(msg)

        if separate_files is None:
            separate_files = True
        self._netcdf_separate = separate_files

        self._netcdf_initialized = True
        self._netcdf_output_dir = pl.Path(output_dir)

        if output_vars is None:
            self._netcdf_output_vars = self.variables
        else:
            self._netcdf_output_vars = list(
                set(output_vars).intersection(set(self.variables))
            )

        if addtl_output_vars is not None:
            self._netcdf_output_vars += addtl_output_vars

        if len(self._netcdf_output_vars) == 0:
            msg = f"No output variables found for process: {self.name}."
            warn(msg)
            self._netcdf_initialized = False
            return

        self._netcdf = {}

        if self._netcdf_separate:
            self._netcdf_output_dir.mkdir(parents=True, exist_ok=True)
            for variable_name in self._netcdf_output_vars:
                nc_path = self._netcdf_output_dir / f"{variable_name}.nc"
                self._netcdf[variable_name] = NetCdfWrite(
                    name=nc_path,
                    coordinates=self._params.coords,
                    variables=[variable_name],
                    var_meta={variable_name: self.meta[variable_name]},
                    extra_coords=extra_coords,
                    global_attrs={"process class": self.name},
                )

        else:
            if self._netcdf_output_vars is None:
                the_out_vars = self.variables
            else:
                the_out_vars = self._netcdf_output_vars

            initial_variable = the_out_vars[0]
            self._netcdf_output_dir.mkdir(parents=True, exist_ok=True)
            self._netcdf[initial_variable] = NetCdfWrite(
                name=self._netcdf_output_dir / f"{self.name}.nc",
                coordinates=self._params.coords,
                variables=self._netcdf_output_vars,
                var_meta=self.meta,
                extra_coords=extra_coords,
                global_attrs={"process class": self.name},
            )
            for variable in the_out_vars[1:]:
                self._netcdf[variable] = self._netcdf[initial_variable]

        return

    def _output_netcdf(self) -> None:
        """Output variable data to NetCDF for a time step.

        Returns:
            None

        """
        if self._netcdf_initialized:
            time_added = False
            for variable in self._netcdf_output_vars:
                if not time_added or self._netcdf_separate:
                    time_added = True
                    self._netcdf[variable].add_simulation_time(
                        self.control.itime_step, self.control.current_datetime
                    )
                self._netcdf[variable].add_data(
                    variable,
                    self._itime_step,
                    getattr(self, variable),
                )

        return

    def _output_restart(self) -> None:
        from xarray import DataArray

        # preamble is logic for outputting restarts, or not.
        if (
            hasattr(self, "_restart_write")
            and self._restart_write is not False
            and self.control.itime_step >= 0
        ):
            if self._restart_write_strf_code == "f":
                if self.control.itime_step != (self.control.n_times - 1):
                    return
            else:
                next_count = int(
                    # write restarts on the LAST day of the period, so
                    # add a day to current time
                    (self.control.current_time + np.timedelta64(24, "h"))
                    .astype("datetime64[D]")
                    .item()
                    .strftime(self._restart_write_strf_code)
                )
                if self._restart_write_strf_code != "%H":
                    # because hours are counted from zero but days are not
                    next_count -= 1
                if next_count != 0:
                    return

        else:
            return

        cur_time = self.control.current_time
        time = np.atleast_1d(np.array(cur_time.astype("datetime64[ns]")))

        for rv in self.restart_variables:
            meta = self.meta[rv]
            data = self[rv]
            dims = meta["dims"]
            if isinstance(data, TimeseriesArray):
                data = data.current
                dims = dims[1:]
            da = DataArray(
                data=np.expand_dims(data, 0),
                dims=("time", *dims),
                coords=dict(
                    time=time,
                ),
                attrs=dict(
                    description=meta["desc"],
                    units=meta["units"],
                ),
                name=rv,
            )
            cur_strft = cur_time.item().strftime("%Y-%m-%d")
            file = self._restart_write / f"{cur_strft}-{rv}.nc"
            da.to_netcdf(file)

    def _finalize_netcdf(self) -> None:
        """Finalize NetCDF output to disk.

        Returns:
            None
        """
        if self._netcdf_initialized:
            for idx, variable in enumerate(self._netcdf_output_vars):
                if (self._netcdf_output_vars is not None) and (
                    variable not in self._netcdf_output_vars
                ):
                    continue

                self._netcdf[variable].close()
                if not self._netcdf_separate:
                    break
        return

    def _reconcile_nc_args_w_control_opts(
        self, output_dir, output_vars, separate_files
    ):
        # can treat the other args but they are not yet in the available opts
        arg_opt_name_map = {
            "output_dir": "netcdf_output_dir",
            "output_vars": "netcdf_output_var_names",
            "separate_files": "netcdf_output_separate_files",
        }

        args = {
            "output_dir": output_dir,
            "output_vars": output_vars,
            "separate_files": separate_files,
        }

        self_vars = set(self.get_variables())

        def is_not_str_iteratable(it: Iterable):
            return isinstance(it, Iterable) and not isinstance(it, str)

        for vv in args.keys():
            arg_val = args[vv]
            opt_name = arg_opt_name_map[vv]
            opts = self.control.options
            if opt_name in opts.keys():
                opt_val = opts[opt_name]
            else:
                opt_val = None

            # the 4 cases:
            if opt_val is None and arg_val is None:
                pass

            elif opt_val is None:
                pass

            elif arg_val is None:
                args[vv] = opt_val

            elif is_not_str_iteratable(opt_val) and (
                (set(opt_val) & self_vars) == self_vars
            ):
                args[vv] = self_vars

            elif (
                is_not_str_iteratable(opt_val)
                and len(set(opt_val) & self_vars) == 0
            ):
                args[vv] = None

            elif (
                is_not_str_iteratable(opt_val)
                and is_not_str_iteratable(arg_val)
            ) and (set(opt_val) & set(arg_val)) == set(arg_val):
                args[vv] = arg_val

            elif opt_val is not None and arg_val is not None:
                if opt_val == arg_val:
                    pass
                else:
                    msg = (
                        f"control.option '{opt_name}' conflicts with "
                        f"initialize_netcdf() argument {vv}"
                    )
                    raise ValueError(msg)

        return args["output_dir"], args["output_vars"], args["separate_files"]
