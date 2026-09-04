import pathlib as pl
from datetime import datetime
from typing import Type, Union

from tqdm.auto import tqdm

from ..base.adapter import adapter_factory
from ..base.control import Control
from ..base.output import Output
from ..constants import fileish
from ..parameters import Parameters
from ..utils.path import path_rel_to_yaml

# This is a convenience
process_order_nhm = [
    "PRMSSolarGeometry",
    "PRMSAtmosphere",
    "PRMSCanopy",
    "PRMSSnow",
    "PRMSRunoff",
    "PRMSRunoffAg",
    "PRMSRunoffCascadesNoDprst",
    "PRMSRunoffNoDprst",
    "PRMSSoilzone",
    "PRMSSoilzoneAg",
    "PRMSSoilzoneAgObsET",
    "PRMSSoilzoneCascadesNoDprst",
    "PRMSSoilzoneNoDprst",
    "PRMSEt",
    "PRMSGroundwater",
    "PRMSGroundwaterNoDprst",
    "PRMSChannel",
    "PRMSHydraulicGeometryFull",
    "PRMSHydraulicGeometryWidthOnly",
    "PRMSStreamTemp",
    "PRMSStreamTempHumidityCBH",
]


class Model:
    """Build a model in pywatershed.

    This is the class that helps execute sets of Processes in order.

    There are two distinct ways of instatniating the Model class described
    below: 1) PRMS-legacy instantation, 2) pywatershed-centric instatiation.

    Args:
        process_list_or_model_dict: a process list (for PRMS-legacy
          instantiation) or a model dictionary (for pywatershed style
          instatiation), see ways of instatiation below.
        control: Control object or None, see below.
        parameters: A Parameters object of a dictionary of Parameters objects.
            Default is None. See ways of instatnation below.
        find_input_files: Search/find input file on __init__ or delay until run
           or advance of the model. Delaying (False) allows ModelGraph of the
           specified model without the need for input files.
        write_control: bool, str, or pl.Path a directory into which a copy of
           the passed control is to be written, default is False. This is for
           convenience when lost of in-memory manipulations may be made before
           passing to the model. The output file name has the form
           %Y-%m-%dT%H:%M:%S.model_control.yaml

    PRMS-legacy instantiation
    -----------------------------

    This method of instantiation supports PRMS files with few modifications.
    The first three arguments to instantiate a Model this way are:

    * process_list_or_model_dict: A process list of PRMS model components.
    * control: A Control object.
    * parameters: A Parameters object (e.g. PrmsParameters).

    The first example below provides details. An extended example is given by
    `examples/02_prms_legacy_models.ipynb <https://github.com/DOI-USGS/pywatershed/blob/develop/examples/02_prms_legacy_models.ipynb>`__.

    pywatershed-centric instatiation
    ------------------------------------

    The pywatershed style instatniation handles aribtrary and sundry
    processes. It is loosely based on how MF6 structures its inputs. All of the
    work is contained in the first argument, the model dictionary, and the
    control and parameters are both None.

    Instantating a Model this way, only the first of the first three args is
    used:

    - process_list_or_model_dict: a "model dictionary", detailed below.
    - control: Do not pass, None is default.
    - parameters: Do not pass, None is default.

    This means that the user must understand the model dictionary supplied. A
    model dictionary has aribtrary keys but prescribed values. Values may be
    the following kinds of objects: Control, discretization dictionary, process
    dictionary, process order list, and exchanges dictionaries. (Exchanges are
    not yet supported). Each of these is described below. Model dictionaries
    can also be specfied via yaml files. Notes on yaml versus in-memory
    requirements for the values are described as they are (necessarily)
    different.

    See the second and third examples below for more details and see
    `examples/01_multi-process_models.ipynb <https://github.com/DOI-USGS/pywatershed/blob/develop/examples/01_multi-process_models.ipynb>`__
    for an extended example.

    Model dictionary values description:
    ====================================

    - **control** - The control object supplies two things for the model 1) the
      global time discretization (start and stop times, as well as time
      step), and 2) default global options for all model processes.
      Only one control object can be included in the model dictionary. Though
      the key for the control can be arbitrary, the value is either an instance
      of class Control or, in the case of a yaml model dictionary, a control
      yaml file to be loaded by Control.from_yaml() (todo: link to this
      staticmethod).
    - **discretizations** - Multiple discretizations may be supplied to the
      model dictionary, each with arbitrary names. These provide spatial
      discretization information which may be shared by multiple processes
      specified later in `process_list`. Each process will refer to its
      required discretization using the key for that discretization in the
      model dict. The value of each is an instance of the class Parameters. In
      a yaml model dictionary, the value of each discretization is a path
      to a netcdf file for that discretization.
    - **processes** - Multiple processes with arbitrary keys are to be
      supplied to model dictionary (if running just one process, there's no
      need for the Model class). The values for each process are dictionaries
      and have the required keys:

      - *class* - The desired class, in the case of a yaml model dictionary,
        the string name of a class in the pywatershed namespace
      - *parameters* - Either a Parameters object or, in the case of a yaml
        model dictionary, the path of a netcdf file with the parameters
      - *dis* - The key of the dis to use as specified at the top level of
        model dictionary.

    - **model_order** - A list of the processess *keys* in the order the model
      is to be executed.
    - **exchanges** - Future.

    Examples:
    ---------

    These examples will work if you have an editable install of the repository
    (not installed from pypi).

    Construct a PRMS-legacy based model:

    >>> import pywatershed as pws
    >>> test_data_dir = pws.constants.__pywatershed_root__ / "../test_data"
    >>> domain_dir = test_data_dir / "drb_2yr"
    >>> # A PRMS-native control file
    >>> control_file = domain_dir / "control.test"
    >>> # PRMS-native parameter file
    >>> parameter_file = domain_dir / "myparam.param"
    >>> control = pws.Control.load_prms(
    ...     control_file, warn_unused_options=False
    ... )
    >>> control.options["input_dir"] = domain_dir / "output"
    >>> params = pws.parameters.PrmsParameters.load(parameter_file)
    >>> model_procs = [
    ...     pws.PRMSGroundwater,
    ...     pws.PRMSChannel,
    ... ]
    >>> model = pws.Model(
    ...     model_procs,
    ...     control=control,
    ...     parameters=params,
    ... )
    PRMSGroundwater jit compiling with numba
    PRMSChannel jit compiling with numba
    >>> model.run()
    100%|█████████████████████████████████████████████████████████| 731/731 [00:00<00:00, 1249.26it/s]
    model.run(): finalizing


    Construct a model the pywatershed-centric way, in memory:

    >>> import pywatershed as pws
    >>> test_data_dir = pws.constants.__pywatershed_root__ / "../test_data"
    >>> domain_dir = test_data_dir / "drb_2yr"
    >>> dis_hru = pws.Parameters.from_netcdf(
    ...     domain_dir / "parameters_dis_hru.nc", encoding=False
    ... )
    >>> control_file = domain_dir / "control.test"
    >>> control = pws.Control.load_prms(
    ...     control_file, warn_unused_options=False
    ... )
    >>> control.options["input_dir"] = domain_dir
    >>> params = {}
    >>> for proc in ["SolarGeometry", "Atmosphere", "Canopy", "Snow"]:
    ...     param_file = domain_dir / f"parameters_PRMS{proc}.nc"
    ...     params[proc.lower()] = pws.Parameters.from_netcdf(param_file)
    ...
    >>> model_dict = {
    ...     "control": control,
    ...     "dis_hru": dis_hru,
    ...     "model_order": [
    ...         "prmssolargeometry",
    ...         "prmsatmosphere",
    ...         "prmscanopy",
    ...         "prmssnow",
    ...     ],
    ...     "prmssolargeometry": {
    ...         "class": pws.PRMSSolarGeometry,
    ...         "parameters": params["solargeometry"],
    ...         "dis": "dis_hru",
    ...     },
    ...     "prmsatmosphere": {
    ...         "class": pws.PRMSAtmosphere,
    ...         "parameters": params["atmosphere"],
    ...         "dis": "dis_hru",
    ...     },
    ...     "prmscanopy": {
    ...         "class": pws.PRMSCanopy,
    ...         "parameters": params["canopy"],
    ...         "dis": "dis_hru",
    ...     },
    ...     "prmssnow": {
    ...         "class": pws.PRMSSnow,
    ...         "parameters": params["snow"],
    ...         "dis": "dis_hru",
    ...     },
    ... }
    >>> model = pws.Model(model_dict)
    PRMSCanopy jit compiling with numba
    PRMSSnow jit compiling with numba
    >>> model.run()
    100%|███████████████████████████████████████████████████████████| 731/731 [00:07<00:00, 60.76it/s]
    model.run(): finalizing

    Construct a model the pywatershed-centric way, from a yaml file definition:

    >>> import yaml
    >>> import pywatershed as pws
    >>> test_data_dir = pws.constants.__pywatershed_root__ / "../test_data"
    >>> domain_dir = test_data_dir / "drb_2yr"
    >>> control = {
    ...     "start_time": "1979-01-01T00:00:00",
    ...     "end_time": "1980-12-31T00:00:00",
    ...     "time_step": 24,
    ...     "time_step_units": "h",
    ...     "verbosity": 0,
    ...     "imbalance_behavior": "warn",
    ...     "input_dir": str(domain_dir),
    ... }
    >>> control_file = domain_dir / "example_control.yaml"
    >>> model_dict = {
    ...     "control": str(control_file),
    ...     "dis_hru": "parameters_dis_hru.nc",
    ...     "dis_both": "parameters_dis_both.nc",
    ...     "solargeometry": {
    ...         "class": "PRMSSolarGeometry",
    ...         "parameters": "parameters_PRMSSolarGeometry.nc",
    ...         "dis": "dis_hru",
    ...     },
    ...     "atmosphere": {
    ...         "class": "PRMSAtmosphere",
    ...         "parameters": "parameters_PRMSAtmosphere.nc",
    ...         "dis": "dis_hru",
    ...     },
    ...     "canopy": {
    ...         "class": "PRMSCanopy",
    ...         "parameters": "parameters_PRMSCanopy.nc",
    ...         "dis": "dis_hru",
    ...     },
    ...     "snow": {
    ...         "class": "PRMSSnow",
    ...         "parameters": "parameters_PRMSSnow.nc",
    ...         "dis": "dis_hru",
    ...     },
    ...     "model_order": ["solargeometry", "atmosphere", "canopy", "snow"],
    ... }
    >>> model_dict_file = domain_dir / "example_model_dict.yaml"
    >>> dump_dict = {control_file: control, model_dict_file: model_dict}
    >>> for key, val in dump_dict.items():
    ...     with open(key, "w") as file:
    ...         documents = yaml.dump(val, file)
    ...
    >>> model = pws.Model.from_yaml(model_dict_file)
    PRMSCanopy jit compiling with numba
    PRMSSnow jit compiling with numba
    >>> model.run()
    100%|████████████████████████████████████████████████████████████| 731/731 [00:07<00:00, 92.30it/s]
    model.run(): finalizing
    >>> control_file.unlink()
    >>> model_dict_file.unlink()

    """  # noqa: E501

    def __init__(
        self,
        process_list_or_model_dict: Union[list, dict],
        control: Control = None,
        parameters: Union[Parameters, dict[Parameters]] = None,
        find_input_files: bool = True,
        write_control: Union[bool, str, pl.Path] = False,
        output_obj_kwargs_dict: dict = None,
        input_aliases: dict = None,
    ):
        self.control = control
        self.parameters = parameters

        # This is for backwards compatibility: make a method?
        msg = "Inputs are inconsistent"
        if isinstance(process_list_or_model_dict, (list, tuple)):
            # take the old-school-style inputs and convert to new-school inputs
            # may be deprecated in the future.
            assert control is not None, msg
            assert isinstance(parameters, Parameters), msg

            # eventually handle a file for parameters?
            # proc_param_file = domain["dir"] / f"parameters_{proc_name}.nc"
            # proc["parameters"] = PrmsParameters.from_netcdf(
            #     proc_param_file
            # )

            self._input_aliases = input_aliases or {}
            self.model_dict = {}
            model_dict = self.model_dict
            model_dict["control"] = self.control
            for process in process_list_or_model_dict:
                proc_name = process.__name__
                model_dict[proc_name] = {}
                model_dict[proc_name]["class"] = process
                model_dict[proc_name]["parameters"] = parameters

            model_dict["model_order"] = [
                key for key in process_order_nhm if key in model_dict.keys()
            ]

        elif isinstance(process_list_or_model_dict, dict):
            assert control is None, msg
            assert parameters is None, msg
            self.model_dict = process_list_or_model_dict
            # Extract input_aliases from model_dict before
            # _categorize_model_dict (any dict-valued top-level key
            # would be mis-classified as a process)
            model_dict_aliases = self.model_dict.pop("input_aliases", {}) or {}
            self._input_aliases = {
                **model_dict_aliases,
                **(input_aliases or {}),
            }

        else:
            raise ValueError("Invalid type of process_list_or_model_dict")

        self._categorize_model_dict()
        self._validate_model_dict()
        self._set_input_path()
        self._solve_inputs()
        self._init_procs()
        self._connect_procs()

        self._found_input_files = False
        if find_input_files:
            self._find_input_files()

        self._netcdf_initialized = False
        opts = self.control.options
        if "netcdf_output_dir" in opts.keys():
            self._default_nc_out_dir = opts["netcdf_output_dir"]
        else:
            self._default_nc_out_dir = None

        if write_control or isinstance(write_control, (pl.Path, str)):
            if isinstance(write_control, bool):
                write_control = pl.Path(".")
            format_fn = "%Y-%m-%dT%H.%M.%S.model_control.yaml"
            yaml_fn = write_control / datetime.now().strftime(format_fn)
            if not yaml_fn.parent.exists():
                yaml_fn.parent.mkdir(parents=True)
            self.control.to_yaml(yaml_fn)

        self._output_obj_kwargs_dict = output_obj_kwargs_dict
        self._handle_output_obj_kwargs_dict()

        return

    def _handle_output_obj_kwargs_dict(self):
        if not self._output_obj_kwargs_dict:
            self._output_obj_kwargs_dict = {}
            self._output_obj = None
            return

        for kk, vv in {"control": self.control, "model": self}.items():
            if kk in self._output_obj_kwargs_dict.keys():
                if self._output_obj_kwargs_dict[kk] is not vv:
                    raise ValueError(
                        f"An inappropriate (and unnecessary) {kk} object "
                        "was passed via output_obj_kwargs_dict."
                    )
            else:
                self._output_obj_kwargs_dict[kk] = vv

        self._output_obj = Output(**self._output_obj_kwargs_dict)

    @property
    def output_obj(self) -> "Output | None":
        """Output object collector for the model (if configured)."""
        return self._output_obj

    def _categorize_model_dict(self):
        """Categorize model_dict entries

        Categorize entries into [control, dis, order, process, exchange]
        categories based on types and attributes
        """
        # TODO: add exchanges
        category_type_dict = {
            "dis": Parameters,
            "control": Control,
            "order": list,
            "process": dict,
            # "exchange": exchange class?
        }
        category_key_dict = {}
        for cat, typ in category_type_dict.items():
            category_key_dict[cat] = [
                kk for kk, vv in self.model_dict.items() if isinstance(vv, typ)
            ]

        self._category_key_dict = category_key_dict

        return

    def _validate_model_dict(self):
        cat_key_dict = self._category_key_dict
        # * a single control object
        assert len(cat_key_dict["control"]) == 1
        cat_key_dict["control"] = cat_key_dict["control"][0]
        self._control_name = cat_key_dict["control"]
        if self.control is None:
            self.control = self.model_dict[self._control_name]
        # * a single order list
        assert len(cat_key_dict["order"]) == 1
        cat_key_dict["order"] = cat_key_dict["order"][0]
        self._order_name = cat_key_dict["order"]
        self.process_order = self.model_dict[self._order_name]
        # * all processes are in order and vice-versa
        assert set(cat_key_dict["process"]) == set(
            self.model_dict[cat_key_dict["order"]]
        )
        # * all required proc dis are present
        # * all required proc in exchanges are present

        return

    @staticmethod
    def solve_inputs(process_list_or_model_dict) -> dict:
        """Determine where each process input comes from, per class.

        Answers, without instantiating a Model (or requiring any files to
        exist): which inputs are supplied by other processes and which
        must come from file. All information required is class-level.

        Args:
            process_list_or_model_dict: as for Model. Either a list of
                Process classes, or a model dictionary whose process
                specifications are dicts with a "class" key (non-dict and
                class-less entries, e.g. control or discretizations, are
                ignored). In the dictionary form, a specification key
                naming an input that is also in the class's ``__init__``
                signature counts as a passed input, satisfied by the
                specification itself (e.g. ``ag_frac``).

        Returns:
            A dict with keys:

            - ``"inputs_from"``: ``{process_name: {input_name: source}}``
              where source is the producing process's name, or None if
              the input must come from file. Passed inputs are omitted.
            - ``"from_file"``: sorted list of the input names that must
              come from file (the union of the None-sourced inputs).

        Notes:
            - Process order is not considered (matching Model
              construction; a later process can supply an earlier one).
            - input_aliases are not considered; names are variable names.

        Examples:
            >>> import pywatershed as pws
            >>> below_soil = [pws.PRMSGroundwater, pws.PRMSChannel]
            >>> pws.Model.solve_inputs(below_soil)["from_file"]
            ['dprst_seep_hru', 'soil_to_gw', 'sroff_vol', 'ssr_to_gw', 'ssres_flow_vol']
        """  # noqa: E501
        import inspect

        if isinstance(process_list_or_model_dict, (list, tuple)):
            specs = {
                cls.__name__: {"class": cls}
                for cls in process_list_or_model_dict
            }
        else:
            specs = {
                kk: vv
                for kk, vv in process_list_or_model_dict.items()
                if isinstance(vv, dict) and "class" in vv.keys()
            }

        proc_dict = {kk: vv["class"] for kk, vv in specs.items()}
        proc_inputs = {kk: vv.get_inputs() for kk, vv in proc_dict.items()}
        proc_vars = {kk: vv.get_variables() for kk, vv in proc_dict.items()}

        def get_passed_args(
            proc_class: Type, proc_passed_arg_names: list
        ) -> set:
            signature_args = set(
                inspect.signature(proc_class.__init__).parameters.keys()
            )
            return (
                signature_args
                & set(proc_passed_arg_names)
                & set(proc_class.get_inputs())
            )

        proc_passed_inputs = {
            kk: get_passed_args(proc_dict[kk], specs[kk].keys())
            for kk in proc_dict.keys()
        }

        # Solve where inputs come from
        inputs_from = {}
        for comp in proc_dict.keys():
            inputs_from[comp] = {}
            for input in proc_inputs[comp]:
                if input in proc_passed_inputs[comp]:
                    continue
                producers = [
                    other
                    for other in proc_dict.keys()
                    if other != comp and input in proc_vars[other]
                ]
                # generally zero or one producer; the first is used
                inputs_from[comp][input] = producers[0] if producers else None

        from_file = sorted(
            {
                input
                for wired in inputs_from.values()
                for input, source in wired.items()
                if source is None
            }
        )

        return {"inputs_from": inputs_from, "from_file": from_file}

    def _solve_inputs(self):
        """What processes supply inputs to others, what files are needed?

        The pure logic lives in the public staticmethod solve_inputs;
        here its answer is adapted to the internal representations.

        TODO: this does not currently take order into account, which could
              be important and is now possible with order specified.
        """
        specs = {
            proc_name: self.model_dict[proc_name]
            for proc_name in self._category_key_dict["process"]
        }
        solved = Model.solve_inputs(specs)

        # internal representation: source as a list, empty if from file
        inputs_from = {
            comp: {
                input: [] if source is None else [source]
                for input, source in wired.items()
            }
            for comp, wired in solved["inputs_from"].items()
        }

        # If inputs dont come from other processes or self, assume they come
        # from file in input_dir or input_file. Exception is that
        # PRMSAtmosphere requires its files on init, so dont adapt these
        input_names = set(solved["from_file"])

        # initiate the file inputs here rather than in the processes
        # Use dummy names for now
        file_inputs = {name: pl.Path(name) for name in input_names}

        self._proc_dict = {kk: vv["class"] for kk, vv in specs.items()}
        self._inputs_from = inputs_from
        self._input_names = input_names
        self._file_inputs = file_inputs
        return

    def _init_procs(self):
        # instantiate processes: instance dict
        self.processes = {}
        for proc_name in self.model_dict[self._order_name]:
            # establish the args for init
            proc_specs = self.model_dict[proc_name]
            proc_specs["control"] = self.control

            dis = None
            if "dis" in proc_specs:
                if proc_specs["dis"] is None:
                    dis = None
                else:
                    dis = self.model_dict[proc_specs["dis"]]

            proc_specs["discretization"] = dis

            process_inputs = {}
            for input in self._proc_dict[proc_name].get_inputs():
                # set the inputs, allowing for passed already inputs
                if input in proc_specs.keys():
                    continue
                process_inputs[input] = None

            proc_specs = proc_specs | process_inputs
            not_args = ["class", "dis", "input_aliases"]
            proc_args = {
                kk: vv for kk, vv in proc_specs.items() if kk not in not_args
            }

            # Merge model-level aliases (filtered to this process's inputs)
            # with any process-level aliases from the proc spec.
            # Process-level takes precedence over model-level.
            proc_inputs = set(self._proc_dict[proc_name].get_inputs())
            merged_aliases = {
                k: v
                for k, v in self._input_aliases.items()
                if k in proc_inputs
            }
            proc_level_aliases = proc_specs.get("input_aliases") or {}
            merged_aliases.update(proc_level_aliases)
            if merged_aliases:
                proc_args["input_aliases"] = merged_aliases

            self.processes[proc_name] = self._proc_dict[proc_name](**proc_args)

        # <
        return

    def _connect_procs(self):
        self.process_input_from = {}
        for process in self.process_order:
            self.process_input_from[process] = {}
            for input, frm in self._inputs_from[process].items():
                if not frm:
                    self.process_input_from[process][input] = (
                        self._file_inputs[input]
                    )
                else:
                    self.process_input_from[process][input] = frm[0]
                    self.processes[process].set_input_to_adapter(
                        input,
                        adapter_factory(
                            self.processes[frm[0]][input],
                            control=self.control,
                        ),  # drop list above
                    )
        # <<<
        return

    def _set_input_path(self):
        opt_keys = self.control.options.keys()
        if ("input_dir" in opt_keys and "input_file" in opt_keys) or (
            "input_dir" not in opt_keys and "input_file" not in opt_keys
        ):
            msg = (
                "Exactly one of 'input_dir' or 'input_file' must be specified"
            )
            raise ValueError(msg)

        for ii in ["input_dir", "input_file"]:
            if ii in opt_keys:
                self._input_path = pl.Path(self.control.options[ii]).resolve()

        return

    def _find_input_files(self) -> None:
        # Build a combined alias map from all processes for file lookup.
        # Process-level aliases take precedence over model-level.
        alias_map = dict(self._input_aliases)
        for proc_name in self.process_order:
            proc_aliases = getattr(
                self.processes[proc_name], "_input_aliases", {}
            )
            alias_map.update(proc_aliases)

        input_adapters = {}
        for name in self._input_names:
            file_var_name = alias_map.get(name, name)
            if self._input_path.is_dir():
                nc_path = self._input_path / f"{file_var_name}.nc"
                # currently netcdf files or dynamic parameter files accepted
                if not nc_path.exists():
                    param_path = self._input_path / f"{file_var_name}.param"
                    if not param_path.exists():
                        aka = "" if file_var_name == name else f" (as {name})"
                        msg = (
                            f"No input file found for '{file_var_name}'"
                            f"{aka} in {self._input_path}: looked for "
                            f"'{nc_path.name}' and '{param_path.name}'."
                        )
                        raise FileNotFoundError(msg)
                    nc_path = param_path
            else:
                nc_path = self._input_path
            input_adapters[name] = adapter_factory(
                nc_path,
                file_var_name,
                control=self.control,
            )
        for process in self.process_order:
            for input, frm in self._inputs_from[process].items():
                if not frm:
                    fname = input_adapters[input]._fname
                    self.process_input_from[process][input] = fname
                    self.processes[process].set_input_to_adapter(
                        input, input_adapters[input]
                    )

        self._found_input_files = True
        return

    @staticmethod
    def model_dict_from_yaml(yaml_file: Union[str, pl.Path]) -> dict:
        """Generate a model dictionary from a yaml file.

        Instead of Model.from_yaml() it can be useful to get the model
        dictionary before passing it to Model.

        Args:
            yaml_file: a yaml file

        Returns:
            A model dictionary.
        """
        import yaml

        import pywatershed

        with pl.Path(yaml_file).open("r") as file_stream:
            model_dict = yaml.load(file_stream, Loader=yaml.Loader)

        for key, val in model_dict.items():
            if isinstance(val, str):
                val_pl = path_rel_to_yaml(val, yaml_file)
                if (val.endswith(".yml")) or (val.endswith(".yaml")):
                    model_dict[key] = Control.from_yaml(val_pl)
                elif val.endswith(".nc"):
                    model_dict[key] = Parameters.from_netcdf(val_pl)
                else:
                    msg = (
                        "Unsupported file extension for control (.yml/.yaml)"
                        "and parameter (.nc) file paths in model yaml file"
                    )
                    raise ValueError(msg)

            elif isinstance(val, dict):
                # Until exchanges are introduced processes are the only
                # accepted dictionaries
                if "class" in val.keys():
                    # processes
                    cls = val["class"]
                    val["class"] = getattr(pywatershed, cls)
                    par = val["parameters"]
                    par_pl = path_rel_to_yaml(par, yaml_file)
                    val["parameters"] = Parameters.from_netcdf(
                        par_pl, encoding=False
                    )
                    # dis = val["dis"]
                    # val["dis"] = model_dict[dis]
                    for subkey in val.keys():
                        if "_class" in subkey:
                            val[subkey] = getattr(pywatershed, val[subkey])

            elif isinstance(val, list):
                pass

        return model_dict

    @staticmethod
    def from_yaml(yaml_file: Union[str, pl.Path]):
        """Instantiate a Model from a yaml file

        A yaml file that specifies a model_dict as the first argument of Model.

        Args:
           yaml_file: str or pathlib.Path

        Returns:
           An instance of Model.

        Yaml file structure (strict order not required, but suggested):

        * Control object: Any name can be used but the value must be a control
          yaml file specified with the suffix ".yaml". E.g
          "name: control.yaml"
          would appear in the passed yaml file. Only one control
          specification is allowed in the yaml_file. For details on the
          requirements of the control.yaml file see `Control.from_yaml`

        * Discretization objects: Any number of discretization objects can be
          supplied with arbitrary (though unique) names. The values supplied
          for each discretization must be a valid netcdf file with suffix
          ".nc". These can generally be obtained by calling
          `parameters.to_netcdf()` on subclasses of Parameters.

        * Process objects: Any number of processes may be specified with
          arbitrary (though unique) names. Processes are specified as
          dictionaries in the yaml file, they have additonal key:value pairs.
          Required key:value pairs are:

          * class: a class that can be `from pywatershed import class`
          * parameters: a netcdf file that specifies the parameters for
            the class
          * dis: the name of the discretization for the class as given by
            a discretization object speficication above.

          Optional key:value pairs: TBD

        * Model order list: a list supplying the order in which the processes
          are to be executed.

        Note: To get a model_dict specfied by the yaml_file, call
        `model_dict_from_yaml` instead.

        """
        return Model(Model.model_dict_from_yaml(yaml_file))

    def initialize_netcdf(
        self,
        output_dir: str = None,
        separate_files: bool = None,
        budget_args: dict = None,
        output_vars: list = None,
    ):
        """Initialize NetCDF output files for model (all processes).

        Args:
            output_dir: pl.Path or str of the directory where to write files
            separate_files: For a given Process, write a single file or
                separate files for the process' variables. DEFAULTS to True
                for performance reasons.
            budget_args: see Budget.initialize_netcdf(). defaults to None
            output_vars: A list of variables to write. Unrecognized variable
                names are silently skipped. Defaults to None which writes
                all variables for all Processes.
        """
        print("model initializing NetCDF output")

        if not self._found_input_files:
            self._find_input_files()

        for cls in self.process_order:
            self.processes[cls].initialize_netcdf(
                output_dir=output_dir,
                separate_files=separate_files,
                budget_args=budget_args,
                output_vars=output_vars,
            )
        self._netcdf_initialized = True
        return

    def run(
        self,
        netcdf_dir: fileish | None = None,
        finalize: bool = True,
        n_time_steps: int | None = None,
        output_vars: list | None = None,
        output_obj: "Output | None" = None,
    ):
        """Run the model.

        Running the model wraps:
            * initializing netcdf output (optional on netcdf_dir argument)
            * full time loop (from control object) with progress reports
                * advance
                * calculate
                * output (optional on output methods requested)
            * finalize (optional)

        Args:
            netcdf_dir: optional directory to netcdf output files (initializes
               netcdf and outputs at each timestep).
            finalize: option to not finalize at the end of the time loop.
               Default is to finalize.
            n_time_steps: the number of timesteps to run
            output_vars: the vars to output to the netcdf_dir
        """
        if self._output_obj is not None:
            if output_obj is not None:
                raise ValueError("Output previously defined on self")
            output_obj = self._output_obj

        if netcdf_dir or (
            not self._netcdf_initialized
            and self._default_nc_out_dir is not None
        ):
            self.initialize_netcdf(netcdf_dir, output_vars=output_vars)

        if not n_time_steps:
            n_time_steps = self.control.n_times

        for istep in tqdm(range(n_time_steps)):
            self.advance()
            self.calculate()
            self.output()
            if output_obj is not None:
                output_obj.calculate()

        if finalize:
            print("model.run(): finalizing")
            self.finalize()
            if output_obj is not None:
                output_obj.finalize()

        return

    def advance(self):
        """Advance the model in time."""
        if not self._found_input_files:
            self._find_input_files()

        if (
            not self._netcdf_initialized
            and self._default_nc_out_dir is not None
        ):
            self.initialize_netcdf()

        self.control.advance()
        for cls in self.process_order:
            self.processes[cls].advance()
        return

    def calculate(self):
        """Calculate the model."""
        for cls in self.process_order:
            self.processes[cls].calculate(1.0)
        return

    def output(self):
        """Output the model at the current time."""
        for cls in self.process_order:
            self.processes[cls].output()
        return

    def finalize(self):
        """Finalize the model."""
        for cls in self.process_order:
            self.processes[cls].finalize()
        return
