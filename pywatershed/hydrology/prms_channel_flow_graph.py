import pathlib as pl
from copy import deepcopy
from typing import Literal, Union

import numba as nb
import numpy as np
import xarray as xr

from pywatershed.base.adapter import (
    Adapter,
    AdapterNetcdf,
    adaptable,
    adapter_factory,
)
from pywatershed.base.conservative_process import ConservativeProcess
from pywatershed.base.control import Control
from pywatershed.base.flow_graph import (
    FlowGraph,
    FlowNode,
    FlowNodeMaker,
    inflow_exchange_factory,
)
from pywatershed.constants import SegmentType, nan, zero
from pywatershed.hydrology.prms_channel import PRMSChannel
from pywatershed.parameters import Parameters


class PRMSChannelFlowNode(FlowNode):
    r"""A FlowNode for the Muskingum-Mann method of PRMSChannel

    This is a :class:`FlowNode` implementation of :class:`PRMSChannel` where
    the solution is the so-called Muskingum-Mann method.

    See :class:`FlowGraph` for discussion and a worked example. The notebook
    `examples/06_flow_graph_starfit.ipynb <https://github.com/DOI-USGS/pywatershed/blob/develop/examples/06_flow_graph_starfit.ipynb>`__
    highlights adding a StarfitFlowNode a :class:`FlowGraph` otherwised
    comprised of :class:`PRMSChannelFlowNode`\ s using the helper functions
    :func:`prms_channel_flow_graph_to_model_dict`
    and :func:`prms_channel_flow_graph_postprocess`.

    """

    def __init__(
        self,
        control: Control,
        tsi: np.int64,
        ts: np.float64,
        c0: np.float64,
        c1: np.float64,
        c2: np.float64,
        calc_method: Literal["numba", "numpy"] = None,
    ):
        """Initialize a PRMSChannelFlowNode.

        Args:
          control: A :class:`Control` object.
          tsi: Parameter of :class:`PRMSChannel`.
          ts: Parameter of :class:`PRMSChannel`.
          c0: Parameter of :class:`PRMSChannel`.
          c1: Parameter of :class:`PRMSChannel`.
          c2: Parameter of :class:`PRMSChannel`.
          calc_method: One of "numba", "numpy" (default).
        """
        self.control = control

        self._tsi = tsi
        self._ts = ts
        self._c0 = c0
        self._c1 = c1
        self._c2 = c2

        self._outflow_ts = zero
        self.inflow_ts_prev = zero
        self._seg_inflow = zero
        self._inflow_ts = zero
        self._seg_current_sum = zero

        if calc_method is None or calc_method == "numba":
            self._calculate_subtimestep = _calculate_subtimestep_numba
        elif calc_method == "numpy":
            self._calculate_subtimestep = _calculate_subtimestep_numpy
        else:
            raise ValueError(f"Invalid choice of calc_method: {calc_method}")
        return

    def prepare_timestep(self):
        # self._simulation_time = simulation_time  # add argument?
        self._seg_inflow = zero
        self._seg_outflow = zero
        # self._seg_outflow0 = zero
        self._inflow_ts = zero
        self._seg_current_sum = zero

        return

    def calculate_subtimestep(self, ihr, inflow_upstream, inflow_lateral):
        (
            self._seg_inflow,
            self._inflow_ts,
            self._outflow_ts,
            self.inflow_ts_prev,
            self._seg_outflow,
        ) = self._calculate_subtimestep(
            ihr,
            inflow_upstream,
            inflow_lateral,
            self.inflow_ts_prev,
            self._seg_inflow,  # implied on RHS by +=
            self._inflow_ts,
            self._seg_outflow,
            self._outflow_ts,
            self._tsi,
            self._ts,
            self._c0,
            self._c1,
            self._c2,
        )

    def finalize_timestep(self):
        # get rid of the magic 24 with argument?
        self._seg_outflow = self._seg_outflow / 24.0
        self._seg_inflow = self._seg_inflow / 24.0
        self.seg_stor_change = self._seg_inflow - self._seg_outflow

        return

    def advance(self):
        self.inflow_ts_prev = self._seg_inflow

        return

    @property
    def outflow(self):
        """The average outflow over the timestep in cubic feet per second."""
        return self._seg_outflow

    @property
    def outflow_substep(self):
        """The outflow over the sub-timestep in cubic feet per second."""
        return self._outflow_ts

    @property
    def storage_change(self):
        """The volumetric storage change in cubic feet."""
        return self.seg_stor_change

    @property
    def storage(self):
        """The volumetric storage in millions of cubic feet.
        Not defined for PRMSChannel.
        """
        return nan

    @property
    def sink_source(self):
        return zero


class PRMSChannelFlowNodeMaker(FlowNodeMaker):
    r"""A FlowNodeMaker for PRMSChannelFlowNodes.

    See :class:`PRMSChannelFlowNode` for additional details and the required
    parameters.

    See :class:`FlowGraph` for discussion and a worked example. The notebook
    `examples/06_flow_graph_starfit.ipynb <https://github.com/DOI-USGS/pywatershed/blob/develop/examples/06_flow_graph_starfit.ipynb>`__
    highlights adding a StarfitFlowNode a :class:`FlowGraph` otherwised
    comprised of :class:`PRMSChannelFlowNode`\ s using the helper functions
    :func:`prms_channel_flow_graph_to_model_dict`
    and :func:`prms_channel_flow_graph_postprocess`.
    """

    def __init__(
        self,
        discretization: Parameters,
        parameters: Parameters,
        calc_method: Literal["numba", "numpy"] = None,
        verbose: bool = None,
    ) -> None:
        """Instantiate a PRMSChannelFlowNodeMaker.

        Args:
          discretization: a discretization of class Parameters
          parameters: a parameter object of class Parameters
          calc_method: one of ["fortran", "numba", "numpy"]. None defaults to
              "numba".
          verbose: Print extra information or not?
        """
        self.name = "PRMSChannelFlowNodeMaker"
        self._calc_method = calc_method
        self._set_data(discretization, parameters)
        self._init_data()

        return

    def get_node(self, control, index) -> PRMSChannelFlowNode:
        # could pass initial conditons here but they arent currently used in
        # PRMS
        return PRMSChannelFlowNode(
            control=control,
            tsi=self._tsi[index],
            ts=self._ts[index],
            c0=self._c0[index],
            c1=self._c1[index],
            c2=self._c2[index],
            calc_method=self._calc_method,
        )

    def _set_data(self, discretization, parameters):
        self._parameters = parameters
        self._discretization = discretization
        for param in self.get_parameters():
            if param in self._parameters.parameters.keys():
                self[param] = self._parameters.parameters[param]
            else:
                self[param] = self._discretization.parameters[param]

        self.nsegment = len(self.seg_length)
        return

    @staticmethod
    def get_dimensions() -> tuple:
        """Get a tuple of dimension names for this PRMSChannelFlowNodeMaker."""
        return ("nsegment",)

    @staticmethod
    def get_parameters() -> tuple:
        """Get a tuple of parameter names for PRMSChannelFlowNodeMaker."""
        return (
            "mann_n",
            "seg_depth",
            "seg_length",
            "seg_slope",
            "segment_type",
            "tosegment",
            "tosegment_nhm",
            "x_coef",
            "segment_flow_init",
        )

    def _init_data(self) -> None:
        """Initialize internal variables from raw channel data"""

        # calculate the Muskingum parameters
        velocity = (
            (
                (1.0 / self.mann_n)
                * np.sqrt(self.seg_slope)
                * self.seg_depth ** (2.0 / 3.0)
            )
            * 60.0
            * 60.0
        )
        # JLM: This is a bad idea and should throw an error rather than edit
        # inputs in place during run
        self.seg_slope = np.where(
            self.seg_slope < 1e-7, 0.0001, self.seg_slope
        )  # not in prms6

        # initialize Kcoef to 24.0 for segments with zero velocities
        # this is different from PRMS, which relied on divide by zero resulting
        # in a value of infinity that when evaluated relative to a maximum
        # desired Kcoef value of 24 would be reset to 24. This approach is
        # equivalent and avoids the occurence of a divide by zero.
        Kcoef = np.full(self.nsegment, 24.0, dtype=float)

        # only calculate Kcoef for cells with velocities greater than zero
        idx = velocity > 0.0
        Kcoef[idx] = self.seg_length[idx] / velocity[idx]
        Kcoef = np.where(
            self.segment_type == SegmentType.LAKE.value, 24.0, Kcoef
        )
        Kcoef = np.where(Kcoef < 0.01, 0.01, Kcoef)
        self._Kcoef = np.where(Kcoef > 24.0, 24.0, Kcoef)

        self._ts = np.ones(self.nsegment, dtype=float)
        self._tsi = np.ones(self.nsegment, dtype="int64")

        # todo: vectorize this
        for iseg in range(self.nsegment):
            k = self._Kcoef[iseg]
            if k < 1.0:
                self._tsi[iseg] = -1
            elif k < 2.0:
                self._ts[iseg] = 1.0
                self._tsi[iseg] = 1
            elif k < 3.0:
                self._ts[iseg] = 2.0
                self._tsi[iseg] = 2
            elif k < 4.0:
                self._ts[iseg] = 3.0
                self._tsi[iseg] = 3
            elif k < 6.0:
                self._ts[iseg] = 4.0
                self._tsi[iseg] = 4
            elif k < 8.0:
                self._ts[iseg] = 6.0
                self._tsi[iseg] = 6
            elif k < 12.0:
                self._ts[iseg] = 8.0
                self._tsi[iseg] = 8
            elif k < 24.0:
                self._ts[iseg] = 12.0
                self._tsi[iseg] = 12
            else:
                self._ts[iseg] = 24.0
                self._tsi[iseg] = 24

        d = self._Kcoef - (self._Kcoef * self.x_coef) + (0.5 * self._ts)
        d = np.where(np.abs(d) < 1e-6, 0.0001, d)
        self._c0 = (-(self._Kcoef * self.x_coef) + (0.5 * self._ts)) / d
        self._c1 = ((self._Kcoef * self.x_coef) + (0.5 * self._ts)) / d
        self._c2 = (
            self._Kcoef - (self._Kcoef * self.x_coef) - (0.5 * self._ts)
        ) / d

        # Short travel time
        idx = self._c2 < 0.0
        self._c1[idx] += self._c2[idx]
        self._c2[idx] = 0.0

        # Long travel time
        idx = self._c0 < 0.0
        self._c1[idx] += self._c0[idx]
        self._c0[idx] = 0.0

        # local flow variables
        self._seg_inflow = np.zeros(self.nsegment, dtype=float)
        self.inflow_ts_prev = np.zeros(self.nsegment, dtype=float) * nan
        self._inflow_ts = np.zeros(self.nsegment, dtype=float)
        self._outflow_ts = np.zeros(self.nsegment, dtype=float)
        self._seg_current_sum = np.zeros(self.nsegment, dtype=float)

        return


# <
# module scope functions
def _calculate_subtimestep_numpy(
    ihr,
    inflow_upstream,
    inflow_lateral,
    inflow_ts_prev,
    _seg_inflow,
    _inflow_ts,
    _seg_outflow,
    _outflow_ts,
    _tsi,
    _ts,
    _c0,
    _c1,
    _c2,
):
    # some way to enforce that ihr is actually an hour?
    seg_current_inflow = inflow_lateral + inflow_upstream
    _seg_inflow += seg_current_inflow
    _inflow_ts += seg_current_inflow

    remainder = (ihr + 1) % _tsi
    if remainder == 0:
        # segment routed on current hour
        _inflow_ts /= _ts

        if _tsi > 0:
            # Muskingum routing equation
            _outflow_ts = (
                _inflow_ts * _c0 + inflow_ts_prev * _c1 + _outflow_ts * _c2
            )
        else:
            _outflow_ts = _inflow_ts

        inflow_ts_prev = _inflow_ts
        _inflow_ts = 0.0

    _seg_outflow += _outflow_ts

    return (
        _seg_inflow,
        _inflow_ts,
        _outflow_ts,
        inflow_ts_prev,
        _seg_outflow,
    )


# numba_msg = "prms_channel_flow_graph jit compiling with numba"
# print(numba_msg, flush=True)

_calculate_subtimestep_numba = nb.njit(
    nb.types.UniTuple(nb.float64, 5)(
        nb.int64,  # ihr
        nb.float64,  # inflow_upstream
        nb.float64,  # inflow_lateral
        nb.float64,  # inflow_ts_prev
        nb.float64,  # _seg_inflow
        nb.float64,  # _inflow_ts
        nb.float64,  # _seg_outflow
        nb.float64,  # _outflow_ts
        nb.int64,  # _tsi
        nb.float64,  # _ts
        nb.float64,  # _c0
        nb.float64,  # _c1
        nb.float64,  # _c2
    ),
    fastmath=True,
    parallel=False,
)(_calculate_subtimestep_numpy)


class HruSegmentFlowAdapter(Adapter):
    """Adapt volumetric flows from HRUs to lateral inflows on PRMS segments.

    This class specifically maps from PRMS HRU outflows to PRMS segment inflows
    using the parameters known to `PRMSChannel`. This reproduces the PRMS
    variable hru_streamflow_out.

    This class is a subclass of :class:`Adapter` which means that it makes
    existing or known flows available over time (but dosent calculate a
    process). This class is meant to force a stand-alone :class:`FlowGraph`
    runoff outside the context of a :class:`Model`. The calculated lateral
    flows (in cubic feet per second) are availble from the current_value
    property.

    See :class:`FlowGraph` for discussion and an example.
    """

    def __init__(
        self,
        parameters: Parameters,
        sroff_vol: Adapter,
        ssres_flow_vol: Adapter,
        gwres_flow_vol: Adapter,
    ):
        """Instantiate an HruSegmentFlowAdapter.

        Args:
          parameters: A :class:`Parameters` object for :class:`PRMSChannel`.
          sroff_vol: An :class:`Adapter` of surface runoff volume.
          ssres_flow_vol: An Adapter of subsurface reservoir outflow volume
          gwres_flow_vol: An Adapter of groundwater reservoir outflow volume
        """
        self._variable = "inflows"
        self._parameters = parameters

        self._inputs = {}
        for key in [
            "sroff_vol",
            "ssres_flow_vol",
            "gwres_flow_vol",
        ]:
            self._inputs[key] = locals()[key]

        self._nhru = self._parameters.dims["nhru"]
        self._nsegment = self._parameters.dims["nsegment"]

        # sum of flows from hrus, these are flows and not volumes
        self._inflows = np.zeros(self._nhru) * nan

        # self.current_value (provided in the super) is the lateral flows
        self._current_value = np.zeros(self._nsegment) * nan

        self._hru_segment = self._parameters.parameters["hru_segment"] - 1

        return

    def advance(self):
        """Advance the Segement inflows in time.
        Here we move to flows from flow volumes.
        """
        for vv in self._inputs.values():
            vv.advance()

        # Move to a flow basis here instead of a volume basis.
        # s_per_time could vary with timestep so leave here.
        s_per_time = self._inputs["sroff_vol"].control.time_step_seconds
        self._inflows = (
            sum([vv.current for vv in self._inputs.values()]) / s_per_time
        )

        self._calculate_segment_lateral_inflows()

        return

    def _calculate_segment_lateral_inflows(self):
        """Map HRU inflows to lateral inflows on segments/nodes"""
        self._current_value[:] = zero
        for ihru in range(self._nhru):
            iseg = self._hru_segment[ihru]
            if iseg < 0:
                # This is bad, selective handling of fluxes is not cool,
                # mass is being discarded in a way that has to be
                # coordinated with other parts of the code.
                # This code should be removed evenutally.
                continue

            self._current_value[iseg] += self._inflows[ihru]

        return


class HruNodeFlowExchange(ConservativeProcess):
    """Process to map PRMS HRU outflows to lateral inflows on nodes.

    This class maps PRMS HRU outflows to :class:`FlowGraph` node inflows in
    the context of a :class:`Model`. To map HRU outflows to PRMS segments, see
    :class:`HruSegmentFlowExchange`.
    """

    def __init__(
        self,
        control: Control,
        discretization: Parameters,
        parameters: Parameters,
        sroff_vol: adaptable,
        ssres_flow_vol: adaptable,
        gwres_flow_vol: adaptable,
        imbalance_behavior: Literal["defer", None, "warn", "error"] = "defer",
        input_aliases: dict = None,
        verbose: bool = None,
    ) -> None:
        """Instantiate a HruNodeFlowExchange.

        Args:
            control: A :class:`Control` object.
            discretization: A discretizaion (:class:`Parameters`) for
              `PRMSChannel`.
            parameters: A :class:`Parameters` for `PRMSChannel`.
            sroff_vol: An :class:`Adaptable` of volumetric surface runoff.
            ssres_flow_vol: An :class:`Adaptable` of volumetric subsurface
              reservoir flow.
            gwres_flow_vol: An :class:`Adaptable` of volumetric groundwater
              reservoir flow.
            imbalance_behavior: one of ["defer", None, "warn", "error"]
              with "defer" being the default and defering to
              control.options["imbalance_behavior"] when available.
              When control.options["imbalance_behavior"] is not
              avaiable, imbalance_behavior is set to "warn".
            verbose: Boolean for the amount of messages to be printed.
        """
        super().__init__(
            control=control,
            discretization=discretization,
            parameters=parameters,
            input_aliases=input_aliases,
        )
        self.name = "HruNodeFlowExchange"

        self._set_inputs(locals())
        self._set_options(locals())

        self._hru_segment = self.hru_segment - 1

        self._set_budget(basis="global")

        return

    @staticmethod
    def get_dimensions() -> tuple:
        return ("nhru", "nnodes")

    @staticmethod
    def get_parameters() -> tuple:
        return ("hru_segment", "node_coord")

    @staticmethod
    def get_inputs() -> tuple:
        return (
            "sroff_vol",
            "ssres_flow_vol",
            "gwres_flow_vol",
        )

    @staticmethod
    def get_init_values() -> dict:
        return {
            "inflows": nan,
            "inflows_vol": nan,
            "sinks": nan,
            "sinks_vol": nan,
        }

    @staticmethod
    def get_mass_budget_terms():
        return {
            "inputs": [
                "sroff_vol",
                "ssres_flow_vol",
                "gwres_flow_vol",
            ],
            "outputs": ["inflows_vol", "sinks_vol"],
            "storage_changes": [],
        }

    def _set_initial_conditions(self) -> None:
        pass

    def _advance_variables(self) -> None:
        # could vary with timesstep some day
        return

    def _calculate(self, simulation_time: float) -> None:
        self._simulation_time = simulation_time

        s_per_time = self.control.time_step_seconds
        self._inputs_sum = (
            sum([vv.current for vv in self._input_variables_dict.values()])
            / s_per_time
        )

        self.inflows[:] = zero
        self.sinks[:] = zero  # this is an HRU variable
        for ihru in range(self.nhru):
            iseg = self._hru_segment[ihru]
            if iseg < 0:
                self.sinks[ihru] += self._inputs_sum[ihru]
            else:
                self.inflows[iseg] += self._inputs_sum[ihru]

        self.inflows_vol[:] = self.inflows * s_per_time
        self.sinks_vol[:] = self.sinks * s_per_time

        return


def prms_channel_flow_graph_postprocess(
    control: Control,
    input_dir: Union[str, pl.Path],
    prms_channel_dis: Parameters,
    prms_channel_params: Parameters,
    new_nodes_maker_dict: dict,
    new_nodes_maker_names: list,
    new_nodes_maker_indices: list,
    new_nodes_maker_ids: list,
    new_nodes_flow_to_nhm_seg: list,
    addtl_output_vars: list[str] = None,
    allow_disconnected_nodes: bool = False,
    imbalance_behavior: Literal["defer", None, "warn", "error"] = "defer",
    prms_channel_node_maker_name: str = "prms_channel",
    type_check_nodes: bool = False,
) -> FlowGraph:
    r"""Add nodes to a PRMSChannel-based FlowGraph to run from known inputs.

    This function helps construct a :class:`FlowGraph` starting from existing
    data for a :class:`PRMSChannel` simulation. The resulting FlowGraph is
    meant to run by itself, forced by known input flows.

    One limitation is that this FlowGraph currently has no-inflow to
    non-PRMSChannel nodes (but this could be added/accomodated).

    Note that if you want to run a :class:`FlowGraph` along with other
    :class:`Process`\ es in a :class:`Model`, then the helper function
    :func:`prms_channel_flow_graph_to_model_dict` is for you.

    See :class:`FlowGraph` for additional details and discussion.

    Please see the example notebook `examples/06_flow_graph_starfit.ipynb <https://github.com/DOI-USGS/pywatershed/blob/develop/examples/06_flow_graph_starfit.ipynb>`__
    which highlights both this helper function and
    :func:`prms_channel_flow_graph_to_model_dict`.

    Args:
        control: The control object for the run
        input_dir: the directory where the input files are found
        prms_channel_dis: the PRMSChannel discretization object
        prms_channel_params: the PRMSChannel parameters object.
        new_nodes_maker_dict: a dictionary of key/name and and instantiated
            NodeMakers.
        new_nodes_maker_names: collated list of what node makers to use.
        new_nodes_maker_indices: collated list of indices relative to each
            NodeMaker.
        new_nodes_maker_ids: Collated list of ids relative to each NodeMaker.
        new_nodes_flow_to_nhm_seg: collated list describing the nhm_segs to
            which the new nodes will flow. Use of non-positive entries specifies
            the zero-based index for flowing to nodes specified in these
            collated parameters, allowing these new nodes to be added in
            groups, in series to the existing NHM FlowGraph. Note that a new
            node may not be placed below any outflow point of the domain.
        imbalance_behavior: one of ["defer", None, "warn", "error"]
            with "defer" being the default and defering to
            control.options["imbalance_behavior"] when available. When
            control.options["imbalance_behavior"] is not avaiable,
            imbalance_behavior is set to "warn".

    Returns:
        An instantiated FlowGraph object.

    """  # noqa: E501
    if imbalance_behavior == "defer":
        if "imbalance_behavior" in control.options.keys():
            imbalance_behavior = control.options["imbalance_behavior"]
        else:
            imbalance_behavior = "warn"

    params_flow_graph, node_maker_dict = _build_flow_graph_inputs(
        prms_channel_dis,
        prms_channel_params,
        new_nodes_maker_dict,
        new_nodes_maker_names,
        new_nodes_maker_indices,
        new_nodes_maker_ids,
        new_nodes_flow_to_nhm_seg,
        prms_channel_node_maker_name=prms_channel_node_maker_name,
    )

    # combine PRMS lateral inflows to a single non-volumetric inflow
    input_variables = {}
    for key in PRMSChannel.get_inputs():
        nc_path = input_dir / f"{key}.nc"
        input_variables[key] = AdapterNetcdf(nc_path, key, control)

    inflows_prms = HruSegmentFlowAdapter(
        prms_channel_params, **input_variables
    )

    class GraphInflowAdapter(Adapter):
        def __init__(
            self,
            prms_inflows: Adapter,
            variable: str = "inflows",
        ):
            self._variable = variable
            self._prms_inflows = prms_inflows

            self._nnodes = params_flow_graph.dims["nnodes"]
            self._nseg = prms_channel_params.dims["nsegment"]
            self._current_value = np.zeros(self._nnodes) * nan
            return

        def advance(self) -> None:
            self._prms_inflows.advance()
            nseg = self._nseg
            self._current_value[0:nseg] = self._prms_inflows.current
            # no inflow non-prms-channel nodes
            self._current_value[nseg:] = zero
            return

    inflows_graph = GraphInflowAdapter(inflows_prms)

    flow_graph = FlowGraph(
        control=control,
        discretization=prms_channel_dis,
        parameters=params_flow_graph,
        inflows=inflows_graph,
        node_maker_dict=node_maker_dict,
        addtl_output_vars=addtl_output_vars,
        imbalance_behavior=imbalance_behavior,
        type_check_nodes=type_check_nodes,
        allow_disconnected_nodes=allow_disconnected_nodes,
    )
    return flow_graph


def prms_channel_flow_graph_to_model_dict(
    model_dict: dict,
    prms_channel_dis: Parameters,
    prms_channel_dis_name: str,
    prms_channel_params: Parameters,
    new_nodes_maker_dict: dict,
    new_nodes_maker_names: list,
    new_nodes_maker_indices: list,
    new_nodes_maker_ids: list,
    new_nodes_flow_to_nhm_seg: list,
    addtl_output_vars: list[str] = None,
    graph_imbalance_behavior: Literal[
        "defer", None, "warn", "error"
    ] = "defer",
    allow_disconnected_nodes: bool = False,
    prms_channel_node_maker_name: str = "prms_channel",
) -> dict:
    r"""Add nodes to a PRMSChannel-based FlowGraph within a Model's model_dict.

    This function helps construct a :class:`FlowGraph` starting from existing
    data for a :class:`PRMSChannel` simulation. The resulting FlowGraph is
    inserted into a model_dict used to instantiate a :class:`Model`.

    One limitation is that this FlowGraph currently has no-inflow to
    non-PRMSChannel nodes (but this could be added/accomodated).

    Note that if you want to run a :class:`FlowGraph` by itself, simply
    forced by known inflows (and not in the context of other
    :class:`base.Process`\ es in a :class:`Model`), then the helper function
    :func:`prms_channel_flow_graph_postprocess` is for you.

    Please see the example notebook `examples/06_flow_graph_starfit.ipynb <https://github.com/DOI-USGS/pywatershed/blob/develop/examples/06_flow_graph_starfit.ipynb>`__
    which highlights both this helper function and
    :func:`prms_channel_flow_graph_postprocess`.

    Args:
        model_dict: an existing model_dict to which to add the FlowGraph
        prms_channel_dis: the PRMSChannel discretization object
        prms_channel_dis_name: the name of the above discretization in the
            model_dict,
        prms_channel_params: the PRMSChannel parameters object,
        new_nodes_maker_dict: a dictionary of key/name and and instantiated
            NodeMakers
        new_nodes_maker_names: collated list of what node makers to use
        new_nodes_maker_indices: collated list of indices relative to each
            NodeMaker
        new_nodes_maker_ids: Collated list of ids relative to each NodeMaker.
        new_nodes_flow_to_nhm_seg: collated list describing the nhm_segs to
            which the new nodes will flow. Use of non-positive entries specifies
            the zero-based index for flowing to nodes specified in these
            collated parameters, allowing these new nodes to be added in
            groups, in series to the existing NHM FlowGraph. Note that a new
            node may not be placed below any outflow point of the domain.
        graph_imbalance_behavior: one of ["defer", None, "warn", "error"] with
            "defer" being the default and defering to
            control.options["imbalance_behavior"] when available. When
            control.options["imbalance_behavior"] is not avaiable, imbalance_behavior is set
            to "warn".

    Returns:
        A model dictionary.

    """  # noqa: E501
    import xarray as xr

    params_flow_graph, node_maker_dict = _build_flow_graph_inputs(
        prms_channel_dis,
        prms_channel_params,
        new_nodes_maker_dict,
        new_nodes_maker_names,
        new_nodes_maker_indices,
        new_nodes_maker_ids,
        new_nodes_flow_to_nhm_seg,
        prms_channel_node_maker_name=prms_channel_node_maker_name,
    )

    def exchange_calculation(self) -> None:
        _hru_segment = self.hru_segment - 1
        s_per_time = self.control.time_step_seconds
        self._inputs_sum = (
            sum([vv.current for vv in self._input_variables_dict.values()])
            / s_per_time
        )

        # This zeros in at the end mean that zero inflows and sinks in added
        # nodes
        self.inflows[:] = zero
        # sinks is an HRU variable, its accounting in budget is fine because
        # global collapses it to a scalar before summing over variables
        self.sinks[:] = zero

        for ihru in range(self.nhru):
            iseg = _hru_segment[ihru]
            if iseg < 0:
                self.sinks[ihru] += self._inputs_sum[ihru]
            else:
                self.inflows[iseg] += self._inputs_sum[ihru]

        self.inflows_vol[:] = self.inflows * s_per_time
        self.sinks_vol[:] = self.sinks * s_per_time

    Exchange = inflow_exchange_factory(
        dimension_names=("nhru", "nnodes"),
        parameter_names=("hru_segment", "node_coord"),
        input_names=PRMSChannel.get_inputs(),
        init_values={
            "inflows": np.nan,
            "inflows_vol": np.nan,
            "sinks": np.nan,
            "sinks_vol": np.nan,
        },
        mass_budget_terms={
            "inputs": [
                "sroff_vol",
                "ssres_flow_vol",
                "gwres_flow_vol",
            ],
            "outputs": ["inflows_vol", "sinks_vol"],
            "storage_changes": [],
        },
        calculation=exchange_calculation,
    )  # get the budget type into the exchange too: exchange_imbalance_behavior

    # Exchange parameters
    nnodes = params_flow_graph.dims["nnodes"]
    params_ds = prms_channel_params.to_xr_ds().copy()
    params_ds["node_coord"] = xr.Variable(
        dims="nnodes",
        data=np.arange(nnodes),
    )
    params_ds = params_ds.set_coords("node_coord")
    params_exchange = Parameters.from_ds(params_ds)

    graph_dict = {
        "inflow_exchange": {
            "class": Exchange,
            "parameters": params_exchange,
            "dis": prms_channel_dis_name,
        },
        "prms_channel_flow_graph": {
            "class": FlowGraph,
            "node_maker_dict": node_maker_dict,
            "parameters": params_flow_graph,
            "dis": None,
            "imbalance_behavior": graph_imbalance_behavior,
            "addtl_output_vars": addtl_output_vars,
            "allow_disconnected_nodes": allow_disconnected_nodes,
        },
    }

    model_dict["model_order"] += list(graph_dict.keys())
    model_dict = model_dict | graph_dict
    return model_dict


def _build_flow_graph_inputs(
    prms_channel_dis: Parameters,
    prms_channel_params: Parameters,
    new_nodes_maker_dict: dict,
    new_nodes_maker_names: list,
    new_nodes_maker_indices: list,
    new_nodes_maker_ids: list,
    new_nodes_flow_to_nhm_seg: list,
    prms_channel_node_maker_name: str = "prms_channel",
):
    prms_channel_flow_makers = [
        type(vv)
        for vv in new_nodes_maker_dict.values()
        if isinstance(vv, PRMSChannelFlowNodeMaker)
    ]
    assert len(prms_channel_flow_makers) == 0

    msg = "Inconsistency in collated inputs"
    assert len(new_nodes_maker_names) == len(new_nodes_maker_indices), msg
    assert len(new_nodes_maker_names) == len(new_nodes_flow_to_nhm_seg), msg
    # I think this is the only condition to check with
    # new_nodes_flow_to_nhm_seg
    assert len(new_nodes_flow_to_nhm_seg) == len(
        np.unique(new_nodes_flow_to_nhm_seg)
    ), "Cant have more than one new node flowing to an existing or new node."

    nseg = prms_channel_params.dims["nsegment"]
    nnew = len(new_nodes_maker_names)
    nnodes = nseg + nnew

    node_maker_name = [
        prms_channel_node_maker_name
    ] * nseg + new_nodes_maker_names
    maxlen = np.array([len(nn) for nn in node_maker_name]).max()
    # need this to be unicode U# for keys and searching below
    node_maker_name = np.array(node_maker_name, dtype=f"|U{maxlen}")
    node_maker_index = np.array(
        np.arange(nseg).tolist() + new_nodes_maker_indices
    )
    node_maker_id = np.append(
        prms_channel_dis.coords["nhm_seg"], np.array(new_nodes_maker_ids)
    )

    to_graph_index = np.zeros(nnodes, dtype=np.int64)
    dis_params = prms_channel_dis.parameters
    tosegment = dis_params["tosegment"] - 1  # fortan to python indexing
    to_graph_index[0:nseg] = tosegment

    # The new nodes which flow to other new_nodes have to be added after
    # the nodes flowing to existing nodes with nhm_seg ids.
    to_new_nodes_inds_in_added = {}
    added_new_nodes_inds_in_graph = {}
    for ii, nhm_seg in enumerate(new_nodes_flow_to_nhm_seg):
        if nhm_seg < 1:
            # negative indes are indices into the collated inputs lists
            # for new nodes being in-series.
            to_new_nodes_inds_in_added[ii] = -1 * nhm_seg
            continue

        wh_intervene_above_nhm = np.where(dis_params["nhm_seg"] == nhm_seg)
        wh_intervene_below_nhm = np.where(
            tosegment == wh_intervene_above_nhm[0][0]
        )
        # have to map to the graph from an index found in prms_channel
        wh_intervene_above_graph = np.where(
            (node_maker_name == prms_channel_node_maker_name)
            & (node_maker_index == wh_intervene_above_nhm[0][0])
        )
        wh_intervene_below_graph = np.where(
            (node_maker_name == prms_channel_node_maker_name)
            & np.isin(node_maker_index, wh_intervene_below_nhm)
        )

        to_graph_index[nseg + ii] = wh_intervene_above_graph[0][0]
        to_graph_index[wh_intervene_below_graph] = nseg + ii
        added_new_nodes_inds_in_graph[ii] = nseg + ii

    # <
    to_new_nodes_inds_remaining = deepcopy(to_new_nodes_inds_in_added)
    # worst case scenario is that we have to iterate the length of this list
    # if the items in the list are in the wrong order
    for itry in range(len(to_new_nodes_inds_remaining)):
        # for input_ind, to_ind_remain in to_new_nodes_inds_remaining.items():
        for input_ind in list(to_new_nodes_inds_remaining.keys()):
            to_ind_remain = to_new_nodes_inds_remaining[input_ind]
            if to_ind_remain not in added_new_nodes_inds_in_graph.keys():
                continue
            flows_to_ind = added_new_nodes_inds_in_graph[to_ind_remain]
            flows_from_inds = np.where(to_graph_index == flows_to_ind)

            to_graph_index[nseg + input_ind] = flows_to_ind
            to_graph_index[flows_from_inds] = nseg + input_ind
            added_new_nodes_inds_in_graph[input_ind] = nseg + input_ind
            del to_new_nodes_inds_remaining[input_ind]

    if len(to_new_nodes_inds_remaining):
        msg = "Unable to connect some new nodes in-series."
        raise ValueError(msg)

    # <
    param_dict = dict(
        dims={
            "nnodes": nnodes,
        },
        coords={
            "node_coord": np.arange(nnodes),
        },
        data_vars={
            "node_maker_name": node_maker_name,
            "node_maker_index": node_maker_index,
            "node_maker_id": node_maker_id,
            "to_graph_index": to_graph_index,
        },
        metadata={
            "node_coord": {"dims": ["nnodes"]},
            "node_maker_name": {"dims": ["nnodes"]},
            "node_maker_index": {"dims": ["nnodes"]},
            "node_maker_id": {"dims": ["nnodes"]},
            "to_graph_index": {"dims": ["nnodes"]},
        },
    )
    params_flow_graph = Parameters(**param_dict, validate=True)

    # make available at top level __init__
    node_maker_dict = {
        prms_channel_node_maker_name: PRMSChannelFlowNodeMaker(
            prms_channel_dis, prms_channel_params
        ),
    } | new_nodes_maker_dict

    return (params_flow_graph, node_maker_dict)


def prms_segment_lateral_inflow_components_to_netcdf(
    control: Control,
    parameters: Parameters,
    input_dir: pl.Path,
    nc_out_file_path: str,
    output_sum: bool = False,
) -> None:
    """Write to NetCDf the components of lateral flow on PRMS segments.

    This helper function takes the PRMS lateral flows from HRUs (sroff_vol,
    ssres_flow_vol, and gwres_flow_vol) and maps them individually to the
    PRMS segments.

    Args:
      control: A Control object for the input files selected.
      parameters: A Parameters object with both nhru and nsegment dimensions.
      input_dir: The directory to look for the inputs files: sroff_vol.nc,
        ssres_flow_vol.nc, and gwres_flow_vol.nc.
      nc_out_file_path: The path of the output netcdf file.
      output_sum: Also include the sum of the lateral flow components in the
        output file (named 'lateral_inflow_vol').

    Examples
    ---------

    This example will work if you have an editable install of the repository
    (not installed from pypi) and you have generated the test data for the
    drb_2yr domain.

    >>> import pathlib as pl
    >>>
    >>> import pywatershed as pws
    >>> import xarray as xr
    >>>
    >>> domain_dir = (
    ...     pws.constants.__pywatershed_root__ / "../test_data/drb_2yr"
    ... )
    >>>
    >>> control = pws.Control.load_prms(domain_dir / "nhm.control")
    >>> params = pws.parameters.PrmsParameters.load(
    ...     domain_dir / control.options["parameter_file"]
    ... )
    >>>
    >>> nc_out_file_path = pl.Path(".") / "segment_lateral_inflows.nc"
    >>> pws.prms_segment_lateral_inflow_components_to_netcdf(
    ...     control,
    ...     params,
    ...     input_dir=domain_dir
    ...     / "output",  # PRMS/pywatershed outputs are here
    ...     nc_out_file_path=nc_out_file_path,
    ...     output_sum=True,
    ... )
    >>>
    >>> lat_inflow_vols = xr.load_dataset(nc_out_file_path)
    >>> display(lat_inflow_vols)
    <xarray.Dataset> Size: 11MB
    Dimensions:             (time: 731, nhm_seg: 456)
    Coordinates:
      * time                (time) datetime64[ns] 6kB 1979-01-01 ... 1980-12-31
      * nhm_seg             (nhm_seg) int64 4kB 2048 4182 4183 ... 2045 2046 2047
    Data variables:
        sroff_vol           (time, nhm_seg) float64 3MB 10.35 10.96 ... 0.0 0.0
        ssres_flow_vol      (time, nhm_seg) float64 3MB 0.0 0.0 ... 0.01131 0.01401
        gwres_flow_vol      (time, nhm_seg) float64 3MB 4.181 1.993 ... 1.762 2.33
        lateral_inflow_vol  (time, nhm_seg) float64 3MB 14.53 12.95 ... 1.774 2.344
    Attributes :
        description:  Daily lateral inflow volumes to PRMS Channel
        units:        cubic feet


    """  # noqa: E501

    time = np.arange(
        control.start_time,
        control.end_time + control.time_step,  # per arange construction
        control.time_step,
    ).astype("datetime64[ns]")
    nhm_seg = parameters.parameters["nhm_seg"]
    nhm_id = parameters.parameters["nhm_id"]

    ntime = len(time)
    nsegment = len(nhm_seg)
    nhru = len(nhm_id)

    zero_adapter = adapter_factory(np.zeros(nhru), "zero", control)

    input_adapters = {}
    for vv in ["sroff_vol", "ssres_flow_vol", "gwres_flow_vol"]:
        nc_path = input_dir / f"{vv}.nc"
        input_adapters[vv] = AdapterNetcdf(nc_path, vv, control)

    exch_sroff = HruSegmentFlowAdapter(
        parameters=parameters,
        sroff_vol=input_adapters["sroff_vol"],
        ssres_flow_vol=zero_adapter,
        gwres_flow_vol=zero_adapter,
    )

    exch_ssres = HruSegmentFlowAdapter(
        parameters=parameters,
        sroff_vol=zero_adapter,
        ssres_flow_vol=input_adapters["ssres_flow_vol"],
        gwres_flow_vol=zero_adapter,
    )

    exch_gwres = HruSegmentFlowAdapter(
        parameters=parameters,
        sroff_vol=zero_adapter,
        ssres_flow_vol=zero_adapter,
        gwres_flow_vol=input_adapters["gwres_flow_vol"],
    )

    data_vars = dict(
        sroff_vol=(
            ["time", "nhm_seg"],
            np.zeros((ntime, nsegment)) * np.nan,
        ),
        ssres_flow_vol=(
            ["time", "nhm_seg"],
            np.zeros((ntime, nsegment)) * np.nan,
        ),
        gwres_flow_vol=(
            ["time", "nhm_seg"],
            np.zeros((ntime, nsegment)) * np.nan,
        ),
    )
    if output_sum:
        data_vars["lateral_inflow_vol"] = (
            ["time", "nhm_seg"],
            np.zeros((ntime, nsegment)) * np.nan,
        )

    seg_lateral_inflow = xr.Dataset(
        data_vars=data_vars,
        coords=dict(
            time=(["time"], time),
            nhm_seg=(["nhm_seg"], nhm_seg),
        ),
        attrs=dict(
            description="Daily lateral inflow volumes to PRMS Channel",
            units="cubic feet",
        ),
    )
    for tt in range(control.n_times):
        control.advance()
        exch_sroff.advance()
        exch_ssres.advance()
        exch_gwres.advance()
        seg_lateral_inflow["sroff_vol"][tt, :] = exch_sroff.current
        seg_lateral_inflow["ssres_flow_vol"][tt, :] = exch_ssres.current
        seg_lateral_inflow["gwres_flow_vol"][tt, :] = exch_gwres.current
        # sum
        if output_sum:
            seg_lateral_inflow["lateral_inflow_vol"][tt, :] = (
                exch_sroff.current + exch_ssres.current + exch_gwres.current
            )

    seg_lateral_inflow.to_netcdf(nc_out_file_path)

    return None
