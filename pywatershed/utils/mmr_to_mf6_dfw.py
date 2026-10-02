import pathlib as pl
from warnings import warn

import numpy as np
import xarray as xr

from pywatershed import Control, meta

from ..constants import fileish, zero
from ..parameters import PrmsParameters
from .network_hydraulics import calculate_seg_mid_elevations
from .optional_import import import_optional_dependency

flopy = import_optional_dependency("flopy", errors="ignore")
gpd = import_optional_dependency("geopandas", errors="ignore")
pint = import_optional_dependency("pint", errors="ignore")
shapely = import_optional_dependency("shapely", errors="ignore")
mpsplines = import_optional_dependency("mpsplines", errors="warn")


# Note: there are some headaches in handling time because PRMS's end day
# is included, the beginning of that day is not the end of the run.

# TODOS:
#     * Deal with time/sim being optional
#     * are start_time and end_time supported to change these times?


class MmrToMf6Dfw:
    """PRMS Muskingum-Mann Routing (MMR) to MF6 Diffusive Wave (DFW).

    Terms:

    * MMR: Muskingum-Mann Routing in PRMS
    * DFW: Diffusive Wave in MF6 (develop branch)

    This class builds a MF6 simulation with Diffusive Wave (DFW) routing from
    PRMS NHM input files for Muskingum-Mann Routing (MMR) and a few simple
    assumptions. The lateral (to-channel) fluxes from a PRMS are used as
    time varying boundary conditions.

    Please see the example notebook
    `examples/mmr_to_mf6_dfw.ipynb <https://github.com/DOI-USGS/pywatershed/blob/develop/examples/mmr_to_mf6_dfw.ipynb>`__ which demonstrates running the Delaware River
    Basin in MF6 DFW based on the PRMS data and its lateral inflows.

    In addition to standard MF6 packages and their input files (e.g. IMS, OC,
    etc), the surface water flow package is created with the diffusive wave
    (DFW) package and depends on the DISV1D and FLW boundary conditions. The
    cross sectional area (CXS) package is optional.

    The daily inflows from PRMS can optionally be smoothed to desired sub-daily
    resolution of MF6 stress period length using mean preserving splines from
    the mpsplines python package https://github.com/jararias/mpsplines:

    `Ruiz-Arias, J. A. (2022). Mean-preserving interpolation with splines
    for solar radiation modeling. Solar Energy, Vol. 248, pp.
    121-127. <https://doi.org/10.1016/j.solener.2022.10.038>`__


    PRMS MMR Parameters used and how:

    * tosegment:
       Used to give the reach/segment connectivity (transformed to zero based
       for use in python and flopy). No units.
    * seg_length:
        Used for SfwDisl, the discretization of the reaches, directly. Also
        used to calculate seg_mid_elevation if not present in parameters.
        Units of meters from metadata (these are converted to units requested
        for modflow, which we currently hardcode to meters and seconds.)
    * seg_slope:
        Passed directly to CHF DFW. Also used to calculate seg_mid_elevation if
        not present in parameters. See its description below. Unitless slope.
    * hru_segment:
        Used to calculate seg_mid_elevation if not present in parameters. See
        its description below. Unitless index mapping from PRMS HRUs to
        segments.
    * hru_elev:
        Used to calculate seg_mid_elevation if not present in parameters. See
        its description below. Units are in meters (though not documented in
        the metadata as such, just "elev_units")
    * seg_width:
        This is (apparently) the NHD bank-full width present in the PRMS
        parameter files but unused in PRMS. Units are in meters.
    * mann_n:
        Mannings N coefficient for MMR. Passed directly to CHF DFW. Units in
        seconds / meter ** (1/3) according to the metadata.

    The following optional parameters can be user-supplied but are NOT part of
    PRMS parameters:

    * seg_mid_elevation:
        The elevation at the mid-point of a segment. This is calculated always
        starting from one of the basin outlets. The outlet elevation is
        calculated as per chd_options["stress_period_data"], the downstream
        end of the segment is assumed to have the elevation of the lowest HRU
        which maps to this segment

            ``outlet_downstream_elev = lowest_hru_elev``

        The upstream end of the outlet segment is then calculated

            ``outlet_upstream_elev =  lowest_hru_elev + seg_length * seg_slope``.

        The midpoint of the of the outlet is the average of upstream and
        downstream segment elevations. For all other segments above outlets,
        its downstream elevation is the same as its downstream segment's
        upstream elevation and so

            ``segment_upstream_elev = downstream_seg_upstream_elev + seg_length * seg_slope``.

        And its midpoint is again the average elevation between upstream and
        downstream elevations of the segment.
    * chd_options["stress_period_data"]:
        At each outlet in the domain, this is taken as the lowest hru elevation
        over all hrus which map to this segment.

    The MF6 IO descrption can be found here
    https://modflow6.readthedocs.io/en/latest/mf6io.html

    """  # noqa: E501

    def __init__(
        self,
        control_file: fileish = None,
        control: Control = None,
        start_time: np.datetime64 = None,
        end_time: np.datetime64 = None,
        tdis_perlen: int = 86400,
        tdis_nstp: int = 1,
        param_file: fileish = None,
        params: PrmsParameters = None,
        segment_shp_file: fileish = None,
        output_dir: fileish = pl.Path("."),
        bc_binary_files: bool = False,
        bc_flows_combined: bool = False,
        sim_name: str = "mmr_to_dfw",
        inflow_dir: fileish = None,
        inflow_from_PRMS: bool = True,
        # intial flows over ride from file?
        # length_units="meters",
        # time_units="seconds",
        save_flows: bool = True,
        time_zone: str = "UTC",
        write_on_init: bool = False,
        chd_options: dict = None,
        cxs_options: dict = None,
        disv1d_options: dict = None,
        dfw_options: dict = None,
        dfw_griddata: dict = None,
        flw_options: dict = None,
        ic_options: dict = None,
        ims_options: dict = None,
        oc_options: dict = None,
        sto_options: dict = None,
    ):
        """Instantiate a MmrToMf6Dfw object.

        Args:
          control_file: The path to a PRMS control file. Exactly one of this
            and control are required. If start and end times are the same, then
            the run is considered a steady state run.
          control: a Control object. Exactly one of this and control are
            required.
          start_time: np.datetime64 = None, to edit the start time.
          end_time: np.datetime64 = None, to edit the simulation end time.
          tdis_perlen: The number of seconds in the MF6 stress periods. The
            value must evenly dvide 1 day or 86400 seconds. Currently the value
            must be the same for all stress periods. Default is 1 day = 86400s.
            For values less than the default, mean-preserving splines are used
            to smooth the flows to the desired resolution.
          tdis_nstp: The number of substeps in each stress period. The value
            must be the same for all stress periods and the default is 1.
          param_file: The filepath to the domain parameters. Exactly one of
            this and the params argument is required.
          params: a Parameter object. Exactly one of this and the params
            argument is required.
          segment_shp_file: a shapefile of segments assumptions are that this
            file has a column "nsevment_v" which is the same as "nhm_seg" in
            the parameter file and another column "model_idx" corresponding to
            the 1-based index of nhm_seg in the parameter file.
          output_dir: Where to write and run the model.
          bc_binary_files: Write binary files for the FLW boundary conditions.
          bc_flows_combined: A single, combined boundary flow file, or 3
            individual files for the 3 prms flux terms?
          sim_name: A name for the simulation.
          inflow_dir: Where to find the PRMS to-channel flux files.
          inflow_from_PRMS: The PRMS inflows are in depth while the inflows
            from pywatershed are volumetric and they have different names
            indicating this. The default is PRMS inflows.
          save_flows: Set as a general MF6 option on all packages.
          time_zone: the timezone to use.
          write_on_init: Write the MF6 files on initialization?
          chd_options: Options to pass to flopy when making each MF6 package.
            These are not always applied and their effect should be checked on
            the MF6 input files written before running the model. Still a work
            in progress.
          cxs_options: See above.
          disv1d_options: See above.
          dfw_options: See above.
          dfw_griddata: See above.
          flw_options: See above.
          ic_options: See above.
          ims_options: See above.
          oc_options: See above.
          sto_options: See above.

        """
        self._params = params
        self._param_file = param_file
        self._control = control
        self._control_file = control_file

        self._start_time = start_time
        self._end_time = end_time
        self._time_zone = time_zone
        self._tdis_perlen = tdis_perlen
        self._tdis_nstp = tdis_nstp

        self._written = False
        self._sim_name = sim_name
        self._output_dir = pl.Path(output_dir)
        self._segment_shp_file = segment_shp_file
        self._save_flows = save_flows

        self._inflow_from_PRMS = inflow_from_PRMS
        self._inflow_dir = inflow_dir
        self._bc_flows_combined = bc_flows_combined
        self._bc_binary_files = bc_binary_files

        self._units = pint.UnitRegistry(system="mks")
        # these are not really optional at the moment, but hope to make so soon
        # making the system definition using input units.
        self._length_units = "meters"
        self._time_units = "seconds"

        self._chd_options = chd_options
        self._cxs_options = cxs_options
        self._dfw_options = dfw_options
        self._disv1d_options = disv1d_options
        self._dfw_griddata = dfw_griddata
        self._flw_options = flw_options
        self._ic_options = ic_options
        self._ims_options = ims_options
        self._oc_options = oc_options
        self._sto_options = sto_options

        self._handle_control_parameters()
        self._set_sim()
        self._set_tdis()
        self._set_chf()
        self._set_disv1d()
        self._set_flw()
        self._set_ims()
        self._set_dfw()
        self._set_sto()
        self._set_ic()
        self._set_cxs()
        self._set_oc()
        self._set_chd()

        if write_on_init:
            self.write()

        return

    def write(self, *args, rewrite: bool = False, **kwargs):
        """Write the simulation to disk.

        Args:
          *args: Arguments to pass to flopy's sim.write_simulation.
          rewrite: Allow the simulation to be written more than once.
          **kwargs: Keyword arguments to pass to sim.write_simulation.
        """
        if self._written and not rewrite:
            msg = (
                "The simulation was already written, "
                "use rewrite=True to force."
            )
            warn(msg)
            return

        print(f"\nWriting simulation files to: {self._output_dir}")
        self._written = True
        return self._sim.write_simulation(*args, **kwargs)

    def run(self, *args, **kwargs):
        """Run the simulation.

        Args:
          *args: Arguments to pass to flopy's sim.write_simulation.
          **kwargs: Keyword arguments to pass to sim.write_simulation.
        """
        print(f"\nRunning simulation files in: {self._output_dir}")
        return self._sim.run_simulation(*args, **kwargs)

    def _handle_control_parameters(self):
        inputs_dict = {
            "parameters": {"obj": self._params, "file": self._param_file},
            "control": {"obj": self._control, "file": self._control_file},
        }

        for key, val_dict in inputs_dict.items():
            obj_file = val_dict["file"]
            obj = val_dict["obj"]

            if (obj_file is not None) and (obj is not None):
                if key == "params":
                    msg = "Can only specify one of param_file or params"
                else:
                    msg = "Can only specify one of control_file or control"
                raise ValueError(msg)

            elif (obj_file is None) and (obj is None):
                if key == "params":
                    msg = "Must specify (exactly) one of param_file or params"
                    raise ValueError(msg)
                else:
                    msg = (
                        "When control is not passed to MMRToMF6, it does "
                        "not create MF6 .nam, .flw, nor .obs files"
                    )
                    setattr(self, "control_file", None)
                    setattr(self, "control", None)
                    warn(msg)

            elif obj_file:
                setattr(self, f"{key}_file", obj_file)
                if key == "params":
                    setattr(self, "params", PrmsParameters.load(obj_file))
                else:
                    setattr(
                        self,
                        "control",
                        Control.load_prms(obj_file, warn_unused_options=False),
                    )

            else:
                setattr(self, f"{key}_file", None)
                setattr(self, key, obj)

        # set dimensions on self
        self._nsegment = self.parameters.dims["nsegment"]
        self._hru_segment = self.parameters.parameters["hru_segment"] - 1

        return

    def _set_sim(self):
        self._sim = flopy.mf6.MFSimulation(
            sim_ws=str(self._output_dir),
            sim_name=self._sim_name,
            # version="mf6",
            # exe_name="mf6",
            memory_print_option="all",
        )

    def _set_tdis(self):
        if self._start_time is None:
            self._start_time = self.control.start_time
        if self._end_time is None:
            self._end_time = self.control.end_time

        assert self._tdis_perlen <= 86400
        ratio = 86400 / self._tdis_perlen
        assert ratio.is_integer()

        if self._start_time == self._end_time:
            # this is steady state, not a 1 day run
            self._nper = 1
        else:
            # The last day is INCLUDED in PRMS, so have to add a day
            run_duration_n_seconds = (
                (self._end_time + np.timedelta64(1, "D")) - self._start_time
            ) / np.timedelta64(1, "s")
            self._nper = int((run_duration_n_seconds / self._tdis_perlen))

        if hasattr(self, "control"):
            # perlen, nstp, stmult
            tdis_rc = [
                (self._tdis_perlen, self._tdis_nstp, 1.0)
                for ispd in range(self._nper)
            ]
            _ = flopy.mf6.ModflowTdis(
                self._sim,
                pname="tdis",
                time_units=self._time_units,
                start_date_time=str(self._start_time) + self._time_zone,
                nper=self._nper,
                perioddata=tdis_rc,
            )

    def _set_chf(self):
        self._chf = flopy.mf6.ModflowChf(
            self._sim, modelname=self._sim_name, save_flows=self._save_flows
        )

    def _set_disv1d(self):
        params = self.parameters
        parameters = params.parameters
        opt_dict_name = "_disv1d_options"
        method_name = "_set_disv1d"
        if self._disv1d_options is None:
            self._disv1d_options = {}

        if "idomain" not in self._disv1d_options.keys():
            self._disv1d_options["idomain"] = 1

        if "width" not in self._disv1d_options.keys():
            # meters, per metadata
            self._disv1d_options["width"] = parameters["seg_width"]

        if "length_units" not in self._disv1d_options.keys():
            self._disv1d_options["length_units"] = self._length_units

        if "bottom" not in self._disv1d_options.keys():
            self._warn_option_overwrite("bottom", opt_dict_name, method_name)
            # calculate the bottom elevations
            if "seg_mid_elevation" in parameters.keys():
                self._seg_mid_elevation = parameters["seg_mid_elevation"]
            else:
                self._calculate_seg_mid_elevations(check=False)
            # <
            self._disv1d_options["bottom"] = self._seg_mid_elevation

        # < vertices and cell1d
        if self._segment_shp_file is not None:
            opts_set = [
                "nodes",
                "nvert",
                "vertices",
                "cell1d",
            ]
            for oo in opts_set:
                self._warn_option_overwrite(oo, opt_dict_name, method_name)

            segment_gdf = gpd.read_file(self._segment_shp_file)
            nreach = params.dims["nsegment"]

            reach_vert_inds = {}
            reach_vert_geoms = {}
            vert_count = 0
            for rr in range(nreach):
                reach_id = parameters["nhm_seg"][rr]
                wh_geom = np.where(segment_gdf.nsegment_v.values == reach_id)[
                    0
                ][0]
                reach_geom = shapely.get_coordinates(
                    segment_gdf.iloc[wh_geom].geometry
                ).tolist()
                # if it has a "to" drop the last vertex, reassign in a second
                # This shouldnt be necessary..
                to_reach_id = parameters["tosegment_nhm"][rr]
                # if to_reach_id != 0:
                #     reach_geom = reach_geom[0:-1]
                reach_vert_geoms[reach_id] = reach_geom
                reach_vert_inds[reach_id] = list(
                    range(vert_count, vert_count + len(reach_geom))
                )
                vert_count = vert_count + len(reach_geom)

            # check the count and collation
            vert_counts_sum = sum([len(rr) for rr in reach_vert_inds.values()])
            vert_geoms_sum = sum([len(rr) for rr in reach_vert_geoms.values()])
            assert vert_geoms_sum == vert_counts_sum
            assert (
                vert_counts_sum == list(reach_vert_inds.values())[-1][-1] + 1
            )

            # build the vertices
            vertices = []
            for rr in range(nreach):
                reach_id = parameters["nhm_seg"][rr]
                inds = reach_vert_inds[reach_id]
                geoms = reach_vert_geoms[reach_id]
                assert len(inds) == len(geoms)
                for vv in range(len(inds)):
                    vertices += [[inds[vv], geoms[vv][0], geoms[vv][1]]]

            # cell1d: for each reach, if it has a "to", get the to's first ind
            # as the last vertex ind for the reach
            cell1d = []
            for rr in range(nreach):
                reach_id = parameters["nhm_seg"][rr]
                icvert = reach_vert_inds[reach_id]
                to_reach_id = parameters["tosegment_nhm"][rr]
                if to_reach_id != 0:
                    to_reach_start_ind = reach_vert_inds[to_reach_id][0]
                    icvert += [to_reach_start_ind]
                # <
                ncvert = len(icvert)
                cell1d += [[rr, 0.5, ncvert, *icvert]]

            # < Just a check
            check_connectivity = True
            if check_connectivity:
                for rr in range(nreach):
                    reach_id = parameters["nhm_seg"][rr]

                    reach_ind = rr
                    reach_cell1d_start_ind = cell1d[reach_ind][3]
                    reach_cell1d_end_ind = cell1d[reach_ind][-1]

                    wh_geom = np.where(
                        segment_gdf.nsegment_v.values == reach_id
                    )
                    wh_geom = wh_geom[0][0]
                    reach_geom = shapely.get_coordinates(
                        segment_gdf.iloc[wh_geom].geometry
                    ).tolist()

                    # check "TO" connectivity
                    to_reach_id = parameters["tosegment_nhm"][rr]
                    if to_reach_id > 0:
                        to_reach_ind = np.where(
                            parameters["nhm_seg"] == to_reach_id
                        )[0][0]
                        to_reach_cell1d_start_ind = cell1d[to_reach_ind][3]
                        assert (
                            reach_cell1d_end_ind == to_reach_cell1d_start_ind
                        )

                        # wh_down_geom = np.where(
                        #     segment_gdf.nsegment_v.values == to_reach_id
                        # )[0][0]
                        # reach_down_geom = shapely.get_coordinates(
                        #     segment_gdf.iloc[wh_down_geom].geometry
                        # ).tolist()
                        # assert reach_geom[-1] == reach_down_geom[0]

                    # check all "FROM" connectivity
                    wh_from_reach = np.where(
                        parameters["tosegment_nhm"] == reach_id
                    )
                    from_reach_ids = parameters["nhm_seg"][
                        wh_from_reach
                    ].tolist()
                    for from_reach_id in from_reach_ids:
                        from_reach_ind = np.where(
                            parameters["nhm_seg"] == from_reach_id
                        )[0][0]
                        from_reach_cell1d_end_ind = cell1d[from_reach_ind][-1]
                        assert (
                            from_reach_cell1d_end_ind == reach_cell1d_start_ind
                        )

                        # wh_up_geom = np.where(
                        #     segment_gdf.nsegment_v.values == from_reach_id
                        # )[0][0]
                        # reach_up_geom = shapely.get_coordinates(
                        #     segment_gdf.iloc[wh_up_geom].geometry
                        # ).tolist()
                        # assert reach_up_geom[-1] == reach_geom[0]

            nnodes = len(cell1d)
            assert nnodes == self._nsegment
            nvert = len(vertices)

            self._nsegment = params.dims["nsegment"]
            self._tosegment = parameters["tosegment"] - 1

            self._disv1d_options["nodes"] = self._nsegment
            self._disv1d_options["nvert"] = nvert
            self._disv1d_options["vertices"] = vertices
            self._disv1d_options["cell1d"] = cell1d

        # <
        self._disv1d = flopy.mf6.ModflowChfdisv1D(
            self._chf, **self._disv1d_options
        )

    def _set_flw(self, **kwargs):
        if self._flw_options is None:
            self._flw_options = {}

        # Boundary conditions / FLW
        # aggregate inflows over the contributing fluxes

        # For non-binary data, this method has to be called after
        # flopy.mf6.Chfdisl is set on self._chf

        parameters = self.parameters.parameters

        if self._inflow_from_PRMS:
            inflow_list = ["sroff", "ssres_flow", "gwres_flow"]
        else:
            inflow_list = ["sroff_vol", "ssres_flow_vol", "gwres_flow_vol"]

        # check they all have the same units before summing
        inflow_units = list(meta.get_units(inflow_list, to_pint=True).values())
        inflow_unit = inflow_units[0]
        assert [inflow_unit] * len(inflow_units) == inflow_units
        inflow_unit = self._units(inflow_unit)

        def read_inflow(vv, start_time, end_time):
            ff = xr.load_dataset(pl.Path(self._inflow_dir) / f"{vv}.nc")[vv]
            return ff.sel(time=slice(start_time, end_time)).values

        inflows = {
            vv: read_inflow(vv, self._start_time, self._end_time)
            for vv in inflow_list
        }

        if self._bc_flows_combined:
            inflows["combined"] = sum(inflows.values())
            for kk in list(inflows.keys()):
                if kk != "combined":
                    del inflows[kk]

        # add the units
        inflows = {kk: vv * inflow_unit for kk, vv in inflows.items()}

        # flopy adds one to the index, but if we write binary we have to do it
        add_one = int(self._bc_binary_files)

        for flow_name in inflows.keys():
            # if from pywatershed, inflows are already volumes in cubicfeet
            if "inch" in str(inflow_unit):  # PRMS style need hru areas
                hru_area_unit = self._units(
                    list(meta.get_units("hru_area").values())[0]
                )
                hru_area = parameters["hru_area"] * hru_area_unit
                inflows[flow_name] *= hru_area

            prms_timestep_s = 24 * 60 * 60 * self._units("seconds")
            inflows[flow_name] /= prms_timestep_s.to("seconds")
            new_inflow_unit = inflows[flow_name].units

            if self._nper > 1:
                prms_mf6_ts_ratio = (
                    prms_timestep_s
                    / (self._tdis_perlen * self._units("seconds"))
                ).magnitude
                prms_nper = int((self._nper) / prms_mf6_ts_ratio)
            else:
                prms_nper = 1

            # print(f"{self._nper=}")
            # print(f"{prms_nper=}")

            # calculate lateral flow term to the REACH/segment from HRUs
            lat_inflow_prms = (
                np.zeros((prms_nper, self._nsegment)) * new_inflow_unit
            )

            for ihru in range(self.parameters.dims["nhru"]):
                iseg = self._hru_segment[ihru]
                if iseg < 0:
                    # This is bad, selective handling of fluxes is not cool,
                    # mass is being discarded in a way that has to be
                    # coordinated
                    # with other parts of the code.
                    # This code shuold be removed evenutally.
                    inflows[flow_name][:, ihru] = zero * new_inflow_unit
                    continue

                lat_inflow_prms[:, iseg] += inflows[flow_name][:, ihru]

            # the target
            lat_inflow = np.zeros((self._nper, self._nsegment))
            time_prms = np.arange(0, prms_nper)
            time_mf6 = np.linspace(
                0, prms_nper - 1, num=self._nper, endpoint=True
            )

            if len(time_prms) > 0 and len(time_prms) != len(time_mf6):
                for iseg in range(lat_inflow_prms.shape[1]):
                    lat_inflow[:, iseg] = (
                        mpsplines.MeanPreservingInterpolation(
                            yi=lat_inflow_prms[:, iseg].magnitude,
                            xi=time_prms,
                            periodic=False,
                        )(time_mf6)
                    )
            else:
                lat_inflow = lat_inflow_prms.magnitude

            lat_inflow = lat_inflow * new_inflow_unit

            # convert to output units in the output data structure
            flw_spd = {}
            for ispd in range(self._nper):
                flw_ispd = [
                    (
                        irch + add_one,
                        lat_inflow[ispd, irch].to_base_units().magnitude,
                    )
                    for irch in range(self._nsegment)
                ]

                if self._bc_binary_files:
                    # should put the time in the file names?
                    ra = np.array(
                        flw_ispd, dtype=[("irch", "<i4"), ("q", "<f8")]
                    )
                    i_time_str = str(
                        self.control.start_time + ispd * self.control.time_step
                    )
                    bin_name = (
                        f"chf_flw_bc/"
                        f"flw_{flow_name}_{i_time_str.replace(':', '_')}.bin"
                    )
                    bin_name_pl = self._output_dir / bin_name
                    if not bin_name_pl.parent.exists():
                        bin_name_pl.parent.mkdir()
                    _ = ra.tofile(bin_name_pl)

                    flw_spd[ispd] = {
                        "filename": str(bin_name),
                        "binary": True,
                        "data": None,
                    }

                else:
                    flw_spd[ispd] = flw_ispd

            for key in ["print_input", "print_flows"]:
                if key not in self._flw_options.keys():
                    self._flw_options[key] = True

            _ = flopy.mf6.ModflowChfflw(
                self._chf,
                save_flows=self._save_flows,
                stress_period_data=flw_spd,
                maxbound=self._nsegment,
                pname=flow_name,
                **self._flw_options,
            )

        return

    def _set_ims(self):
        self._ims = flopy.mf6.ModflowIms(self._sim, **self._ims_options)

    def _set_dfw(self):
        parameters = self.parameters.parameters
        if "save_flows" not in self._dfw_options.keys():
            self._dfw_options["save_flows"] = self._save_flows

        if "manningsn" in self._dfw_options.keys():
            warn("Using supplied Manning's N and that from PRMS.")
        else:
            # seconds / meter ** (1/3)
            self._dfw_options["manningsn"] = parameters["mann_n"]

        _ = flopy.mf6.ModflowChfdfw(self._chf, **self._dfw_options)

    def _set_sto(self):
        if "save_flows" not in self._sto_options:
            self._sto_options["save_flows"] = self._save_flows
        _ = flopy.mf6.ModflowChfsto(
            self._chf,
            **self._sto_options,
        )

    def _set_ic(self):
        if self._ic_options is None:
            self._ic_options = {}
        else:
            # TODO JLM REVISIT
            ic_options = {
                "strt": self._seg_mid_elevation + self._ic_options["strt"]
            }

        self._ic = flopy.mf6.ModflowChfic(self._chf, **ic_options)

    def _set_cxs(self):
        if self._cxs_options is None:
            pass
        else:
            self._cxs = flopy.mf6.ModflowChfcxs(self._chf, **self._cxs_options)

    def _set_oc(self):
        if self._oc_options is None:
            self._oc_options = {}

        self._oc = flopy.mf6.ModflowChfoc(
            self._chf,
            budget_filerecord=f"{self._sim_name}.bud",
            stage_filerecord=f"{self._sim_name}.stage",
            # qoutflow_filerecord=f"{self._sim_name}.qoutflow",
            **self._oc_options,
        )

    def _set_chd(self):
        if self._chd_options is None:
            self._chd_options = {}
        if "stress_period_data" not in self._chd_options.keys():
            if hasattr(self, "_outlet_chds"):
                self._chd_options["stress_period_data"] = list(
                    self._outlet_chds.items()
                )

            else:
                msg = (
                    "CHD without calculation of seg_mid_elevation is "
                    "not yet implemented."
                )
                raise NotImplementedError(msg)
                # This approach just will not be consistent with
                # randomly supplied data for seg_mid_elevation,
                # likely not possible and would have to be passed in.

        # <<
        self._chd_options["maxbound"] = len(
            self._chd_options["stress_period_data"]
        )
        self._chd = flopy.mf6.ModflowChfchd(self._chf, **self._chd_options)

    def _calculate_seg_mid_elevations(self, check=False):
        mid, outlet_mid = calculate_seg_mid_elevations(self.parameters)
        # constant head at each outlet: 1 m above the midpoint elevation
        self._outlet_chds = {kk: 1.0 + vv for kk, vv in outlet_mid.items()}
        if check:
            params = self.parameters.parameters
            seg_dy = params["seg_slope"] * params["seg_length"]
            tosegment0 = params["tosegment"] - 1
            hru_seg0 = params["hru_segment"] - 1
            hru_elev = params["hru_elev"]
            for ss in range(len(seg_dy)):
                down = tosegment0[ss]
                if down == -1:
                    # an outlet's downstream end is the lowest elevation
                    # of the HRUs draining to it
                    down_end = mid[ss] - seg_dy[ss] / 2
                    outlet_elev = hru_elev[np.where(hru_seg0 == ss)].min()
                    if abs(down_end - outlet_elev) >= 1.0e-7:
                        raise ValueError(
                            f"Outlet segment {ss} downstream elevation "
                            f"{down_end} does not equal the minimum "
                            f"elevation {outlet_elev} of its HRUs"
                        )
                    continue
                # upstream end of ss equals upstream end of down + rise
                up_ss = mid[ss] + seg_dy[ss] / 2
                up_down = mid[down] + seg_dy[down] / 2
                if abs((up_ss - up_down) - seg_dy[ss]) >= 1.0e-7:
                    raise ValueError(
                        f"Segment {ss} upstream elevation {up_ss} is not "
                        f"its rise {seg_dy[ss]} above the upstream "
                        f"elevation {up_down} of downstream segment {down}"
                    )
        self._seg_mid_elevation = mid
        return

    def _warn_option_overwrite(self, opt_name, opt_dict_name, method_name):
        opt_dict = getattr(self, opt_dict_name)
        if opt_name in opt_dict.keys():
            msg = (
                f'Option "{opt_name}" in "{opt_dict_name}" '
                f'will be overwritten in method "{method_name}".'
            )
            warn(msg)
