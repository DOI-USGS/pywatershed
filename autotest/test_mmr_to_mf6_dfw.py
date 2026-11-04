import pathlib as pl
import shutil

import flopy
import numpy as np
import pytest
import xarray as xr

import pywatershed as pws
from pywatershed.utils.mmr_to_mf6_dfw import MmrToMf6Dfw

# Needs mf6 in PATH on mac/linux
mf6_bin_unavailable = shutil.which("mf6") is None

# Getting the gis files causes problems in parallel in CI. See regression test.

# See below to check if these answers are still up-to-date with
# mf6/autotest/test_chf_dfw.py
# Note the answers are from the binary FLW files in
# mf6/autotest/test_chf_dfw.py, if you switch to text files, the line should be
# flw_list = [
#     (int(binary), 100),
# ]  # one-based cell numbers here if binary, zero-based if text

answers_chf_dfw = {
    "ia": np.array([0, 2, 5, 7]),
    "ja": np.array([0, 1, 1, 0, 2, 2, 1], dtype=np.int32),
    "stage": np.array([[[[1.00196123, 1.00003366, 1.0]]]]),
    "flowja": np.array(
        [
            [
                [
                    5.49604806e-12,
                    -1.93841991e00,
                    2.45701848e-10,
                    1.93841991e00,
                    -2.55350330e-01,
                    0.00000000e00,
                    2.55350330e-01,
                ]
            ]
        ]
    ),
    "qstorage": np.array([[[-98.06158009, -1.68306958, 0.0]]]),
    "qflw": np.array([1.0, 1.0, 100.0]),
    "qchd": np.array([3.0, 1.0, -0.25535033]),
    "qresidual": 0.0,
}


@pytest.mark.skipif(mf6_bin_unavailable, reason="mf6 binary not available")
@pytest.mark.domainless
@pytest.mark.parametrize("binary_flw", [True, False])
def test_mmr_to_mf6_chf_dfw(tmp_path, binary_flw):
    # The point of this test is to reproduce the
    # modflow6/autotest/test_chf_dfw.py

    # Here we supply "seg_mid_elevation" in the parameter data and
    # "stress_period_data" in chd options. This bypasses the calculation of
    # these given PRMS parameter information. In this case, it would be
    # impossible to supply PRMS-like data that would reproduce this test.

    # This example shows how to formulate pywatershed inputs to match MF6
    # inputs unfortunately, these have to be translated through some of the
    # assumptions of PRMS to go through MMRToMF6DFW
    # One is that there are flows coming from potentially multiple HRUs to
    # stream segments, the other is about the units of volume/flow being in
    # cubicfeet.

    name = "chf-dfw01"
    output_dir = tmp_path / "test_chf_dfw01"
    save_flows = True
    print_flows = True

    control = pws.Control(
        start_time=np.datetime64("2023-01-01T00:00:00"),
        end_time=np.datetime64("2023-01-01T00:00:00"),
        time_step=np.timedelta64(1, "s"),
    )

    # ims
    ims_options = {
        "print_option": "all",
        "linear_acceleration": "BICGSTAB",
        "outer_dvclose": 1.0e-7,
        "inner_dvclose": 1.0e-8,
    }

    # disv1d
    dx = 1000.0
    nreach = 3

    # mannings coefficients are actually considered part of the dfw package
    params = pws.parameters.PrmsParameters(
        dims={
            "nsegment": nreach,
            "nhru": nreach,
        },
        coords={
            "seg_id": np.array(range(nreach)),  # todo: rename seg_id or such
        },
        data_vars={
            # "tosegment" no longer used
            "seg_length": np.ones(nreach) * dx,
            "mann_n": np.ones(nreach) * 0.035,
            # "seg_slope": np.ones(nreach) * 0.001,
            "seg_width": np.ones(nreach) * 50.0,
            "seg_mid_elevation": np.zeros(nreach),
            "hru_segment": np.array(range(nreach)) + 1,
        },
        metadata={
            "seg_id": {"dims": ["nsegment"]},
            "seg_length": {"dims": ["nsegment"]},
            "mann_n": {"dims": ["nsegment"]},
            "seg_width": {"dims": ["nsegment"]},
            "seg_mid_elevation": {"dims": ["nsegment"]},
            "hru_segment": {"dims": ["nsegment"]},
        },
        validate=True,
    )

    # vertices could also be supplied by shapefiles
    vertices = []
    vertices = [[j, j * dx, 0.0] for j in range(nreach + 1)]
    cell1d = []
    for j in range(nreach):
        cell1d.append([j, 0.5, 2, j, j + 1])
    nodes = len(cell1d)
    nvert = len(vertices)
    disv1d_options = {
        "nodes": nodes,
        "nvert": nvert,
        "vertices": vertices,
        "cell1d": cell1d,
    }

    # dfw
    dfw_options = {
        "print_flows": print_flows,
        "save_flows": save_flows,
        "idcxs": 0,  # zero based in flopy, None for hydraulically wide
    }

    # sto
    sto_options = {"save_flows": save_flows}

    # ic
    ic_options = {"strt": 1.0}

    # CXS
    # mf6 > 6.7 requires the cross section (feature) number ifno as the
    # first field of each crosssectiondata entry (modflow6 PR #2892);
    # zero-based here, as for packagedata below.
    ifno = [0, 0, 0, 0]
    xfraction = [0.0, 0.0, 1.0, 1.0]
    height = [100.0, 0.0, 0.0, 100.0]
    mannfraction = [1.0, 1.0, 1.0, 1.0]
    cxsdata = list(zip(ifno, xfraction, height, mannfraction))
    cxs_options = {
        "nsections": 1,
        "npoints": 4,
        "packagedata": [(0, 4)],
        "crosssectiondata": cxsdata,
    }

    # oc
    oc_options = {
        "saverecord": [
            ("STAGE", "ALL"),
            ("BUDGET", "ALL"),
            # ("QOUTFLOW", "ALL"),
        ],
        "printrecord": [
            ("STAGE", "LAST"),
            ("BUDGET", "ALL"),
            # ("QOUTFLOW", "ALL"),
        ],
    }

    # boundary conditions / FLW
    # if we use the volumes from pws rather than the inches from
    # PRMS, we dont have to supply 'hru_area' in the parameters for
    # depth to volume conversion.
    inflow_from_PRMS = False
    # this will aggregate to a single flw variable when mf6 writes to file
    bc_flows_combined = True
    # we have to create files with the boundary conditions in them
    inflow_list = ["sroff_vol", "ssres_flow_vol", "gwres_flow_vol"]
    inflow_dir = output_dir / "inflow_dir"
    inflow_dir.mkdir(parents=True)
    for inflw in inflow_list:
        if inflw == "sroff_vol":
            flw_vol = 3531.4666721489 * 86400  # 100.0 m3/s in ft^3/day
        else:
            flw_vol = 0.0

        # <
        time_coord_data = np.array([control.start_time]).astype(
            "datetime64[ns]"
        )

        _ = xr.Dataset(
            coords=dict(
                time=time_coord_data,
                nsegment=params.parameters["seg_id"],
            ),
            data_vars={
                f"{inflw}": (
                    ["time", "nsegment"],
                    np.array([[flw_vol, 0.0, 0.0]]),
                ),
            },
        ).to_netcdf(inflow_dir / f"{inflw}.nc")

    flw_options = {
        "print_input": True,
        "print_flows": True,
    }

    chd_options = {
        "maxbound": 1,
        "print_input": True,
        "print_flows": True,
        "stress_period_data": [(2, 1.0)],
    }

    dfw = MmrToMf6Dfw(
        control=control,
        params=params,
        tdis_perlen=1,  # does this matter
        tdis_nstp=1,
        output_dir=output_dir,
        bc_binary_files=binary_flw,
        sim_name=name,
        save_flows=save_flows,
        ims_options=ims_options,
        dfw_options=dfw_options,
        sto_options=sto_options,
        ic_options=ic_options,
        oc_options=oc_options,
        chd_options=chd_options,
        disv1d_options=disv1d_options,
        cxs_options=cxs_options,
        inflow_from_PRMS=inflow_from_PRMS,
        bc_flows_combined=bc_flows_combined,
        inflow_dir=inflow_dir,
        flw_options=flw_options,
    )

    dfw.write()
    success, buff = dfw.run(silent=False, report=True)

    # ======================
    # Checks
    assert success

    # one can verify the answers match the current mf6 results for test_chf_dfw
    # by placing the path to its output here. Run the following lines to
    # generate the mf6 results:
    #   cd modflow6_for_pws_ci/autotest  # or your mf6 repo location
    #   pytest -s -vv test_chf_dfw.py --keep=keepers
    # output_dir = pl.Path(
    #     "../../modflow6_for_pws_ci/autotest/keepers/"
    #     "test_mf6model[0-chf-dfw01]0"
    # )

    # check binary grid file
    grb = flopy.mf6.utils.MfGrdFile(output_dir / f"{name}.disv1d.grb")
    ia = grb.ia
    ja = grb.ja
    assert (answers_chf_dfw["ia"] == ia).all()
    assert (answers_chf_dfw["ja"] == ja).all()
    assert ia.shape[0] == grb.nodes + 1, "ia in grb file is not correct size"

    # check stage file
    qobj = flopy.utils.HeadFile(
        output_dir / f"{name}.stage", precision="double", text="STAGE"
    )
    stage = qobj.get_alldata()
    assert ((answers_chf_dfw["stage"] - stage) < 1.0e-7).all()

    # read the budget file
    budobj = flopy.utils.binaryfile.CellBudgetFile(output_dir / f"{name}.bud")
    flowja = budobj.get_data(text="FLOW-JA-FACE")[0]
    qstorage = budobj.get_data(text="STORAGE")[0]
    qflw = np.array(budobj.get_data(text="FLW")[0].tolist()[0])
    qchd = np.array(budobj.get_data(text="CHD")[0].tolist()[0])
    qresidual = np.zeros(grb.nodes)[0]

    assert (answers_chf_dfw["flowja"] - flowja < 1.0e-7).all()
    assert (answers_chf_dfw["qstorage"] - qstorage < 1.0e-7).all()
    assert (answers_chf_dfw["qflw"] - qflw < 1.0e-7).all()
    assert (answers_chf_dfw["qchd"] - qchd < 1.0e-7).all()
    assert (answers_chf_dfw["qresidual"] - qresidual < 1.0e-7).all()


# <
answers_regression_means = {
    "stage_all": 1.0372047024253908,
    "flow_all": 44.70210250910929,
}


@pytest.mark.skipif(mf6_bin_unavailable, reason="mf6 binary not available")
@pytest.mark.domain
def test_mmr_to_mf6_dfw_regression(simulation, tmp_path):
    # this test is based on the notebook examples/mmr_to_mf6_dfw.ipynb
    if simulation["name"] != "drb_2yr:nhm":
        pytest.skip("test_mmr_to_mf6_dfw_regression only runs for drb_2yr_nhm")

    domain_name = simulation["name"].split(":")[0]

    # Fail if segment shp file not present
    gis_dir = pws.utils.gis_files.gis_dir
    seg_shp_file = gis_dir / f"{domain_name}/Segments_subset.shp"
    msg = (
        f"GIS data and segment shapefile missing for {domain_name}. Run "
        "`python pywatershed/pywatershed/utils/gis_files.py` to install these"
    )
    assert seg_shp_file.exists(), msg

    test_data_dir = pl.Path("../test_data")

    domain_name = simulation["name"].split(":")[0]
    domain_dir = test_data_dir / f"{domain_name}"
    run_dir = tmp_path
    inflow_dir = domain_dir / "output"

    control_file = domain_dir / "nhm.control"
    control = pws.Control.load_prms(control_file, warn_unused_options=False)
    ndays_run = 18
    # subtract one becase end day/time is included in the PRMS run
    control.edit_end_time(
        control.start_time + ((ndays_run - 1) * control.time_step)
    )

    dis_both_file = domain_dir / "parameters_dis_both.nc"
    dis_both = pws.Parameters.from_netcdf(dis_both_file)
    seg_params = pws.Parameters.from_netcdf(
        domain_dir / "parameters_PRMSChannel.nc"
    )
    params = pws.Parameters.merge(dis_both, seg_params)

    # IMS options
    nouter, ninner = 100, 50
    hclose, rclose, relax = 1e-5, 1.0, 0.97

    ims_options = {
        "print_option": "SUMMARY",
        "outer_dvclose": hclose,
        "outer_maximum": nouter,
        "under_relaxation": "DBD",
        "under_relaxation_theta": 0.95,
        "under_relaxation_kappa": 0.0001,
        "under_relaxation_gamma": 0.0,
        "under_relaxation_momentum": 0.0,
        "inner_maximum": ninner,
        "inner_dvclose": hclose,
        "linear_acceleration": "BICGSTAB",
        "scaling_method": "NONE",
        "reordering_method": "NONE",
        "relaxation_factor": relax,
        "filename": "drb.ims",
    }

    dfw_options = {
        "print_flows": True,
        "save_flows": True,
        "idcxs": None,  # zero based in flopy, None for hydraulically wide
    }

    sto_options = {"save_flows": True}

    oc_options = {
        "saverecord": [
            ("STAGE", "ALL"),
            ("BUDGET", "ALL"),
        ],
        "printrecord": [
            ("STAGE", "LAST"),
            ("BUDGET", "ALL"),
        ],
    }

    # Initial water depth, units of meters
    ic_options = {"strt": 0.5}

    chd_options = {
        "print_input": True,
        "print_flows": False,
    }

    bc_binary_files = True
    bc_flows_combined = True

    tdis_perlen = 2 * 3600  # stress period length
    tdis_nstp = 3  # substeps per stress period

    run_dir.mkdir(parents=True, exist_ok=True)

    dfw = MmrToMf6Dfw(
        control=control,
        segment_shp_file=seg_shp_file,
        params=params,
        tdis_perlen=tdis_perlen,
        tdis_nstp=tdis_nstp,
        output_dir=run_dir,
        sim_name="drb_dfw",
        ims_options=ims_options,
        dfw_options=dfw_options,
        sto_options=sto_options,
        oc_options=oc_options,
        ic_options=ic_options,
        chd_options=chd_options,
        bc_binary_files=bc_binary_files,
        bc_flows_combined=bc_flows_combined,
        inflow_dir=inflow_dir,
    )

    dfw.write()
    success, buff = dfw.run(silent=False, report=True)
    assert success

    sim = flopy.mf6.MFSimulation.load(
        "drb_mf6_dfw",
        sim_ws=str(run_dir),
        exe_name="mf6",
        load_only=["disv1d"],
    )
    sim.model_names
    model = sim.get_model("drb_dfw")

    # time
    tdis = sim.tdis
    sim_start_time = np.datetime64(
        tdis.start_date_time.get_data().upper()[0:19]
    )
    n_substeps = int((ndays_run) * 24 * 60 * 60 / tdis_perlen * tdis_nstp)
    substep_len = np.timedelta64(int(tdis_perlen / tdis_nstp), "s")
    sim_end_time = sim_start_time + n_substeps * substep_len
    sim_times = np.arange(sim_start_time, sim_end_time, substep_len)
    perioddata = tdis.perioddata.get_data()
    assert len(sim_times) == len(perioddata) * perioddata[0][1]

    # stage
    stage_file = run_dir / "drb_dfw.stage"
    sobj = flopy.utils.HeadFile(stage_file, text="STAGE", verbose=False)
    stage_all = sobj.get_alldata().squeeze()
    disv1d = model.get_package("disv1d")
    bottom_ele = disv1d.bottom.get_data()
    stage_all = np.maximum(stage_all - bottom_ele, 0)

    # getting flow is more complicated and could be improved/refined
    budget_file = run_dir / "drb_dfw.bud"
    budobj = flopy.utils.binaryfile.CellBudgetFile(budget_file)
    flowja = budobj.get_data(text="FLOW-JA-FACE")
    # qstorage = budobj.get_data(text="STORAGE")
    # qflw = budobj.get_data(text="FLW")
    # qextoutflow = budobj.get_data(text="EXT-OUTFLOW")

    grb_file = run_dir / "drb_dfw.disv1d.grb"
    grb = flopy.mf6.utils.MfGrdFile(grb_file)
    ia = grb.ia
    ja = grb.ja

    def get_outflow(itime):
        outflow = np.zeros(ia.shape[0] - 1)
        flowjaflat = flowja[itime].flatten()
        for n in range(grb.nodes):
            for ipos in range(ia[n] + 1, ia[n + 1]):
                q = flowjaflat[ipos]
                if q < 0:
                    outflow[n] += -q
        # <<<
        return outflow

    flow_all = stage_all.copy() * np.nan
    for tt in range(flow_all.shape[0]):
        flow_all[tt, :] = get_outflow(tt)

    for kk, vv in answers_regression_means.items():
        # Relative: mf6 nightlies on linux/windows drift from these
        # (mac-generated) means by ~4e-6 relative (Nov 2026).
        rel_diff = abs(locals()[kk].mean() - vv) / abs(vv)

        assert rel_diff < 1e-5, f"results for {kk} are not close"
