import os
import pathlib as pl
import warnings
from fnmatch import fnmatch
from typing import List
from warnings import warn

import pytest

import pywatershed as pws

# Note: Either PRMS or GSFLOW exectuables may be executed. The choice is
# triggered by the control file field executable_desc. If "gsflow" is
# found in the lower case version of its value, then gsflow is used. Otherwise,
# PRMS is used.

test_data_dir = pl.Path("..")

# Subset this to speed up tests by eliminating some domains
all_domain_dirs = sorted(
    [path for path in test_data_dir.iterdir() if path.is_dir()]
)

# This would change to handle other/additional schedulers
domain_globs_schedule = ["*conus*"]

final_var_names = ["through_rain", "infil", "seg_lateral_inflow"]


def get_ctl_exe_desc(ctl_file):
    import warnings

    with warnings.catch_warnings():
        # This is the only way to silence "invalid" options.
        warnings.simplefilter("ignore")
        ctl = pws.Control.load_prms(
            ctl_file,
            keep_unused_options=True,
            warn_unused_options=False,
        )

    if "executable_desc" in ctl.options.keys():
        exe_desc = ctl.options["executable_desc"][0].lower()
    else:
        exe_desc = "prms"

    return exe_desc


def pytest_addoption(parser):
    parser.addoption(
        "--force",
        required=False,
        action="store_true",
        default=False,
        help=("Force run without scheduler present"),
    )

    parser.addoption(
        "--domain",
        required=False,
        action="append",
        default=[],
        help=(
            "Domain(s) to run (name of domain dir and NOT path to it). "
            "You can pass multiples of this argument. If not used, "
            "all self-contained domains in test_data/ will be run."
        ),
    )

    parser.addoption(
        "--control_pattern",
        required=False,
        action="append",
        default=[],
        help=(
            "Control file glob(s) (NOT path. Can drop '.control'). "
            "You can pass multiples of this argument. If not used, "
            "all control files in each domain will be run."
        ),
    )

    parser.addoption(
        "--write_log",
        help=(
            "Write the PRMS/GSFLOW stdout to log matching control file name."
        ),
        action="store_true",
    )

    parser.addoption(
        "--suppress-control-warnings",
        required=False,
        action="store_true",
        default=False,
        help=(
            "Suppress UserWarnings about unrecognized control options "
            "(e.g., 'executable_model', 'model_mode')"
        ),
    )

    parser.addoption(
        "--exe",
        required=False,
        default=None,
        help=("Path to PRMS or GSFLOW executable to use"),
    )


def pytest_configure(config):
    """Configure pytest with warning filters based on command line options."""
    if config.getoption("suppress_control_warnings"):
        warnings.filterwarnings(
            "ignore",
            message=r".*is not an available control option",
            category=UserWarning,
        )


@pytest.fixture(scope="function")
def exe(simulation, request):
    # Check if exe was provided on command line
    exe_path = request.config.getoption("exe")
    if exe_path:
        exe_pth = pl.Path(exe_path).resolve()
        if not exe_pth.exists():
            pytest.fail(f"Executable not found: {exe_pth}")
        return exe_pth

    exe_desc = get_ctl_exe_desc(simulation["control_file"])
    # PRMS binaries are compiled from prms_src/ on first use; the GSFLOW
    # binaries are checked in to bin/ because their source is not in this
    # repository.
    return pws.utils.get_or_compile_prms_exe(exe_desc)


def scheduler_active():
    slurm = os.getenv("SLURM_JOB_ID") is not None
    pbs = os.getenv("PBS_JOBID") is not None
    # add more scheduler checks
    return slurm or pbs


def enforce_scheduler(test_dir):
    """Enforce the use of scheduler

    Args:
        test_dir: the domain directory to run or schedule

    Return:
        True if must use a scheduler, False if not
    """

    if scheduler_active():
        return False
    glob_match = list(
        fnmatch(str(test_dir), gg) for gg in domain_globs_schedule
    )
    if any(glob_match):
        msg = (
            f"Skipping domain '{test_dir}' which must be scheduled or use "
            "--force to override skip"
        )
        warn(msg, UserWarning)
        return True

    return False


def collect_simulations(
    domain_list: list,
    control_pattern_list,
    force: bool = True,
    verbose: bool = False,
    write_log: bool = False,
):
    simulations = {}
    for dom_dir in all_domain_dirs:
        if dom_dir.name not in domain_list:
            continue

        # ensure this is a self-contained run (all files in repo)
        # domains with a make_cbh_only control generate their own CBH
        # forcing files with PRMS and are also self-contained
        if not (
            (dom_dir / "prcp.cbh").exists()
            or (dom_dir / "prcp.day").exists()
            or (dom_dir / "precip.cbh").exists()
            or (dom_dir / "precip.day").exists()
            or len(list(dom_dir.glob("*make_cbh_only*.control")))
        ):
            # this is kind of a silly check... until something better needed
            warn(f"prcp/precip.cbh/day not found in {dom_dir}, skipping")
            continue

        # filter selected domains
        if len(domain_list) and (dom_dir.name not in domain_list):
            continue

        # optionally enforce scheduler
        if not force:
            skip = enforce_scheduler(dom_dir)
            if skip:
                continue

        control_file_candidates = sorted(dom_dir.glob("*.control"))

        # filter against control pattern
        control_files = []
        for control in control_file_candidates:
            if not len(control_pattern_list):
                control_files += [control]
            else:
                for gg in control_pattern_list:
                    if gg in control.name:
                        control_files += [control]

        for control in control_files:
            id = f"{dom_dir.name}:{control.with_suffix('').name}"
            ctl = pws.Control.load_prms(control, warn_unused_options=False)
            output_dir = dom_dir / ctl.options["netcdf_output_dir"]
            simulations[id] = {
                "ws": dom_dir,
                "control_file": control,
                "output_dir": output_dir,
                "write_log": write_log,
            }

    # make sure all requested domains were found
    if len(domain_list) and (len(simulations) < len(domain_list)):
        requested = set(domain_list)
        found = [pl.Path(dd).name for dd in simulations.keys()]
        requested_not_found = requested.difference(found)
        msg = (
            f"The following requested domains were not found: "
            f"{requested_not_found}"
        )
        pytest.exit(msg)

    if verbose:
        print("\nrun_domains.py found the following domains to run:\n")
        print(f"{list(simulations.keys())}")

    return simulations


def collect_csv_files(simulations: dict) -> List[tuple]:
    csv_files = []
    for key, value in simulations.items():
        control = value["control_file"]
        csv_files_dom = sorted(value["output_dir"].glob("*.csv"))
        csv_files += [
            (control, ff) for ff in csv_files_dom if ff.name != "stats.csv"
        ]
    return csv_files


def pytest_generate_tests(metafunc):
    domain_list = metafunc.config.getoption("domain")
    control_pattern_list = metafunc.config.getoption("control_pattern")
    force = metafunc.config.getoption("force")
    write_log = metafunc.config.getoption("write_log")

    simulations = collect_simulations(
        domain_list, control_pattern_list, force=force, write_log=write_log
    )
    control_csv_files = collect_csv_files(simulations)

    if "control_csv_file" in metafunc.fixturenames:
        ids = [
            ff.parent.parent.name
            + ":"
            + cc.with_suffix("").name
            + ":"
            + ff.name
            for cc, ff in control_csv_files
        ]
        metafunc.parametrize("control_csv_file", control_csv_files, ids=ids)

    if "control_soltab_file" in metafunc.fixturenames:
        # The single/combined soltab output file was from PRMS 5.2.1. In that
        # case this parameterization doesnt really make sense because it's
        # always the same, but this is useful because it also identifies the
        # ws/domain_dir. The processing of this parameter returns 3 individual
        # output netcdf files: soltab_potsw, soltab_horad_potsw, soltab_sunhrs.
        # In PRMS 5.3+ the soltab file was split into 4 output files:
        # soltab_sunhrs.csv, soltab_potsw.csv, obliquity.csv, and
        # solar_declination.csv. I've modified GSFLOW 2.4.1 to also output
        #  soltab_horad_potsw.csv. Sadly, the output format does not match
        # PRMS output files, there is no date column, presumably because it
        # has a doy dimension 1-366.
        control_soltab_files = []
        for kk, vv in simulations.items():
            exe_desc = get_ctl_exe_desc(vv["control_file"])
            if "5.2.1.1" in exe_desc:
                soltab_name = "soltab_debug_5.2.1.1"
            else:
                soltab_name = "soltab_debug"

            control_soltab_files += [
                (vv["control_file"], vv["ws"] / soltab_name)
            ]

        # <
        ids = [
            ff.parent.name + ":" + cc.with_suffix("").name + ":" + ff.name
            for cc, ff in control_soltab_files
        ]

        metafunc.parametrize(
            "control_soltab_file",
            control_soltab_files,
            ids=ids,
            scope="session",
        )

    if "simulation" in metafunc.fixturenames:
        metafunc.parametrize(
            "simulation",
            list(simulations.values()),
            ids=list(simulations.keys()),
            scope="session",
        )

    if "control_final_file" in metafunc.fixturenames:
        control_final_files = [
            (vv["control_file"], vv["output_dir"] / var)
            for kk, vv in simulations.items()
            for var in final_var_names
        ]
        ids = [
            ff.parent.parent.name
            + ":"
            + cc.with_suffix("").name
            + ":"
            + ff.name
            for cc, ff in control_final_files
        ]

        # these are not really file names they are domain/key_var
        metafunc.parametrize(
            "control_final_file", control_final_files, ids=ids, scope="session"
        )
