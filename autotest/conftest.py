import pathlib as pl
from warnings import warn

import pywatershed as pws

# Domain inputs and answers are populated here from a pws-prms-oracle
# answer set by test_data/generate/oracle_populate.py (links to the
# oracle's files, derived variables written alongside, key markers).
test_data_dir = pl.Path(__file__).parent.parent / "test_data"
all_domain_dirs = sorted(
    [path for path in test_data_dir.iterdir() if path.is_dir()]
)


def _read_marker(path: pl.Path) -> str | None:
    return path.read_text().strip() if path.exists() else None


def check_answers(cid: str, output_dir: pl.Path) -> None:
    """The answers for a simulation are populated (.derived_okey written
    by oracle_populate.py after deriving) and were derived from the
    answers currently linked (.oracle_okey, copied from the oracle)."""
    populate = (
        "run test_data/generate/oracle_populate.py "
        f"--domain={cid.split(':')[0]} --control_pattern={cid.split(':')[1]}"
    )
    if not output_dir.is_dir():
        raise ValueError(f"{cid}: no answers at {output_dir}; {populate}")
    oracle_key = _read_marker(output_dir / ".oracle_okey")
    derived_key = _read_marker(output_dir / ".derived_okey")
    if derived_key is None:
        raise ValueError(f"{cid}: derived variables missing; {populate}")
    if oracle_key is None:
        warn(f"{cid}: answers carry no .oracle_okey (unkeyed oracle run)")
    elif oracle_key != derived_key:
        raise ValueError(
            f"{cid}: derived variables are from answers {derived_key}, "
            f"the linked answers are {oracle_key}; {populate}"
        )


def pytest_addoption(parser):
    parser.addoption(
        "--domain",
        required=False,
        action="append",
        default=[],
        help=(
            "Domain(s) to run (name of domain dir and NOT path to it). "
            "You can pass multiples of this argument. Required for tests "
            "using the simulation fixture (unless --all_domains is used)."
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
        "--print_ans",
        required=False,
        action="store_true",
        default=False,
        help=("Print results and assert False for all domain tests"),
    )

    parser.addoption(
        "--all_domains",
        required=False,
        action="store_true",
        default=False,
        help=(
            "Run all test domains hru1, drb_2yr, ucb_2yr. This option can"
            "be expanded but is hardwired."
        ),
    )


def collect_simulations(
    domain_list: list,
    control_pattern_list,
    verbose: bool = False,
):
    simulations = {}
    for dom_dir in all_domain_dirs:
        if dom_dir.name not in domain_list:
            continue

        if not (
            (dom_dir / "prcp.cbh").exists()
            or (dom_dir / "prcp.day").exists()
            or (dom_dir / "prcp.nc").exists()
            or (dom_dir / "precip.cbh").exists()
            or (dom_dir / "precip.day").exists()
        ):
            # this is kind of a silly check... until something better needed
            warn(f"prcp/precip.cbh/day not found in {dom_dir}, skipping")
            continue

        # filter selected domains
        if len(domain_list) and (dom_dir.name not in domain_list):
            continue

        control_file_candidates = sorted(dom_dir.glob("*.control"))
        # a control that only generates input data for the other controls
        # (the CBH forcing files of sagehen_gridded_5yr) is not a
        # simulation to test
        control_file_candidates = [
            cc
            for cc in control_file_candidates
            if "make_cbh_only" not in cc.name
        ]

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
            cid = f"{dom_dir.name}:{control.with_suffix('').name}"
            ctl = pws.Control.load_prms(control, warn_unused_options=False)
            output_dir = dom_dir / ctl.options["netcdf_output_dir"]
            check_answers(cid, output_dir)
            simulations[cid] = {
                "name": cid,
                "dir": dom_dir,
                "control_file": control,
                "output_dir": output_dir,
            }

    return simulations


def pytest_generate_tests(metafunc):
    domain_list = metafunc.config.getoption("domain")
    all_domains_option = metafunc.config.getoption("all_domains")
    # reconcile the above
    if len(domain_list) and all_domains_option:
        raise ValueError("--domain arguments conflicts with --all_domains.")
    if all_domains_option:
        # This is somewhat arbitrary
        domain_list = ["hru_1", "drb_2yr", "ucb_2yr"]

    control_pattern_list = metafunc.config.getoption("control_pattern")

    if "domainless" == metafunc.config.option.markexpr:
        return

    if "simulation" in metafunc.fixturenames:
        if not len(domain_list):
            raise ValueError(
                "Tests using the simulation fixture require --domain "
                "(may be repeated) or --all_domains."
            )
        simulations = collect_simulations(domain_list, control_pattern_list)

        # Put --print_ans in the domain fixture as it applies only to the
        # domain tests. It is a run time attribute, not actually an attribute
        # of the domain however, so the "domain" fixture is more like a
        # "domain_test" fixture (domain + runtime test options)
        # Not sure I love this, maybe have a domain_opts fixture later?
        print_ans = metafunc.config.getoption("print_ans")
        for sk, sv in simulations.items():
            sv["print_ans"] = print_ans

        if not len(simulations):
            raise ValueError("No simulations found, check your input.")

        metafunc.parametrize(
            "simulation", simulations.values(), ids=simulations.keys()
        )

    return
