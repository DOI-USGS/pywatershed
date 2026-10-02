"""Write per-process and discretization parameter netCDF files for domains.

This is the one place the ``parameters_<Process>.nc`` and
``parameters_dis_*.nc`` files in ``test_data/<domain>/`` come from. It
wraps :func:`pywatershed.utils.separate_domain_params_dis_to_ncdf`. For
every ``*.control`` in a domain whose ``cascade_flag`` is set, the cascade
parameters are derived first and the cascade process classes are written
from that.

Usage (from test_data/generate/):
    python separate_params_to_nc.py sagehen_5yr sagehen_gridded_5yr
    python separate_params_to_nc.py --cascades_only sagehen_5yr

With ``--cascades_only`` the plain (non-cascade) pass is skipped and only
the cascade process files are (re)written.
"""

import argparse
import pathlib as pl

import pywatershed as pws
from pywatershed.utils import separate_domain_params_dis_to_ncdf

cascade_processes = [
    pws.PRMSRunoffCascadesNoDprst,
    pws.PRMSSoilzoneCascadesNoDprst,
]
# only derivable from a control with cascadegw_flag set
gw_cascade_processes = [pws.PRMSGroundwaterCascadesNoDprst]


def separate_domain(domain_dir: pl.Path, cascades_only: bool) -> None:
    param_file = domain_dir / "myparam.param"
    if not cascades_only:
        separate_domain_params_dis_to_ncdf(
            prms_param_file=param_file,
            domain_name=None,
            out_dir=domain_dir,
        )

    # one control per cascade process class suffices, the parameters are
    # the same; the groundwater cascade class needs a control with
    # cascadegw_flag set
    written = set()
    for control_file in sorted(domain_dir.glob("*.control")):
        control = pws.Control.load_prms(
            control_file, warn_unused_options=False
        )
        if not control.options.get("cascade_flag", 0):
            continue
        process_list = list(cascade_processes)
        if control.options.get("cascadegw_flag", 0):
            process_list += gw_cascade_processes
        process_list = [pp for pp in process_list if pp not in written]
        if not len(process_list):
            continue
        print(f"cascade parameters from {control_file.name}")
        separate_domain_params_dis_to_ncdf(
            prms_param_file=param_file,
            domain_name=None,
            out_dir=domain_dir,
            process_list=process_list,
            control=control,
            write_dis=False,
        )
        written.update(process_list)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("domains", nargs="+", help="domain names")
    parser.add_argument("--cascades_only", action="store_true")
    args = parser.parse_args()

    test_data_dir = pl.Path(__file__).parent.parent
    for domain_name in args.domains:
        domain_dir = test_data_dir / domain_name
        assert domain_dir.exists(), domain_dir
        separate_domain(domain_dir, args.cascades_only)
