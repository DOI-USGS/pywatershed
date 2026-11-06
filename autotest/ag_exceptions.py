"""HRU-time exceptions for the agricultural (ag) parity tests.

Single source of truth for the domain HRU-time exceptions (previously
duplicated in test_prms_soilzone_ag.py and
test_prms_runoff_soilzone_ag.py), with validation so entries can never
go silently inert.

Exception keys are simulation names, "{domain_dir}:{control_stem}"
(see conftest.collect_simulations). Every key must refer either to a
domain directory present in test_data/ (in which case the control stem
must name an actual .control file there and HRU indices must be in
range) or to a domain listed in OFFLINE_DOMAINS. Anything else is a
hard error at import (= pytest collection) time: a renamed or typo'd
key would otherwise simply never fire, silently disabling the
carve-outs it encodes.
"""

import pathlib as pl
import warnings

import xarray as xr

TEST_DATA_DIR = pl.Path(__file__).parent.parent / "test_data"

# Domains the exceptions may reference that are NOT shipped in
# test_data/: the original, larger UCB agricultural domains the ag
# tests were developed on. Too large for CI; they may return. Keys
# referencing them are retained but cannot fire until the domain
# directories exist locally.
OFFLINE_DOMAINS = {
    "ucb_ag_spinup_2yr",
    "ucb_ag_analysis_2yr",
}

# Domain-specific HRU-time exceptions for threshold-crossing divergences
# When accumulated floating-point errors cause Python (double precision) and
# Fortran (single precision) to cross thresholds differently, affected
# variables diverge. At these points, Python values are replaced with Fortran
# values before comparison to allow the test to continue.
#
# Format: {simulation_name:
#     {hru_index: (start_timestep, affected_vars, reason)}
# }
DOMAIN_HRU_TIME_EXCEPTIONS = {
    "ucb_ag_analysis_2yr:nhm_dynamic_2000_2020_w_output_subset_no_restart": {
        # HRU 1279, timestep 472:
        #   - ag_soil_moist = 2.000002 (Fortran) vs 1.9999997 (Python)
        #   - pcts = 0.5000004 (Fortran) vs 0.49999991 (Python)
        #   - Crosses 0.5 threshold for LOAM soil potet_lower reduction
        #   - Fortran: pcts >= 0.5, no reduction, ag_potet_lower = 0.05
        #   - Python: pcts < 0.5, reduction applied, ag_potet_lower = 0.025
        #   - This is a floating-point precision boundary issue
        1279: (
            472,
            ["ag_potet_lower"],
            "pcts 0.5 threshold crossing for LOAM soil ag_potet_lower",
        ),
    },
    "ucb_ag_spinup_2yr:nhm_ic_w_output_subset": {
        # HRU 3642, timestep 40:
        #   - snow_free = 1.0 - snowcov_area = 0.010000000000000009 in Python
        #     vs ~0.00999999 in Fortran (single precision)
        #   - Crosses 0.01 threshold for snow_free ET calculation
        #   - Python: snow_free >= 0.01, et_type = EVAP_ONLY, ET computed
        #   - Fortran: snow_free < 0.01, et_type = ET_DEFAULT, et = 0
        #   - This affects downstream ET calculations and soil moisture
        #     accounting
        3642: (
            40,
            [
                "potet_lower",
                "potet_rechr",
                "soil_moist",
                "soil_rechr",
                "perv_actet",
                "hru_actet",
                "unused_potet",
                "soil_moist_tot",
                "soil_rechr_change",
            ],
            "snow_free 0.01 threshold crossing for ET calculation",
        ),
    },
}


def _validate_exceptions() -> None:
    """Fail at import (= collection) time on unmatchable keys."""
    for key, entries in DOMAIN_HRU_TIME_EXCEPTIONS.items():
        domain, _, control = key.partition(":")
        if not control:
            raise ValueError(
                f"exception key '{key}' is not 'domain:control_stem'"
            )
        domain_dir = TEST_DATA_DIR / domain
        if not domain_dir.is_dir():
            if domain not in OFFLINE_DOMAINS:
                raise ValueError(
                    f"exception key '{key}': domain '{domain}' is neither "
                    f"in test_data/ nor declared in OFFLINE_DOMAINS -- "
                    f"a renamed or typo'd key would never fire"
                )
            continue  # offline: nothing more is checkable
        if not (domain_dir / f"{control}.control").exists():
            raise ValueError(
                f"exception key '{key}': no {control}.control in "
                f"{domain_dir} -- this entry would never fire"
            )
        with xr.open_dataset(domain_dir / "parameters_dis_hru.nc") as ds:
            nhru = ds.sizes["nhru"]
        bad = [hh for hh in entries if hh >= nhru]
        if bad:
            raise ValueError(
                f"exception key '{key}': HRU indices {bad} out of range "
                f"(nhru = {nhru})"
            )


_validate_exceptions()


def get_hru_time_exceptions(simulation: dict) -> dict:
    """Exceptions for one collected simulation; loud when they apply.

    Returns {hru_index: (start_timestep, affected_vars, reason)} for
    this simulation ({} for most). When entries apply, a UserWarning
    puts them in the pytest warnings summary: silence means the
    comparison ran clean, with no Fortran-value substitutions.
    """
    entries = DOMAIN_HRU_TIME_EXCEPTIONS.get(simulation["name"], {})
    if entries:
        reasons = "; ".join(vv[2] for vv in entries.values())
        warnings.warn(
            f"Applying {len(entries)} HRU-time exception(s) for "
            f"{simulation['name']}: {reasons}",
            UserWarning,
        )
    return entries
