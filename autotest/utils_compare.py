import pathlib as pl
from types import MappingProxyType
from typing import Union

import numpy as np
import xarray as xr

import pywatershed as pws


def assert_dicts_equal(dic1, dic2):
    assert set(dic1.keys()) == set(dic2.keys())

    # add cases as needed
    for kk, vv in dic1.items():
        if isinstance(vv, (dict, MappingProxyType)):
            assert_dicts_equal(dic1[kk], dic2[kk])
        elif isinstance(vv, np.ndarray):
            np.testing.assert_equal(dic1[kk], dic2[kk])
        else:
            if (
                isinstance(vv, float)
                and np.isnan(dic1[kk])
                and np.isnan(dic2[kk])
            ):
                continue
            assert dic1[kk] == dic2[kk]


def assert_allclose(
    actual: np.ndarray,
    desired: np.ndarray,
    rtol: float = 1.0e-15,
    atol: float = 1.0e-15,
    equal_nan: bool = True,
    strict: bool = False,
    also_check_w_np: bool = True,
    print_max_errs: bool = False,
    var_name: str = "",
    sentinel_to_nan: Union[float, list, None] = None,
):
    """Reinvent np.testing.assert_allclose to get useful diagnostincs in debug

    Args:
        actual: Array obtained.
        desired: Array desired.
        rtol: Relative tolerance.
        atol: Absolute tolerance.
        equal_nan: If True, NaNs will compare equal.
        strict: If True, raise an ``AssertionError`` when either the shape or
            the data type of the arguments does not match. The special
            handling of scalars mentioned in the Notes section is disabled.
        also_check_w_np: first check using np.testing.assert_allclose using
            the same options.
        print_max_errs: bool=False. Print max abs and rel err for each var.
        sentinel_to_nan: Optional sentinel value(s) to convert to NaN before
            comparison. Can be a single float or a list of floats. Values
            less than or equal to the sentinel will be converted to NaN.
            This is useful for comparing outputs where one system uses
            sentinel values (e.g., -99.9) and another uses NaN.
    """

    # Convert sentinel values to NaN in both arrays
    if sentinel_to_nan is not None:
        actual = actual.copy()
        desired = desired.copy()
        if isinstance(sentinel_to_nan, (int, float)):
            sentinel_to_nan = [sentinel_to_nan]
        for sentinel in sentinel_to_nan:
            actual = np.where(actual <= sentinel, np.nan, actual)
            desired = np.where(desired <= sentinel, np.nan, desired)

    if also_check_w_np:
        np.testing.assert_allclose(
            actual,
            desired,
            rtol=rtol,
            atol=atol,
            equal_nan=equal_nan,
            # strict=strict,  # to add to newer versions of numpy
        )

    if strict:
        assert actual.shape == desired.shape
        assert isinstance(actual, type(desired))
        assert isinstance(desired, type(actual))

    if equal_nan:
        actual_nan = np.where(np.isnan(actual), True, False)
        desired_nan = np.where(np.isnan(desired), True, False)
        assert (actual_nan == desired_nan).all()
        if len(actual_nan) == len(actual):
            return

    abs_diff = abs(actual - desired)
    with np.errstate(divide="ignore", invalid="ignore"):
        rel_abs_diff = abs_diff / desired

    abs_close = abs_diff < atol
    rel_close = rel_abs_diff < rtol
    rel_close = np.where(np.isnan(rel_close), False, rel_close)

    close = abs_close | rel_close

    if print_max_errs:
        sp = "" if len(var_name) == 0 else " "
        print(f"{var_name}{sp}max abs err: {abs_diff.max()}")
        print(f"{var_name}{sp}max rel err: {rel_abs_diff.max()}")

    assert close.all()
    return


def compare_in_memory(
    process: pws.base.Process,
    answers: dict[pws.base.adapter.AdapterNetcdf],
    rtol: float = 1.0e-15,
    atol: float = 1.0e-15,
    mask_dict: np.array = None,
    equal_nan: bool = True,
    strict: bool = False,
    also_check_w_np: bool = True,
    skip_missing_ans: bool = False,
    fail_after_all_vars: bool = True,
    verbose: bool = False,
    var_tolerances: dict = None,
    var_sentinel_to_nan: dict = None,
):
    # TODO: docstring

    fail_list = []
    skip_list = []
    for var in process.get_variables():
        if var not in answers.keys():
            if skip_missing_ans:
                skip_list += [var]
                continue
            else:
                msg = f"Variable '{var}' not found in the answers provided."
                raise KeyError(msg)

        if verbose:
            print(f"checking {var}")

        if not isinstance(answers[var], np.ndarray):
            answers[var].advance()

        if isinstance(process[var], pws.base.timeseries.TimeseriesArray):
            actual = process[var].current
        else:
            actual = process[var]

        if isinstance(answers[var], pws.base.adapter.Adapter):
            desired = answers[var].current.data
        else:
            desired = answers[var]

        if mask_dict is not None:
            actual = actual[mask_dict[var]]
            desired = np.array(desired)[mask_dict[var]]
        elif hasattr(process, "_active_hru_mask") and (
            np.shape(actual) == np.shape(process._active_hru_mask)
        ):
            # compare only at active HRUs; inactive HRUs are masked to
            # nan by pywatershed but generally not by PRMS output.
            actual = actual[process._active_hru_mask]
            desired = np.array(desired)[process._active_hru_mask]

        # Get variable-specific tolerances if provided
        var_rtol = rtol
        var_atol = atol
        if var_tolerances is not None and var in var_tolerances:
            var_rtol = var_tolerances[var].get("rtol", rtol)
            var_atol = var_tolerances[var].get("atol", atol)

        # Get variable-specific sentinel value to convert to NaN
        sentinel = None
        if var_sentinel_to_nan is not None and var in var_sentinel_to_nan:
            sentinel = var_sentinel_to_nan[var]

        if not fail_after_all_vars:
            assert_allclose(
                actual,
                desired,
                atol=var_atol,
                rtol=var_rtol,
                equal_nan=equal_nan,
                strict=strict,
                also_check_w_np=also_check_w_np,
                var_name=var,
                sentinel_to_nan=sentinel,
            )

        else:
            try:
                assert_allclose(
                    actual,
                    desired,
                    atol=var_atol,
                    rtol=var_rtol,
                    equal_nan=equal_nan,
                    strict=strict,
                    also_check_w_np=also_check_w_np,
                    var_name=var,
                    sentinel_to_nan=sentinel,
                )
                if verbose:
                    print(f"compare_in_memory all close for variable: {var}")

            except AssertionError:
                fail_list += [var]
                if verbose:
                    print(
                        f"compare_in_memory NOT all close for variable: {var}"
                    )

    if len(fail_list) > 0:
        success_vars = sorted(
            set(process.get_variables())
            - (set(fail_list).union(set(skip_list)))
        )
        msg = (
            "compare_in_memory:\n"
            f"    SUCCESSFUL for variables: {success_vars}\n"
            f"    SKIPPED variables: {sorted(skip_list)}\n"
            f"    FAILED for variables: {sorted(fail_list)}"
        )
        assert False, msg


def compare_model_in_memory(
    model: pws.base.Model,
    answers: dict,
    rtol: dict,
    atol: dict,
    equal_nan: bool = True,
    strict: bool = False,
    also_check_w_np: bool = True,
    skip_missing_ans: bool = False,
    fail_after_all_vars: bool = True,
    verbose: bool = False,
) -> None:
    """Compare model values to answer dictionaries.

    Parameters:
        model: a pws.Model object.
        answers: A dictionary where each process name contains a dictionary of
          variable names with answers.
        atol: A dictionary of tolerances to use for each process.
        rtol: As for atol.
    """

    for process_name in model.processes.keys():
        if process_name not in answers.keys():
            continue

        # <
        _ = compare_in_memory(
            process=model.processes[process_name],
            answers=answers[process_name],
            atol=atol[process_name],
            rtol=rtol[process_name],
            equal_nan=equal_nan,
            strict=strict,
            also_check_w_np=also_check_w_np,
            skip_missing_ans=skip_missing_ans,
            fail_after_all_vars=fail_after_all_vars,
            verbose=verbose,
        )


def compare_netcdfs(
    var_list: list,
    results_dir: pl.Path,
    answers_dir: pl.Path,
    rtol: float = 1.0e-15,
    atol: float = 1.0e-15,
    equal_nan: bool = True,
    strict: bool = False,
    also_check_w_np: bool = True,
    print_var_max_errs: bool = False,
    fail_after_all_vars: bool = True,
    verbose: bool = False,
    var_tolerances: dict = None,
    var_sentinel_to_nan: dict = None,
):
    # TODO: docstring
    # TODO: improve error message
    # TODO: collect failures in a try and report at end
    fail_list = []
    for var in var_list:
        answer = xr.open_dataarray(
            answers_dir / f"{var}.nc", decode_timedelta=False
        )
        result = xr.open_dataarray(
            results_dir / f"{var}.nc", decode_timedelta=False
        )

        # Get variable-specific tolerances if provided
        var_rtol = rtol
        var_atol = atol
        if var_tolerances is not None and var in var_tolerances:
            var_rtol = var_tolerances[var].get("rtol", rtol)
            var_atol = var_tolerances[var].get("atol", atol)

        # Get variable-specific sentinel value to convert to NaN
        sentinel = None
        if var_sentinel_to_nan is not None and var in var_sentinel_to_nan:
            sentinel = var_sentinel_to_nan[var]

        if not fail_after_all_vars:
            assert_allclose(
                actual=result.values,
                desired=answer.values,
                rtol=var_rtol,
                atol=var_atol,
                equal_nan=equal_nan,
                strict=strict,
                also_check_w_np=also_check_w_np,
                print_max_errs=print_var_max_errs,
                var_name=var,
                sentinel_to_nan=sentinel,
            )
            if verbose:
                print(f"compare_netcdfs all close for variable: {var}")

        else:
            try:
                assert_allclose(
                    actual=result.values,
                    desired=answer.values,
                    rtol=var_rtol,
                    atol=var_atol,
                    equal_nan=equal_nan,
                    strict=strict,
                    also_check_w_np=also_check_w_np,
                    print_max_errs=print_var_max_errs,
                    var_name=var,
                    sentinel_to_nan=sentinel,
                )
                if verbose:
                    print(f"compare_netcdfs all close for variable: {var}")

            except AssertionError:
                fail_list += [var]
                if verbose:
                    print(f"compare_netcdfs NOT all close for variable: {var}")

    if len(fail_list) > 0:
        assert False, (
            f"compare_netcdfs failed for variables: {sorted(fail_list)}"
        )
