import numpy as np
import pytest

from pywatershed.base.budget import Budget
from pywatershed.base.control import Control

# TODO
# * Test restart more robustly
# * "Real test cases" in indvidual storage units (e.g. Canopy).

time_dict = {
    "start_time": np.datetime64("1979-01-03T00:00:00.00"),
    "end_time": np.datetime64("1979-01-06T00:00:00.00"),
    "time_step": np.timedelta64(1, "D"),
}

time_dict["init_time"] = time_dict["start_time"] - time_dict["time_step"]


@pytest.fixture(scope="function")
def control_simple():
    return Control(**time_dict)


@pytest.mark.domainless
@pytest.mark.filterwarnings("ignore:Metadata unavailable")
def test_budget(control_simple):
    nhru = 5
    inputs = {
        "in1": np.ones([nhru]),
        "in2": np.ones([nhru]),
    }
    outputs = {
        "out1": np.zeros([nhru]),
        "out2": np.ones([nhru]),
    }
    storage_changes = {
        "stor1": np.ones([nhru]),
        "stor2": np.zeros([nhru]),
    }

    terms = {
        "inputs": inputs,
        "outputs": outputs,
        "storage_changes": storage_changes,
    }

    terms_keys = {key: list(val.keys()) for key, val in terms.items()}

    budget = Budget(
        control_simple,
        **terms_keys,
        time_unit="D",
        description="simple_test",
        units="m*3/D",
        verbose=False,
    )
    budget.set(terms)

    assert isinstance(budget, Budget)

    for component in budget.components:
        assert budget[component] == terms[component]

    control_simple.advance()
    budget.advance()
    budget.calculate()

    assert (budget._inputs_sum == 2).all()
    assert (budget._outputs_sum == 1).all()
    assert (budget._storage_changes_sum == 1).all()

    for component in budget.components:
        for var in budget[component].keys():
            assert (
                budget.accumulations[component][var]
                == terms[component][var] * 1
            ).all()

    control_simple.advance()
    budget.advance()
    budget.calculate()

    assert (budget._inputs_sum == 2).all()
    assert (budget._outputs_sum == 1).all()
    assert (budget._storage_changes_sum == 1).all()
    assert (budget.balance == 1).all()

    for component in budget.components:
        for var in budget[component].keys():
            assert (
                budget.accumulations[component][var]
                == terms[component][var] * 2
            ).all()

    # change inputs
    inputs["in1"][:] = -1 * np.ones([nhru])
    storage_changes["stor2"][:] = -2 * np.ones([nhru])

    for comp_name in budget.components:
        for var_key, var_val in budget[comp_name].items():
            assert budget[comp_name][var_key] is var_val

    control_simple.advance()
    budget.advance()
    budget.calculate()

    print(budget)

    assert (budget._inputs_sum == 0).all()
    assert (budget._outputs_sum == 1).all()
    assert (budget._storage_changes_sum == -1).all()
    assert (budget.balance == -1).all()

    accum_answers = {
        "inputs": {"in1": np.ones([nhru]), "in2": np.ones([nhru]) * 3},
        "outputs": {"out1": np.zeros([nhru]), "out2": np.ones([nhru]) * 3},
        "storage_changes": {
            "stor1": np.ones([nhru]) * 3,
            "stor2": np.ones([nhru]) * -2,
        },
    }
    for component in budget.components:
        for var in budget[component].keys():
            assert (
                budget.accumulations[component][var]
                == accum_answers[component][var]
            ).all()

    with pytest.warns(UserWarning):
        storage_changes["stor2"][:] = -23 * np.ones([nhru])
        control_simple.advance()
        budget.advance()
        budget.calculate()
        print(budget)

    with pytest.raises(ValueError):
        budget.calculate()

    assert budget._accum_start_time == time_dict["init_time"]
    budget.reset_accumulations()
    assert budget._accum_start_time == control_simple.current_time
    print(budget)

    return


@pytest.mark.domainless
@pytest.mark.filterwarnings("ignore:Metadata unavailable")
def test_budget_active_mask_imbalance_indices(control_simple):
    # inactive hru 0 and 2 shift the compressed positions; the reported
    # locations must still be full hru indices (here 3 and 4).
    nhru = 5
    active_mask = np.array([False, True, False, True, True])
    inputs = {"in1": np.ones([nhru])}
    outputs = {"out1": np.ones([nhru])}
    storage_changes = {"stor1": np.zeros([nhru])}
    storage_changes["stor1"][3:] = 7.0
    # mirror _mask_inactive_hrus: inactive hrus hold nan in every term
    for arr in (inputs["in1"], outputs["out1"], storage_changes["stor1"]):
        arr[~active_mask] = np.nan
    terms = {
        "inputs": inputs,
        "outputs": outputs,
        "storage_changes": storage_changes,
    }
    terms_keys = {key: list(val.keys()) for key, val in terms.items()}
    budget = Budget(
        control_simple,
        **terms_keys,
        time_unit="D",
        description="masked_test",
        units="m*3/D",
        verbose=False,
        active_mask=active_mask,
        imbalance_fatal=True,
    )
    budget.set(terms)
    control_simple.advance()
    budget.advance()
    with pytest.raises(ValueError, match=r"\[3 4\]"):
        budget.calculate()
