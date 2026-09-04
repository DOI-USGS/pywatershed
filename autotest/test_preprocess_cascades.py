import numpy as np
import pytest

from pywatershed.base.control import Control
from pywatershed.parameters import Parameters, PrmsParameters
from pywatershed.utils.preprocess_cascades import (
    calc_hru_route_order,
    init_cascade_params,
    order_hrus,
)

# TODO: These tests are only valid for sagehen_5yr and certain configurations
#       add skips

# None of the answer variables are output by PRMS, so they are culled from
# the diagnostic messages printed to cascade.msgs for the sagehen_5yr domain.


@pytest.fixture(scope="function")
def control(simulation):
    if simulation["name"].split(":")[0] != "sagehen_5yr":
        pytest.skip(
            "test_preprocess_cascades answers are hard-coded for the "
            "sagehen_5yr domain"
        )
    control = Control.load_prms(
        simulation["control_file"], warn_unused_options=False
    )
    if not control.options.get("cascade_flag", 0):
        pytest.skip("cascade_flag absent or 0")
    return control


@pytest.fixture(scope="function")
def parameters(simulation, control):
    param_file = simulation["dir"] / control.options["parameter_file"]
    params = PrmsParameters.load(param_file)
    return params


def test_preprocess(control, parameters):
    new_params = calc_hru_route_order(parameters)
    assert "hru_route_order" in new_params.variables
    assert isinstance(new_params, Parameters)
    newer_params = init_cascade_params(control, new_params, verbosity=100)

    # fmt: off
    answer = np.array(
        [
            1,   4,   6,   7,   8,   10,  11,  13,  14,  16,
            17,  18,  19,  20,  21,  22,  23,  25,  26,  27,
            28,  29,  31,  32,  33,  35,  37,  43,  44,  46,
            48,  50,  52,  56,  57,  58,  59,  67,  68,  70,
            73,  78,  81,  82,  84,  87,  88,  91,  92,  95,
            96,  98,  99, 100, 101, 102, 104, 108, 114, 116,
            118, 119, 120, 123, 124, 125, 127, 128,   3,   9,
            12,  24,  30,  34,  38,  45,  47,  54,  60,  66,
            72,  76,  77,  79,  83,  86,  97, 105, 106, 107,
            109, 110, 111, 112, 113, 115, 117, 121, 122, 126,
            2,   5,  15,  39,  40,  41,  42,  49,  53,  62,
            63,  69,  71,  74,  75,  80,  85,  89,  90,  93,
            94, 103,  55,  61,  64,  65,  51,  36,
        ],
        dtype='int64'
    )
    # fmt: on
    hru_route_order = newer_params.parameters["hru_route_order"]
    assert (hru_route_order == answer).all()

    # fmt: off
    answer_hru_down_frac_flat = np.array(
        [
            87.6314304123478, 12.3685695876522, 11.7745831117949,
            82.2331823105820, 5.99223457762308, 100.000000000000,
            100.000000000000, 100.000000000000, 100.000000000000,
            13.9913989496653, 9.92099232701961, 76.0876087233151,
            58.8217004604251, 41.1782995395749, 100.000000000000,
            66.5827121488632, 33.4172878511368, 21.1921896964111,
            78.8078103035889, 100.000000000000, 100.000000000000,
            100.000000000000, 100.000000000000, 100.000000000000,
            100.000000000000, 100.000000000000, 100.000000000000,
            100.000000000000, 100.000000000000, 15.8131625636568,
            19.0638126823525, 65.1230247539907, 49.3853383517303,
            50.6146616482697, 100.000000000000, 14.0491460679301,
            85.9508539320699, 100.000000000000, 100.000000000000,
            100.000000000000, 100.000000000000, 100.000000000000,
            100.000000000000, 89.5889591619249, 10.4110408380751,
            100.000000000000, 100.000000000000, 100.000000000000,
            100.000000000000, 37.0337034794352, 62.9662965205648,
            100.000000000000, 78.0778318915780, 21.9221681084220,
            100.000000000000, 6.43605027341388, 9.29993805051148,
            84.2640116760746, 53.4903182220562, 46.5096817779438,
            100.000000000000, 7.65076506041115, 92.3492349395889,
            72.3744751543619, 27.6255248456381, 22.2570204628608,
            77.7429795371392, 100.000000000000, 100.000000000000,
            100.000000000000, 100.0000000000000, 23.3724985013258,
            76.6275014986742, 12.4346920021060, 87.5653079978940,
            54.7872357289642, 45.2127642710358, 100.000000000000,
            100.000000000000, 100.000000000000, 100.000000000000,
            100.000000000000, 100.000000000000, 31.8610990434774,
            68.1389009565226, 100.000000000000, 100.000000000000,
            68.1186785198300, 20.2911445861286, 11.5901768940415,
            25.1788270550906, 7.82436419087800, 54.2863427633817,
            12.7104659906497, 100.000000000000, 100.000000000000,
            100.000000000000, 100.000000000000, 16.9416945989572,
            83.0583054010428, 100.000000000000, 100.000000000000,
            22.0673033243474, 77.9326966756526, 100.000000000000,
            100.000000000000, 83.8767758377653, 16.1232241622347,
            100.000000000000, 10.0163130841634, 89.9836869158366,
            41.5083009662538, 14.6229247584391, 43.8687742753072,
            7.40431771131174, 92.5956822886883, 100.000000000000,
            6.34872121464874, 9.87465633645974, 83.7766224488915,
            85.9521646413924, 4.92481567990151, 9.12301967870610,
            100.000000000000, 22.1917816504327, 77.8082183495673,
            24.4648928611306, 42.9085808839281, 32.6265262549414,
            11.9142421207032, 32.1286379599253, 43.5222042972141,
            12.4349156221574, 100.000000000000, 100.000000000000,
            88.5723101515525, 11.4276898484475, 10.1910187591173,
            89.8089812408827, 100.000000000000, 100.000000000000,
            100.000000000000, 100.000000000000, 11.0137154687524,
            48.0251631549998, 40.9611213762479, 31.0862168874637,
            54.8109618179743, 14.1028212945619, 72.5528635405319,
            27.4471364594681, 100.000000000000, 100.000000000000,
            100.000000000000, 89.3619647480041, 10.6380352519959,
            100.000000000000, 100.000000000000, 42.6194780741360,
            57.3805219258640, 49.4898975478837, 10.3020603053467,
            40.2080421467696, 28.5728562454228, 71.4271437545772,
            83.4406434339559, 6.97183151271610, 9.58752505332795,
            100.000000000000, 100.000000000000, 100.000000000000,
            3.88005271653781, 5.73396829650825, 90.3859789869539,
            100.000000000000, 100.000000000000, 22.3149866356031,
            11.3790970235735, 66.3059163408234, 71.4342874863003,
            17.5835159357747, 10.9821965779250, 10.9379968781041,
            89.0620031218959, 100.000000000000, 25.0000000000000,
            75.0000000000000, 100.000000000000, 100.000000000000,
            100.000000000000, 100.000000000000, 100.000000000000,
            100.000000000000, 100.000000000000, 100.000000000000,
            9.77097724078999, 90.2290227592100, 100.000000000000,
        ],
        dtype="float64"
    ) / 100.0

    answer_hru_down_flat = np.array(
        [
            2, 11, 3, 8, 15, 9, 14, 15, -7, 5, -8, 22, 12, -7, 19, 26, 27, 9,
            20, 24, -7, 21, -7, -8, 29, 33, 30, 30, 59, 24, 34, 36, 35, -3,
            57, 36, -3, -3, 44, 38, -8, 48, -8, 42, 61, 70, -4, -11, 47, -12,
            -13, 54, 78, 88, -13, 45, 39, 82, -14, -15, 75, 39, -2, -6, 95,
            76, 90, -15, -1, 85, -6, -1, 104, 77, 83, 72, 102, 79, 93, 72,
            -1, 107, 112, 109, 118, 119, 112, 115, 121, 113, 106, 122, 93,
            117, 125, -5, 105, 126, 2, -8, -7, -7, 30, 34, -10, 36, -10,
            -4, 39, 40, 60, -1, 55, 63, 41, 66, -1, -1, 62, 71, 53, -4,
            65, 69, 63, 86, 75, -6, 94, -2, 63, 97, 109, 63, 103, 93, 107,
            94, -6, 110, 111, -2, -2, 90, -4, 121, 74, -4, 121, 93, -5,
            -4, -6, -5, 5, 15, -8, -8, -10, -2, 49, -3, 41, -3, 42, -3,
            -1, 61, -3, -11, 61, 55, 51, 80, -12, -1, 64, 65, 89, -13,
            -6, 85, 51, -2, -14, 65, -4, -4, -5, -6, -5, 51, -1, 65,
            -4, 36, -2, -10,
        ],
        dtype='int64'
    )
    # fmt: on

    hru_down = newer_params.parameters["hru_down"]
    hru_down_frac = newer_params.parameters["hru_down_frac"]
    ncascade_hru = newer_params.parameters["ncascade_hru"]
    flat_hru_down = []
    flat_hru_down_frac = []
    ni, nj = hru_down.shape
    for jj in range(nj):
        oo = hru_route_order[jj] - 1
        ncasc = ncascade_hru[oo]
        for ii in range(ncasc):
            down_val = hru_down[ii, oo]
            frac_val = hru_down_frac[ii, oo]
            # if down_val == 0:
            #     break
            flat_hru_down.append(down_val)
            flat_hru_down_frac.append(frac_val)

    assert (flat_hru_down == answer_hru_down_flat).all()
    assert (abs(flat_hru_down_frac - answer_hru_down_frac_flat) < 1e-8).all()


def test_order_hrus_circle():
    # HRU 1 is a swale root; HRU 2 cascades to HRU 3 and HRU 1; HRU 3
    # cascades back to HRU 2.
    nhru = 3
    hru_route_order = np.array([1, 2, 3], dtype="int64")
    ncascade_hru = np.array([0, 2, 1], dtype="int64")
    hru_down = np.array([[0, 3, 2], [0, 1, 0]], dtype="int64")
    hru_type = np.array([3, 1, 1], dtype="int64")
    with pytest.raises(ValueError, match="Circular cascading path"):
        order_hrus(
            nhru,
            nhru,
            hru_route_order,
            ncascade_hru,
            hru_down,
            hru_type,
            circle_switch=1,
        )


def test_calc_hru_route_order_lake():
    # a lake HRU without an nlake dimension gets the PRMS diagnostic
    nhru = 2
    params = Parameters(
        dims={"nhru": nhru},
        coords={"nhru": np.arange(nhru)},
        data_vars={"hru_type": np.array([1, 2], dtype="int64")},
        metadata={"nhru": {"dims": ["nhru"]}, "hru_type": {"dims": ["nhru"]}},
        validate=True,
    )
    with pytest.raises(ValueError, match="nlake = 0"):
        calc_hru_route_order(params)
