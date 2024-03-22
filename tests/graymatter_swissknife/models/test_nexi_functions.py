import os
import sys
import numpy as np

# Get the absolute path to the root directory of your project
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
# Append the project root to sys.path
sys.path.append(project_root)

from src.graymatter_swissknife.models.NEXI.functions.scipy_nexi import (
    M,
    nexi_signal,
    nexi_jacobian,
    nexi_hessian,
    nexi_signal_from_vector,
    nexi_optimized_mse_jacobian,
)


def test_M():
    x = 0.5
    b = np.array([1, 2, 5])
    t = np.array([20, 30, 40])
    tex = 30
    Di = 2
    De = 1.0
    Dp = 1.0
    f = 0.4

    result = M(x, b, t, tex, Di, De, Dp, f)
    expected_result = np.array([0.46064021, 0.22113262, 0.03022508])

    assert np.allclose(result, expected_result)


def test_nexi_signal():
    tex = 30
    Di = 2
    De = 1.0
    Dp = 1.0
    f = 0.4
    b = np.array([1, 2, 5])
    t = np.array([20, 30, 40])

    result = nexi_signal(tex, Di, De, Dp, f, b, t)
    expected_result = np.array([0.45451206, 0.24024062, 0.07948643])

    assert np.allclose(result, expected_result)


def test_nexi_signal_from_vector():
    tex = 30
    Di = 2
    De = 1.0
    Dp = 1.0
    f = 0.4

    param_vector = np.array([tex, Di, De, f])
    b = np.array([1, 2, 5])
    t = np.array([20, 30, 40])

    result = nexi_signal_from_vector(param_vector, b, t)
    expected_result = np.array([0.45451206, 0.24024062, 0.07948643])

    assert np.allclose(result, expected_result)


def test_nexi_jacobian():
    tex = 30
    Di = 2
    De = 1.0
    Dp = 1.0
    f = 0.4
    b = np.array([1, 2, 5])
    t = np.array([20, 30, 40])

    result = nexi_jacobian(tex, Di, De, Dp, f, b, t)
    expected_result = np.array(
        [
            [1.54514401e-04, -4.62179508e-02, -2.28483192e-01, 2.25580692e-01],
            [4.50341890e-04, -3.96858817e-02, -1.84227752e-01, 2.89960595e-01],
            [8.58253213e-04, -1.91050038e-02, -4.50295796e-02, 2.34286955e-01],
        ]
    )

    assert np.allclose(result, expected_result)


def test_nexi_optimized_mse_jacobian():
    tex = 30
    Di = 2
    De = 1.0
    f = 0.4
    param = np.array([tex, Di, De, f])
    b = np.array([1, 2, 5])
    t = np.array([20, 30, 40])
    measured_signal = np.array([0.5, 0.2, 0.07])
    b_td_dimensions = b.ndim

    result = nexi_optimized_mse_jacobian(param, b, t, measured_signal, b_td_dimensions)
    expected_result = np.array([3.84705102e-05, 6.48273196e-04, 5.10524168e-03, 7.25908042e-03])

    assert np.allclose(result, expected_result)


def test_nexi_hessian():
    tex = 30
    Di = 2
    De = 1.0
    Dp = 1.0
    f = 0.4
    b = np.array([1, 2, 5])
    t = np.array([20, 30, 40])

    result = nexi_hessian(tex, Di, De, Dp, f, b, t)
    expected_result = np.array(
        [
            [
                [-8.67335582e-06, -1.55651863e-06, 2.18045281e-04, 1.36086092e-04],
                [-1.55651863e-06, 1.91030931e-02, 5.27110397e-03, -1.15578206e-01],
                [2.18045281e-04, 5.27110397e-03, 2.07696229e-01, 3.60959263e-01],
                [1.36086092e-04, -1.15578206e-01, 3.60959263e-01, 4.54012844e-02],
            ],
            [
                [-2.30124301e-05, -6.61735471e-05, 5.59240171e-04, 4.46433446e-04],
                [-6.61735471e-05, 2.39876238e-02, 1.10851785e-02, -1.03052775e-01],
                [5.59240171e-04, 1.10851785e-02, 3.10333760e-01, 2.48903525e-01],
                [4.46433446e-04, -1.03052775e-01, 2.48903525e-01, 1.42589349e-01],
            ],
            [
                [-3.88576179e-05, -2.08384428e-04, 5.53097451e-04, 1.17287377e-03],
                [-2.08384428e-04, 1.41591595e-02, 7.59396884e-03, -6.03884118e-02],
                [5.53097451e-04, 7.59396884e-03, 1.33604022e-01, -3.88932993e-04],
                [1.17287377e-03, -6.03884118e-02, -3.88932993e-04, 2.88758269e-01],
            ],
        ]
    )

    assert np.allclose(result, expected_result)
