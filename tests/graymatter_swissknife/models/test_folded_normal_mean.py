import os
import sys
import numpy as np

# Get the absolute path to the root directory of your project
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
# Append the project root to sys.path
sys.path.append(project_root)

from src.graymatter_swissknife.models.noise.folded_normal_mean import folded_normal_mean, folded_normal_mean_and_jacobian


def test_folded_normal_mean():
    nu = np.array([0.8, 0.5, 0.3, 0])
    sigma = 0.2

    result = folded_normal_mean(nu, sigma)
    expected_result = np.array([0.80000286, 0.50080165, 0.31172272, 0.15957691])

    assert np.allclose(result, expected_result)

    nu = np.array([0.8, 0.5, 0.3, 0])
    sigma = np.array([0.2, 0, 0.3, 0])

    result = folded_normal_mean(nu, sigma)
    expected_result = np.array([0.80000286, 0.5       , 0.34998928, 0.        ])

    assert np.allclose(result, expected_result)


def test_folded_normal_mean_and_jacobian():

    # Case sigma is an array

    nu = np.array([0.8, 0.5, 0.3, 0])
    sigma = np.array([0.2, 0, 0.3, 0])
    dnu = np.array([[1, 2], [1, 2], [1, 2], [1, 2]])

    result_mu, result_dmu_dnu = folded_normal_mean_and_jacobian(nu, sigma, dnu)
    expected_mu = np.array([0.80000286, 0.5, 0.34998928, 0.0])
    expected_dmu_dnu = np.array(
        [[0.99993666, 1.99987332, 0.0], [1.0, 2.0, 0.0], [0.68268949, 1.36537898, 0.0], [1.0, 2.0, 0.0]]
    )

    assert np.allclose(result_mu, expected_mu, equal_nan=True)
    assert np.allclose(result_dmu_dnu, expected_dmu_dnu)

    # Case sigma is a scalar

    sigma = 0.2
    result_mu, result_dmu_dnu = folded_normal_mean_and_jacobian(nu, sigma, dnu)
    expected_mu = np.array([0.80000286, 0.50080165, 0.31172272, 0.15957691])
    expected_dmu_dnu = np.array(
        [[0.99993666, 1.99987332, 0.0], [0.98758067, 1.97516134, 0.0], [0.8663856 , 1.73277119, 0.0], [0.0, 0.0, 0.0]]
    )

    assert np.allclose(result_mu, expected_mu)
    assert np.allclose(result_dmu_dnu, expected_dmu_dnu)

    # Case sigma = 0

    sigma = 0
    result_mu, result_dmu_dnu = folded_normal_mean_and_jacobian(nu, sigma, dnu)
    expected_mu = np.array([0.8, 0.5, 0.3, 0])
    expected_dmu_dnu = np.array([[1, 2, 0.0], [1, 2, 0.0], [1, 2, 0.0], [1, 2, 0.0]])

    assert np.allclose(result_mu, expected_mu)
    assert np.allclose(result_dmu_dnu, expected_dmu_dnu)
