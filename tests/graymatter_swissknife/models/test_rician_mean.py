import os
import sys
import numpy as np

# Get the absolute path to the root directory of your project
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
# Append the project root to sys.path
sys.path.append(project_root)

from src.graymatter_swissknife.models.rice_noise.rice_mean import rice_mean, rice_mean_and_jacobian


def test_rice_mean():
    nu = np.array([0.8, 0.5, 0.3, 0])
    sigma = 0.2

    result = rice_mean(nu, sigma)
    expected_result = np.array([0.82543871, 0.54224029, 0.37498715, 0.25066283])

    assert np.allclose(result, expected_result)

    nu = np.array([0.8, 0.5, 0.3, 0])
    sigma = np.array([0.2, 0, 0.3, 0])

    result = rice_mean(nu, sigma)
    expected_result = np.array([0.82543871, 0.5, 0.46457174, 0.0])

    assert np.allclose(result, expected_result)


def test_rice_mean_and_jacobian():

    # Case sigma is an array

    nu = np.array([0.8, 0.5, 0.3, 0])
    sigma = np.array([0.2, 0, 0.3, 0])
    dnu = np.array([[1, 2], [1, 2], [1, 2], [1, 2]])

    result_mu, result_dmu_dnu = rice_mean_and_jacobian(nu, sigma, dnu)
    expected_mu = np.array([0.82543871, np.nan, 0.46457174, np.nan])
    expected_dmu_dnu = np.array(
        [[0.96693878, 1.93387755, 0.0], [1.0, 2.0, 0.0], [0.55717947, 1.11435894, 0.0], [1.0, 2.0, 0.0]]
    )

    assert np.allclose(result_mu, expected_mu, equal_nan=True)
    assert np.allclose(result_dmu_dnu, expected_dmu_dnu)

    # Case sigma is a scalar

    sigma = 0.2
    result_mu, result_dmu_dnu = rice_mean_and_jacobian(nu, sigma, dnu)
    expected_mu = np.array([0.82543871, 0.54224029, 0.37498715, 0.25066283])
    expected_dmu_dnu = np.array(
        [[0.96693878, 1.93387755, 0.0], [0.90478279, 1.80956557, 0.0], [0.73546946, 1.47093891, 0.0], [0.0, 0.0, 0.0]]
    )

    assert np.allclose(result_mu, expected_mu)
    assert np.allclose(result_dmu_dnu, expected_dmu_dnu)

    # Case sigma = 0

    sigma = 0
    result_mu, result_dmu_dnu = rice_mean_and_jacobian(nu, sigma, dnu)
    expected_mu = np.array([0.8, 0.5, 0.3, 0])
    expected_dmu_dnu = np.array([[1, 2, 0.0], [1, 2, 0.0], [1, 2, 0.0], [1, 2, 0.0]])

    assert np.allclose(result_mu, expected_mu)
    assert np.allclose(result_dmu_dnu, expected_dmu_dnu)
