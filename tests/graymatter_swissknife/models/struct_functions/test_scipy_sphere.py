import os
import sys
import numpy as np
import logging

# Get the absolute path to the root directory of your project
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
# Append the project root to sys.path
sys.path.append(project_root)

from src.graymatter_swissknife.models.struct_functions.scipy_sphere import (
    sphere_murdaycotts,
    sphere_jacobian,
    moving_d0_sphere_jacobian
)

def test_sphere_murdaycotts():
    r, D0 = 9.5, 2.5
    b = np.array([1, 2, 5, 1, 2, 7])
    big_delta = np.array([30, 30, 30, 60, 60, 60])
    small_delta = 15
    sphere_signal = sphere_murdaycotts(r, D0, b, big_delta, small_delta)
    expected_signal = np.array([0.66963017, 0.44840456, 0.1346403 , 
                                0.82406686, 0.6790862 , 0.25806979])
    np.testing.assert_allclose(sphere_signal, expected_signal, rtol=1e-5)

def test_sphere_jacobian():
    r, D0, = 9.5, 2.5
    b = np.array([1, 2, 5, 1, 2, 7])
    big_delta = np.array([30, 30, 30, 60, 60, 60])
    small_delta = 15
    eps = 1e-8
    sphere_signal_r_mdr = sphere_murdaycotts(r - eps, D0, b, big_delta, small_delta)
    sphere_signal_r_pdr = sphere_murdaycotts(r + eps, D0, b, big_delta, small_delta)
    deriv_sphere_signal_r = (sphere_signal_r_pdr - sphere_signal_r_mdr) / (2 * eps)
    # sphere_signal_D0_mdr = sphere_murdaycotts(r, D0 - eps, b, big_delta, small_delta)
    # sphere_signal_D0_pdr = sphere_murdaycotts(r, D0 + eps, b, big_delta, small_delta)
    # deriv_sphere_signal_D0 = (sphere_signal_D0_pdr - sphere_signal_D0_mdr) / (2 * eps)
    # Our implementation
    _, jacobian_dr = sphere_jacobian(r, D0, b, big_delta, small_delta)

    # Test the Jacobian with respect to r
    np.testing.assert_allclose(jacobian_dr, deriv_sphere_signal_r, rtol=1e-5)

def test_moving_d0_sphere_jacobian():
    r, D0, = 9.5, 2.5
    b = np.array([1, 2, 5, 1, 2, 7])
    big_delta = np.array([30, 30, 30, 60, 60, 60])
    small_delta = 15
    eps = 1e-8
    sphere_signal_r_mdr = sphere_murdaycotts(r - eps, D0, b, big_delta, small_delta)
    sphere_signal_r_pdr = sphere_murdaycotts(r + eps, D0, b, big_delta, small_delta)
    deriv_sphere_signal_r = (sphere_signal_r_pdr - sphere_signal_r_mdr) / (2 * eps)
    sphere_signal_D0_mdD0 = sphere_murdaycotts(r, D0 - eps, b, big_delta, small_delta)
    sphere_signal_D0_pdD0 = sphere_murdaycotts(r, D0 + eps, b, big_delta, small_delta)
    deriv_sphere_signal_D0 = (sphere_signal_D0_pdD0 - sphere_signal_D0_mdD0) / (2 * eps)
    # Our implementation
    _, jacobian_dr, jacobian_dD0 = moving_d0_sphere_jacobian(r, D0, b, big_delta, small_delta)
    # Test the Jacobian with respect to r
    np.testing.assert_allclose(jacobian_dr, deriv_sphere_signal_r, rtol=1e-5)
    # Test the Jacobian with respect to D0
    np.testing.assert_allclose(jacobian_dD0, deriv_sphere_signal_D0, rtol=1e-5)
    
# Run the tests
if __name__ == "__main__":
    test_sphere_murdaycotts()
    test_sphere_jacobian()
    test_moving_d0_sphere_jacobian()
    logging.info("Scipy sphere: All tests passed successfully!")


