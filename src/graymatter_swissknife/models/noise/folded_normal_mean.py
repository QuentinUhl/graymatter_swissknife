import numpy as np
from scipy.special import erf


def folded_normal_mean(mu, sigma):
    """
    Mean/Expectation of the Folded Normal distribution

    :param nu: array_like, shape (n_samples, ), mean of the Gaussian distribution
    :param sigma: scalar or array_like with shape (n_samples, ), standard deviation of the Gaussian distribution
    """
    # Scalar case
    if np.isscalar(sigma):
        if sigma == 0:
            return mu
        else:
            return np.sqrt(2 / np.pi) * sigma * np.exp(-0.5 * mu ** 2 / sigma ** 2) + mu * erf(mu / (np.sqrt(2) * sigma))
    # Array case
    nan_sigma = np.where(sigma == 0, np.nan, sigma)
    fn_mean = np.sqrt(2 / np.pi) * nan_sigma * np.exp(-0.5 * mu ** 2 / nan_sigma ** 2) + mu * erf(mu / (np.sqrt(2) * nan_sigma))
    return np.where(sigma == 0, mu, fn_mean)


def folded_normal_mean_and_jacobian(mu, sigma, dmu):
    """
    Mean/Expectation of the Folded Normal distribution and its Jacobian. 

    :param mu: array_like, shape (n_samples, ), mean of the Gaussian distribution
    :param sigma: scalar or array_like with shape (n_samples, ), standard deviation of the Gaussian distribution
    :param dmu: array_like, shape (n_samples, n_parameters), Jacobian of the mean of the Gaussian distribution
    """
    # Scalar case
    if np.isscalar(sigma):
        if sigma == 0:
            dmu_extended = np.hstack((dmu, np.zeros((dmu.shape[0], 1))))
            return mu, dmu_extended
        else:
            mufd = np.sqrt(2 / np.pi) * sigma * np.exp(-0.5 * mu ** 2 / sigma ** 2) + mu * erf(mu / (np.sqrt(2) * sigma))
            dmufd_dmu = np.tile((erf(mu / (np.sqrt(2) * sigma)))[:, np.newaxis], dmu.shape[-1]) * dmu
            dmufd_dmu = np.hstack((dmufd_dmu, np.zeros((dmufd_dmu.shape[0], 1))))
            # Uncomment to allow change in sigma (and comment the previous line)
            # dmu_dsigma = np.sqrt(np.pi / 2) * (l12(x) + (nu / sigma)**2 * dK)
            # dmu_dnu = np.hstack((dmu_dnu, np.expand_dims(dmu_dsigma, axis=1)))
            return mufd, dmufd_dmu
    # Array case
    else:
        nan_sigma = np.where(sigma == 0, np.nan, sigma)
        fn_mean = np.sqrt(2 / np.pi) * nan_sigma * np.exp(-0.5 * mu ** 2 / nan_sigma ** 2) + mu * erf(mu / (np.sqrt(2) * nan_sigma))
        mufd = np.where(sigma == 0, mu, fn_mean)
        fn_der_mean = erf(mu / (np.sqrt(2) * nan_sigma))
        dmufd_dmu = np.tile(fn_der_mean[:, np.newaxis], dmu.shape[-1]) * dmu
        dmufd_dmu = np.where(np.tile(sigma[:, np.newaxis], dmu.shape[-1]) == 0, dmu, dmufd_dmu)
        dmufd_dmu = np.hstack((dmufd_dmu, np.zeros((dmufd_dmu.shape[0], 1))))
        # Uncomment to allow change in sigma (and comment the previous line)
        # dmu_dsigma = np.sqrt(np.pi / 2) * (l12(x) + (nu / sigma)**2 * dK)
        # dmu_dnu = np.hstack((dmu_dnu, np.expand_dims(dmu_dsigma, axis=1)))
        return mufd, dmufd_dmu


broad5 = lambda matrix: np.tile(matrix[..., np.newaxis], 5)
broad6 = lambda matrix: np.tile(matrix[..., np.newaxis], 6)
