import numpy as np
from scipy.special import hyp1f1


def l12(x):
    # Confluent hypergeometric function of the first kind, or Kummer's function
    # Another way to define it would be:
    # np.float64(np.exp(x / 2) * ((1-x) * np.float64(scipy.special.iv(0, -x / 2)) - x * np.float64(scipy.special.iv(1, -x / 2))))
    # However, the following version is more stable.
    return hyp1f1(-0.5, 1, x)


def l12_derivate(x):
    # Derivative of the confluent hypergeometric function of the first kind, or Kummer's function
    # z . (d / dz) M(a, b, z) = z.(a-b).M(a, b + 1, z)/b + z.M(a, b, z)
    # z . (d / dz) M(a, b, z) = z.(a/b).M(a + 1, b + 1, z)
    # a = -0.5
    # b = 1
    # (a-b)/b = -1.5
    return -0.5 * hyp1f1(0.5, 2, x)


def rice_mean(nu, sigma):
    """
    Mean/Expectation of the Rice distribution

    :param nu: array_like, shape (n_samples, ), mean of the Gaussian distribution
    :param sigma: scalar or array_like with shape (n_samples, ), standard deviation of the Gaussian distribution
    """
    # Scalar case
    if np.isscalar(sigma):
        if sigma == 0:
            return nu
        else:
            x = -0.5 * nu ** 2 / sigma ** 2
            return np.sqrt(np.pi / 2) * sigma * l12(x)
    # Array case
    nan_sigma = np.where(sigma == 0, np.nan, sigma)
    return np.where(sigma == 0, nu, np.sqrt(np.pi / 2) * nan_sigma * l12(-0.5 * nu ** 2 / nan_sigma ** 2))


def rice_mean_and_jacobian(nu, sigma, dnu):
    """
    Mean/Expectation of the Rice distribution and its Jacobian. They are computed together to avoid
    recomputing the confluent hypergeometric function of the first kind, or Kummer's function.

    :param nu: array_like, shape (n_samples, ), mean of the Gaussian distribution
    :param sigma: scalar or array_like with shape (n_samples, ), standard deviation of the Gaussian distribution
    :param dnu: array_like, shape (n_samples, n_parameters), Jacobian of the mean of the Gaussian distribution
    """
    # Scalar case
    if np.isscalar(sigma):
        if sigma == 0:
            dnu_extended = np.hstack((dnu, np.zeros((dnu.shape[0], 1))))
            return nu, dnu_extended
        else:
            x = -(nu ** 2) / (2 * sigma ** 2)
            K = hyp1f1(-0.5, 1, x)
            dK = -0.5 * hyp1f1(0.5, 2, x)
            mu = np.sqrt(np.pi / 2) * sigma * K
            dmu_dnu = (
                np.tile((np.sqrt(np.pi / 2) * sigma * dK * (-nu / sigma ** 2))[:, np.newaxis], dnu.shape[-1]) * dnu
            )
            dmu_dnu = np.hstack((dmu_dnu, np.zeros((dmu_dnu.shape[0], 1))))
            # Uncomment to allow change in sigma (and comment the previous line)
            # dmu_dsigma = np.sqrt(np.pi / 2) * (l12(x) + (nu / sigma)**2 * dK)
            # dmu_dnu = np.hstack((dmu_dnu, np.expand_dims(dmu_dsigma, axis=1)))
            return mu, dmu_dnu
    # Array case
    else:
        nan_sigma = np.where(sigma == 0, np.nan, sigma)
        x = -(nu ** 2) / (2 * nan_sigma ** 2)
        K = hyp1f1(-0.5, 1, x)
        dK = -0.5 * hyp1f1(0.5, 2, x)
        mu = np.sqrt(np.pi / 2) * nan_sigma * K
        dmu_dnu = (
            np.tile((np.sqrt(np.pi / 2) * nan_sigma * dK * (-nu / nan_sigma ** 2))[:, np.newaxis], dnu.shape[-1]) * dnu
        )
        dmu_dnu = np.where(np.tile(sigma[:, np.newaxis], dnu.shape[-1]) == 0, dnu, dmu_dnu)
        dmu_dnu = np.hstack((dmu_dnu, np.zeros((dmu_dnu.shape[0], 1))))
        # Uncomment to allow change in sigma (and comment the previous line)
        # dmu_dsigma = np.sqrt(np.pi / 2) * (l12(x) + (nu / sigma)**2 * dK)
        # dmu_dnu = np.hstack((dmu_dnu, np.expand_dims(dmu_dsigma, axis=1)))
        return mu, dmu_dnu


broad5 = lambda matrix: np.tile(matrix[..., np.newaxis], 5)
broad6 = lambda matrix: np.tile(matrix[..., np.newaxis], 6)
