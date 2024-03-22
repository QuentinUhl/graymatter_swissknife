import numpy as np
from ...models.microstructure_models import MicroStructModel
from .smex import Smex
from .functions.scipy_smex import smex_signal_from_vector, \
    smex_jacobian_concatenated_from_vector
from ...models.rice_noise.rice_mean import rice_mean, rice_mean_and_jacobian, broad5


class SmexRiceMean(MicroStructModel):
    """SMEX / Neurite Exchange Imaging accounting for Wide Pulses model for diffusion MRI. Corrected for Rician noise."""

    # Class attributes
    n_params = 5
    param_names = ["t_ex", "Di", "De", "f", "sigma"]
    classic_limits = np.array([[1, 150], [0.1, 3.5], [0.1, 3.5], [0.1, 0.9], [0, 100]])
    grid_search_nb_points = [15, 12, 8, 8]
    has_rician_mean_correction = True
    non_corrected_model = Smex()

    def __init__(self, param_lim=classic_limits, invert_tex=False):
        super().__init__(name='SMEX_Rice_Mean')
        self.param_lim = param_lim
        self.invert_tex = invert_tex
        self.constraints = [self.constr_on_diffusivities]

    @staticmethod
    def constr_on_diffusivities(param):
        if np.array(param).ndim == 1:
            # Put Di>De
            if param[1] < param[2]:
                # exchange them
                param[1], param[2] = param[2], param[1]
        elif np.array(param).ndim == 2:
            # Put Di>De
            for iset in range(len(param)):
                # if Di < De of the set iset
                if param[iset, 1] < param[iset, 2]:
                    # exchange them
                    param[iset, 1], param[iset, 2] = param[iset, 2], param[iset, 1]
        else:
            raise ValueError('Wrong dimension of parameters')
        return param

    def set_param_lim(self, param_lim):
        self.param_lim = param_lim

    @classmethod
    def get_signal(cls, parameters, acq_parameters):
        """Get signal from single Ground Truth."""
        return rice_mean(smex_signal_from_vector(parameters[:4],
                                                               acq_parameters.b, acq_parameters.td,
                                                               acq_parameters.small_delta),
                         parameters[4])

    @classmethod
    def get_jacobian(cls, parameters, acq_parameters):
        """Get jacobian from single Ground Truth."""
        nexi_fp_signal_vec, nexi_fp_vec_jac = smex_jacobian_concatenated_from_vector(parameters[:4], acq_parameters)
        # nexi_fp_signal_vec = nexi_fp_vec_jac_concatenation[..., 0]
        # nexi_fp_vec_jac = nexi_fp_vec_jac_concatenation[..., 1:5]
        # Turn last parameter jacobian to 0 to avoid updates
        _, nexi_fp_rm_vec_jac = rice_mean_and_jacobian(nexi_fp_signal_vec, parameters[4], dnu=nexi_fp_vec_jac)
        return nexi_fp_rm_vec_jac


    @classmethod
    def get_hessian(cls, parameters, acq_parameters):
        return None


    # Optimized methods for Non-linear Least Squares
    @classmethod
    def get_mse_jacobian(cls, parameters, acq_parameters, signal_gt):
        """Get jacobian of Mean Square Error from single Ground Truth."""
        nexi_fp_signal_vec, nexi_fp_vec_jac = smex_jacobian_concatenated_from_vector(parameters[:-1], acq_parameters)
        # nexi_fp_signal_vec = nexi_fp_vec_jac_concatenation[..., 0]
        # nexi_fp_vec_jac = nexi_fp_vec_jac_concatenation[..., 1:]
        nexi_fp_rm_signal_vec, nexi_fp_rm_vec_jac = rice_mean_and_jacobian(nexi_fp_signal_vec, parameters[-1], dnu=nexi_fp_vec_jac)
        if acq_parameters.ndim == 1:
            mse_jacobian = np.sum(2 * nexi_fp_rm_vec_jac * broad5(nexi_fp_rm_signal_vec - signal_gt), axis=0)
        elif acq_parameters.ndim == 2:
            mse_jacobian = np.sum(2 * nexi_fp_rm_vec_jac * broad5(nexi_fp_rm_signal_vec - signal_gt), axis=(0, 1))
        else:
            raise NotImplementedError
        # Turn last parameter jacobian to 0 to avoid updates
        mse_jacobian[..., -1] = np.zeros_like(mse_jacobian[..., -1])
        return mse_jacobian
