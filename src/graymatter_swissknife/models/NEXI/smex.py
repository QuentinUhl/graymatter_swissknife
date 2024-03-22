import numpy as np
from ...models.microstructure_models import MicroStructModel
from .functions.scipy_smex import smex_signal_from_vector, \
    smex_jacobian_from_vector, smex_optimized_mse_jacobian

class Smex(MicroStructModel):
    """SMEX / Neurite Exchange Imaging accounting for Finite Pulses model for diffusion MRI."""

    # Class attributes
    n_params = 4
    param_names = ["t_ex", "Di", "De", "f"]
    classic_limits = np.array([[1, 150], [0.1, 3.5], [0.1, 3.5], [0.1, 0.9]])
    grid_search_nb_points = [15, 12, 8, 8]
    has_rician_mean_correction = False

    def __init__(self, param_lim=classic_limits, invert_tex=False):
        super().__init__(name='SMEX')
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
        return smex_signal_from_vector(parameters, acq_parameters.b, acq_parameters.td, acq_parameters.small_delta)

    @classmethod
    def get_jacobian(cls, parameters, acq_parameters):
        """Get jacobian from single Ground Truth."""
        return smex_jacobian_from_vector(parameters, acq_parameters.b, acq_parameters.td, acq_parameters.small_delta)

    @classmethod
    def get_hessian(cls, parameters, acq_parameters):
        return None


    # Optimized methods for Non-linear Least Squares
    @classmethod
    def get_mse_jacobian(cls, parameters, acq_parameters, signal_gt):
        """Get jacobian of Mean Square Error from single Ground Truth."""
        return smex_optimized_mse_jacobian(parameters, acq_parameters, signal_gt)
