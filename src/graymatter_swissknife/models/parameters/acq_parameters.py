import numpy as np
from abc import ABC


class AcquisitionParameters(ABC):
    """Acquisition parameters."""

    def __init__(self, b, td, small_delta=None):
        """Initialize the acquisition parameters."""
        # b-values or shells
        self.b = np.array(b)
        # diffusion time Δ
        self.td = np.array(td)
        # gradient duration 𝛿
        if small_delta is not None:
            self.small_delta = small_delta
        else:
            self.small_delta = None
        # resulting number of acquisitions
        self.nb_acq = np.prod(self.b.shape)
        # resulting number of dimension of acquisition shape
        self.ndim = self.b.ndim
        # resulting shape of acquisition
        self.shape = self.b.shape


class AcquisitionParametersException(Exception):
    """Handle exceptions related to acquisition parameters."""

    pass


class InvalidAcquisitionParameters(AcquisitionParametersException):
    """Handle exceptions related to wrong acquisition parameters."""

    pass
