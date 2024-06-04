import numpy as np
from abc import ABC
import logging


class AcquisitionParameters(ABC):
    """Acquisition parameters."""

    def __init__(self, b, delta, small_delta=None):
        """Initialize the acquisition parameters."""
        # b-values or shells
        self.b = np.array(b)
        # diffusion time Δ
        self.delta = np.array(delta)
        # gradient duration 𝛿
        if small_delta is not None:
            self.small_delta = small_delta
            self.td = self.delta - self.small_delta / 3
        else:
            logging.warning("The gradient duration 𝛿 is not provided. The diffusion time td=Δ-δ/3 will be equal to Δ only.")
            self.small_delta = None
            self.td = self.delta
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
