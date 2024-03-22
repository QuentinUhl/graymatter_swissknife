from abc import ABC

class MicroStructModel(ABC):
    """Microstructure model."""

    def __init__(self, name):
        """Initialize the name of the microstructure model."""
        self.name = name

    def __str__(self):
        """Representation of the model."""
        return self.name


class MicroStructModelException(Exception):
    """Handle exceptions related to microstructure models."""

    pass


class InvalidMicroStructModel(MicroStructModelException):
    """Handle exceptions related to wrong microstructure models."""

    pass
