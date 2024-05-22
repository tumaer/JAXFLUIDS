from abc import ABC, abstractmethod

import jax.numpy as jnp
from jax import Array

from jaxfluids.domain.domain_information import DomainInformation
from jaxfluids.config import precision

class ShockSensor(ABC):
    """Abstract base class for shock sensors. Shock sensors indicate 
    the presence of discontinuities by a marker.
    """

    def __init__(self, domain_information: DomainInformation) -> None:

        self.eps = precision.get_eps()
        self.domain_information = domain_information
        self.cell_sizes = self.domain_information.get_local_cell_sizes()
        self.active_axes_indices = [{"x": 0, "y": 1, "z": 2}[axis] for axis in domain_information.active_axes]
    
    @abstractmethod
    def compute_sensor_function(self):
        """Computes the sensor function which is a marker (0/1)
        indicating the presence of shock discontinuities.

        Implementation in child classes.
        """
        pass