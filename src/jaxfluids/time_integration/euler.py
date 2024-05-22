from typing import List

import jax.numpy as jnp
from jax import Array

from jaxfluids.time_integration.time_integrator import TimeIntegrator

class Euler(TimeIntegrator):
    """First-order explicit Euler time integration scheme
    """
    def __init__(
            self,
            nh: int,
            inactive_axes: List,
            offset: int = 0
            ) -> None:
            
        super(Euler, self).__init__(nh, inactive_axes, offset)
        self.no_stages = 1
        self.timestep_multiplier = (1.0,)
        self.timestep_increment_factor = (1.0,) 

    def prepare_buffer_for_integration(
            self,
            buffer: Array,
            init_buffer: Array,
            stage: int
            ) -> Array:
        pass

    def integrate(
            self,
            buffer: Array,
            rhs: Array,
            timestep: float,
            stage: int,
            ) -> Array:
        timestep = timestep * self.timestep_multiplier[stage]
        buffer = self.integrate_buffer(buffer, rhs, timestep)
        return buffer


