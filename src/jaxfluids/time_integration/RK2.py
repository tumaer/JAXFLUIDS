from typing import List

import jax.numpy as jnp
from jax import Array

from jaxfluids.time_integration.time_integrator import TimeIntegrator

class RungeKutta2(TimeIntegrator):
    """2nd-order TVD RK2 scheme
    """
    def __init__(
            self,
            nh: int,
            inactive_axes: List,
            offset: int = 0
            ) -> None:
            
        super(RungeKutta2, self).__init__(nh, inactive_axes, offset)
        self.no_stages = 2
        self.timestep_multiplier = (1.0, 0.5)
        self.timestep_increment_factor = (1.0, 1.0) 

    def prepare_buffer_for_integration(
            self,
            buffer: Array,
            init_buffer: Array,
            stage: int
            ) -> Array:
        """ u_cons = 0.5 u^n + 0.5 u^* """
        return 0.5*buffer + 0.5*init_buffer

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

