from typing import List

import jax.numpy as jnp
from jax import Array

from jaxfluids.time_integration.time_integrator import TimeIntegrator
class RungeKutta2_LS4(TimeIntegrator):
    """2nd-order 4-step LS RK scheme

    u^{1}   = u^{n} + 0.11   * dt * L(u^{n})
    u^{2}   = u^{n} + 0.2766 * dt * L(u^{1})
    u^{3}   = u^{n} + 0.5    * dt * L(u^{2})
    u^{n+1} = u^{n} + 1.0    * dt * L(u^{3})

    """
    def __init__(
            self,
            nh: int,
            inactive_axes: List,
            offset: int = 0
            ) -> None:

        super(RungeKutta2_LS4, self).__init__(nh, inactive_axes, offset)
        self.no_stages = 4
        self.timestep_multiplier = (0.11, 0.2766, 0.5, 1.0) 
        self.timestep_increment_factor = (0.11, 0.2766, 0.5, 1.0) 
        self.conservatives_multiplier = [ (0.0, 1.0), (0.0, 1.0), (0.0, 1.0) ]

    def prepare_buffer_for_integration(
            self,
            buffer: Array,
            init_buffer: Array,
            stage: int
            ) -> Array:
        """stage 1: u_cons = 3/4 u^n + 1/4 u^*
        stage 2: u_cons = 1/3 u^n + 2/3 u^** 

        :param conservatives: _description_
        :type conservatives: Array
        :param init: _description_
        :type init: Array
        :param stage: _description_
        :type stage: int
        :return: _description_
        :rtype: Array
        """
        buffer = self.conservatives_multiplier[stage-1][0]*buffer \
            + self.conservatives_multiplier[stage-1][1]*init_buffer
        return buffer

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
