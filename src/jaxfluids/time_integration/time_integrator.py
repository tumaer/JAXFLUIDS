#*------------------------------------------------------------------------------*
#* JAX-FLUIDS -                                                                 *
#*                                                                              *
#* A fully-differentiable CFD solver for compressible two-phase flows.          *
#* Copyright (C) 2022  Deniz A. Bezgin, Aaron B. Buhendwa, Nikolaus A. Adams    *
#*                                                                              *
#* This program is free software: you can redistribute it and/or modify         *
#* it under the terms of the GNU General Public License as published by         *
#* the Free Software Foundation, either version 3 of the License, or            *
#* (at your option) any later version.                                          *
#*                                                                              *
#* This program is distributed in the hope that it will be useful,              *
#* but WITHOUT ANY WARRANTY; without even the implied warranty of               *
#* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the                *
#* GNU General Public License for more details.                                 *
#*                                                                              *
#* You should have received a copy of the GNU General Public License            *
#* along with this program.  If not, see <https://www.gnu.org/licenses/>.       *
#*                                                                              *
#*------------------------------------------------------------------------------*
#*                                                                              *
#* CONTACT                                                                      *
#*                                                                              *
#* deniz.bezgin@tum.de // aaron.buhendwa@tum.de // nikolaus.adams@tum.de        *
#*                                                                              *
#*------------------------------------------------------------------------------*
#*                                                                              *
#* Munich, April 15th, 2022                                                     *
#*                                                                              *
#*------------------------------------------------------------------------------*

from abc import ABC, abstractmethod
from functools import partial
from typing import List

import jax
import jax.numpy as jnp

class TimeIntegrator(ABC):
    """Abstract base class for explicit time integration schemes.
    All time intergration schemes are derived from TimeIntegrator.
    """

    def __init__(self, nh: int, inactive_axis: List) -> None:
        
        self.no_stages          = None
        self.nhx                = jnp.s_[:] if "x" in inactive_axis else jnp.s_[nh:-nh]    
        self.nhy                = jnp.s_[:] if "y" in inactive_axis else jnp.s_[nh:-nh]    
        self.nhz                = jnp.s_[:] if "z" in inactive_axis else jnp.s_[nh:-nh]

        self.timestep_multiplier = ()
        self.timestep_increment_factor = ()

    def integrate_conservatives(self, cons: jnp.ndarray, rhs: jnp.ndarray, timestep: float) -> jnp.ndarray:
        """Integrates the conservative variables.

        :param cons: conservative variables buffer before integration
        :type cons: jnp.ndarray
        :param rhs: right-hand side buffer 
        :type rhs: jnp.ndarray
        :param timestep: timestep adjusted according to sub-stage in Runge-Kutta
        :type timestep: float
        :return: conservative variables buffer after integration
        :rtype: DeviceArray
        """
        cons = cons.at[..., self.nhx, self.nhy, self.nhz].add(timestep * rhs)
        return cons

    @abstractmethod
    def integrate(self, cons: jnp.ndarray, rhs: jnp.ndarray, timestep: float, stage: int) -> jnp.ndarray:
        """Wrapper function around integrate_conservatives. Adjusts the timestep
        according to current RK stage and calls integrate_conservatives.
        Implementation in child class.

        :param cons: conservative variables buffer before integration
        :type cons: jnp.ndarray
        :param rhs: right-hand side buffer 
        :type rhs: jnp.ndarray
        :param timestep: timestep to be integrated
        :type timestep: float
        :return: conservative variables buffer after integration
        :rtype: DeviceArray
        """
        pass

    def prepare_buffer_for_integration(self, cons: jnp.ndarray, init: jnp.ndarray, stage: int) -> jnp.ndarray:
        """In multi-stage Runge-Kutta methods, prepares the buffer for integration.
        Implementation in child class.

        :param cons: Buffer of conservative variables.
        :type cons: jnp.ndarray
        :param init: Initial conservative buffer.
        :type init: jnp.ndarray
        :param stage: Current stage of the RK time integrator.
        :type stage: int
        :return: Sum of initial buffer and current buffer.
        :rtype: jnp.ndarray
        """
        pass