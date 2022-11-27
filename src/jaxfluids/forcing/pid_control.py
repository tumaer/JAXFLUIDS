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

from functools import partial
from typing import List, Tuple

import jax
import jax.numpy as jnp

class PIDControl:
    """Standard PID controller.
    Used for example in the computation of the mass flow forcing in channel flows.

    u = K_s * (K_p * e + K_i * e_int + K_d de/dt)
    """
    
    def __init__(self, K_static: float = 1.0, K_P: float = 1.0, K_I: float = 1.0, K_D: float = 0.0, T_N: float = 0.5, T_V: float = 0.5) -> None:
        
        self.K_static   = K_static
        self.K_P        = K_P
        self.K_I        = K_I
        self.K_D        = K_D

        self.T_N = T_N
        self.T_V = T_V

    @partial(jax.jit, static_argnums=(0))
    def compute_output(self, current_value: float, target_value: float, dt: float, e_old: float, e_int: float) -> Tuple[float, float, float]:
        """Computes the control variable based on a standard PID controller.

        :param current_value: Current value of the control variable.
        :type current_value: float
        :param target_value: Target value for the control variable.
        :type target_value: float
        :param dt: Time step size.
        :type dt: float
        :param e_old: Previous instantaneous error of the control variable.
        :type e_old: float
        :param e_int: Previous integral error of the control variable.
        :type e_int: float
        :return: Updated control variable, updated instantaneous and integral errors
        :rtype: Tuple[float, float, float]
        """
  
        e_new = (target_value - current_value) / (target_value + jnp.finfo(jnp.float64).eps)

        de = (e_new - e_old) * self.T_V / dt
        e_int += e_new * dt / self.T_N
        
        output = self.K_static * (self.K_P * e_new + self.K_I * e_int + self.K_D * de) 

        return output, e_new, e_int