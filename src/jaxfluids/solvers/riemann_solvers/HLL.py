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

from typing import Callable

import jax
import jax.numpy as jnp

from jaxfluids.materials.material_manager import MaterialManager
from jaxfluids.solvers.riemann_solvers.riemann_solver import RiemannSolver
from jaxfluids.utilities import get_fluxes_xi

class HLL(RiemannSolver):
    """HLL Riemann Solver by Harten, Lax and van Leer
    Harten et al. 1983
    """

    def __init__(self, material_manager: MaterialManager, signal_speed: Callable) -> None:
        super().__init__(material_manager, signal_speed)

    def solve_riemann_problem_xi(self, primes_L: jnp.ndarray, primes_R: jnp.ndarray, 
        cons_L: jnp.ndarray, cons_R: jnp.ndarray, axis: int, **kwargs) -> jnp.ndarray:
        fluxes_left = get_fluxes_xi(primes_L, cons_L, axis)
        fluxes_right = get_fluxes_xi(primes_R, cons_R, axis)

        speed_of_sound_left = self.material_manager.get_speed_of_sound(p = primes_L[4], rho = primes_L[0])
        speed_of_sound_right = self.material_manager.get_speed_of_sound(p = primes_R[4], rho = primes_R[0])
        
        wave_speed_simple_L, wave_speed_simple_R = self.signal_speed(primes_L[axis+1], primes_R[axis+1], speed_of_sound_left, speed_of_sound_right, 
            rho_L = primes_L[0], rho_R = primes_R[0], p_L = primes_L[4], p_R = primes_R[4], gamma = self.material_manager.gamma)
        wave_speed_left  = jnp.minimum( wave_speed_simple_L, 0.0 )
        wave_speed_right = jnp.maximum( wave_speed_simple_R, 0.0 )

        fluxes_xi = (wave_speed_right * fluxes_left - wave_speed_left * fluxes_right +
            wave_speed_left * wave_speed_right * ( cons_R - cons_L ) ) / ( wave_speed_right - wave_speed_left + self.eps)
            
        return fluxes_xi