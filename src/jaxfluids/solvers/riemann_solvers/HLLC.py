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

from jaxfluids.utilities import get_fluxes_xi
from jaxfluids.solvers.riemann_solvers.riemann_solver import RiemannSolver
from jaxfluids.solvers.riemann_solvers.signal_speeds import compute_sstar
from jaxfluids.materials.material_manager import MaterialManager

class HLLC(RiemannSolver):
    """HLLC Riemann Solver
    Toro et al. 1994
    """

    def __init__(self, material_manager: MaterialManager, signal_speed: Callable) -> None:
        super().__init__(material_manager, signal_speed)
        self.s_star = compute_sstar

        # MINOR AXIS DIRECTIONS 
        self.minor = [
            [2, 3],
            [3, 1],
            [1, 2],
        ]

    def solve_riemann_problem_xi(self, primes_L: jnp.ndarray, primes_R: jnp.ndarray, 
        cons_L: jnp.ndarray, cons_R: jnp.ndarray, axis: int, **kwargs) -> jnp.ndarray:
        fluxes_left = get_fluxes_xi(primes_L, cons_L, axis)
        fluxes_right = get_fluxes_xi(primes_R, cons_R, axis)
        
        speed_of_sound_left = self.material_manager.get_speed_of_sound(p = primes_L[4], rho = primes_L[0])
        speed_of_sound_right = self.material_manager.get_speed_of_sound(p = primes_R[4], rho = primes_R[0])

        wave_speed_simple_L, wave_speed_simple_R = self.signal_speed(primes_L[axis+1], primes_R[axis+1], speed_of_sound_left, speed_of_sound_right, 
            rho_L = primes_L[0], rho_R = primes_R[0], p_L = primes_L[4], p_R = primes_R[4], gamma = self.material_manager.gamma)
        wave_speed_contact = self.s_star(primes_L[axis+1], primes_R[axis+1], primes_L[4], primes_R[4], primes_L[0], primes_R[0],
            wave_speed_simple_L, wave_speed_simple_R)

        wave_speed_left  = jnp.minimum( wave_speed_simple_L, 0.0 )
        wave_speed_right = jnp.maximum( wave_speed_simple_R, 0.0 )

        ''' Toro 10.73 '''
        pre_factor_L = (wave_speed_simple_L - primes_L[axis+1]) / (wave_speed_simple_L - wave_speed_contact) * primes_L[0]
        pre_factor_R = (wave_speed_simple_R - primes_R[axis+1]) / (wave_speed_simple_R - wave_speed_contact) * primes_R[0]

        u_star_L = [pre_factor_L, pre_factor_L, pre_factor_L, pre_factor_L, pre_factor_L * (cons_L[4] / cons_L[0] + (wave_speed_contact - primes_L[axis+1]) * (wave_speed_contact + primes_L[4] / primes_L[0] / (wave_speed_simple_L - primes_L[axis+1]) )) ]
        u_star_L[axis+1]             *= wave_speed_contact
        u_star_L[self.minor[axis][0]] *= primes_L[self.minor[axis][0]]
        u_star_L[self.minor[axis][1]] *= primes_L[self.minor[axis][1]]
        u_star_L = jnp.stack(u_star_L)

        u_star_R = [pre_factor_R, pre_factor_R, pre_factor_R, pre_factor_R, pre_factor_R * (cons_R[4] / cons_R[0] + (wave_speed_contact - primes_R[axis+1]) * (wave_speed_contact + primes_R[4] / primes_R[0] / (wave_speed_simple_R - primes_R[axis+1]) )) ]
        u_star_R[axis+1]             *= wave_speed_contact
        u_star_R[self.minor[axis][0]] *= primes_R[self.minor[axis][0]]
        u_star_R[self.minor[axis][1]] *= primes_R[self.minor[axis][1]]
        u_star_R = jnp.stack(u_star_R)

        ''' Toro 10.72 '''
        flux_star_L = fluxes_left + wave_speed_left * (u_star_L - cons_L)
        flux_star_R = fluxes_right + wave_speed_right * (u_star_R - cons_R)

        ''' Kind of Toro 10.71 '''
        fluxes_xi = 0.5 * (1 + jnp.sign(wave_speed_contact)) * flux_star_L + 0.5 * (1 - jnp.sign(wave_speed_contact)) * flux_star_R
        return fluxes_xi