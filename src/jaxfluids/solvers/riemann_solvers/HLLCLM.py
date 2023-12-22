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

class HLLCLM(RiemannSolver):
    """HLLCLM Riemann Solver
    Fleischmann et al. 2020
    """

    def __init__(self, material_manager: MaterialManager, signal_speed: Callable) -> None:
        super().__init__(material_manager, signal_speed)
        self.s_star = compute_sstar
        self.Ma_limit = 0.1

    def solve_riemann_problem_xi(self, cell_state_L: jnp.ndarray, cell_state_R: jnp.ndarray, 
        conservative_L: jnp.ndarray, conservative_R: jnp.ndarray, axis: int, **kwargs) -> jnp.ndarray:
        fluxes_left = get_fluxes_xi(cell_state_L, conservative_L, axis)
        fluxes_right = get_fluxes_xi(cell_state_R, conservative_R, axis)
        
        speed_of_sound_left = self.material_manager.get_speed_of_sound(p = cell_state_L[4], rho = cell_state_L[0])
        speed_of_sound_right = self.material_manager.get_speed_of_sound(p = cell_state_R[4], rho = cell_state_R[0])

        wave_speed_simple_L, wave_speed_simple_R = self.signal_speed(cell_state_L[axis+1], cell_state_R[axis+1], speed_of_sound_left, speed_of_sound_right, 
            rho_L = cell_state_L[0], rho_R = cell_state_R[0], p_L = cell_state_L[4], p_R = cell_state_R[4], gamma = self.material_manager.gamma)
        wave_speed_contact = self.s_star(cell_state_L[axis+1], cell_state_R[axis+1], cell_state_L[4], cell_state_R[4], cell_state_L[0], cell_state_R[0],
            wave_speed_simple_L, wave_speed_simple_R)

        ''' Toro 10.73 '''
        pre_factor_L = (wave_speed_simple_L - cell_state_L[axis+1]) / (wave_speed_simple_L - wave_speed_contact) * cell_state_L[0]
        pre_factor_R = (wave_speed_simple_R - cell_state_R[axis+1]) / (wave_speed_simple_R - wave_speed_contact) * cell_state_R[0]

        # ORDERING !!! 
        shear_dirs = np.roll([1, 2, 3], 3 - (axis+1))[:2]
        u_star_L = [pre_factor_L, pre_factor_L, pre_factor_L, pre_factor_L, pre_factor_L * (conservative_L[4] / conservative_L[0] + (wave_speed_contact - cell_state_L[axis+1]) * (wave_speed_contact + cell_state_L[4] / cell_state_L[0] / (wave_speed_simple_L - cell_state_L[axis+1]) )) ]
        u_star_L[axis+1] *= wave_speed_contact
        u_star_L[shear_dirs[0]] *= cell_state_L[shear_dirs[0]]
        u_star_L[shear_dirs[1]] *= cell_state_L[shear_dirs[1]]
        u_star_L = jnp.stack(u_star_L)

        u_star_R = [pre_factor_R, pre_factor_R, pre_factor_R, pre_factor_R, pre_factor_R * (conservative_R[4] / conservative_R[0] + (wave_speed_contact - cell_state_R[axis+1]) * (wave_speed_contact + cell_state_R[4] / cell_state_R[0] / (wave_speed_simple_R - cell_state_R[axis+1]) )) ]
        u_star_R[axis+1] *= wave_speed_contact
        u_star_R[shear_dirs[0]] *= cell_state_R[shear_dirs[0]]
        u_star_R[shear_dirs[1]] *= cell_state_R[shear_dirs[1]]
        u_star_R = jnp.stack(u_star_R)

        ''' Fleischmann et al. - 2020 - Eq (23 - 25) '''
        Ma_local = jnp.maximum(jnp.abs(cell_state_L[axis+1] / speed_of_sound_left), jnp.abs(cell_state_R[axis+1] / speed_of_sound_right))
        phi = jnp.sin(jnp.minimum(1.0, Ma_local / self.Ma_limit) * jnp.pi * 0.5)
        wave_speed_left  = phi * wave_speed_simple_L
        wave_speed_right = phi * wave_speed_simple_R

        ''' Fleischmann et al. - 2020 - Eq. (19) '''
        flux_star = 0.5 * (fluxes_left + fluxes_right) + \
                    0.5 * (wave_speed_left * (u_star_L - conservative_L) + jnp.abs(wave_speed_contact) * (u_star_L - u_star_R) + wave_speed_right * (u_star_R - conservative_R) )


        ''' Fleischmann et al. - 2020 - Eq. (18) '''
        fluxes_xi = 0.5 * (1 + jnp.sign(wave_speed_simple_L)) * fluxes_left + \
                    0.5 * (1 - jnp.sign(wave_speed_simple_R)) * fluxes_right + \
                    0.25 * (1 - jnp.sign(wave_speed_simple_L)) * (1 + jnp.sign(wave_speed_simple_R)) * flux_star

        return fluxes_xi
