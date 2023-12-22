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

from typing import Callable, Dict

import jax
import jax.numpy as jnp

from jaxfluids.materials.material_manager import MaterialManager
from jaxfluids.solvers.riemann_solvers.riemann_solver import RiemannSolver
from jaxfluids.utilities import get_fluxes_xi

class RusanovNN(RiemannSolver):
    """RusanovNN Riemann Solver which is optimied in the JAX-FLUIDS paper.
    The dissipation is calculated by a neural network. See details in 
    Bezgin et al. 2022.
    """

    def __init__(self, material_manager: MaterialManager, signal_speed: Callable) -> None:
        super().__init__(material_manager, signal_speed)

    def solve_riemann_problem_xi(self, primes_L: jnp.ndarray, primes_R: jnp.ndarray, 
        cons_L: jnp.ndarray, cons_R: jnp.ndarray, axis: int, 
        ml_parameters_dict: Dict, ml_networks_dict: Dict, **kwargs) -> jnp.ndarray:        
        params = ml_parameters_dict["riemannsolver"]
        net    = ml_networks_dict["riemannsolver"]

        # PHYSICAL FLUXES
        fluxes_left  = get_fluxes_xi(primes_L, cons_L, axis)
        fluxes_right = get_fluxes_xi(primes_R, cons_R, axis)

        # BUILD NEURAL NETWORK INPUTS
        speed_of_sound_left  = self.material_manager.get_speed_of_sound(p = primes_L[4], rho = primes_L[0])
        speed_of_sound_right = self.material_manager.get_speed_of_sound(p = primes_R[4], rho = primes_R[0])
        speed_of_sound = 0.5 * (speed_of_sound_left + speed_of_sound_right)

        delta_vel = jnp.abs(primes_R[axis+1] - primes_L[axis+1])
        mean_vel  = 0.5 * (primes_L[axis+1] + primes_R[axis+1])

        entropy_L = primes_L[4] / (primes_L[0])**self.material_manager.gamma
        entropy_R = primes_R[4] / (primes_R[0])**self.material_manager.gamma
        delta_s = jnp.abs(entropy_R - entropy_L)

        # EVALUATE NEURAL NETWORK FOR RUSANOVNN DISSIPATION
        vec = jnp.stack([delta_vel, mean_vel, speed_of_sound, delta_s])
        dissipation = net.apply(params, vec)

        # FINAL FLUX
        fluxes_xi = 0.5 * ( (fluxes_left + fluxes_right) - dissipation * (cons_R - cons_L) )
        return fluxes_xi
