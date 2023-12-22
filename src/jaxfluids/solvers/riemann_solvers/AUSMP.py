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

class AUSMP(RiemannSolver):
    """AUSM+ Scheme - M Liou - 1996
    Advetion-Upstream Method Plus according to Liou.
    """

    def __init__(self, material_manager: MaterialManager, signal_speed: Callable) -> None:
        super().__init__(material_manager, signal_speed)

        self.interface_speed_of_sound = "ARITHMETIC" 
        self.alpha = 3.0 / 16.0
        self.beta  = 1.0 / 8.0

    def solve_riemann_problem_xi(self, primes_L: jnp.ndarray, primes_R: jnp.ndarray, 
        cons_L: jnp.ndarray, cons_R: jnp.ndarray, axis: int, **kwargs) -> jnp.ndarray:
        phi_left  = self.get_phi(primes_L, cons_L)
        phi_right = self.get_phi(primes_R, cons_R)

        speed_of_sound_left  = self.material_manager.get_speed_of_sound(p = primes_L[4], rho = primes_L[0])
        speed_of_sound_right = self.material_manager.get_speed_of_sound(p = primes_R[4], rho = primes_R[0])
        
        if self.interface_speed_of_sound == "CRITICAL": # Eq. 40
            a_star_L  = jnp.sqrt( 2.0 * (self.material_manager.gamma - 1.0) / (self.material_manager.gamma + 1.0) * phi_left[4] / phi_left[0] )
            a_star_R  = jnp.sqrt( 2.0 * (self.material_manager.gamma - 1.0) / (self.material_manager.gamma + 1.0) * phi_right[4] / phi_right[0] )
            a_tilde_L = a_star_L * jnp.where(a_star_L > jnp.abs(primes_L[axis+1]), 1.0, 1.0 / jnp.abs(primes_L[axis+1]))
            a_tilde_R = a_star_R * jnp.where(a_star_R > jnp.abs(primes_R[axis+1]), 1.0, 1.0 / jnp.abs(primes_R[axis+1]))
            speed_of_sound_ausm = jnp.minimum(a_tilde_L, a_tilde_R)
        
        if self.interface_speed_of_sound == "ARITHMETIC":   # Eq. 41a
            speed_of_sound_ausm = 0.5 * (speed_of_sound_left + speed_of_sound_right) 
        
        if self.interface_speed_of_sound == "SQRT": # Eq. 41b
            speed_of_sound_ausm = jnp.sqrt(speed_of_sound_left * speed_of_sound_right)

        # Eq. A1
        M_l = primes_L[axis+1] / speed_of_sound_ausm
        M_r = primes_R[axis+1] / speed_of_sound_ausm

        # Eq. 19
        M_plus  = jnp.where(jnp.abs(M_l) >= 1, 0.5 * (M_l + jnp.abs(M_l)),  0.25 * (M_l + 1.0) * (M_l + 1.0) + self.beta * (M_l * M_l - 1.0) * (M_l * M_l - 1.0))
        M_minus = jnp.where(jnp.abs(M_r) >= 1, 0.5 * (M_r - jnp.abs(M_r)), -0.25 * (M_r - 1.0) * (M_r - 1.0) - self.beta * (M_r * M_r - 1.0) * (M_r * M_r - 1.0))  

        # Eq. A2
        M_ausm = M_plus + M_minus
        M_ausm_plus  = 0.5 * (M_ausm + jnp.abs(M_ausm))
        M_ausm_minus = 0.5 * (M_ausm - jnp.abs(M_ausm))

        # Eq. 21
        P_plus  = jnp.where(jnp.abs(M_l) >= 1.0, 0.5 * (1 + jnp.sign(M_l)), 0.25 * (M_l + 1) * (M_l + 1.0) * (2.0 - M_l) + self.alpha * M_l * (M_l * M_l - 1.0) * (M_l * M_l - 1.0))
        P_minus = jnp.where(jnp.abs(M_r) >= 1.0, 0.5 * (1 - jnp.sign(M_r)), 0.25 * (M_r - 1) * (M_r - 1.0) * (2.0 + M_r) - self.alpha * M_r * (M_r * M_r - 1.0) * (M_r * M_r - 1.0))  
        # Eq. A2
        pressure_ausm = P_plus * primes_L[4] + P_minus * primes_R[4]
   
        # Eq. A3
        fluxes_xi = speed_of_sound_ausm * (M_ausm_plus * phi_left + M_ausm_minus * phi_right)
        fluxes_xi = fluxes_xi.at[axis+1].add(pressure_ausm)

        return fluxes_xi

    def get_phi(self, primes: jnp.ndarray, cons: jnp.ndarray) -> jnp.ndarray:
        """Computes the phi vector from primitive and conservative variables
        in which energy is replaced by enthalpy.
        phi = [rho, rho * velX, rho * velY, rho * velZ, H]

        :param primes: Buffer of primitive variables.
        :type primes: jnp.ndarray
        :param cons: Buffer of conservative variables.
        :type cons: jnp.ndarray
        :return: Buffer of phi variable.
        :rtype: jnp.ndarray
        """
        rho =  cons[0] 
        rhou = cons[1] 
        rhov = cons[2] 
        rhow = cons[3] 
        ht   = cons[4] + primes[4]
        phi = jnp.stack([rho, rhou, rhov, rhow, ht], axis=0)
        return phi
