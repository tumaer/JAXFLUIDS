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

from typing import Tuple

import jax.numpy as jnp

from jaxfluids.domain_information import DomainInformation
from jaxfluids.materials.material_manager import MaterialManager
from jaxfluids.shock_sensor.ducros import Ducros

from jaxfluids.iles.ALDM_WENO1 import ALDM_WENO1
from jaxfluids.iles.ALDM_WENO3 import ALDM_WENO3
from jaxfluids.iles.ALDM_WENO5 import ALDM_WENO5

class ALDM:
    """ Adaptive Local Deconvolution Method - ALDM - Hickel et al. 2014

    ALDM is a numerical scheme for computation of convective fluxes. It consits
    of a combined reconstruction and flux-function. ALDM is optimized to model
    subgrid-scale terms in underresolved LES. 

    ALDM consists of a  
    1) cell face reconstruction based on a convex sum of 
    adapted WENO1, WENO3, and WENO5 
    
    2) flux-function with adjusted dissipation of SGS modeling 
    and low Mach number consistency.
    """

    def __init__(self, domain_information: DomainInformation, material_manager: MaterialManager) -> None:
        self.material_manager = material_manager

        self.domain_information     = domain_information
        self.active_axis            = domain_information.active_axis
        self.active_axis_indices    = [{"x": 0, "y": 1, "z": 2}[axis] for axis in self.active_axis]

        self._sigma_rho     =   0.615
        self._sigma_rhou    =   0.125
        self._sigma_rhoe    =   0.615

        self.ALDM_WENO1 = ALDM_WENO1(nh=self.domain_information.nh_conservatives, inactive_axis=self.domain_information.inactive_axis)
        self.ALDM_WENO3 = ALDM_WENO3(nh=self.domain_information.nh_conservatives, inactive_axis=self.domain_information.inactive_axis)
        self.ALDM_WENO5 = ALDM_WENO5(nh=self.domain_information.nh_conservatives, inactive_axis=self.domain_information.inactive_axis)

        self.shock_sensor = Ducros(domain_information)

    def compute_fluxes_xi(self, prime: jnp.ndarray, cons: jnp.ndarray, axis: int, **kwargs) -> jnp.ndarray:
        """Computes the numerical flux in the axis direction.

        :param prime: Buffer of primitive variables.
        :type prime: jnp.ndarray
        :param cons: Buffer of conservative variables.
        :type cons: jnp.ndarray
        :param axis: Spatial direction in which the flux is computed.
        :type axis: int
        :return: Numerical flux in specified direction.
        :rtype: jnp.ndarray
        """
        # Evaluate shock sensor
        fs = self.shock_sensor.compute_sensor_function(prime[1:4], axis)

        # Solution adaptive alpha parameters
        alpha_1 = (1.0 - fs) / 3.0
        alpha_2 = (1.0 - fs) / 3.0
        alpha_3 = 1.0 - alpha_1 - alpha_2

        # Reconstruct phi at the cell face
        phi         = self.compute_phi(prime, cons)
        phi_L, p3_L = self.reconstruct_xi(phi, alpha_1, alpha_2, alpha_3, fs, axis, 0)
        phi_R, p3_R = self.reconstruct_xi(phi, alpha_1, alpha_2, alpha_3, fs, axis, 1)

        return self.solve_riemann_problem_xi(phi_L, phi_R, p3_L, p3_R, alpha_3, fs, axis)

    def solve_riemann_problem_xi(self, phi_L: jnp.ndarray, phi_R: jnp.ndarray, 
        p3_L: jnp.ndarray, p3_R: jnp.ndarray, alpha_3: jnp.ndarray, 
        fs: jnp.ndarray, axis: int) -> jnp.ndarray:
        """Solves the Riemann problem, i.e., calculates the numerical flux, in 
        the direction specified by axis.

        phi = [rho, u1, u2, u3, p, rho_e]
        p3_K is third-order pressure reconstruction

        :param phi_L: Phi vector of left neighboring state
        :type phi_L: jnp.ndarray
        :param phi_R: Phi vector of right neighboring state
        :type phi_R: jnp.ndarray
        :param p3_L: Third-order pressure reconstruction of left neighboring state
        :type p3_L: jnp.ndarray
        :param p3_R: Third-order pressure reconstruction of right neighboring state
        :type p3_R: jnp.ndarray
        :param alpha_3: Third-order reconstruction weight
        :type alpha_3: jnp.ndarray
        :param fs: Shock sensor.
        :type fs: jnp.ndarray
        :param axis: Spatial direction along which flux is calculated.
        :type axis: int
        :return: Numerical flux in axis drection.
        :rtype: jnp.ndarray
        """
        
        phi_delta = phi_R - phi_L
        phi_sum   = phi_R + phi_L

        # Interface pressure and transport velocity
        # Eq. (34a)
        p_star = 0.5 * phi_sum[4]

        speed_of_sound_left  = self.material_manager.get_speed_of_sound(p = phi_L[4], rho = phi_L[0])
        speed_of_sound_right = self.material_manager.get_speed_of_sound(p = phi_R[4], rho = phi_R[0])
        c = jnp.maximum(speed_of_sound_left, speed_of_sound_right)
        # Eq. (34b)
        u_star = 0.5 * phi_sum[axis+1] - alpha_3 / c * (p3_R - p3_L) / phi_sum[0] 

        # Dissipation matrix 
        R_diss = jnp.stack([
            self._sigma_rho  * jnp.abs(phi_delta[axis+1]) + fs * 0.5 * (jnp.abs(u_star) + jnp.abs(phi_delta[axis+1])),
            self._sigma_rhou * jnp.abs(phi_delta[1])      + fs * 0.5 * (jnp.abs(u_star) + jnp.abs(phi_delta[axis+1])),
            self._sigma_rhou * jnp.abs(phi_delta[2])      + fs * 0.5 * (jnp.abs(u_star) + jnp.abs(phi_delta[axis+1])),
            self._sigma_rhou * jnp.abs(phi_delta[3])      + fs * 0.5 * (jnp.abs(u_star) + jnp.abs(phi_delta[axis+1])),
            self._sigma_rhoe * jnp.abs(phi_delta[axis+1]) + fs * 0.5 * (jnp.abs(u_star) + jnp.abs(phi_delta[axis+1])),
        ])        

        # Flux computation
        flux_rho  = u_star * 0.5 * (phi_R[0] + phi_L[0]) - R_diss[0] * (phi_R[0] - phi_L[0])
        flux_ui   = [
            flux_rho * 0.5 * (phi_R[1] + phi_L[1]) - R_diss[1] * 0.5 * (phi_R[0] + phi_L[0]) * (phi_R[1] - phi_L[1]),
            flux_rho * 0.5 * (phi_R[2] + phi_L[2]) - R_diss[2] * 0.5 * (phi_R[0] + phi_L[0]) * (phi_R[2] - phi_L[2]),
            flux_rho * 0.5 * (phi_R[3] + phi_L[3]) - R_diss[3] * 0.5 * (phi_R[0] + phi_L[0]) * (phi_R[3] - phi_L[3]),
        ]
        flux_rhoe = u_star * 0.5 * (phi_R[5] + phi_L[5]) \
            + 0.5 * (phi_R[1] + phi_L[1]) * (flux_ui[0] - 0.25 * (phi_R[1] + phi_L[1]) * flux_rho) \
            + 0.5 * (phi_R[2] + phi_L[2]) * (flux_ui[1] - 0.25 * (phi_R[2] + phi_L[2]) * flux_rho) \
            + 0.5 * (phi_R[3] + phi_L[3]) * (flux_ui[2] - 0.25 * (phi_R[3] + phi_L[3]) * flux_rho) \
            - R_diss[4] * (phi_R[5] - phi_L[5])

        fluxes_xi = [flux_rho, flux_ui[0], flux_ui[1], flux_ui[2], flux_rhoe]
        
        # Add pressure flux
        fluxes_xi[axis+1] += p_star
        fluxes_xi[4]    += u_star * p_star

        return jnp.stack(fluxes_xi)

    def reconstruct_xi(self, phi: jnp.ndarray, alpha_1: jnp.ndarray, alpha_2: jnp.ndarray, alpha_3: jnp.ndarray, 
        fs: jnp.ndarray, axis: int, j: int, dx: float = None) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Reconstructs the phi vector along the axis direction. Reconstruction is done
        via a convex combination of modified WENO1, WENO3 and WENO5.

        :param phi: Buffer of phi vector.
        :type phi: jnp.ndarray
        :param alpha_1: First-order reconstruction weight.
        :type alpha_1: jnp.ndarray
        :param alpha_2: Second-order reconstruction weight.
        :type alpha_2: jnp.ndarray
        :param alpha_3: Third-order reconstruction weight.
        :type alpha_3: jnp.ndarray
        :param fs: Shock sensor.
        :type fs: jnp.ndarray
        :param axis: Spatial direction along which reconstruction is done.
        :type axis: int
        :param j: Bit indicating whether reconstruction is left (j=0) or right (j=1)
            of the cell face.
        :type j: int
        :param dx: Vector of cell sizes in axis direction, defaults to None
        :type dx: float, optional
        :return: Reconstructed phi vector and reconstructed third-oder pressure value.
        :rtype: Tuple[jnp.ndarray, jnp.ndarray]
        """
        cell_state_1 = self.ALDM_WENO1.reconstruct_xi(phi, axis, j)             # WENO 1
        cell_state_2 = self.ALDM_WENO3.reconstruct_xi(phi, axis, j)             # WENO 3
        cell_state_3 = self.ALDM_WENO5.reconstruct_xi(phi, axis, j, fs=fs)      # WENO 5
        
        cell_state_xi_j = alpha_1 * cell_state_1 + alpha_2 * cell_state_2 + alpha_3 * cell_state_3

        return cell_state_xi_j, cell_state_3[4]

    def compute_phi(self, primes: jnp.ndarray, cons: jnp.ndarray) -> jnp.ndarray:
        """Computes the phi vector which is the quantity that is reconstructed
        in the ALDM scheme.

        phi vector notation different from paper,
            \bar{phi} = {\bar{rho}, \bar{u1}, \bar{u2}, \bar{u3}, \bar{p}, \bar{rho_e}}

        :param primes: Buffer of primitive variables.
        :type primes: jnp.ndarray
        :param cons: Buffer of conservative variables.
        :type cons: jnp.ndarray
        :return: Buffer of the phi vector.
        :rtype: jnp.ndarray
        """
        rho_e = cons[4] - 0.5 * primes[0] * (primes[1] * primes[1] + primes[2] * primes[2] + primes[3] * primes[3])
        phi = jnp.stack([primes[0], primes[1], primes[2], primes[3], primes[4], rho_e], axis=0)
        return phi
