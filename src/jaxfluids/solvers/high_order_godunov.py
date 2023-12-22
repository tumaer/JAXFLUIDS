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

from typing import Dict, Union

import jax.numpy as jnp
import jax

from jaxfluids.domain_information import DomainInformation
from jaxfluids.materials.material_manager import MaterialManager
from jaxfluids.solvers.riemann_solvers.eigendecomposition import Eigendecomposition
from jaxfluids.solvers.riemann_solvers.riemann_solver import RiemannSolver
from jaxfluids.stencils.spatial_reconstruction import SpatialReconstruction
from jaxfluids.stencils.reconstruction.weno1_js import WENO1
from jaxfluids.utilities import get_conservatives_from_primitives, get_primitives_from_conservatives

class HighOrderGodunov:
    """The HighOrderGodunov class implements the flux calculation
    according to the high-order Godunov approach in the finite volume
    framework.

    The calculation of the fluxes consists of three steps:
    1) RECONSTRUCT STATE ON CELL FACE
    2) CONVERT PRIMITIVES TO CONSERVATIVES AND VICE VERSA
    3) SOLVE RIEMANN PROBLEM

    The reconstruction step can be done on primitive or conservative variables
    in either physical or characteristic space. The safe reconstruction guards
    against reconstruction of inadmissible states (e.g., negative pressure or
    density) by resorting to first-order upwind reconstruction in problematic
    cells.

    """

    def __init__(self, material_manager: MaterialManager, domain_information: DomainInformation, 
        riemann_solver: RiemannSolver, reconstruction_stencil: SpatialReconstruction, 
        reconstruction_var: str, is_safe_reconstruction: str) -> None:

        self.material_manager = material_manager

        self.reconstruction_var     = reconstruction_var
        self.reconstruction_stencil = reconstruction_stencil
        self.riemann_solver         = riemann_solver

        self.is_safe_reconstruction = is_safe_reconstruction
        if self.is_safe_reconstruction:
            self.reconstruction_stencil_safe = WENO1(nh=domain_information.nh_conservatives, inactive_axis=domain_information.inactive_axis)

        if self.reconstruction_var in ['CHAR-PRIMITIVE', 'CHAR-CONSERVATIVE']:
            self.reconstruction_stencil.set_slices_stencil()
            self.eigendecomposition = Eigendecomposition(self.material_manager, self.reconstruction_stencil._stencil_size, domain_information)

        self.cell_sizes = domain_information.cell_sizes

    def compute_fluxes_xi(self, primes: jnp.ndarray, cons: jnp.ndarray, axis: int,
        ml_parameters_dict: Union[Dict, None] = None, ml_networks_dict: Union[Dict, None] = None) -> jnp.ndarray:
        """Computes the numerical flux in a specified spatial direction.

        :param primes: Buffer of primitive variables.
        :type primes: jnp.ndarray
        :param cons: Buffer of conservative variables.
        :type cons: jnp.ndarray
        :param axis: Spatial direction along which flux is calculated.
        :type axis: int
        :return: Numerical flux in axis direction.
        :rtype: jnp.ndarray
        """
        if self.reconstruction_var == 'PRIMITIVE':
            primes_xi_L = self.reconstruction_stencil.reconstruct_xi(primes, axis, 0, dx=self.cell_sizes[axis],
                ml_parameters_dict=ml_parameters_dict, ml_networks_dict=ml_networks_dict)
            primes_xi_R = self.reconstruction_stencil.reconstruct_xi(primes, axis, 1, dx=self.cell_sizes[axis],
                ml_parameters_dict=ml_parameters_dict, ml_networks_dict=ml_networks_dict)

            conservative_xi_L = get_conservatives_from_primitives(primes_xi_L, self.material_manager)
            conservative_xi_R = get_conservatives_from_primitives(primes_xi_R, self.material_manager)

        elif self.reconstruction_var == 'CONSERVATIVE':
            conservative_xi_L = self.reconstruction_stencil.reconstruct_xi(cons, axis, 0, dx=self.cell_sizes[axis],
                ml_parameters_dict=ml_parameters_dict, ml_networks_dict=ml_networks_dict)
            conservative_xi_R = self.reconstruction_stencil.reconstruct_xi(cons, axis, 1, dx=self.cell_sizes[axis],
                ml_parameters_dict=ml_parameters_dict, ml_networks_dict=ml_networks_dict)

            primes_xi_L = get_primitives_from_conservatives(conservative_xi_L, self.material_manager)
            primes_xi_R = get_primitives_from_conservatives(conservative_xi_R, self.material_manager)

        elif self.reconstruction_var == 'CHAR-PRIMITIVE':
            stencil_prime_window = self.eigendecomposition.get_stencil_window(primes, axis=axis)

            right_eigs, left_eigs = self.eigendecomposition.eigendecomp_prim(stencil_prime_window, axis)

            char = self.eigendecomposition.transformtochar(stencil_prime_window, left_eigs, axis=axis)

            char_xi_L = self.reconstruction_stencil.reconstruct_xi(char, axis, 0, dx=self.cell_sizes[axis])
            char_xi_R = self.reconstruction_stencil.reconstruct_xi(char, axis, 1, dx=self.cell_sizes[axis])

            primes_xi_L = self.eigendecomposition.transformtophysical(char_xi_L, right_eigs)
            primes_xi_R = self.eigendecomposition.transformtophysical(char_xi_R, right_eigs)
            
            conservative_xi_L = get_conservatives_from_primitives(primes_xi_L, self.material_manager)
            conservative_xi_R = get_conservatives_from_primitives(primes_xi_R, self.material_manager)

        elif self.reconstruction_var == 'CHAR-CONSERVATIVE':
            stencil_cons_window   = self.eigendecomposition.get_stencil_window(cons, axis=axis) 
            stencil_prime_window = self.eigendecomposition.get_stencil_window(primes, axis=axis)

            right_eigs, left_eigs = self.eigendecomposition.eigendecomp_cons(stencil_prime_window, axis=axis)

            char = self.eigendecomposition.transformtochar(stencil_cons_window, left_eigs, axis=axis)
            char_xi_L = self.reconstruction_stencil.reconstruct_xi(char, axis, 0, dx=self.cell_sizes[axis])
            char_xi_R = self.reconstruction_stencil.reconstruct_xi(char, axis, 1, dx=self.cell_sizes[axis])

            conservative_xi_L = self.eigendecomposition.transformtophysical(char_xi_L, right_eigs)
            conservative_xi_R = self.eigendecomposition.transformtophysical(char_xi_R, right_eigs)
            
            primes_xi_L = get_primitives_from_conservatives(conservative_xi_L, self.material_manager)
            primes_xi_R = get_primitives_from_conservatives(conservative_xi_R, self.material_manager)

        if self.is_safe_reconstruction:

            p_L     = primes_xi_L[4]
            p_R     = primes_xi_R[4]
            rho_L   = primes_xi_L[0]
            rho_R   = primes_xi_R[0]

            mask_L = jnp.where(((p_L < 0) | (rho_L < 0)), 0.0, 1.0)
            mask_R = jnp.where(((p_R < 0) | (rho_R < 0)), 0.0, 1.0)

            cell_state_xi_safe_L    = self.reconstruction_stencil_safe.reconstruct_xi(primes, axis, 0, dx=self.cell_sizes[axis])
            cell_state_xi_safe_R    = self.reconstruction_stencil_safe.reconstruct_xi(primes, axis, 1, dx=self.cell_sizes[axis])
            conservative_xi_safe_L  = get_conservatives_from_primitives(primes_xi_L, self.material_manager)
            conservative_xi_safe_R  = get_conservatives_from_primitives(primes_xi_R, self.material_manager)

            primes_xi_L = primes_xi_L*mask_L + cell_state_xi_safe_L*(1 - mask_L)
            primes_xi_R = primes_xi_R*mask_R + cell_state_xi_safe_R*(1 - mask_R)

            conservative_xi_L = conservative_xi_L*mask_L + conservative_xi_safe_L*(1 - mask_L)
            conservative_xi_R = conservative_xi_R*mask_R + conservative_xi_safe_R*(1 - mask_R)

        fluxes_xi = self.riemann_solver.solve_riemann_problem_xi(
            primes_xi_L, primes_xi_R, conservative_xi_L, conservative_xi_R, axis, 
            ml_parameters_dict=ml_parameters_dict, ml_networks_dict=ml_networks_dict)

        return fluxes_xi