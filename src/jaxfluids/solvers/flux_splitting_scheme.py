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

import jax.numpy as jnp

from jaxfluids.domain_information import DomainInformation
from jaxfluids.materials.material_manager import MaterialManager
from jaxfluids.solvers.riemann_solvers.eigendecomposition import Eigendecomposition
from jaxfluids.stencils.spatial_reconstruction import SpatialReconstruction
from jaxfluids.utilities import get_fluxes_xi

class FluxSplittingScheme:
    """Base class for the Flux-Splitting Scheme. The flux-splitting schemes
    transforms conservative variables and physical fluxes into the characteristic
    space, performs a flux-splitting in characteristic space, and transforms
    the final flux back to physical space.

    The eigenvalues - which are used according to the user-specified flux-splitting -
    determine the numerical flux.

    Details are given in Bezgin, Buhendwa, Adams - 2022 - JAX-FLUIDS.
    """

    def __init__(self, material_manager: MaterialManager, domain_information: DomainInformation, 
        flux_splitting: str, reconstruction_stencil: SpatialReconstruction) -> None:
        self.material_manager = material_manager

        self.reconstruction_stencil = reconstruction_stencil
        self.reconstruction_stencil.set_slices_stencil()

        self.flux_splitting = flux_splitting

        self.eigendecomposition = Eigendecomposition(self.material_manager, self.reconstruction_stencil._stencil_size, domain_information, flux_splitting=flux_splitting)

        self.cell_sizes = domain_information.cell_sizes

    def compute_fluxes_xi(self, primes: jnp.ndarray, cons: jnp.ndarray, axis: int, **kwargs) -> jnp.ndarray:
        """Computes the numerical flux in axis direction.

        :param primes: Buffer of primitive variables
        :type primes: jnp.ndarray
        :param cons: Buffer of primitive variables
        :type cons: jnp.ndarray
        :param axis: Spatial direction along which flux is calculated.
        :type axis: int
        :return: Numerical flux in axis direction.
        :rtype: jnp.ndarray
        """
        physical_flux = get_fluxes_xi(primes, cons, axis)

        stencil_cons_window     = self.eigendecomposition.get_stencil_window(cons, axis=axis) 
        stencil_prime_window    = self.eigendecomposition.get_stencil_window(primes, axis=axis) 
        physical_flux_window    = self.eigendecomposition.get_stencil_window(physical_flux, axis=axis)

        right_eigs, left_eigs, eigvals = self.eigendecomposition.eigendecomp_cons(stencil_prime_window, axis=axis)

        char      = self.eigendecomposition.transformtochar(stencil_cons_window, left_eigs, axis=axis)
        char_flux = self.eigendecomposition.transformtochar(physical_flux_window, left_eigs, axis=axis)

        positive_char_flux = 0.5 * (char_flux +  self.eigendecomposition.transformtochar(char, eigvals, axis))
        negative_char_flux = 0.5 * (char_flux -  self.eigendecomposition.transformtochar(char, eigvals, axis))

        char_flux_xi_L = self.reconstruction_stencil.reconstruct_xi(positive_char_flux, axis, 0, dx=self.cell_sizes[axis])
        char_flux_xi_R = self.reconstruction_stencil.reconstruct_xi(negative_char_flux, axis, 1, dx=self.cell_sizes[axis])

        char_flux_xi = char_flux_xi_L + char_flux_xi_R

        fluxes_xi = self.eigendecomposition.transformtophysical(char_flux_xi, right_eigs)

        return fluxes_xi