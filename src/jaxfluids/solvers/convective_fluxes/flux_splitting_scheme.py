from __future__ import annotations
from typing import Dict, Union, TYPE_CHECKING, Tuple

import jax.numpy as jnp
from jax import Array

from jaxfluids.domain.domain_information import DomainInformation
from jaxfluids.materials.material_manager import MaterialManager
from jaxfluids.solvers.convective_fluxes.convective_flux_solver import ConvectiveFluxSolver
from jaxfluids.solvers.riemann_solvers.eigendecomposition import Eigendecomposition
from jaxfluids.stencils.spatial_reconstruction import SpatialReconstruction
from jaxfluids.equation_manager import EquationManager
if TYPE_CHECKING:
    from jaxfluids.data_types.numerical_setup.conservatives import ConvectiveFluxesSetup

class FluxSplittingScheme(ConvectiveFluxSolver):
    """Base class for the Flux-Splitting Scheme. The flux-splitting schemes
    transforms conservative variables and physical fluxes into the characteristic
    space, performs a flux-splitting in characteristic space, and transforms
    the final flux back to physical space.

    The eigenvalues - which are used according to the user-specified flux-splitting -
    determine the numerical flux.

    Details are given in Bezgin, Buhendwa, Adams - 2022 - JAX-FLUIDS.
    """

    def __init__(
            self,
            convective_fluxes_setup: ConvectiveFluxesSetup,
            material_manager: MaterialManager,
            domain_information: DomainInformation,
            equation_manager: EquationManager,
            **kwargs
            ) -> None:

        super(FluxSplittingScheme, self).__init__(
            convective_fluxes_setup, material_manager, domain_information, equation_manager)

        reconstruction_stencil = convective_fluxes_setup.reconstruction_stencil
        reconstruction_stencil: SpatialReconstruction = reconstruction_stencil(
            nh = domain_information.nh_conservatives, 
            inactive_axes = domain_information.inactive_axes,
            is_mesh_stretching = domain_information.is_mesh_stretching,
            cell_sizes = domain_information.get_global_cell_sizes_halos())

        self.reconstruction_stencil = reconstruction_stencil
        self.reconstruction_stencil.set_slices_stencil()

        self.eigendecomposition = Eigendecomposition(
            material_manager = self.material_manager,
            stencil_size = self.reconstruction_stencil._stencil_size, 
            domain_information = domain_information,
            equation_information = equation_manager.equation_information, 
            flux_splitting = convective_fluxes_setup.flux_splitting,
            frozen_state = convective_fluxes_setup.frozen_state)

    def compute_flux_xi(
            self,
            primitives: Array,
            conservatives: Array,
            axis: int,
            ml_parameters_dict: Union[Dict, None] = None,
            ml_networks_dict: Union[Dict, None] = None,
            **kwargs,
            ) -> Tuple[Array, None, None, None]:
        """Computes the numerical flux in axis direction.

        :param primitives: Buffer of primitive variables
        :type primitives: Array
        :param conservatives: Buffer of primitive variables
        :type conservatives: Array
        :param axis: Spatial direction along which flux is calculated.
        :type axis: int
        :return: Numerical flux in axis direction.
        :rtype: Array
        """
        cell_sizes = self.domain_information.get_device_cell_sizes()
        right_eigs, left_eigs, eigvals = self.eigendecomposition.eigendecomposition_conservatives(primitives, axis=axis)
        stencil_cons_window  = self.eigendecomposition.get_stencil_window(conservatives, axis=axis) 
        char = self.eigendecomposition.transformtochar(stencil_cons_window, left_eigs, axis=axis)
        physical_flux = self.equation_manager.get_fluxes_xi(primitives, conservatives, axis)
        physical_flux_window = self.eigendecomposition.get_stencil_window(physical_flux, axis=axis)
        char_flux = self.eigendecomposition.transformtochar(physical_flux_window, left_eigs, axis=axis)

        positive_char_flux = 0.5 * (char_flux + self.eigendecomposition.transformtochar(char, eigvals, axis))
        negative_char_flux = 0.5 * (char_flux - self.eigendecomposition.transformtochar(char, eigvals, axis))

        char_flux_xi_L = self.reconstruction_stencil.reconstruct_xi(
            positive_char_flux, 
            axis, 
            0, 
            dx=cell_sizes[axis]
        )
        char_flux_xi_R = self.reconstruction_stencil.reconstruct_xi(
            negative_char_flux, 
            axis, 
            1, 
            dx=cell_sizes[axis]
        )

        char_flux_xi = char_flux_xi_L + char_flux_xi_R

        fluxes_xi = self.eigendecomposition.transformtophysical(char_flux_xi, right_eigs)

        return fluxes_xi, None, None, None, None