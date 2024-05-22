from typing import Dict, Tuple, Union, List
import jax.numpy as jnp
from jax import Array

from jaxfluids.config import precision
from jaxfluids.domain.domain_information import DomainInformation 
from jaxfluids.materials.material_manager import MaterialManager
from jaxfluids.equation_manager import EquationManager

from jaxfluids.stencils import WENO1

class PositivityLimiterInterpolation:
    """The PositivityLimiterInterpolation class implementes functionality
    which ensures that reconstructed states are physically admissible.
    """
    
    def __init__(
            self,
            domain_information: DomainInformation,
            material_manager: MaterialManager,
            equation_manager: EquationManager,
            is_logging: bool = False
            ) -> None:
        self.eps = precision.get_interpolation_limiter_eps()
        
        self.equation_manager = equation_manager
        self.material_manager = material_manager
        self.domain_information = domain_information
        self.is_logging = is_logging
        
        equation_information = equation_manager.equation_information
        self.equation_type = equation_information.equation_type
        self.mass_slices = equation_information.mass_slices
        self.mass_ids = equation_information.mass_ids
        self.vf_slices = equation_information.vf_slices
        self.energy_ids = equation_information.energy_ids
    
        self.weno1_stencil = WENO1(
            nh=domain_information.nh_conservatives, 
            inactive_axes=domain_information.inactive_axes,)
    
    def compute_positivity_preserving_interpolation_xi(
            self,
            primitives: Array,
            primitives_xi_j: Array,
            j: int,
            cell_sizes: List,
            axis: int,
            apertures: Tuple[Array] = None
            ) -> Tuple[Array, Array, int]:
        """Limits the reconstructed values left or right of the cell-faces (i.e.,
        primitives_xi_j) to first order. This is done for 
        reconstructed values with

        - negative (phasic) densities
        - negative pressures
        - negative volume fractions

        :param primitives: Buffer of primitive variables
        :type primitives: Array
        :param primitives_xi_j: Buffer of reconstructed primitives left or right of the cell face
        :type primitives_xi_L: Array
        :param j: Integer indicating reconstruction left or right of cell face
        :type j: int
        :param cell_sizes: List of cell sizes
        :type cell_sizes: List
        :param axis: Integer indicating the axis direction, i.e. (0,1,2)
        :type axis: int
        :return: Returns reconstructed conservatives and primitives (left or right of cell face)
        :rtype: Tuple[Array, Array, Array, Array, int]
        """

        cell_state_xi_safe_j = self.weno1_stencil.reconstruct_xi(
            primitives, axis, j, dx=cell_sizes[axis])
        counter = 0

        if self.equation_type == "DIFFUSE-INTERFACE-5EQM":
            alpharho_j = primitives_xi_j[self.mass_slices]
            alpha_j = primitives_xi_j[self.vf_slices]

            mask = jnp.where(
                (alpharho_j < self.eps.density              ).any(axis=0) | 
                (alpha_j    < self.eps.volume_fraction      ).any(axis=0) | 
                (alpha_j    > 1.0 - self.eps.volume_fraction).any(axis=0) , 0, 1)
            counter += jnp.sum(1 - mask)    # TODO check in parallel

            primitives_xi_j = primitives_xi_j * mask + cell_state_xi_safe_j * (1 - mask)

            alpha_j = primitives_xi_j[self.vf_slices]
            pressure_j = primitives_xi_j[self.energy_ids]
            pb_j = self.material_manager.get_background_pressure(alpha_j)

            # OPTION 1 - CHECK PRESSURE DIRECTLY
            mask = jnp.where(pressure_j + pb_j < self.eps.pressure, 0, 1)
            counter += jnp.sum(1 - mask)    # TODO check in parallel

            # OPTION 2 - CHECK VIA INTERNAL ENERGY
            # alpharho_j = primitives_xi_j[self.mass_slices]
            # rho_j = self.material_manager.get_density(primitives_xi_j)
            # rhoe_j = rho_j * self.material_manager.get_specific_energy(pressure_j, rho=rho_j, alpha_i=alpha_j)

            # mask = jnp.where(rhoe_j - pb_j < self.eps.pressure, 0, 1)
            # counter += jnp.sum(1 - mask)    # TODO check in parallel

            primitives_xi_j = primitives_xi_j * mask + cell_state_xi_safe_j * (1 - mask)

        elif self.equation_type == "SINGLE-PHASE":
            # CHECK DENSITY
            rho_j = primitives_xi_j[self.mass_ids]
            mask = jnp.where(rho_j < self.eps.density, 0, 1)
            counter += jnp.sum(1 - mask)    # TODO check in parallel

            primitives_xi_j = primitives_xi_j * mask + cell_state_xi_safe_j * (1 - mask)

            # CHECK ENERGY / PRESSURE
            pb = self.material_manager.get_background_pressure()
            p_j = primitives_xi_j[self.energy_ids]
            
            # OPTION 1 - CHECK PRESSURE DIRECTLY
            mask = jnp.where(p_j + pb < self.eps.pressure, 0, 1)

            # OPTION 2 - CHECK VIA INTERNAL ENERGY
            # rhoe_j = rho_j * self.material_manager.get_specific_energy(p_j, rho=rho_j)
            # mask_j = jnp.where(rhoe_j - pb < self.eps.pressure, 0, 1)

            counter += jnp.sum(1 - mask)    # TODO check in parallel
            primitives_xi_j = primitives_xi_j * mask + cell_state_xi_safe_j * (1 - mask)

        elif self.equation_type in ("TWO-PHASE-LS", "SINGLE-PHASE-SOLID-LS"):
            rho_j = primitives_xi_j[self.mass_ids]
            p_j = primitives_xi_j[self.energy_ids]
            pb = self.material_manager.get_background_pressure()
            mask = jnp.where( (p_j + pb < self.eps.pressure) | (rho_j < self.eps.density), 1, 0)
            primitives_xi_j = primitives_xi_j * (1 - mask) + cell_state_xi_safe_j * mask
            if self.is_logging:
                nhx_,nhy_,nhz_ = self.domain_information.domain_slices_geometry
                apertures_xi = apertures[axis][nhx_,nhy_,nhz_]
                apertures_xi_mask = jnp.where(apertures_xi > 0.0, 1, 0)
                counter += jnp.sum(mask * jnp.stack([apertures_xi_mask, 1 - apertures_xi_mask], axis=0))

        else:
            raise NotImplementedError

        conservative_xi_j = self.equation_manager.get_conservatives_from_primitives(primitives_xi_j)

        return conservative_xi_j, primitives_xi_j, counter

    def compute_positivity_preserving_thinc_interpolation_xi(
            self,
            primitives: Array,
            primitives_xi_j: Array,
            j: int,
            cell_sizes: List,
            axis: int,
            ) -> Tuple[Array, Array, int]:
        """Interpolation limiter for THINC reconstructed cell face values.
        THINC guarantees that the reconstructed variables conservative are admissible, 
        however, square of speed of sound is not necessarily positive.

        :param primitives: Buffer of primitive variables
        :type primitives: Array
        :param primitives_xi_j: Buffer of reconstructed primitives left or right of the cell face
        :type primitives_xi_L: Array
        :param j: Integer indicating reconstruction left or right of cell face
        :type j: int
        :param cell_sizes: List of cell sizes
        :type cell_sizes: List
        :param axis: Integer indicating the axis direction, i.e. (0,1,2)
        :type axis: int
        :return: Returns reconstructed conservatives and primitives (left or right of cell face)
        :rtype: Tuple[Array, Array, Array, Array, int]
        """

        cell_state_xi_safe_j = self.weno1_stencil.reconstruct_xi(
            primitives, axis, j, dx=cell_sizes[axis])
        counter = 0
        
        if self.equation_type == "DIFFUSE-INTERFACE-5EQM":
            alpharho_j = primitives_xi_j[self.mass_slices]
            alpha_j = primitives_xi_j[self.vf_slices]

            # mask = jnp.where(
            #     (alpharho_j < self.eps.density              ).any(axis=0) | 
            #     (alpha_j    < self.eps.volume_fraction      ).any(axis=0) | 
            #     (alpha_j    > 1.0 - self.eps.volume_fraction).any(axis=0) , 0, 1)
            # counter += jnp.sum(1 - mask)    # TODO check in parallel

            # primitives_xi_j = primitives_xi_j * mask + cell_state_xi_safe_j * (1 - mask)

            # print("THINC DEBUG: MIN PRIM 0 = ", jnp.min(primitives[0]))
            # print("THINC DEBUG: MIN PRIM 1 = ", jnp.min(primitives[1]))
            # print("THINC DEBUG: MIN PRIM -2 = ", jnp.min(primitives[-2]))
            # print("THINC DEBUG: MIN PRIM -1 = ", jnp.min(primitives[-1]))
            # print("THINC DEBUG: MIN ALPHARHO_0 = ", jnp.min(alpharho_j[0]))
            # print("THINC DEBUG: MIN ALPHARHO_1 = ", jnp.min(alpharho_j[1]))
            # print("THINC DEBUG: MIN ALPHA_0 = ", jnp.min(alpha_j))
            # input()

            alpha_j = primitives_xi_j[self.vf_slices]
            p_j = primitives_xi_j[self.energy_ids]
            pb_j = self.material_manager.get_background_pressure(alpha_j)

            # OPTION 1 - CHECK PRESSURE DIRECTLY
            mask = jnp.where(p_j + pb_j < self.eps.pressure, 0, 1)
            counter += jnp.sum(1 - mask)    # TODO check in parallel

            # import matplotlib.pyplot as plt
            # fig, ax = plt.subplots(ncols=3, sharex=True, sharey=True)
            # ax = ax.flatten()
            # ax[0].imshow(jnp.squeeze(p_j + pb_j))
            # ax[1].imshow(jnp.squeeze((p_j + pb_j) < 0))
            # ax[2].imshow(jnp.squeeze(primitives_xi_j[vf_slices]))
            # plt.show()

            # OPTION 2 - CHECK VIA INTERNAL ENERGY
            # alpharho_j = primitives_xi_j[self.mass_slices]
            # rho_j = self.material_manager.get_density(primitives_xi_j)
            # rhoe_j = rho_j * self.material_manager.get_specific_energy(p_j, rho=rho_j, alpha_i=alpha_j)

            # mask = jnp.where(rhoe_j - pb_j < self.eps.pressure, 0, 1)
            # counter += jnp.sum(1 - mask)    # TODO check in parallel

            primitives_xi_j = primitives_xi_j * mask + cell_state_xi_safe_j * (1 - mask)

        conservative_xi_j = self.equation_manager.get_conservatives_from_primitives(primitives_xi_j)

        return conservative_xi_j, primitives_xi_j, counter
