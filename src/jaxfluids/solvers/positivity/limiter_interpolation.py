from typing import Dict, Tuple, Union, List

import jax
import jax.numpy as jnp

from jaxfluids.config import precision
from jaxfluids.data_types.numerical_setup import NumericalSetup
from jaxfluids.domain.domain_information import DomainInformation 
from jaxfluids.equation_manager import EquationManager
from jaxfluids.materials.material_manager import MaterialManager

from jaxfluids.stencils.reconstruction.shock_capturing.weno import WENO1

Array = jax.Array

class PositivityLimiterInterpolation:
    """The PositivityLimiterInterpolation class implementes functionality
    which ensures that reconstructed states are physically admissible.
    """
    
    def __init__(
            self,
            domain_information: DomainInformation,
            material_manager: MaterialManager,
            equation_manager: EquationManager,
            numerical_setup: NumericalSetup,
            ) -> None:
        self.eps = precision.get_interpolation_limiter_eps()
                    
        self.equation_manager = equation_manager
        self.equation_information = equation_manager.equation_information
        self.material_manager = material_manager
        self.domain_information = domain_information
        
        self.equation_type = self.equation_information.equation_type
        self.s_mass = self.equation_information.s_mass
        self.ids_mass = self.equation_information.ids_mass
        self.ids_volume_fraction = self.equation_information.ids_volume_fraction
        self.s_volume_fraction = self.equation_information.s_volume_fraction
        self.ids_energy = self.equation_information.ids_energy
    
        self.limit_velocity = numerical_setup.conservatives.positivity.limit_velocity
        if not self.limit_velocity:
            self.limit_ids = self.get_limit_ids()

        self.first_order_stencil = WENO1(
            nh=domain_information.nh_conservatives, 
            inactive_axes=domain_information.inactive_axes,)
    

    def get_limit_ids(self,):
        if self.equation_type == "DIFFUSE-INTERFACE-5EQM":
            limit_ids = jnp.array([*self.ids_mass, self.ids_energy, *self.ids_volume_fraction])
        elif self.equation_type == "DIFFUSE-INTERFACE-4EQM":
            limit_ids = jnp.array([*self.ids_mass, self.ids_energy])
        elif self.equation_type == "SINGLE-PHASE":
            limit_ids = jnp.array([self.ids_mass, self.ids_energy])
        elif self.equation_type == "TWO-PHASE-LS":
            limit_ids = jnp.array([self.ids_mass, self.ids_energy])
        else:
            raise NotImplementedError

        return limit_ids


    def apply_limiter(self, mask: Array, primitives_xi_j: Array, primitives_weno1_xi_j: Array) -> Array:
        if self.limit_velocity:
            primitives_xi_j = primitives_xi_j * (1 - mask) + primitives_weno1_xi_j * mask
        else:
            primitives_xi_j = primitives_xi_j.at[self.limit_ids].set(
                primitives_xi_j[self.limit_ids] * (1 - mask) + primitives_weno1_xi_j[self.limit_ids] * mask
            )
    
        return primitives_xi_j


    def limit_interpolation_xi(
            self,
            primitives: Array,
            primitives_xi_j: Array,
            j: int,
            cell_sizes: Tuple[Array],
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
        :param cell_sizes: Tuple of cell sizes
        :type cell_sizes: Tuple[Array]
        :param axis: Integer indicating the axis direction, i.e. (0,1,2)
        :type axis: int
        :return: Returns reconstructed conservatives and primitives (left or right of cell face)
        :rtype: Tuple[Array, Array, Array, Array, int]
        """
        
        primitives_weno1_xi_j = self.first_order_stencil.reconstruct_xi(
            primitives, axis, j, dx=cell_sizes[axis])
        counter = 0

        if self.equation_information.is_solid_levelset:
            nhx_,nhy_,nhz_ = self.domain_information.domain_slices_geometry
            apertures_xi = apertures[axis][nhx_,nhy_,nhz_]
            apertures_xi_mask = jnp.where(apertures_xi > 0.0, 1, 0)
        else:
            apertures_xi_mask = 1

        if self.equation_type == "DIFFUSE-INTERFACE-5EQM":
            alpharho_j = primitives_xi_j[self.s_mass]
            alpha_j = primitives_xi_j[self.s_volume_fraction]

            mask = jnp.where(
                (alpharho_j < self.eps.density              ).any(axis=0) | 
                (alpha_j    < self.eps.volume_fraction      ).any(axis=0) | 
                (alpha_j    > 1.0 - self.eps.volume_fraction).any(axis=0) , 1, 0)
            counter += jnp.sum(mask * apertures_xi_mask)
            primitives_xi_j = self.apply_limiter(mask, primitives_xi_j, primitives_weno1_xi_j)

            alpha_j = primitives_xi_j[self.s_volume_fraction]
            pressure_j = primitives_xi_j[self.ids_energy]
            pb_j = self.material_manager.get_background_pressure(alpha_j)

            # OPTION 1 - CHECK PRESSURE DIRECTLY
            mask = jnp.where(pressure_j + pb_j < self.eps.pressure, 1, 0)

            # OPTION 2 - CHECK VIA INTERNAL ENERGY
            # alpharho_j = primitives_xi_j[self.s_mass]
            # rho_j = self.material_manager.get_density(primitives_xi_j)
            # rhoe_j = rho_j * self.material_manager.get_specific_energy(pressure_j, rho=rho_j, alpha_i=alpha_j)
            # mask = jnp.where(rhoe_j - pb_j < self.eps.pressure, 1, 0)

            counter += jnp.sum(mask * apertures_xi_mask)
            primitives_xi_j = self.apply_limiter(mask, primitives_xi_j, primitives_weno1_xi_j)
        
        elif self.equation_type == "DIFFUSE-INTERFACE-4EQM":
            # 1) CHECK MASSES
            alpharho_j = primitives_xi_j[self.s_mass]
            mask = jnp.where((alpharho_j < self.eps.density).any(axis=0), 1, 0)
            counter += jnp.sum(mask * apertures_xi_mask)
            primitives_xi_j = self.apply_limiter(mask, primitives_xi_j, primitives_weno1_xi_j)

            # 2) CHECK SPEED OF SOUND
            pressure_j = primitives_xi_j[self.ids_energy]
            pb_j = self.material_manager.get_phase_background_pressure()

            # OPTION 1 - CHECK PRESSURE DIRECTLY
            mask = jnp.where((pressure_j + pb_j < self.eps.pressure).any(axis=0), 1, 0)

            # OPTION 2 - CHECK VIA INTERNAL ENERGY
            # alpharho_j = primitives_xi_j[self.s_mass]
            # rho_j = self.material_manager.get_density(primitives_xi_j)
            # rhoe_j = rho_j * self.material_manager.get_specific_energy(pressure_j, rho=rho_j, alpha_i=alpha_j)
            # mask = jnp.where(rhoe_j - pb_j < self.eps.pressure, 1, 0)

            counter += jnp.sum(mask * apertures_xi_mask)
            primitives_xi_j = self.apply_limiter(mask, primitives_xi_j, primitives_weno1_xi_j)

        elif self.equation_type == "SINGLE-PHASE":
            # 1) CHECK DENSITY
            rho_j = primitives_xi_j[self.ids_mass]
            mask = jnp.where(rho_j < self.eps.density, 1, 0)
            counter += jnp.sum(mask * apertures_xi_mask)
            primitives_xi_j = self.apply_limiter(mask, primitives_xi_j, primitives_weno1_xi_j)

            # 2) CHECK ENERGY / PRESSURE
            pb = self.material_manager.get_background_pressure()
            p_j = primitives_xi_j[self.ids_energy]
            
            # OPTION 1 - CHECK PRESSURE DIRECTLY
            mask = jnp.where(p_j + pb < self.eps.pressure, 1, 0)

            # OPTION 2 - CHECK VIA INTERNAL ENERGY
            # rhoe_j = rho_j * self.material_manager.get_specific_energy(p_j, rho=rho_j)
            # mask_j = jnp.where(rhoe_j - pb < self.eps.pressure, 1, 0)

            counter += jnp.sum(mask * apertures_xi_mask)
            primitives_xi_j = self.apply_limiter(mask, primitives_xi_j, primitives_weno1_xi_j)

        elif self.equation_type == "TWO-PHASE-LS":
            rho_j = primitives_xi_j[self.ids_mass]
            p_j = primitives_xi_j[self.ids_energy]
            pb = self.material_manager.get_background_pressure()
            mask = jnp.where((p_j + pb < self.eps.pressure) | (rho_j < self.eps.density), 1, 0)
            primitives_xi_j = self.apply_limiter(mask, primitives_xi_j, primitives_weno1_xi_j)

            nhx_,nhy_,nhz_ = self.domain_information.domain_slices_geometry
            apertures_xi = apertures[axis][nhx_,nhy_,nhz_]
            apertures_xi = jnp.stack([apertures_xi, 1.0 - apertures_xi], axis=0)
            apertures_xi_mask = jnp.where(apertures_xi > 0.0, 1, 0)
            counter += jnp.sum(mask * apertures_xi_mask)

        else:
            raise NotImplementedError

        conservative_xi_j = self.equation_manager.get_conservatives_from_primitives(primitives_xi_j)

        return conservative_xi_j, primitives_xi_j, counter


    def compute_positivity_preserving_thinc_interpolation_xi(
            self,
            primitives: Array,
            primitives_xi_j: Array,
            j: int,
            cell_sizes: Tuple[Array],
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
        :param cell_sizes: Tuple of cell sizes
        :type cell_sizes: Tuple[Array]
        :param axis: Integer indicating the axis direction, i.e. (0,1,2)
        :type axis: int
        :return: Returns reconstructed conservatives and primitives (left or right of cell face)
        :rtype: Tuple[Array, Array, Array, Array, int]
        # TODO get rid of this
        """

        # TODO apertures_mask for is_solid_levelset

        primitives_weno1_xi_j = self.first_order_stencil.reconstruct_xi(
            primitives, axis, j, dx=cell_sizes[axis])
        counter = 0
        
        if self.equation_type == "DIFFUSE-INTERFACE-5EQM":
            alpharho_j = primitives_xi_j[self.s_mass]
            alpha_j = primitives_xi_j[self.s_volume_fraction]

            # mask = jnp.where(
            #     (alpharho_j < self.eps.density              ).any(axis=0) | 
            #     (alpha_j    < self.eps.volume_fraction      ).any(axis=0) | 
            #     (alpha_j    > 1.0 - self.eps.volume_fraction).any(axis=0) , 1, 0)
            # counter += jnp.sum(mask)

            # primitives_xi_j = primitives_xi_j * (1 - mask) + primitives_weno1_xi_j * mask


            alpha_j = primitives_xi_j[self.s_volume_fraction]
            p_j = primitives_xi_j[self.ids_energy]
            pb_j = self.material_manager.get_background_pressure(alpha_j)

            # OPTION 1 - CHECK PRESSURE DIRECTLY
            mask = jnp.where(p_j + pb_j < self.eps.pressure, 1, 0)

            # OPTION 2 - CHECK VIA INTERNAL ENERGY
            # alpharho_j = primitives_xi_j[self.s_mass]
            # rho_j = self.material_manager.get_density(primitives_xi_j)
            # rhoe_j = rho_j * self.material_manager.get_specific_energy(p_j, rho=rho_j, alpha_i=alpha_j)
            # mask = jnp.where(rhoe_j - pb_j < self.eps.pressure, 1, 0)

            counter += jnp.sum(mask)
            primitives_xi_j = primitives_xi_j * (1 - mask) + primitives_weno1_xi_j * mask
        
        elif self.equation_type == "DIFFUSE-INTERFACE-4EQM":
            #TODO 4EQM
            raise NotImplementedError

        else:
            raise NotImplementedError

        conservative_xi_j = self.equation_manager.get_conservatives_from_primitives(primitives_xi_j)

        return conservative_xi_j, primitives_xi_j, counter
