from functools import partial
from typing import List, Tuple

import jax
import jax.numpy as jnp 

from jaxfluids.equation_information import EquationInformation
from jaxfluids.materials.material_manager import MaterialManager
from jaxfluids.math.sum_consistent import sum3_consistent

Array = jax.Array

class EquationManager:
    """ The EquationManager stores information on the system of equations that is being solved.
    Besides providing indices for the different variables, the EquationManager provides the 
    equation-specific primitive/conservative conversions and the flux calculation.
    """

    def __init__(
            self,
            material_manager: MaterialManager,
            equation_information: EquationInformation
            ) -> None:

        self.material_manager = material_manager
        self.equation_information = equation_information

        self.ids_mass = equation_information.ids_mass 
        self.vel_ids = equation_information.ids_velocity 
        self.ids_energy = equation_information.ids_energy 
        self.ids_volume_fraction = equation_information.ids_volume_fraction
        self.ids_species = equation_information.ids_species
        
        self.s_mass = equation_information.s_mass
        self.vel_slices = equation_information.s_velocity
        self.s_energy = equation_information.s_energy
        self.s_volume_fraction = equation_information.s_volume_fraction
        self.s_species = equation_information.s_species
        
        self.equation_type = equation_information.equation_type
        self.diffuse_interface_model = equation_information.diffuse_interface_model
        self.no_fluids = equation_information.no_fluids

    def get_conservatives_from_primitives(self, primitives: Array) -> Array:
        """Converts primitive variables to conservative ones.
        Wrapper for 5 equation DIM and single-phase/level-set model.

        :param primitives: _description_
        :type primitives: Array
        :return: _description_
        :rtype: Array
        """
        if self.equation_type == "DIFFUSE-INTERFACE-5EQM":
            velocity_vec = primitives[self.vel_slices]
            rho  = self.material_manager.get_density(primitives)
            momentum_vec = rho * velocity_vec       # = rho * u_i

            e = self.material_manager.get_specific_energy(
                p=primitives[self.ids_energy], 
                rho=rho, alpha_i=primitives[self.s_volume_fraction])
            # E = rho * (1/2 u^2 + e)
            E = rho * (0.5 * sum3_consistent(*jnp.square(velocity_vec)) + e)
            
            # TODO
            # rhoe = self.material_manager.get_specific_energy(
            #     p       = primitives[self.ids_energy], 
            #     rho     = 1.0,
            #     alpha_i = primitives[self.s_volume_fraction])
            # E    = rhoe + rho * 0.5 * ( primitives[self.vel_ids[0]] * primitives[self.vel_ids[0]] + primitives[self.vel_ids[1]] * primitives[self.vel_ids[1]] + primitives[self.vel_ids[2]] * primitives[self.vel_ids[2]])
            
            conservatives = jnp.stack([
                *primitives[self.s_mass],
                *momentum_vec, E,
                *primitives[self.s_volume_fraction]],
            axis=0)

        elif self.equation_type == "DIFFUSE-INTERFACE-4EQM":
            velocity_vec = primitives[self.vel_slices]
            rho = self.material_manager.get_density(primitives)
            momentum_vec = rho * velocity_vec   # = rho * u_i

            e = self.material_manager.get_specific_energy(
                p=primitives[self.ids_energy],
                alpha_rho_i=primitives[self.s_mass])
            # E = rho * (1/2 u^2 + e)
            E = rho * (0.5 * sum3_consistent(*jnp.square(velocity_vec)) + e)
            
            conservatives = jnp.stack([
                *primitives[self.s_mass],
                *momentum_vec, E],
            axis=0)

        elif self.equation_type in ("SINGLE-PHASE", "TWO-PHASE-LS"):
            rho = primitives[self.ids_mass] # = rho
            velocity_vec = primitives[self.vel_slices]

            e = self.material_manager.get_specific_energy(primitives[self.ids_energy], rho)
            momentum_vec = rho * velocity_vec   # = rho * u_i

            E = rho * (0.5 * sum3_consistent(*jnp.square(velocity_vec)) + e)  # E = rho * (1/2 u^2 + e)
            conservatives = jnp.stack([rho, *momentum_vec, E], axis=0)
        
        else:
            raise NotImplementedError

        return conservatives

    def get_primitives_from_conservatives(
            self,
            conservatives: Array,
            fluid_mask: Array = None
            ) -> Array:
        """Converts conservative variables to primitive variables.

        :param conservatives: Buffer of conservative variables
        :type conservatives: Array
        :return: Buffer of primitive variables
        :rtype: Array
        """           

        if self.equation_type == "DIFFUSE-INTERFACE-5EQM":
            rho = self.material_manager.get_density(conservatives)
            one_rho = 1.0 / rho
            velocity_vec = conservatives[self.vel_slices] * one_rho # u_i = (rho * u)_i / rho
            e = conservatives[self.ids_energy] * one_rho - 0.5 * sum3_consistent(*jnp.square(velocity_vec))
            pressure = self.material_manager.get_pressure(
                e=e, rho=rho,
                alpha_i=conservatives[self.s_volume_fraction]) # p = (gamma-1) * ( E - 1/2 * (rho*u) * u)

            # TODO
            # rhoe = conservatives[self.ids_energy] - 0.5 * (conservatives[self.vel_ids[0]] * conservatives[self.vel_ids[0]] + conservatives[self.vel_ids[1]] * conservatives[self.vel_ids[1]] + conservatives[self.vel_ids[2]] * conservatives[self.vel_ids[2]]) / rho
            # p   = self.material_manager.get_pressure(
            #     e       = rhoe, 
            #     rho     = 1.0,
            #     alpha_i = conservatives[self.s_volume_fraction])

            primitives = jnp.stack([
                *conservatives[self.s_mass],
                *velocity_vec, pressure,
                *conservatives[self.s_volume_fraction]],
            axis=0)

        elif self.equation_type == "DIFFUSE-INTERFACE-4EQM":
            rho = self.material_manager.get_density(conservatives)
            one_rho = 1.0 / rho
            velocity_vec = conservatives[self.vel_slices] * one_rho # u_i = (rho * u)_i / rho

            e = conservatives[self.ids_energy] * one_rho - 0.5 * sum3_consistent(*jnp.square(velocity_vec))
            pressure = self.material_manager.get_pressure(
                e=e, alpha_rho_i=conservatives[self.s_mass])
            # p = (gamma-1) * ( E - 1/2 * (rho*u) * u)

            # TODO
            # rhoe   = conservatives[self.ids_energy] - 0.5 * (conservatives[self.vel_ids[0]] * conservatives[self.vel_ids[0]] + conservatives[self.vel_ids[1]] * conservatives[self.vel_ids[1]] + conservatives[self.vel_ids[2]] * conservatives[self.vel_ids[2]]) / rho
            # p   = self.material_manager.get_pressure(
            #     e       = rhoe, 
            #     rho     = 1.0,
            #     alpha_i = conservatives[self.s_volume_fraction])

            primitives = jnp.stack([
                *conservatives[self.s_mass],
                *velocity_vec, pressure,], axis=0)

        elif self.equation_type in ("SINGLE-PHASE", "TWO-PHASE-LS"):
            rho = conservatives[self.ids_mass]  # rho = rho
            one_rho = 1.0 / rho
            velocity_vec = conservatives[self.vel_slices] * one_rho # u_i = (rho * u)_i / rho
            e = conservatives[self.ids_energy] * one_rho - 0.5 * sum3_consistent(*jnp.square(velocity_vec))
            pressure = self.material_manager.get_pressure(e, rho) # p = (gamma-1) * ( E - 1/2 * (rho*u) * u)

            primitives = jnp.stack([rho, *velocity_vec, pressure], axis=0)

        else:
            raise NotImplementedError

        return primitives

    def get_fluxes_xi(
            self,
            primitives: Array,
            conservatives: Array,
            axis: int
            ) -> Array:
        """Computes the physical flux in a specified spatial direction.
        Cf. Eq. (3.65) in Toro.

        :param primitives: Buffer of primitive variables
        :type primitives: Array
        :param conservatives: Buffer of conservative variables
        :type conservatives: Array
        :param axis: Spatial direction along which fluxes are calculated
        :type axis: int
        :return: Physical fluxes in axis direction
        :rtype: Array
        """

        vel_id_axis = self.vel_ids[axis]

        if self.equation_type == "DIFFUSE-INTERFACE-5EQM":
            rho_alpha_ui = primitives[vel_id_axis] * conservatives[self.s_mass] # (u_i * alpha_1 rho_1 )
            rho_ui_u1 = conservatives[vel_id_axis] * primitives[self.vel_ids[0]]     # (rho u_i) * u_1
            rho_ui_u2 = conservatives[vel_id_axis] * primitives[self.vel_ids[1]]     # (rho u_i) * u_2
            rho_ui_u3 = conservatives[vel_id_axis] * primitives[self.vel_ids[2]]     # (rho u_i) * u_3
            ui_Ep = primitives[vel_id_axis] * (conservatives[self.ids_energy] + primitives[self.ids_energy])
            if axis == 0:
                rho_ui_u1 += primitives[self.ids_energy]
            elif axis == 1:
                rho_ui_u2 += primitives[self.ids_energy]
            elif axis == 2:
                rho_ui_u3 += primitives[self.ids_energy]
            
            alpha_ui = primitives[vel_id_axis] * conservatives[self.s_volume_fraction]

            flux_xi = jnp.stack([
                *rho_alpha_ui, rho_ui_u1,
                rho_ui_u2, rho_ui_u3, ui_Ep,
                *alpha_ui], axis=0)

        elif self.equation_type == "DIFFUSE-INTERFACE-4EQM":
            rho_alpha_ui = primitives[vel_id_axis] * conservatives[self.s_mass] # (u_i * alpha_1 rho_1 )
            rho_ui_u1 = conservatives[vel_id_axis] * primitives[self.vel_ids[0]]     # (rho u_i) * u_1
            rho_ui_u2 = conservatives[vel_id_axis] * primitives[self.vel_ids[1]]     # (rho u_i) * u_2
            rho_ui_u3 = conservatives[vel_id_axis] * primitives[self.vel_ids[2]]     # (rho u_i) * u_3
            ui_Ep = primitives[vel_id_axis] * (conservatives[self.ids_energy] + primitives[self.ids_energy])
            if axis == 0:
                rho_ui_u1 += primitives[self.ids_energy]
            elif axis == 1:
                rho_ui_u2 += primitives[self.ids_energy]
            elif axis == 2:
                rho_ui_u3 += primitives[self.ids_energy]
            
            flux_xi = jnp.stack([
                *rho_alpha_ui, rho_ui_u1,
                rho_ui_u2, rho_ui_u3, ui_Ep,],
                axis=0)

        elif self.equation_type in ("SINGLE-PHASE", "TWO-PHASE-LS"):
            rho_ui = conservatives[axis+1] # (rho u_i)
            rho_ui_u1 = conservatives[axis+1] * primitives[self.vel_ids[0]] # (rho u_i) * u_1
            rho_ui_u2 = conservatives[axis+1] * primitives[self.vel_ids[1]] # (rho u_i) * u_2
            rho_ui_u3 = conservatives[axis+1] * primitives[self.vel_ids[2]] # (rho u_i) * u_3
            ui_Ep = primitives[axis+1] * (conservatives[self.ids_energy] + primitives[self.ids_energy])
            if axis == 0:
                rho_ui_u1 += primitives[self.ids_energy]
            elif axis == 1:
                rho_ui_u2 += primitives[self.ids_energy]
            elif axis == 2:
                rho_ui_u3 += primitives[self.ids_energy]

            flux_xi = jnp.stack([
                rho_ui, rho_ui_u1, rho_ui_u2, rho_ui_u3,
                ui_Ep], axis=0)
        
        else:
            raise NotImplementedError

        return flux_xi
