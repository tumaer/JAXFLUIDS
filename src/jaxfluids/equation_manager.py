from functools import partial
from typing import List

import jax
import jax.numpy as jnp
from jax import Array 

from jaxfluids.equation_information import EquationInformation
from jaxfluids.materials.material_manager import MaterialManager

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

        self.mass_ids = equation_information.mass_ids 
        self.vel_ids = equation_information.velocity_ids 
        self.energy_ids = equation_information.energy_ids 
        self.vf_ids = equation_information.vf_ids
        self.species_ids = equation_information.species_ids
        
        self.mass_slices = equation_information.mass_slices
        self.vel_slices = equation_information.velocity_slices
        self.energy_slices = equation_information.energy_slices
        self.vf_slices = equation_information.vf_slices
        self.species_slices = equation_information.species_slices
        
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
            rho  = self.material_manager.get_density(primitives)
            rhou = rho * primitives[self.vel_ids[0]] # = rho * u
            rhov = rho * primitives[self.vel_ids[1]] # = rho * v
            rhow = rho * primitives[self.vel_ids[2]] # = rho * w

            e = self.material_manager.get_specific_energy(
                p       = primitives[self.energy_ids], 
                rho     = rho,
                alpha_i = primitives[self.vf_slices])

            E = rho * (
                0.5 * (
                    primitives[self.vel_ids[0]] * primitives[self.vel_ids[0]] \
                    + primitives[self.vel_ids[1]] * primitives[self.vel_ids[1]] \
                    + primitives[self.vel_ids[2]] * primitives[self.vel_ids[2]]
                    ) + e)
            
            conservatives = jnp.stack([
                *primitives[self.mass_slices],
                rhou, rhov, rhow,
                E,
                *primitives[self.vf_slices]],
            axis=0)


        elif self.equation_type in ("SINGLE-PHASE",
                                    "TWO-PHASE-LS",
                                    "SINGLE-PHASE-SOLID-LS",):
            rho = primitives[self.mass_ids] # = rho
            e = self.material_manager.get_specific_energy(primitives[self.energy_ids], rho)
            rhou = rho * primitives[self.vel_ids[0]] # = rho * u
            rhov = rho * primitives[self.vel_ids[1]] # = rho * v
            rhow = rho * primitives[self.vel_ids[2]] # = rho * w
            E = rho * (0.5 * (
                primitives[self.vel_ids[0]] * primitives[self.vel_ids[0]] \
                + primitives[self.vel_ids[1]] * primitives[self.vel_ids[1]] \
                + primitives[self.vel_ids[2]] * primitives[self.vel_ids[2]]) + e)  # E = rho * (1/2 u^2 + e)
            conservatives = jnp.stack([rho, rhou, rhov, rhow, E], axis=0)
        
        else:
            raise NotImplementedError

        return conservatives

    def get_primitives_from_conservatives(self, conservatives: Array) -> Array:
        """Converts conservative variables to primitive variables.

        :param conservatives: Buffer of conservative variables
        :type conservatives: Array
        :return: Buffer of primitive variables
        :rtype: Array
        """           

        if self.equation_type == "DIFFUSE-INTERFACE-5EQM":
            rho = self.material_manager.get_density(conservatives)
            one_rho = 1.0 / rho
            u   = conservatives[self.vel_ids[0]] * one_rho  # u = rho*u / rho
            v   = conservatives[self.vel_ids[1]] * one_rho  # v = rho*v / rho
            w   = conservatives[self.vel_ids[2]] * one_rho  # w = rho*w / rho
            e   = conservatives[self.energy_ids] * one_rho - 0.5 * (u * u + v * v + w * w)
            p   = self.material_manager.get_pressure(
                e       = e, 
                rho     = rho,
                alpha_i = conservatives[self.vf_slices]) # p = (gamma-1) * ( E - 1/2 * (rho*u) * u)

            primitives = jnp.stack([
                *conservatives[self.mass_slices],
                u, v, w,
                p,
                *conservatives[self.vf_slices]],
            axis=0)

        elif self.equation_type in ("SINGLE-PHASE",
                                    "TWO-PHASE-LS",
                                    "SINGLE-PHASE-SOLID-LS",):
            rho = conservatives[self.mass_ids]  # rho = rho
            one_rho = 1.0 / rho
            u = conservatives[self.vel_ids[0]] * one_rho  # u = rho*u / rho
            v = conservatives[self.vel_ids[1]] * one_rho  # v = rho*v / rho
            w = conservatives[self.vel_ids[2]] * one_rho  # w = rho*w / rho
            e = conservatives[self.energy_ids] * one_rho - 0.5 * (u * u + v * v + w * w)
            p = self.material_manager.get_pressure(e, rho) # p = (gamma-1) * ( E - 1/2 * (rho*u) * u)

            primitives = jnp.stack([rho, u, v, w, p], axis=0)

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

        if self.equation_type == "DIFFUSE-INTERFACE-5EQM":
            rho_alpha_ui = primitives[self.vel_ids[axis]] * conservatives[self.mass_slices] # (u_i * alpha_1 rho_1 )
            rho_ui_u1 = conservatives[self.vel_ids[axis]] * primitives[self.vel_ids[0]]     # (rho u_i) * u_1
            rho_ui_u2 = conservatives[self.vel_ids[axis]] * primitives[self.vel_ids[1]]     # (rho u_i) * u_2
            rho_ui_u3 = conservatives[self.vel_ids[axis]] * primitives[self.vel_ids[2]]     # (rho u_i) * u_3
            ui_Ep = primitives[self.vel_ids[axis]] * (conservatives[self.energy_ids] + primitives[self.energy_ids])
            if axis == 0:
                rho_ui_u1 += primitives[self.energy_ids]
            elif axis == 1:
                rho_ui_u2 += primitives[self.energy_ids]
            elif axis == 2:
                rho_ui_u3 += primitives[self.energy_ids]
            
            alpha_ui = primitives[self.vel_ids[axis]] * conservatives[self.vf_slices]

            flux_xi = jnp.stack([
                *rho_alpha_ui,
                rho_ui_u1,
                rho_ui_u2,
                rho_ui_u3,
                ui_Ep,
                *alpha_ui],
                axis=0)

        elif self.equation_type in ("SINGLE-PHASE",
                                    "TWO-PHASE-LS",
                                    "SINGLE-PHASE-SOLID-LS",):
            rho_ui = conservatives[axis+1] # (rho u_i)
            rho_ui_u1 = conservatives[axis+1] * primitives[self.vel_ids[0]] # (rho u_i) * u_1
            rho_ui_u2 = conservatives[axis+1] * primitives[self.vel_ids[1]] # (rho u_i) * u_2
            rho_ui_u3 = conservatives[axis+1] * primitives[self.vel_ids[2]] # (rho u_i) * u_3
            ui_Ep = primitives[axis+1] * (conservatives[self.energy_ids] + primitives[self.energy_ids])
            if axis == 0:
                rho_ui_u1 += primitives[self.energy_ids]
            elif axis == 1:
                rho_ui_u2 += primitives[self.energy_ids]
            elif axis == 2:
                rho_ui_u3 += primitives[self.energy_ids]

            flux_xi = jnp.stack([
                rho_ui,
                rho_ui_u1,
                rho_ui_u2,
                rho_ui_u3,
                ui_Ep],
                axis=0)
        
        else:
            raise NotImplementedError

        return flux_xi
