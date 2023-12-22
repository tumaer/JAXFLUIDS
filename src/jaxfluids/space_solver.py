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

from typing import Dict, List, Union, Tuple

import jax.numpy as jnp

from jaxfluids.domain_information import DomainInformation
from jaxfluids.flux_computation import FluxComputer
from jaxfluids.materials.material_manager import MaterialManager
from jaxfluids.solvers.source_term_solver import SourceTermSolver
from jaxfluids.levelset.levelset_handler import LevelsetHandler
from jaxfluids.stencils import DICT_CENTRAL_RECONSTRUCTION, DICT_FIRST_DERIVATIVE_CENTER, DICT_DERIVATIVE_FACE

class SpaceSolver:
    """The Space Solver class manages the calculation of the righ-hand-side (i.e., fluxes) of the NSE
    and, for two-phase simulations, manages the calculation of the rhs of the level-set advection.

    Depending on the numerical setup, the calculation of the fluxes for the NSE has contributions from:
    1) Convective flux
    2) Viscous flux
    3) Heat flux
    4) Interface exchange flux
    5) Volume force flux
    6) External forcing 
    """

    def __init__(self, domain_information: DomainInformation, material_manager: MaterialManager, numerical_setup: Dict,
        gravity: jnp.ndarray, levelset_type: Tuple[None, str], levelset_handler: Union[LevelsetHandler, None]) -> None:
        
        self.flux_computer      = FluxComputer(
            numerical_setup,
            material_manager,
            domain_information,
            )

        self.source_term_solver = SourceTermSolver(
            material_manager            = material_manager,
            gravity                     = gravity,
            domain_information          = domain_information,
            derivative_stencil_center   = DICT_FIRST_DERIVATIVE_CENTER[numerical_setup["conservatives"]["dissipative_fluxes"]["derivative_stencil_center"]](nh=domain_information.nh_conservatives, inactive_axis=domain_information.inactive_axis, offset=2 if numerical_setup["conservatives"]["dissipative_fluxes"]["reconstruction_stencil"] == "R4" else 1) if numerical_setup["active_physics"]["is_viscous_flux"] else None,
            derivative_stencil_face     = DICT_DERIVATIVE_FACE[numerical_setup["conservatives"]["dissipative_fluxes"]["derivative_stencil_face"]](nh=domain_information.nh_conservatives, inactive_axis=domain_information.inactive_axis, offset=0) if numerical_setup["active_physics"]["is_viscous_flux"] or numerical_setup["active_physics"]["is_heat_flux"] else None,
            reconstruct_stencil_duidxi  = DICT_CENTRAL_RECONSTRUCTION[numerical_setup["conservatives"]["dissipative_fluxes"]["reconstruction_stencil"]](nh=domain_information.nh_conservatives, inactive_axis=domain_information.inactive_axis, offset=domain_information.nh_conservatives-2 if numerical_setup["conservatives"]["dissipative_fluxes"]["reconstruction_stencil"] == "R4" else domain_information.nh_conservatives-1) if numerical_setup["active_physics"]["is_viscous_flux"] else None,
            reconstruct_stencil_ui      = DICT_CENTRAL_RECONSTRUCTION[numerical_setup["conservatives"]["dissipative_fluxes"]["reconstruction_stencil"]](nh=domain_information.nh_conservatives, inactive_axis=domain_information.inactive_axis, offset=0) if numerical_setup["active_physics"]["is_viscous_flux"] or numerical_setup["active_physics"]["is_heat_flux"] else None,
            levelset_type               = levelset_type
            )

        self.levelset_handler   = levelset_handler
        self.material_manager   = material_manager

        self.is_convective_flux     = numerical_setup["active_physics"]["is_convective_flux"]
        self.is_viscous_flux        = numerical_setup["active_physics"]["is_viscous_flux"]
        self.is_heat_flux           = numerical_setup["active_physics"]["is_heat_flux"]
        self.is_volume_force        = numerical_setup["active_physics"]["is_volume_force"]
        self.levelset_type          = levelset_type

        self.active_axis_indices                = domain_information.active_axis_indices
        cell_size_x, cell_size_y, cell_size_z   = domain_information.cell_sizes
        self.one_cell_size                      = [ 1.0/cell_size_x, 1.0/cell_size_y, 1.0/cell_size_z ]
        self.nx, self.ny, self.nz               = domain_information.number_of_cells
        self.nhx_, self.nhy_, self.nhz_         = domain_information.domain_slices_geometry


        self.flux_slices    = [ 
            [jnp.s_[...,1:,:,:], jnp.s_[...,:-1,:,:]],
            [jnp.s_[...,:,1:,:], jnp.s_[...,:,:-1,:]],
            [jnp.s_[...,:,:,1:], jnp.s_[...,:,:,:-1]],
        ]

    def compute_rhs(self, cons: jnp.ndarray, primes: jnp.ndarray, current_time: float, 
        levelset: jnp.ndarray = None, volume_fraction: jnp.ndarray = None,
        apertures: List = None, forcings_dictionary: Union[Dict, None] = None,
        ml_parameters_dict: Union[Dict, None] = None, ml_networks_dict: Union[Dict, None] = None) -> Tuple[jnp.ndarray, Union[jnp.ndarray, None], Union[float, None]]:
        """Computes the right-hand-side of the Navier-Stokes equations depending 
        on active physics and active axis. For levelset simulations with FLUID-FLUID or FLUID-SOLID-DYNAMIC
        interface interactions, also computes the right-hand-side of the levelset advection.

        :param cons: Buffer of conservative variables
        :type cons: jnp.ndarray
        :param primes: Buffer of primitive variables
        :type primes: jnp.ndarray
        :param current_time: Current physical simulation time
        :type current_time: float
        :param levelset: Levelset buffer, defaults to None
        :type levelset: jnp.ndarray, optional
        :param volume_fraction: Volume fraction buffer, defaults to None
        :type volume_fraction: jnp.ndarray, optional
        :param apertures: Aperture buffers, defaults to None
        :type apertures: List, optional
        :param forcings_dictionary: Forcings dictionary, defaults to None
        :type forcings_dictionary: Union[Dict, None], optional
        :param ml_parameters_dict: Dictionary containing NN weights, defaults to None
        :type ml_parameters_dict: Union[Dict, None], optional
        :param ml_networks_dict: Dictionary containing NN architectures, defaults to None
        :type ml_networks_dict: Union[Dict, None], optional
        :return: Tuple containing the right-hand-side buffer of the Navier-Stokes equations, the
        right-hand-side buffer of the levelset advection equation and the maximum extension residual of the interface quantities 
        :rtype: Tuple[jnp.ndarray, Union[jnp.ndarray, None], Union[float, None]]
        """

        # COMPUTE TEMPERATURE
        if self.is_viscous_flux or self.is_heat_flux:
            temperature = self.material_manager.get_temperature(primes[4:5], primes[0:1])

        # COMPUTE INTERFACE VELOCITY AND INTERFACE PRESSURE
        if self.levelset_type == "FLUID-FLUID":
            interface_velocity, interface_pressure, residual_interface = self.levelset_handler.compute_interface_quantities(primes, levelset, volume_fraction)
        elif self.levelset_type == "FLUID-SOLID-DYNAMIC":
            interface_velocity, interface_pressure, residual_interface = self.levelset_handler.compute_solid_interface_velocity(current_time), None, None
        else:
            interface_velocity, interface_pressure, residual_interface = None, None, None

        # INITIALIZE RHS VARIABLES
        rhs_cons, rhs_levelset = 0.0, 0.0 if self.levelset_type != None else None

        # CELL FACE FLUX
        for axis in self.active_axis_indices:

            flux_xi = 0

            # CONVECTIVE CONTRIBUTION
            if self.is_convective_flux:
                flux_xi += self.flux_computer.compute_convective_flux_xi(primes, cons, axis, ml_parameters_dict, ml_networks_dict)
            
            # VISCOUS CONTRIBUTION
            if self.is_viscous_flux:
                flux_xi -= self.source_term_solver.compute_viscous_flux_xi(primes[1:4], temperature, axis)

            # HEAT CONTRIBUTION
            if self.is_heat_flux:
                flux_xi += self.source_term_solver.compute_heat_flux_xi(temperature, axis)

            # WEIGHT FLUXES
            if self.levelset_type != None:
                flux_xi = self.levelset_handler.weight_cell_face_flux_xi(flux_xi, apertures[axis])

            # SUM RIGHT HAND SIDE
            if self.is_convective_flux or self.is_viscous_flux or self.is_heat_flux:
                rhs_cons += self.one_cell_size[axis] * (flux_xi[self.flux_slices[axis][1]] - flux_xi[self.flux_slices[axis][0]])

            # INTERFACE FLUXES
            if self.levelset_type != None:
                interface_flux_xi = self.levelset_handler.compute_interface_flux_xi(primes, levelset, interface_velocity, interface_pressure, volume_fraction, apertures[axis], axis)
                rhs_cons += self.one_cell_size[axis] * interface_flux_xi

            # LEVELSET ADVECTION
            if self.levelset_type in ["FLUID-FLUID", "FLUID-SOLID-DYNAMIC"]:
                rhs_contribution_levelset = self.levelset_handler.compute_levelset_advection_rhs(levelset, interface_velocity, axis)
                rhs_levelset += rhs_contribution_levelset

        # VOLUME FORCES
        if self.is_volume_force:
            volume_forces = self.source_term_solver.compute_gravity_forces(primes)
            if self.levelset_type != None:
                volume_forces = self.levelset_handler.weight_volume_force(volume_forces, volume_fraction)
            rhs_cons += volume_forces

        # FORCINGS
        if forcings_dictionary:
            for key in forcings_dictionary:
                forcing = forcings_dictionary[key]["force"]
                if self.levelset_type != None:
                    forcing = self.levelset_handler.weight_volume_force(forcing, volume_fraction)
                rhs_cons += forcing

        # CLEAN RHS
        if self.levelset_type != None:
            mask_real, _ = self.levelset_handler.compute_masks(levelset, volume_fraction)
            rhs_cons *= mask_real[...,self.nhx_,self.nhy_,self.nhz_]

        return rhs_cons, rhs_levelset, residual_interface