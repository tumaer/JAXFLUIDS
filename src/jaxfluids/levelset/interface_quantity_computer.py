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

from typing import Tuple, Dict
import types

import jax.numpy as jnp

from jaxfluids.domain_information import DomainInformation
from jaxfluids.materials.material_manager import MaterialManager
from jaxfluids.unit_handler import UnitHandler


class InterfaceQuantityComputer:
    """The InterfaceQuantityComputer class 
    1) solves the two-material Riemann problem, i.e., computes the interface velocity
        and interface pressure for FLUID-FLUID interface interactions
    2) computes the solid interface velocity for FLUID-SOLID-DYNAMIC interface interactions
    """
    eps = jnp.finfo(jnp.float64).eps

    def __init__(self, domain_information: DomainInformation, material_manager: MaterialManager, unit_handler: UnitHandler,
            solid_interface_velocity: Dict, numerical_setup: Dict) -> None:
        
        self.material_manager           = material_manager
        self.unit_handler               = unit_handler
        self.solid_interface_velocity   = solid_interface_velocity

        self.nhx__, self.nhy__, self.nhz__  = domain_information.domain_slices_conservatives_to_geometry
        self.nhx_, self.nhy_, self.nhz_     = domain_information.domain_slices_geometry
        self.nhx, self.nhy, self.nhz        = domain_information.domain_slices_conservatives
        self.cell_centers                   = domain_information.cell_centers
        self.cell_sizes                     = domain_information.cell_sizes
        self.active_axis_indices            = domain_information.active_axis_indices

        self.is_surface_tension = numerical_setup["active_physics"]["is_surface_tension"]

    def compute_solid_interface_velocity(self, current_time: float) -> jnp.ndarray:
        """Computes the solid interface velocity for FLUID-SOLID-DYNAMIC interface interactions.

        :param current_time: Current physical simulation time  
        :type current_time: float
        :return: Solid interface velocity
        :rtype: jnp.ndarray
        """

        # COMPUTE LAMBDA INPUTS
        mesh_grid = [jnp.meshgrid(*self.cell_centers, indexing="ij")[i] for i in self.active_axis_indices]
        for i in range(len(mesh_grid)):
            mesh_grid[i] = self.unit_handler.dimensionalize(mesh_grid[i], "length")
        current_time = self.unit_handler.dimensionalize(current_time, "time")

        # BUILDING BLOCKS
        if type(self.solid_interface_velocity) == list:
            solid_interface_velocity = jnp.zeros_like(mesh_grid[0])
            for velocity_object in self.solid_interface_velocity:
                velocity_function   = velocity_object["function"]
                velocity_tuple      = velocity_function(*mesh_grid, current_time)
                velocity_array      = jnp.stack([velocity_tuple[i] if i in self.active_axis_indices else jnp.zeros_like(velocity_tuple[0]) for i in range(3)], axis=0)
                bounding_domain     = velocity_object["bounding_domain"]
                mask                = bounding_domain(*mesh_grid, current_time)
                solid_interface_velocity *= (1 - mask)
                solid_interface_velocity += velocity_array * mask

        # LAMBDA FUNCTION
        elif type(self.solid_interface_velocity) == types.LambdaType:
            velocity_tuple              = self.solid_interface_velocity(*mesh_grid, current_time)
            solid_interface_velocity    = jnp.stack([velocity_tuple[i] if i in self.active_axis_indices else jnp.zeros_like(velocity_tuple[0]) for i in range(3)], axis=0)

        solid_interface_velocity = self.unit_handler.non_dimensionalize(solid_interface_velocity, "velocity")

        return solid_interface_velocity


    def solve_interface_interaction(self, primes: jnp.ndarray, normal: jnp.ndarray,
            curvature: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Solves the two-material Riemann problem for FLUID-FLUID interface interactions.

        :param primes: Primitive variable buffer
        :type primes: jnp.ndarray
        :param normal: Interface normal buffer
        :type normal: jnp.ndarray
        :param curvature: Interface curvature buffer
        :type curvature: jnp.ndarray
        :return: Interface velocity and interface pressure
        :rtype: Tuple[jnp.ndarray, jnp.ndarray]
        """
        
        primes      = primes[...,self.nhx__,self.nhy__,self.nhz__]
        pressure    = primes[4]
        density     = primes[0]

        velocity_normal_projection  = jnp.einsum('ijklm, ijklm -> jklm', primes[1:4], jnp.expand_dims(normal, axis=1) )
        speed_of_sound              = self.material_manager.get_speed_of_sound(pressure, density)
        impendance                  = speed_of_sound * density
        inverse_impendace_sum       = 1.0 / ( impendance[0] + impendance[1] + self.eps )

        # CAPILLARY PRESSURE JUMP
        if self.is_surface_tension:
            delta_p = self.material_manager.sigma * curvature
        else:
            delta_p = 0.0

        # INTERFACE QUANTITIES
        interface_velocity              = ( impendance[1] * velocity_normal_projection[1] + impendance[0] * velocity_normal_projection[0] + \
                                            pressure[1] - pressure[0] - delta_p ) * inverse_impendace_sum
        interface_pressure_positive     = (impendance[1] * pressure[0] + impendance[0] * (pressure[1] - delta_p) + \
                                            impendance[0] * impendance[1] * (velocity_normal_projection[1] - velocity_normal_projection[0]) ) * inverse_impendace_sum
        interface_pressure_negative     = (impendance[1] * (pressure[0] + delta_p) + impendance[0] * pressure[1] + \
                                            impendance[0] * impendance[1] * (velocity_normal_projection[1] - velocity_normal_projection[0]) ) * inverse_impendace_sum
        
        interface_pressure = jnp.stack([interface_pressure_positive, interface_pressure_negative], axis=0)

        return interface_velocity, interface_pressure
