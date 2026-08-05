from typing import Callable, Tuple

import jax
import jax.numpy as jnp
import numpy as np

from jaxfluids.halos.outer.boundary_condition import BoundaryCondition, get_signs_symmetry
from jaxfluids.domain.domain_information import DomainInformation
from jaxfluids.equation_manager import EquationManager
from jaxfluids.data_types.case_setup.boundary_conditions import (
    BoundaryConditionsField,
    BoundaryConditionsFace,
)
from jaxfluids.domain import VERTEX_LOCATIONS

Array = jax.Array


class BoundaryConditionTemperature(BoundaryCondition):
    def __init__(
            self,
            domain_information: DomainInformation,
            equation_manager: EquationManager,
            boundary_conditions: BoundaryConditionsField,
        ) -> None:

        super().__init__(domain_information, boundary_conditions)

        self.equation_manager = equation_manager
        self.equation_information = equation_manager.equation_information
        self.material_manager = self.equation_manager.material_manager

        (
            self.face_signs_symmetry,
            self.edge_signs_symmetry,
            self.vertex_signs_symmetry
        ) = get_signs_symmetry(
            self.equation_information.no_primes,
            self.equation_information.equation_type,
            self.equation_information.ids_velocity,
        )


    def face_halo_update(
            self,
            temperature: Array,
            physical_simulation_time: float,
        ) -> Array:
        """Fills the face halos of the temperature buffer."""

        is_parallel = self.domain_information.is_parallel
        active_face_locations = self.domain_information.active_face_locations
        for face_location in active_face_locations:

            boundary_conditions_face_tuple: Tuple[BoundaryConditionsFace] = \
            getattr(self.boundary_conditions, face_location)
            if len(boundary_conditions_face_tuple) > 1:
                multiple_types_at_face = True
            else:
                multiple_types_at_face = False

            for boundary_conditions_face in boundary_conditions_face_tuple:

                boundary_type = boundary_conditions_face.boundary_type
                if boundary_type in ["ISOTHERMALWALL", "ISOTHERMALMASSTRANSFERWALL"]:
                    wall_temperature_callable = boundary_conditions_face.wall_temperature_callable
                    halos = self.wall_temperature(
                        temperature,
                        face_location,
                        wall_temperature_callable,
                        physical_simulation_time,
                    )

                else:
                    continue

                if multiple_types_at_face:
                    meshgrid, axes_to_expand = self.get_boundary_coordinates_at_location(
                        face_location)
                    bounding_domain_callable = boundary_conditions_face.bounding_domain_callable
                    bounding_domain_mask = bounding_domain_callable(*meshgrid)
                    for axis in axes_to_expand:
                        bounding_domain_mask = jnp.expand_dims(bounding_domain_mask, axis)
                else:
                    bounding_domain_mask = 1.0

                slices_fill = self.halo_slices.face_slices_conservatives[face_location]

                if is_parallel:
                    device_id = jax.lax.axis_index(axis_name="i")
                    device_mask = self.face_halo_mask
                    device_mask = device_mask[face_location][device_id]
                    mask = bounding_domain_mask * device_mask
                else:
                    mask = bounding_domain_mask

                temperature = temperature.at[slices_fill].mul(1 - mask)
                temperature = temperature.at[slices_fill].add(halos * mask)

        return temperature

    def edge_halo_update(self, temperature: Array) -> Array:
        """Updates the edge halo cells of the temperature buffer."""

        edge_slices = self.halo_slices.edge_slices_conservatives

        is_parallel = self.domain_information.is_parallel
        active_edge_locations = self.domain_information.active_edge_locations
        for edge_location in active_edge_locations:
            
            edge_boundary_type = self.edge_boundary_types[edge_location]
            
            if edge_boundary_type != "ANY_ANY":
                continue

            location_retrieve_1 = edge_location + "_10"
            location_retrieve_2 = edge_location + "_01"
            slice_retrieve_1 = edge_slices[location_retrieve_1]
            slice_retrieve_2 = edge_slices[location_retrieve_2]
            halos = 0.5 * (temperature[slice_retrieve_1] + temperature[slice_retrieve_2])

            if is_parallel:
                device_id = jax.lax.axis_index(axis_name="i")
                mask = self.edge_halo_mask[edge_location][device_id]
            else:
                mask = 1

            slice_fill = edge_slices[edge_location]
            temperature = temperature.at[slice_fill].mul(1 - mask)
            temperature = temperature.at[slice_fill].add(halos * mask)

        return temperature

    def vertex_halo_update(self, temperature: Array) -> Array:
        """Updates the vertex halo cells of the temperature buffer."""

        vertex_slices = self.halo_slices.vertex_slices_conservatives

        is_parallel = self.domain_information.is_parallel
        for vertex_location in VERTEX_LOCATIONS:
            
            vertex_boundary_type = self.vertex_boundary_types[vertex_location]
            
            if vertex_boundary_type != "ANY_ANY_ANY":
                continue

            location_retrieve_1 = vertex_location + "_100"
            location_retrieve_2 = vertex_location + "_010"
            location_retrieve_3 = vertex_location + "_001"
            slice_retrieve_1 = vertex_slices[location_retrieve_1]
            slice_retrieve_2 = vertex_slices[location_retrieve_2]
            slice_retrieve_3 = vertex_slices[location_retrieve_3]
            halos = 1.0/3.0 * (
                temperature[slice_retrieve_1]
                + temperature[slice_retrieve_2]
                + temperature[slice_retrieve_3]
            )

            if is_parallel:
                device_id = jax.lax.axis_index(axis_name="i")
                mask = self.vertex_halo_mask[vertex_location][device_id]
            else:
                mask = 1

            slice_fill = vertex_slices[vertex_location]
            temperature = temperature.at[slice_fill].mul(1 - mask)
            temperature = temperature.at[slice_fill].add(halos * mask)
        
        return temperature


    def wall_temperature(
            self, 
            temperature: Array,
            face_location: str, 
            wall_temperature_callable: Callable,
            physical_simulation_time: float, 
        ) -> Array:
        """Computes the temperature halos for
        isothermal wall boundaries.

        :param temperature: _description_
        :type temperature: Array
        :param face_location: _description_
        :type face_location: str
        :param temperature_functions: _description_
        :type temperature_functions: Dict
        :param physical_simulation_time: _description_
        :type physical_simulation_time: float
        :return: _description_
        :rtype: Array
        """

        (
            meshgrid,
            axes_to_expand
        ) = self.get_boundary_coordinates_at_location(face_location)
        
        wall_temperature = wall_temperature_callable(
            *meshgrid, physical_simulation_time)
        
        for axis in axes_to_expand:
            wall_temperature = jnp.expand_dims(wall_temperature, axis)

        slices_retrieve = self.face_slices_retrieve_conservatives["SYMMETRY"][face_location]
        halos_temperature = 2 * wall_temperature - temperature[slices_retrieve]

        return halos_temperature