from __future__ import annotations
from typing import Dict, Tuple, Callable, TYPE_CHECKING

import jax
import jax.numpy as jnp
import numpy as np

from jaxfluids.halos.outer.boundary_condition import BoundaryCondition, get_signs_symmetry
from jaxfluids.domain.domain_information import DomainInformation
from jaxfluids.equation_information import EquationInformation
from jaxfluids.data_types.case_setup.boundary_conditions import BoundaryConditionsField, BoundaryConditionsFace
from jaxfluids.domain import EDGE_LOCATIONS, VERTEX_LOCATIONS
if TYPE_CHECKING:
    from jaxfluids.levelset.fluid_solid.solid_properties_manager import SolidPropertiesManager

Array = jax.Array

class BoundaryConditionSolids(BoundaryCondition):

    def __init__(
            self,
            domain_information: DomainInformation,
            boundary_conditions: BoundaryConditionsField,
            solid_properties_manager: SolidPropertiesManager
            ) -> None:

        super().__init__(domain_information, boundary_conditions)

        self.solid_properties_manager = solid_properties_manager

    def face_halo_update(
            self,
            solid_temperature: Array,
            physical_simulation_time: float,
            solid_energy: Array = None
            ) -> Array:
        """Fills the halo cells of the solid field buffers.

        :param solid_temperature: _description_
        :type solid_temperature: Array
        :return: _description_
        :rtype: Array
        """

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

                if boundary_type in ("ZEROGRADIENT", "PERIODIC", "ADIABTIC", "SYMMETRY"):
                    boundary_type == "SYMMETRY" if boundary_type == "ADIABATIC" else boundary_type
                    halos_temperature = self.miscellaneous(
                        solid_temperature, boundary_type, face_location)
                    
                elif boundary_type == "ISOTHERMAL":
                    temperature_callable = boundary_conditions_face.temperature_callable
                    halos_temperature = self.isothermal(solid_temperature, temperature_callable,
                                                        face_location, physical_simulation_time)
                
                elif boundary_type == "HEATFLUX":
                    raise NotImplementedError
                
                elif boundary_type == "HEATTRANSFER":
                    raise NotImplementedError
                
                else:
                    raise NotImplementedError

                if multiple_types_at_face:
                    meshgrid, axes_to_expand = self.get_boundary_coordinates_at_location(
                        face_location)
                    bounding_domain_callable = boundary_conditions_face.bounding_domain_callable
                    bounding_domain_mask = bounding_domain_callable(*meshgrid)
                    for axis in axes_to_expand:
                        bounding_domain_mask = jnp.expand_dims(bounding_domain_mask, axis)
                else:
                    bounding_domain_mask = 1

                if is_parallel:
                    device_id = jax.lax.axis_index(axis_name="i")
                    device_mask = self.face_halo_mask
                    device_mask = device_mask[face_location][device_id]
                    mask = bounding_domain_mask * device_mask
                else:
                    mask = bounding_domain_mask

                slice_fill = self.halo_slices.face_slices_conservatives[face_location]
                solid_temperature = solid_temperature.at[slice_fill].mul(1 - mask)
                solid_temperature = solid_temperature.at[slice_fill].add(halos_temperature * mask)
                if solid_energy is not None:
                    halos_energy = self.solid_properties_manager.compute_internal_energy(halos_temperature)
                    solid_energy = solid_energy.at[slice_fill].mul(1 - mask)
                    solid_energy = solid_energy.at[slice_fill].add(halos_energy * mask)
        
        if solid_energy is not None:
            return solid_temperature, solid_energy
        else:
            return solid_temperature

    def edge_halo_update(
            self,
            solid_temperature: Array,
            solid_energy: Array = None
            ) -> Array:
        """Updates the edge halos of the conserved
        variable solid_temperature.

        :param solid_temperature: _description_
        :type solid_temperature: Array
        :return: _description_
        :rtype: Array
        """

        edge_slices = self.halo_slices.edge_slices_conservatives
        is_parallel = self.domain_information.is_parallel
        active_edge_halos = self.domain_information.active_edge_locations
        for edge_location in active_edge_halos:

            edge_boundary_types = self.edge_boundary_types[edge_location]

            if edge_boundary_types == "ANY_ANY":
                location_retrieve_1 = edge_location + "_10"
                location_retrieve_2 = edge_location + "_01"
                slice_retrieve_1 = edge_slices[location_retrieve_1]
                slice_retrieve_2 = edge_slices[location_retrieve_2]
                halos_temperature = 0.5 * (solid_temperature[slice_retrieve_1] + solid_temperature[slice_retrieve_2])
                
            else:
                location_retrieve = self.edge_types_to_location_retrieve[edge_location][edge_boundary_types]
                slice_retrieve = edge_slices[location_retrieve]
                halos_temperature = solid_temperature[slice_retrieve]

            if is_parallel:
                device_id = jax.lax.axis_index(axis_name="i")
                mask = self.edge_halo_mask[edge_location][device_id]
            else:
                mask = 1
            slice_fill = edge_slices[edge_location]
            solid_temperature = solid_temperature.at[slice_fill].mul(1 - mask)
            solid_temperature = solid_temperature.at[slice_fill].add(halos_temperature * mask)

            if solid_energy is not None:
                halos_energy = self.solid_properties_manager.compute_internal_energy(halos_temperature)
                solid_energy = solid_energy.at[slice_fill].mul(1 - mask)
                solid_energy = solid_energy.at[slice_fill].add(halos_energy * mask)
        
        if solid_energy is not None:
            return solid_temperature, solid_energy
        else:
            return solid_temperature


    def vertex_halo_update(
            self,
            solid_temperature: Array,
            solid_energy: Array = None
            ) -> Array:

        vertex_slices = self.halo_slices.vertex_slices_conservatives
        is_parallel = self.domain_information.is_parallel
        for vertex_location in VERTEX_LOCATIONS:

            vertex_boundary_types = self.vertex_boundary_types[vertex_location]

            if vertex_boundary_types == "ANY_ANY_ANY":
                location_retrieve_1 = vertex_location + "_100"
                location_retrieve_2 = vertex_location + "_010"
                location_retrieve_3 = vertex_location + "_001"
                slice_retrieve_1 = vertex_slices[location_retrieve_1]
                slice_retrieve_2 = vertex_slices[location_retrieve_2]
                slice_retrieve_3 = vertex_slices[location_retrieve_3]
                halos_temperature = 1.0/3.0 * (solid_temperature[slice_retrieve_1] + solid_temperature[slice_retrieve_2]
                                        + solid_temperature[slice_retrieve_3])
            else:
                location_retrieve = self.vertex_types_to_location_retrieve[vertex_location][vertex_boundary_types]
                slice_retrieve = vertex_slices[location_retrieve]
                halos_temperature = solid_temperature[slice_retrieve]

            if is_parallel:
                device_id = jax.lax.axis_index(axis_name="i")
                mask = self.vertex_halo_mask[vertex_location][device_id]
            else:
                mask = 1

            slices_fill = self.halo_slices.vertex_slices_conservatives[vertex_location]
            solid_temperature = solid_temperature.at[slices_fill].mul(1 - mask)
            solid_temperature = solid_temperature.at[slices_fill].add(halos_temperature * mask)

            if solid_energy is not None:
                halos_energy = self.solid_properties_manager.compute_internal_energy(halos_temperature)
                solid_energy = solid_energy.at[slices_fill].mul(1 - mask)
                solid_energy = solid_energy.at[slices_fill].add(halos_energy * mask)
        
        if solid_energy is not None:
            return solid_temperature, solid_energy
        else:
            return solid_temperature

    def isothermal(
            self,
            solid_temperature: Array,
            temperature_callable: Callable,
            face_location: str,
            physical_simulation_time: float
            ) -> Array:
        meshgrid, axes_to_expand = \
        self.get_boundary_coordinates_at_location(
            face_location)
        wall_temperature = temperature_callable(
            *meshgrid, physical_simulation_time)
        for axis in axes_to_expand:
            wall_temperature = jnp.expand_dims(wall_temperature, axis)
        slices_retrieve = self.face_slices_retrieve_conservatives["SYMMETRY"][face_location]
        halos = 2 * wall_temperature - solid_temperature[slices_retrieve]
        return halos

    def miscellaneous(
            self,
            solid_temperature: Array,
            boundary_type: str,
            face_location: str,
            )-> Array:
        """Periodic, zerogradient, symmetry

        :param solid_temperature: _description_
        :type solid_temperature: Array
        :param boundary_type: _description_
        :type boundary_type: str
        :param face_location: _description_
        :type face_location: str
        :return: _description_
        :rtype: Array
        """
        slices_retrieve = self.face_slices_retrieve_conservatives[boundary_type][face_location]
        halos = solid_temperature[slices_retrieve]
        return halos