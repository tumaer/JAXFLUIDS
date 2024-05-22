from typing import Dict, Callable, Tuple

import jax
import jax.numpy as jnp
from jax import Array
import numpy as np

from jaxfluids.halos.outer.boundary_condition import BoundaryCondition
from jaxfluids.domain.domain_information import DomainInformation
from jaxfluids.unit_handler import UnitHandler
from jaxfluids.data_types.case_setup.boundary_conditions import BoundaryConditionsField, BoundaryConditionsFace
from jaxfluids.domain import EDGE_LOCATIONS, VERTEX_LOCATIONS
from jaxfluids.halos.outer import EDGE_TYPES, VERTEX_TYPES

class BoundaryConditionLevelset(BoundaryCondition):
    """ The BoundaryConditionLevelset class implements functionality to enforce user-
    specified boundary conditions on the levelset field.

    Boundary conditions for the level-set field:
    1) Periodic
    2) Symmetry
    3) Zero-Gradient
    """

    def __init__(
            self,
            domain_information: DomainInformation,
            boundary_conditions: BoundaryConditionsField
            ) -> None:

        super().__init__(domain_information, boundary_conditions)

    def face_halo_update(
            self,
            levelset: Array,
            is_geometry_halos: bool = False
            ) -> Array:
        """Fills the face halo cells of
        the levelset buffer.

        :param levelset: Levelset buffer
        :type levelset: Array
        :return: Levelset buffer with filled halo cells
        :rtype: Array
        """

        if is_geometry_halos:
            face_slices_retrieve = self.face_slices_retrieve_geometry
            face_slices_fill = self.halo_slices.face_slices_geometry
        else:
            face_slices_retrieve = self.face_slices_retrieve_conservatives
            face_slices_fill = self.halo_slices.face_slices_conservatives

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

                if boundary_type in ["REINITIALIZATION"]:
                    continue
                elif boundary_type == "DIRICHLET":
                    levelset_callable = boundary_conditions_face.levelset_callable
                    halos = self.dirichlet(face_location, levelset_callable, 0.0)
                elif boundary_type in ["ZEROGRADIENT", "SYMMETRY", "PERIODIC"]:
                    slice_retrieve = face_slices_retrieve[boundary_type][face_location]
                    halos = levelset[slice_retrieve]
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
                    bounding_domain_mask = 1.0

                if is_parallel:
                    device_id = jax.lax.axis_index(axis_name="i")
                    device_mask = self.face_halo_mask
                    device_mask = device_mask[face_location][device_id]
                    mask = bounding_domain_mask * device_mask
                else:
                    mask = bounding_domain_mask

                slices_fill = face_slices_fill[face_location]
                levelset = levelset.at[slices_fill].mul(1 - mask)
                levelset = levelset.at[slices_fill].add(halos * mask)

        return levelset

    def edge_halo_update(
            self,
            levelset: Array,
            is_geometry_halos: bool = False
            ) -> Array:
        """Updates the edge halo cells
        of the levelset buffer.

        :param levelset: Levelset buffer
        :type levelset: Array
        :return: Levelset buffer with filled halo cells at the corners
        :rtype: Array
        """

        if is_geometry_halos:
            edge_slices = self.halo_slices.edge_slices_geometry
        else:
            edge_slices = self.halo_slices.edge_slices_conservatives

        is_parallel = self.domain_information.is_parallel
        active_edge_locations = self.domain_information.active_edge_locations
        for edge_location in active_edge_locations:
            
            edge_boundary_type = self.edge_boundary_types[edge_location]
            
            if edge_boundary_type == "ANY_ANY":
                location_retrieve_1 = edge_location + "_10"
                location_retrieve_2 = edge_location + "_01"
                slice_retrieve_1 = edge_slices[location_retrieve_1]
                slice_retrieve_2 = edge_slices[location_retrieve_2]
                halos = 0.5 * (levelset[slice_retrieve_1] + levelset[slice_retrieve_2])
            else:
                location_retrieve = self.edge_types_to_location_retrieve[edge_location][edge_boundary_type]
                slice_retrieve = edge_slices[location_retrieve]
                halos = levelset[slice_retrieve]
                if "SYMMETRY" in edge_boundary_type:
                    s_ = self.edge_flip_slices_symmetry[edge_location][edge_boundary_type]
                    halos = halos[s_]

            if is_parallel:
                device_id = jax.lax.axis_index(axis_name="i")
                mask = self.edge_halo_mask[edge_location][device_id]
            else:
                mask = 1

            slice_fill = edge_slices[edge_location]
            levelset = levelset.at[slice_fill].mul(1 - mask)
            levelset = levelset.at[slice_fill].add(halos * mask)

        return levelset

    def vertex_halo_update(
            self,
            levelset: Array,
            is_geometry_halos: bool = False
            ) -> Array:
        """Updates the vertex halo
        cells of the levelset buffer

        :param levelset: _description_
        :type levelset: Array
        :param is_geometry_halos: _description_, defaults to False
        :type is_geometry_halos: bool, optional
        :return: _description_
        :rtype: Array
        """
        
        if is_geometry_halos:
            vertex_slices = self.halo_slices.vertex_slices_geometry
        else:
            vertex_slices = self.halo_slices.vertex_slices_conservatives

        is_parallel = self.domain_information.is_parallel
        for vertex_location in VERTEX_LOCATIONS:
            
            vertex_boundary_type = self.vertex_boundary_types[vertex_location]
            
            if vertex_boundary_type == "ANY_ANY_ANY":
                location_retrieve_1 = vertex_location + "_100"
                location_retrieve_2 = vertex_location + "_010"
                location_retrieve_3 = vertex_location + "_001"
                slice_retrieve_1 = vertex_slices[location_retrieve_1]
                slice_retrieve_2 = vertex_slices[location_retrieve_2]
                slice_retrieve_3 = vertex_slices[location_retrieve_3]
                halos = 1.0/3.0 * (levelset[slice_retrieve_1] + levelset[slice_retrieve_2]
                                   + levelset[slice_retrieve_3])
            else:
                location_retrieve = self.vertex_types_to_location_retrieve[vertex_location][vertex_boundary_type]
                slice_retrieve = vertex_slices[location_retrieve]
                halos = levelset[slice_retrieve]
                if "SYMMETRY" in vertex_boundary_type:
                    s_ = self.vertex_flip_slices_symmetry[vertex_location][vertex_boundary_type]
                    halos = halos[s_]

            if is_parallel:
                device_id = jax.lax.axis_index(axis_name="i")
                mask = self.vertex_halo_mask[vertex_location][device_id]
            else:
                mask = 1

            slice_fill = vertex_slices[vertex_location]
            levelset = levelset.at[slice_fill].mul(1 - mask)
            levelset = levelset.at[slice_fill].add(halos * mask)
        
        return levelset

    def dirichlet(
            self,
            face_location: str,
            levelset_callable: Callable,
            physical_simulation_time: float,
            ) -> Array:
        """Computes the halo cells from
        a DIRICHLET condition.

        :param face_location: _description_
        :type face_location: str
        :param levelset_callable: _description_
        :type levelset_callable: Callable
        :param physical_simulation_time: _description_
        :type physical_simulation_time: float
        :return: _description_
        :rtype: Array
        """
        meshgrid, axes_to_expand = \
        self.get_boundary_coordinates_at_location(
            face_location)
        halos = levelset_callable(*meshgrid, physical_simulation_time)
        for axis in axes_to_expand:
            halos = jnp.expand_dims(halos, axis)
        return halos
    
    def get_halo_mask(self, levelset: Array):
        """Generates a mask for the halo cells that require reinitialization

        :param levelset: _description_
        :type levelset: Array
        :return: _description_
        :rtype: _type_
        """
        mask = jnp.zeros_like(levelset, dtype=jnp.uint32)
        locations = [location for location in self.boundary_conditions if self.boundary_conditions[location] == "reinitialization"]
        for loc in locations:
            slice_objects = self.halo_slices.face_slices_geometry[loc]
            mask = mask.at[slice_objects].set(1)
        return mask
    



