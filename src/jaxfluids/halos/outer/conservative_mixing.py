from typing import Dict, Tuple

import jax
import jax.numpy as jnp
from jax import Array
import numpy as np

from jaxfluids.halos.outer.boundary_condition import BoundaryCondition, get_signs_symmetry
from jaxfluids.domain.domain_information import DomainInformation
from jaxfluids.equation_information import EquationInformation
from jaxfluids.data_types.case_setup.boundary_conditions import BoundaryConditionsField, BoundaryConditionsFace
from jaxfluids.domain import EDGE_LOCATIONS, VERTEX_LOCATIONS

class BoundaryConditionConservativeMixing(BoundaryCondition):

    def __init__(
            self,
            domain_information: DomainInformation,
            boundary_conditions: BoundaryConditionsField,
            equation_information: EquationInformation
            ) -> None:

        super().__init__(domain_information, boundary_conditions)

        no_primes = equation_information.no_primes
        equation_type = equation_information.equation_type
        vel_indices = equation_information.velocity_ids
        self.face_signs_symmetry, self.edge_signs_symmetry, \
        self.vertex_signs_symmetry = get_signs_symmetry(
            no_primes, equation_type, vel_indices)

    def face_halo_update(
            self,
            conservatives: Array,
            ) -> Array:
        """Fills the conservatives buffer halo cells.

        :param conservatives: _description_
        :type conservatives: Array
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
                
                if boundary_type in ["ZEROGRADIENT", "PERIODIC", "SYMMETRY"]:
                    pass
                elif boundary_type in ["DIRICHLET", "NEUMANN"]:
                    boundary_type = "ZEROGRADIENT"
                elif "WALL" in boundary_type:
                    boundary_type = "SYMMETRY"
                else:
                    raise NotImplementedError

                slice_retrieve = self.face_slices_retrieve_conservatives[boundary_type][face_location]
                halos = conservatives[slice_retrieve]
                if boundary_type == "SYMMETRY":
                    halos *= self.face_signs_symmetry[face_location]

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
                conservatives = conservatives.at[slice_fill].mul(1 - mask)
                conservatives = conservatives.at[slice_fill].add(halos * mask)

        return conservatives

    def edge_halo_update(
            self,
            conservatives: Array
            ) -> Array:
        """Updates the edge halos of the conserved
        variable buffer.

        :param conservatives: _description_
        :type conservatives: Array
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
                halos = 0.5 * (conservatives[slice_retrieve_1] + conservatives[slice_retrieve_2])
                
            else:
                location_retrieve = self.edge_types_to_location_retrieve[edge_location][edge_boundary_types]
                slice_retrieve = edge_slices[location_retrieve]
                halos = conservatives[slice_retrieve]
                if "SYMMETRY" in edge_boundary_types:
                    s_ = self.edge_flip_slices_symmetry[edge_location][edge_boundary_types]
                    signs = self.edge_signs_symmetry[edge_location][edge_boundary_types]
                    halos = halos[s_]
                    halos *= signs

            if is_parallel:
                device_id = jax.lax.axis_index(axis_name="i")
                mask = self.edge_halo_mask[edge_location][device_id]
            else:
                mask = 1
            slice_fill = edge_slices[edge_location]
            conservatives = conservatives.at[slice_fill].mul(1 - mask)
            conservatives = conservatives.at[slice_fill].add(halos * mask)

        return conservatives


    def vertex_halo_update(
            self,
            conservatives: Array,
            ) -> Array:
        """Updates the vertex halos of
        the conserved variable buffer.

        :param conservatives: _description_
        :type conservatives: Array
        :return: _description_
        :rtype: Array
        """

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
                halos = 1.0/3.0 * (conservatives[slice_retrieve_1] + conservatives[slice_retrieve_2]
                                        + conservatives[slice_retrieve_3])
            else:
                location_retrieve = self.vertex_types_to_location_retrieve[vertex_location][vertex_boundary_types]
                slice_retrieve = vertex_slices[location_retrieve]
                halos = conservatives[slice_retrieve]
                if "SYMMETRY" in vertex_boundary_types:
                    s_ = self.vertex_flip_slices_symmetry[vertex_location][vertex_boundary_types]
                    signs = self.vertex_signs_symmetry[vertex_location][vertex_boundary_types]
                    halos = halos[s_]
                    halos *= signs

            if is_parallel:
                device_id = jax.lax.axis_index(axis_name="i")
                mask = self.vertex_halo_mask[vertex_location][device_id]
            else:
                mask = 1

            slices_fill = self.halo_slices.vertex_slices_conservatives[vertex_location]
            conservatives = conservatives.at[slices_fill].mul(1 - mask)
            conservatives = conservatives.at[slices_fill].add(halos * mask)

        return conservatives