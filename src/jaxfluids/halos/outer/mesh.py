from typing import Dict, Tuple

import jax
import jax.numpy as jnp
from jax import Array

from jaxfluids.halos.outer.boundary_condition import BoundaryCondition
from jaxfluids.domain.domain_information import DomainInformation
from jaxfluids.data_types.case_setup.boundary_conditions import BoundaryConditionsField, BoundaryConditionsFace

class BoundaryConditionMesh(BoundaryCondition):

    def __init__(
            self,
            domain_information: DomainInformation,
            boundary_conditions: BoundaryConditionsField,
            ) -> None:

        super().__init__(domain_information, boundary_conditions)

    def face_halo_update(
            self,
            mesh_xi: Array,
            axis_index: int,
            quantity: str = "cell_sizes"
            ) -> Array:
        """Updates the face halos of the
        mesh in xi direction.

        :param mesh_xi: _description_
        :type mesh_xi: Array
        :param axis_index: _description_
        :type axis_index: int
        :return: _description_
        :rtype: Array
        """
        if quantity not in ("cell_sizes", "cell_centers"):
            raise NotImplementedError
        
        if quantity == "cell_sizes":
            mesh_xi = self.face_halo_update_cell_sizes(mesh_xi, axis_index)
        elif quantity == "cell_centers":
            mesh_xi = self.face_halo_update_cell_centers(mesh_xi, axis_index)
            
        return mesh_xi

    def face_halo_update_cell_sizes(
            self,
            cell_sizes_xi: Array,
            axis_index: int
            ) -> Array:
        """Performs a halo update for the cell sizes

        :param cell_sizes_xi: _description_
        :type cell_sizes_xi: Array
        :param axis_index: _description_
        :type axis_index: int
        :return: _description_
        :rtype: Array
        """

        split_factors = self.domain_information.split_factors
        axis_id_to_axis = self.domain_information.axis_id_to_axis
        axis =  axis_id_to_axis[axis_index]
        axis_to_face_locations = self.domain_information.axis_to_face_locations

        for face_location in axis_to_face_locations[axis]:

            boundary_condition_face: Tuple[BoundaryConditionsFace] = \
                getattr(self.boundary_conditions, face_location)
            
            if len(boundary_condition_face) > 1:
                boundary_type = "ZEROGRADIENT"
            else:
                boundary_type = boundary_condition_face[0].boundary_type
            
            if boundary_type in ["DIRICHLET", "NEUMANN"]:
                boundary_type = "ZEROGRADIENT"
            elif "WALL" in boundary_type:
                boundary_type = "SYMMETRY"
            else:
                pass
            
            slice_retrieve = self.face_slices_retrieve_conservatives[boundary_type][face_location]
            slice_retrieve = slice_retrieve[-3 + axis_index]
            slice_fill = self.halo_slices.face_slices_conservatives[face_location]
            slice_fill = slice_fill[-3 + axis_index]
            halos = cell_sizes_xi[slice_retrieve]

            if split_factors[axis_index] > 1:
                device_id = jax.lax.axis_index(axis_name="i")
                mask = self.face_halo_mask
                mask_value = mask[face_location][device_id]
                cell_sizes_xi = cell_sizes_xi.at[slice_fill].mul(1 - mask_value)
                cell_sizes_xi = cell_sizes_xi.at[slice_fill].add(halos * mask_value)
            else:
                cell_sizes_xi = cell_sizes_xi.at[slice_fill].set(halos)
            
        return cell_sizes_xi

    def face_halo_update_cell_centers(
            self,
            cell_centers_xi: Array,
            axis_index: int
            ) -> Array:
        """Performs a halo update for the cell centers.
        If mesh stretching is active, the cell center
        halos are computed from the cell size halos.

        :param cell_centers_xi: _description_
        :type cell_centers_xi: Array
        :param axis_index: _description_
        :type axis_index: int
        :return: _description_
        :rtype: Array
        """

        is_mesh_stretching = self.domain_information.is_mesh_stretching
        axis_id_to_axis = self.domain_information.axis_id_to_axis
        axis = axis_id_to_axis[axis_index]
        axis_to_face_locations = self.domain_information.axis_to_face_locations
        face_location_to_axis_side = self.domain_information.face_location_to_axis_side
        nh = self.domain_information.nh_conservatives

        cell_sizes_with_halos_xi = self.domain_information.get_device_cell_sizes_halos()[axis_index].flatten()
        cell_sizes_xi = self.domain_information.get_device_cell_sizes()[axis_index].flatten()

        for face_location in axis_to_face_locations[axis]:
            
            axis_side = face_location_to_axis_side[face_location]
            halo_slices = self.halo_slices.face_slices_conservatives[face_location]
            halo_slices = halo_slices[-3 + axis_index]

            if is_mesh_stretching[axis_index]:
                dxi_at_bound = cell_sizes_xi[axis_side]
                halos = cell_sizes_with_halos_xi[halo_slices]

                if axis_side == -1:
                    xi_at_bound = cell_centers_xi[-nh-1]
                    halos = jnp.concatenate([jnp.array([dxi_at_bound]), halos])
                    halos = 0.5*(halos[:-1] + halos[1:])
                    halos = xi_at_bound + jnp.cumsum(halos)
                elif axis_side == 0:
                    xi_at_bound = cell_centers_xi[nh]
                    halos = jnp.concatenate([halos, jnp.array([dxi_at_bound])])
                    halos = 0.5*(halos[:-1] + halos[1:])
                    halos = xi_at_bound - jnp.cumsum(halos[::-1])[::-1]

            else:
                if axis_side == -1:
                    xi_at_bound = cell_centers_xi[-nh-1]
                    halos = xi_at_bound + jnp.arange(1, nh + 1, 1) * cell_sizes_xi
                elif axis_side == 0:
                    xi_at_bound = cell_centers_xi[nh]
                    halos = xi_at_bound - jnp.arange(nh, 0, -1) * cell_sizes_xi

            cell_centers_xi = cell_centers_xi.at[halo_slices].set(halos)

        return cell_centers_xi

    def edge_halo_update(self):
        pass
            
    def vertex_halo_update(self):
        pass