from functools import partial
from typing import List, Tuple, Dict
import types

import jax
import jax.numpy as jnp
from jax import Array
import numpy as np
import matplotlib.pyplot as plt

from jaxfluids.domain.domain_information import DomainInformation
from jaxfluids.equation_manager import EquationManager
from jaxfluids.halos.halo_slices import HaloSlices
from jaxfluids.data_types.case_setup.boundary_conditions import BoundaryConditionsField, BoundaryConditionsFace
from jaxfluids.domain import FACE_LOCATIONS, EDGE_LOCATIONS, VERTEX_LOCATIONS

class HaloCommunication:
    def __init__(
            self,
            domain_information: DomainInformation,
            equation_manager: EquationManager,
            boundary_conditions: BoundaryConditionsField
            ) -> None:
        
        self.domain_information = domain_information
        self.equation_manager = equation_manager
        self.boundary_conditions = boundary_conditions

        self.halo_slices = HaloSlices(
            nh_conservatives = domain_information.nh_conservatives,
            nh_geometry = domain_information.nh_geometry,
            active_axes_indices = domain_information.active_axes_indices)

        # PERMUTATIONS
        self.send_permutations = {
            "east": [],
            "west": [],
            "north": [],
            "south": [],
            "top": [],
            "bottom": [],
        }

        subdomain_ids_grid = domain_information.subdomain_ids_grid
        subdomain_ids_flat = domain_information.subdomain_ids_flat

        for k in range(subdomain_ids_grid.shape[2]):
            for j in range(subdomain_ids_grid.shape[1]):
                for i in range(subdomain_ids_grid.shape[0]):
                    
                    target_index_i = 0 if i == subdomain_ids_grid.shape[0] - 1 else i + 1
                    target_index_j = 0 if j == subdomain_ids_grid.shape[1] - 1 else j + 1
                    target_index_k = 0 if k == subdomain_ids_grid.shape[2] - 1 else k + 1

                    # X
                    source_id = subdomain_ids_grid[i,j,k]
                    target_id = subdomain_ids_grid[target_index_i,j,k]
                    source_id_index = int(jnp.where(subdomain_ids_flat == source_id)[0][0])
                    target_id_index = int(jnp.where(subdomain_ids_flat == target_id)[0][0])
                    self.send_permutations["west"].append( (source_id_index, target_id_index) )
                    self.send_permutations["east"].append( (target_id_index, source_id_index) )

                    # Y
                    source_id = subdomain_ids_grid[i,j,k]
                    target_id = subdomain_ids_grid[i,target_index_j,k]
                    source_id_index = int(jnp.where(subdomain_ids_flat == source_id)[0][0])
                    target_id_index = int(jnp.where(subdomain_ids_flat == target_id)[0][0])
                    self.send_permutations["south"].append( (source_id_index, target_id_index) )
                    self.send_permutations["north"].append( (target_id_index, source_id_index) )

                    # Z
                    source_id = subdomain_ids_grid[i,j,k]
                    target_id = subdomain_ids_grid[i,j,target_index_k]
                    source_id_index = int(jnp.where(subdomain_ids_flat == source_id)[0][0])
                    target_id_index = int(jnp.where(subdomain_ids_flat == target_id)[0][0])
                    self.send_permutations["bottom"].append( (source_id_index, target_id_index) )
                    self.send_permutations["top"].append( (target_id_index, source_id_index) )

        # RETRIEVE SLICES
        dim = domain_information.dim
        nh_conservatives = domain_information.nh_conservatives
        nhx, nhy, nhz = domain_information.domain_slices_conservatives
        self.face_slices_retrieve_conservatives = get_face_slices_retrieve(
            nh_conservatives, nhx, nhy, nhz)
        self.edge_slices_retrieve_conservatives = get_edge_slices_retrieve(
            nh_conservatives, nhx, nhy, nhz)
        self.vertex_slices_retrieve_conservatives = get_vertex_slices_retrieve(
            nh_conservatives)

        nh_geometry = self.domain_information.nh_geometry
        if nh_geometry != None:
            nhx, nhy, nhz = domain_information.domain_slices_geometry
            self.face_slices_retrieve_geometry = get_face_slices_retrieve(
                nh_geometry, nhx, nhy, nhz)
            self.edge_slices_retrieve_geometry = get_edge_slices_retrieve(
                nh_geometry, nhx, nhy, nhz)
            self.vertex_slices_retrieve_geometry = get_vertex_slices_retrieve(
                nh_geometry)
            
        self.halo_mask = self.get_halo_mask()
    
    def get_halo_mask(self) -> Dict[str, Array]:
        """Generates halo masks for the inner halo update.
        The masks indicates whether the subdomain must update
        its halos using communication. This is not the case
        if the halos are located at the outer boundary in the
        present communication (axis) direction unless the boundary type
        is PERIODIC and the domain is split in said
        axis direction. Otherwise, the outer boundary 
        condition is used to update the halos.

        :return: _description_
        :rtype: Dict[str, Array]
        """
    
        subdomain_ids_grid = self.domain_information.subdomain_ids_grid
        subdomain_ids_flat = self.domain_information.subdomain_ids_flat
        split_factors = self.domain_information.split_factors
        face_location_to_axis_index = self.domain_information.face_location_to_axis_index
        face_location_to_axis_side = self.domain_information.face_location_to_axis_side
        active_face_locations = self.domain_information.active_face_locations

        face_to_subdomain_slices = {}
        for face_location in FACE_LOCATIONS:
            s_ = ()
            axis_side = face_location_to_axis_side[face_location]
            axis_index = face_location_to_axis_index[face_location]
            for i in range(3):
                if axis_index == i:
                    s_ += (axis_side,)
                else:
                    s_ += (jnp.s_[:],)
            face_to_subdomain_slices[face_location] = s_

        halo_mask = {}
        for face_location in active_face_locations:
            axis_index = face_location_to_axis_index[face_location]
            split_xi = split_factors[axis_index]
            slices = face_to_subdomain_slices[face_location]
            subdomain_ids_at_face = jnp.reshape(subdomain_ids_grid[slices], -1)
            boundary_conditions_face: Tuple[BoundaryConditionsFace] = getattr(self.boundary_conditions, face_location)
            boundary_type = boundary_conditions_face[0].boundary_type
            if boundary_type == "PERIODIC" and split_xi  > 1:
                mask = jnp.ones_like(subdomain_ids_flat)
            else:
                mask = jnp.ones_like(subdomain_ids_flat).at[subdomain_ids_at_face].set(0)
            halo_mask[face_location] = mask

        return halo_mask

    def face_halo_update(
            self,
            buffer: Array,
            conservatives: Array = None,
            is_geometry_halos: bool = False
            ) -> Tuple[Array, Array]:
        """Updates the inner face halos for the
        specified buffer. Performs a permute communication
        in each active direction and updates the face
        halos accordingly. If conservatives is not None,
        then buffer is the primitive variable buffer
        and conservatives will be computed accordingly.
        The boolean is_geometry_halos specifies the buffers'
        halo size.

        :param buffer: _description_
        :type buffer: Array
        :param primitives: _description_, defaults to None
        :type primitives: Array, optional
        :param is_geometry_halos: _description_, defaults to False
        :type is_geometry_halos: bool, optional
        :return: _description_
        :rtype: Tuple[Array, Array]
        """

        split_factors = self.domain_information.split_factors
        face_locations_to_axis_index = self.domain_information.face_location_to_axis_index
        active_face_locations = self.domain_information.active_face_locations

        if is_geometry_halos:
            face_slices_fill = self.halo_slices.face_slices_geometry
            face_slices_retrieve = self.face_slices_retrieve_geometry
        else:
            face_slices_fill = self.halo_slices.face_slices_conservatives
            face_slices_retrieve = self.face_slices_retrieve_conservatives

        for face_location in active_face_locations:
            
            axis_index = face_locations_to_axis_index[face_location]
            split_xi = split_factors[axis_index]
            if split_xi == 1:
                continue 

            # SLICE OBJECTS
            slice_fill = face_slices_fill[face_location]
            slice_retrieve = face_slices_retrieve[face_location]

            # SEND HALO BUFFER TO NEIGHBOR
            permutation = self.send_permutations[face_location]
            buffer_halos_retrieve = jax.lax.ppermute(buffer[slice_retrieve], perm=permutation, axis_name="i")

            # RESET AND FILL HALOS
            device_id = jax.lax.axis_index(axis_name="i")
            mask = self.halo_mask[face_location]
            mask_value = mask[device_id]
            
            buffer = buffer.at[slice_fill].mul(1.0 - mask_value)
            buffer = buffer.at[slice_fill].add(buffer_halos_retrieve * mask_value)
            if conservatives != None:
                cons_halos_retrieve = \
                self.equation_manager.get_conservatives_from_primitives(
                    buffer_halos_retrieve)
                conservatives = conservatives.at[slice_fill].mul(1.0 - mask_value)
                conservatives = conservatives.at[slice_fill].add(cons_halos_retrieve * mask_value)

        if conservatives != None:
            return buffer, conservatives
        else:
            return buffer

    def edge_halo_update(
            self,
            buffer: Array,
            conservatives: Array = None,
            is_geometry_halos: bool = False
            ) -> Tuple[Array, Array]:
        """Updates the inner edge halos for the
        specified buffer. If conservatives is not None,
        then buffer is the primitive variable buffer
        and conservatives will be computed accordingly.
        The boolean is_geometry_halos specifies the buffers'
        halo size.


        :param conservatives: _description_
        :type conservatives: Array
        :param primitives: _description_
        :type primitives: Array
        :param mask: _description_
        :type mask: Dict
        :return: _description_
        :rtype: Tuple[Array, Array]
        """

        if is_geometry_halos:
            edge_slices_retrieve = self.edge_slices_retrieve_geometry
            edge_slices_fill = self.halo_slices.edge_slices_geometry
        else:
            edge_slices_retrieve = self.edge_slices_retrieve_conservatives
            edge_slices_fill = self.halo_slices.edge_slices_conservatives

        split_factors = self.domain_information.split_factors
        face_locations_to_axis_index = self.domain_information.face_location_to_axis_index
        active_face_locations = self.domain_information.active_face_locations
        active_edge_locations = self.domain_information.active_edge_locations

        # TODO dont loop over edges multiple times
        for face_location in active_face_locations:

            axis_index = face_locations_to_axis_index[face_location]
            split_xi = split_factors[axis_index]
            if split_xi == 1:
                continue 

            # ARANGE RETRIEVE BUFFER
            buffer_halos_retrieve_dict = {}
            for edge_location, slice_retrieve in edge_slices_retrieve[face_location].items():
                if edge_location not in active_edge_locations:
                    continue
                buffer_halos_retrieve_dict[edge_location] = buffer[slice_retrieve]

            # SEND RETRIEVE BUFFER
            permutation = self.send_permutations[face_location]
            buffer_halos_retrieve_dict = jax.lax.ppermute(
                buffer_halos_retrieve_dict, perm=permutation,
                axis_name="i")

            for edge_location in edge_slices_retrieve[face_location].keys():
                if edge_location not in active_edge_locations:
                    continue
                
                # RESET AND FILL BUFFER
                slice_fill = edge_slices_fill[edge_location]
                buffer_halos_retrieve = buffer_halos_retrieve_dict[edge_location]

                device_id = jax.lax.axis_index(axis_name="i")
                mask = self.halo_mask[face_location]
                mask_value = mask[device_id]
                
                buffer = buffer.at[slice_fill].mul(1.0 - mask_value)
                buffer = buffer.at[slice_fill].add(buffer_halos_retrieve * mask_value)
                if conservatives != None:
                    cons_halos_retrieve = \
                    self.equation_manager.get_conservatives_from_primitives(
                            buffer_halos_retrieve)
                    conservatives = conservatives.at[slice_fill].mul(1.0 - mask_value)
                    conservatives = conservatives.at[slice_fill].add(cons_halos_retrieve * mask_value)
        
        if conservatives != None:
            return buffer, conservatives
        else:
            return buffer

    def vertex_halo_update(
            self,
            buffer: Array,
            conservatives: Array = None,
            is_geometry_halos: bool = False
            ) -> Tuple[Array, Array]:
        """Updates the inner vertex halos for the
        specified buffer. If conservatives is not None,
        then buffer is the primitive variable buffer
        and conservatives will be computed accordingly.
        The boolean is_geometry_halos specifies the buffers'
        halo size.

        :param conservatives: _description_
        :type conservatives: Array
        :param primitives: _description_
        :type primitives: Array
        :param mask: _description_
        :type mask: Dict
        :return: _description_
        :rtype: Tuple[Array, Array]
        """

        if is_geometry_halos:
            vertex_slices_retrieve = self.vertex_slices_retrieve_geometry
            vertex_slices_fill = self.halo_slices.vertex_slices_geometry
        else:
            vertex_slices_retrieve = self.vertex_slices_retrieve_conservatives
            vertex_slices_fill = self.halo_slices.vertex_slices_conservatives

        split_factors = self.domain_information.split_factors
        face_locations_to_axis_index = self.domain_information.face_location_to_axis_index
        
        # TODO dont loop over vertices multiple times
        for face_location in FACE_LOCATIONS:

            axis_index = face_locations_to_axis_index[face_location]
            split_xi = split_factors[axis_index]
            if split_xi == 1:
                continue 

            # ARANGE RETRIEVE BUFFER
            buffer_halos_retrieve_dict = {}
            for vertex_location, slice_retrieve in vertex_slices_retrieve[face_location].items():
                buffer_halos_retrieve_dict[vertex_location] = buffer[slice_retrieve]

            # SEND RETRIEVE BUFFER
            permutation = self.send_permutations[face_location]
            buffer_halos_retrieve_dict = jax.lax.ppermute(
                buffer_halos_retrieve_dict, perm=permutation,
                axis_name="i")

            for vertex_location in vertex_slices_retrieve[face_location].keys():

                # RESET AND FILL BUFFER
                buffer_halos_retrieve = buffer_halos_retrieve_dict[vertex_location]
                slice_fill = vertex_slices_fill[vertex_location]

                device_id = jax.lax.axis_index(axis_name="i")
                mask = self.halo_mask[face_location]
                mask_value = mask[device_id]

                buffer = buffer.at[slice_fill].mul(1.0 - mask_value)
                buffer = buffer.at[slice_fill].add(buffer_halos_retrieve * mask_value)
                if conservatives != None:
                    cons_halos_retrieve = \
                    self.equation_manager.get_conservatives_from_primitives(
                            buffer_halos_retrieve)
                    conservatives = conservatives.at[slice_fill].mul(1.0 - mask_value)
                    conservatives = conservatives.at[slice_fill].add(cons_halos_retrieve * mask_value)
        
        if conservatives != None:
            return buffer, conservatives
        else:
            return buffer


    def face_halo_update_mesh(
            self,
            mesh_xi: Array,
            axis_index: int
            ) -> Array:
        """Updates the inner face halos
        of the mesh, i.e., cell centers or
        cell sizes.

        :param mesh_xi: _description_
        :type mesh_xi: Array
        :param axis_index: _description_
        :type axis_index: int
        :return: _description_
        :rtype: Array
        """
        face_slices_fill = self.halo_slices.face_slices_conservatives
        face_slices_retrieve = self.face_slices_retrieve_conservatives
        axis_id_to_axis = self.domain_information.axis_id_to_axis
        axis =  axis_id_to_axis[axis_index]
        axis_to_face_locations = self.domain_information.axis_to_face_locations
        for face_location in axis_to_face_locations[axis]:
            slice_fill = face_slices_fill[face_location]
            slice_retrieve = face_slices_retrieve[face_location]
            slice_fill = slice_fill[-3 + axis_index]
            slice_retrieve = slice_retrieve[-3 + axis_index]
            permutation = self.send_permutations[face_location]
            halos = jax.lax.ppermute(mesh_xi[slice_retrieve], perm=permutation, axis_name="i")
            mesh_xi = mesh_xi.at[slice_fill].set(halos)
        return mesh_xi
    


def get_face_slices_retrieve(
        nh: int,
        nhx: Tuple,
        nhy: Tuple,
        nhz: Tuple
        ) -> Dict[str, Tuple]:
    """Generates the slices for cells 
    that are required by the face
    halo update.

    :param nh: _description_
    :type nh: int
    :param nhx: _description_
    :type nhx: Tuple
    :param nhy: _description_
    :type nhy: Tuple
    :param nhz: _description_
    :type nhz: Tuple
    :return: _description_
    :rtype: Dict
    """
    edge_slices_retrieve = {
        "east"      :   jnp.s_[..., nh:2*nh, nhy, nhz],
        "west"      :   jnp.s_[..., -2*nh:-nh, nhy, nhz],
        "north"     :   jnp.s_[..., nhx, nh:2*nh, nhz],
        "south"     :   jnp.s_[..., nhx, -2*nh:-nh, nhz],
        "top"       :   jnp.s_[..., nhx, nhy, nh:2*nh],
        "bottom"    :   jnp.s_[..., nhx, nhy, -2*nh:-nh],
    }
    return edge_slices_retrieve

def get_edge_slices_retrieve(
        nh: int,
        nhx: Tuple,
        nhy: Tuple,
        nhz: Tuple
        ) -> Dict[str, Dict[str, Tuple]]:
    """Generates the slices for the cells that are 
    required by the edge halo update in 2D.

    :param nh: _description_
    :type nh: int
    :param nhx: _description_
    :type nhx: Tuple
    :param nhy: _description_
    :type nhy: Tuple
    :param nhz: _description_
    :type nhz: Tuple
    :return: _description_
    :rtype: Dict
    """

    edge_slices_retrieve = {

        "east": {
            "east_south"    : jnp.s_[..., nh:2*nh, :nh, nhz], 
            "east_north"    : jnp.s_[..., nh:2*nh, -nh:, nhz],
            "east_bottom"   : jnp.s_[..., nh:2*nh, nhy, :nh],
            "east_top"      : jnp.s_[..., nh:2*nh, nhy, -nh:],
        },
        "west": {
            "west_south"    : jnp.s_[..., -2*nh:-nh, :nh, nhz],
            "west_north"    : jnp.s_[..., -2*nh:-nh, -nh:, nhz],
            "west_bottom"   : jnp.s_[..., -2*nh:-nh, nhy, :nh],
            "west_top"      : jnp.s_[..., -2*nh:-nh, nhy, -nh:],
        },

        "north": {
            "west_north"    : jnp.s_[..., :nh, nh:2*nh, nhz], 
            "east_north"    : jnp.s_[..., -nh:, nh:2*nh, nhz],
            "north_bottom"  : jnp.s_[..., nhx, nh:2*nh, :nh],
            "north_top"     : jnp.s_[..., nhx, nh:2*nh, -nh:],
        },
        "south": {
            "west_south"    : jnp.s_[..., :nh, -2*nh:-nh, nhz],
            "east_south"    : jnp.s_[..., -nh:, -2*nh:-nh, nhz],
            "south_bottom"  : jnp.s_[..., nhx, -2*nh:-nh, :nh],
            "south_top"     : jnp.s_[..., nhx, -2*nh:-nh, -nh:],
        },

        "top": {
            "east_top"      : jnp.s_[..., -nh:, nhy, nh:2*nh], 
            "west_top"      : jnp.s_[..., :nh, nhy, nh:2*nh],
            "north_top"     : jnp.s_[..., nhx, -nh:, nh:2*nh],
            "south_top"     : jnp.s_[..., nhx, :nh, nh:2*nh],
        },
        "bottom": {
            "east_bottom"   : jnp.s_[..., -nh:, nhy, -2*nh:-nh],
            "west_bottom"   : jnp.s_[..., :nh, nhy, -2*nh:-nh],
            "north_bottom"  : jnp.s_[..., nhx, -nh:, -2*nh:-nh],
            "south_bottom"  : jnp.s_[..., nhx, :nh, -2*nh:-nh],
        }

    }

    return edge_slices_retrieve


def get_vertex_slices_retrieve(
        nh: int
        ) -> Dict[str, Dict[str, Tuple]]:
    """Generates the slices for the cells that are 
    required by the vertex halo update.

    :param nh: _description_
    :type nh: int
    :param nhx: _description_
    :type nhx: Tuple
    :param nhy: _description_
    :type nhy: Tuple
    :param nhz: _description_
    :type nhz: Tuple
    :return: _description_
    :rtype: Dict
    """

    vertex_slices_retrieve = {

        "east": {
            "east_north_top"    : jnp.s_[..., nh:2*nh, -nh:, -nh:], 
            "east_south_top"    : jnp.s_[..., nh:2*nh, :nh, -nh:],
            "east_north_bottom" : jnp.s_[..., nh:2*nh, -nh:, :nh],
            "east_south_bottom" : jnp.s_[..., nh:2*nh, :nh, :nh],
        },
        "west": {
            "west_south_bottom" : jnp.s_[..., -2*nh:-nh, :nh, :nh],
            "west_south_top"    : jnp.s_[..., -2*nh:-nh, :nh, -nh:],
            "west_north_bottom" : jnp.s_[..., -2*nh:-nh, -nh:, :nh],
            "west_north_top"    : jnp.s_[..., -2*nh:-nh, -nh:, -nh:],
        },

        "north": {
            "east_north_top"    : jnp.s_[..., -nh:, nh:2*nh, -nh:], 
            "east_north_bottom" : jnp.s_[..., -nh:, nh:2*nh, :nh],
            "west_north_bottom" : jnp.s_[..., :nh, nh:2*nh, :nh],
            "west_north_top"    : jnp.s_[..., :nh, nh:2*nh, -nh:],
        },
        "south": {
            "west_south_bottom" : jnp.s_[..., :nh, -2*nh:-nh, :nh],
            "west_south_top"    : jnp.s_[..., :nh, -2*nh:-nh, -nh:],
            "east_south_top"    : jnp.s_[..., -nh:, -2*nh:-nh, -nh:],
            "east_south_bottom" : jnp.s_[..., -nh:, -2*nh:-nh, :nh],
        },

        "top": {
            "west_south_top"    : jnp.s_[..., :nh, :nh, nh:2*nh], 
            "east_south_top"    : jnp.s_[..., -nh:, :nh, nh:2*nh],
            "west_north_top"    : jnp.s_[..., :nh, -nh:, nh:2*nh],
            "east_north_top"    : jnp.s_[..., -nh:, -nh:, nh:2*nh],
        },
        "bottom": {
            "east_south_bottom" : jnp.s_[..., -nh:, :nh, -2*nh:-nh],
            "west_south_bottom" : jnp.s_[..., :nh, :nh, -2*nh:-nh],
            "east_north_bottom" : jnp.s_[..., -nh:, -nh:, -2*nh:-nh],
            "west_north_bottom" : jnp.s_[..., :nh, -nh:, -2*nh:-nh],
        }
    }

    return vertex_slices_retrieve