from abc import ABC, abstractmethod
from typing import Dict, List, Tuple

import jax
import jax.numpy as jnp
from jax import Array
import numpy as np

from jaxfluids.domain.domain_information import DomainInformation
from jaxfluids.halos.halo_slices import HaloSlices
from jaxfluids.data_types.case_setup.boundary_conditions import BoundaryConditionsField, BoundaryConditionsFace
from jaxfluids.domain import EDGE_LOCATIONS, VERTEX_LOCATIONS, FACE_LOCATIONS
from jaxfluids.halos.outer import EDGE_TYPES, VERTEX_TYPES


class BoundaryCondition(ABC):
    """ The BoundaryCondition class implements functionality to enforce user-
    specified boundary conditions.
    """

    def __init__(
            self,
            domain_information: DomainInformation,
            boundary_conditions: BoundaryConditionsField
            ) -> None:
        
        self.domain_information = domain_information
        self.boundary_conditions = boundary_conditions

        self.edge_types_to_location_retrieve = get_types_to_location_retrieve(EDGE_LOCATIONS, EDGE_TYPES)
        self.vertex_types_to_location_retrieve = get_types_to_location_retrieve(VERTEX_LOCATIONS, VERTEX_TYPES)

        self.edge_flip_slices_symmetry = get_flip_slices_symmetry(EDGE_LOCATIONS, EDGE_TYPES)
        self.vertex_flip_slices_symmetry = get_flip_slices_symmetry(VERTEX_LOCATIONS, VERTEX_TYPES)

        active_edge_locations = self.domain_information.active_edge_locations
        self.edge_boundary_types = self.assign_boundary_type_combinations(active_edge_locations)
        self.vertex_boundary_types = self.assign_boundary_type_combinations(VERTEX_LOCATIONS)

        self.halo_slices = HaloSlices(
            nh_conservatives = domain_information.nh_conservatives,
            nh_geometry = domain_information.nh_geometry,
            active_axes_indices = domain_information.active_axes_indices)

        # FACE LOCATION TO ACTIVE AXIS INDICES
        active_axes_indices = domain_information.active_axes_indices
        self.face_location_to_axis_indices = {}
        face_location_to_axis_indices = { 
            "east"  : [1,2], "west"  : [1,2],
            "north" : [0,2], "south" : [0,2],
            "top"   : [0,1], "bottom": [0,1]
        }
        for key, indices in face_location_to_axis_indices.items():
            active_indices = [i for i in indices if i in active_axes_indices]
            self.face_location_to_axis_indices[key] = active_indices

        # RETRIEVE SLICES FOR EDGE HALOS CONSERVATIVES
        nh_conservatives = domain_information.nh_conservatives
        nhx, nhy, nhz = domain_information.domain_slices_conservatives
        self.face_slices_retrieve_conservatives = get_face_slices_retrieve(
            nh_conservatives, nhx, nhy, nhz)

        # RETRIEVE SLICES FOR EDGE HALOS GEOMETRY
        nh_geometry = domain_information.nh_geometry
        if nh_geometry != None:
            nhx, nhy, nhz = domain_information.domain_slices_geometry
            self.face_slices_retrieve_geometry = get_face_slices_retrieve(
                nh_geometry, nhx, nhy, nhz)

        if domain_information.is_parallel:
            self.face_halo_mask = self.get_face_halo_mask()
            self.edge_halo_mask = self.get_edge_halo_mask()
            self.vertex_halo_mask = self.get_vertex_halo_mask()

    def get_boundary_coordinates_at_location(
            self,
            face_location
            ) -> Tuple:
        """Gets the mesh grid that corresponds
        to the boundary coordinates at the
        present location and the axes to expand.

        :param face_location: _description_
        :type face_location: _type_
        :return: _description_
        :rtype: Tuple
        """
        cell_centers = self.domain_information.get_device_cell_centers()
        cell_centers = [xi.flatten() for xi in cell_centers]
        indices = self.face_location_to_axis_indices[face_location]
        axes_to_expand = [i for i in range(3) if i not in indices]
        coordinates = []
        for axis_index in indices:
            coordinates.append(cell_centers[axis_index])
        meshgrid = jnp.meshgrid(*coordinates, indexing="ij")
        return meshgrid, axes_to_expand

    def assign_boundary_type_combinations(
            self,
            LOCATIONS: Tuple[str]
            ) -> Dict[str,str]:
        """Identifies the boundary types of the faces
        intersecting at the edges/vertices of the computational domain.
        If one of the boundary types is PERIODIC or
        SYMMETRY and multiple boundary types are not
        present at the corresponding face, then the edge halos
        can be filled accordingly, i.e., they are filled
        so that PERIODIC/SYMMETRY is satisfied.
        If this is not the case, then the edge halos
        are filled with a mean value of the neighboring
        halos.

        :param LOCATIONS: _description_
        :type LOCATIONS: Tuple[str]
        :return: _description_
        :rtype: Dict[str,str]
        """

        type_combinations = {}
        for location in LOCATIONS:

            multiple_types_at_face_flags = []
            boundary_types = []

            for face_location in location.split("_"):
                boundary_condition_face: Tuple[BoundaryConditionsFace] = getattr(
                    self.boundary_conditions, face_location)
                if len(boundary_condition_face) > 1:
                    multiple_types_at_face_flags.append(True)
                    boundary_types.append(None)
                else:
                    multiple_types_at_face_flags.append(False)
                    boundary_types.append(boundary_condition_face[0].boundary_type)

            type_list = []
            loop = True
            for boundary_type, multiple_type_flag in zip(boundary_types, multiple_types_at_face_flags):
                if not multiple_type_flag and loop:
                    if boundary_type in ["PERIODIC", "SYMMETRY"]:
                        type_list.append(boundary_type)
                        loop = False
                    else:
                        type_list.append("ANY")
                else:
                    type_list.append("ANY")

            type_combinations[location] = "_".join(type_list)

        return type_combinations

    def get_face_halo_mask(self) -> Dict[str,Array]:
        """Generates the face halo masks for a decomposed domain.
        The face halo masks indicates whether the subdomain 
        must update its face halos according to the
        outer boundary condition. This is not the case if
        1) The face of the subdomain is not located at
        the outer boundary
        2) The boundary type is PERIODIC and the domain
        is split in the corresponding axis direction.
        If one of the aforementioned conditions is True,
        the halos must be updated using communication,
        i.e., an inner halo update.

        :return: _description_
        :rtype: Dict
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

        face_halo_mask = {}
        for face_location in active_face_locations:
            axis_index = face_location_to_axis_index[face_location]
            split_xi = split_factors[axis_index]
            slices = face_to_subdomain_slices[face_location]
            subdomain_ids_at_face = jnp.reshape(subdomain_ids_grid[slices], -1)
            boundary_conditions_face: Tuple[BoundaryConditionsFace] = getattr(self.boundary_conditions, face_location)
            boundary_type = boundary_conditions_face[0].boundary_type
            if boundary_type == "PERIODIC" and split_xi  > 1:
                mask = jnp.zeros_like(subdomain_ids_flat)
            else:
                mask = jnp.zeros_like(subdomain_ids_flat)
                mask = mask.at[subdomain_ids_at_face].set(1)
            face_halo_mask[face_location] = mask
        return face_halo_mask

    def get_edge_halo_mask(self) -> Dict[str,Array]:
        """Generates the edge halo masks for a decomposed domain.
        The edge halo masks indicates whether the subdomain 
        must update its edge halos according to the
        outer boundary condition. This is not the case if
        1) The edge of the subdomain is not located at
        the outer boundary
        2) The boundary type is PERIODIC and the domain
        is split in the corresponding axis direction.
        If one of the aforementioned conditions is True,
        the outer halos must be updated using communication,
        i.e., an inner halo update.

        :return: _description_
        :rtype: Dict
        """

        subdomain_ids_grid = self.domain_information.subdomain_ids_grid
        subdomain_ids_flat = self.domain_information.subdomain_ids_flat
        split_factors = self.domain_information.split_factors
        face_location_to_axis_index = self.domain_information.face_location_to_axis_index
        active_edge_locations = self.domain_information.active_edge_locations
        face_location_to_axis_side = self.domain_information.face_location_to_axis_side

        edge_to_subdomain_slices = {}
        for edge_location in EDGE_LOCATIONS:
            face_1, face_2 = edge_location.split("_")
            s_ = ()
            for i in range(3):
                axis_index_1 = face_location_to_axis_index[face_1]
                axis_index_2 = face_location_to_axis_index[face_2]
                if axis_index_1 == i:
                    axis_side = face_location_to_axis_side[face_1]
                    s_ += (axis_side,)
                elif axis_index_2 == i:
                    axis_side = face_location_to_axis_side[face_2]
                    s_ += (axis_side,)
                else:
                    s_ += (jnp.s_[:],)
            edge_to_subdomain_slices[edge_location] = s_

        edge_halo_mask = {}
        for edge_location in active_edge_locations:
            boundary_types = self.edge_boundary_types[edge_location]
            boundary_types_list = boundary_types.split("_")
            face_location_list = edge_location.split("_")
            flag = False
            for b_type, face_location in zip(boundary_types_list, face_location_list):
                axis_index = face_location_to_axis_index[face_location]
                split_xi = split_factors[axis_index]
                if b_type == "PERIODIC" and split_xi > 1:
                    flag = True
            if flag:
                mask = jnp.zeros_like(subdomain_ids_flat)
            else:
                slices = edge_to_subdomain_slices[edge_location]
                subdomain_ids_at_face = jnp.reshape(subdomain_ids_grid[slices], -1)
                mask = jnp.zeros_like(subdomain_ids_flat).at[subdomain_ids_at_face].set(1)
            edge_halo_mask[edge_location] = mask
        return edge_halo_mask

    def get_vertex_halo_mask(self) -> Dict[str,Array]:
        """Generates the vertex halo mask for a decomposed domain.
        The vertex halo mask indicates whether the subdomain 
        must update its vertex halos according to the
        outer boundary condition. This is not the case if
        1) The vertex of the subdomain is not located at
        the outer boundary
        2) The boundary type is PERIODIC and the domain
        is split in the corresponding axis direction.
        If one of the aforementioned conditions is True,
        the outer halos must be updated using communication,
        i.e., an inner halo update.

        :return: _description_
        :rtype: Dict
        """

        subdomain_ids_grid = self.domain_information.subdomain_ids_grid
        subdomain_ids_flat = self.domain_information.subdomain_ids_flat
        split_factors = self.domain_information.split_factors
        face_location_to_axis_index = self.domain_information.face_location_to_axis_index
        face_location_to_axis_side = self.domain_information.face_location_to_axis_side

        vertex_to_subdomain_slices = {}
        for vertex_location in VERTEX_LOCATIONS:
            face_locations = vertex_location.split("_")
            s_ = ()
            for i in range(3):
                for face in face_locations:
                    axis_index = face_location_to_axis_index[face]
                    axis_side = face_location_to_axis_side[face]
                    if axis_index == i:
                        s_ += (axis_side,)
                    else:
                        continue
            vertex_to_subdomain_slices[vertex_location] = s_

        vertex_halo_mask = {}
        for vertex_location in VERTEX_LOCATIONS:
            boundary_types = self.vertex_boundary_types[vertex_location]
            boundary_types_list = boundary_types.split("_")
            face_location_list = vertex_location.split("_")
            flag = False
            for b_type, face_location in zip(boundary_types_list, face_location_list):
                axis_index = face_location_to_axis_index[face_location]
                split_xi = split_factors[axis_index]
                if b_type == "PERIODIC" and split_xi > 1:
                    flag = True
            if flag:
                mask = jnp.zeros_like(subdomain_ids_flat)
            else:
                slices = vertex_to_subdomain_slices[vertex_location]
                subdomain_ids_at_face = jnp.reshape(subdomain_ids_grid[slices], -1)
                mask = jnp.zeros_like(subdomain_ids_flat).at[subdomain_ids_at_face].set(1)
            vertex_halo_mask[vertex_location] = mask
        return vertex_halo_mask

    @abstractmethod
    def face_halo_update(self, **args): 
        pass

    @abstractmethod
    def edge_halo_update(self, **args): 
        pass

    @abstractmethod
    def vertex_halo_update(self, **args): 
        pass

def get_face_slices_retrieve(
        nh: int,
        nhx: Tuple,
        nhy: Tuple,
        nhz: Tuple
        ) -> Dict:
    """Generates a mapping that assigns
    boundary type plus face location
    to slices that retrieve the
    required cells to update the halos at
    said face.

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

    face_slices_retrieve = {
        "PERIODIC" : {
            "east"      :   jnp.s_[..., nh:2*nh, nhy, nhz], 
            "west"      :   jnp.s_[..., -2*nh:-nh, nhy, nhz], 
            "north"     :   jnp.s_[..., nhx, nh:2*nh, nhz], 
            "south"     :   jnp.s_[..., nhx, -2*nh:-nh, nhz], 
            "top"       :   jnp.s_[..., nhx, nhy, nh:2*nh], 
            "bottom"    :   jnp.s_[..., nhx, nhy, -2*nh:-nh], 
        },
        "SYMMETRY" : {
            "east"      :   jnp.s_[..., -nh-1:-2*nh-1:-1, nhy, nhz], 
            "west"      :   jnp.s_[..., 2*nh-1:nh-1:-1, nhy, nhz], 
            "north"     :   jnp.s_[..., nhx, -nh-1:-2*nh-1:-1, nhz], 
            "south"     :   jnp.s_[..., nhx, 2*nh-1:nh-1:-1, nhz], 
            "top"       :   jnp.s_[..., nhx, nhy, -nh-1:-2*nh-1:-1], 
            "bottom"    :   jnp.s_[..., nhx, nhy, 2*nh-1:nh-1:-1], 
        },
        "NEUMANN" : {
            "east"      :   jnp.s_[..., -nh-1:-nh, nhy, nhz], 
            "west"      :   jnp.s_[..., nh:nh+1, nhy, nhz], 
            "north"     :   jnp.s_[..., nhx, -nh-1:-nh, nhz], 
            "south"     :   jnp.s_[..., nhx, nh:nh+1, nhz], 
            "top"       :   jnp.s_[..., nhx, nhy, -nh-1:-nh], 
            "bottom"    :   jnp.s_[..., nhx, nhy, nh:nh+1], 
        },
        "ZEROGRADIENT": {
            "east"      :   jnp.s_[..., -nh-1:-nh, nhy, nhz], 
            "west"      :   jnp.s_[..., nh:nh+1, nhy, nhz], 
            "north"     :   jnp.s_[..., nhx, -nh-1:-nh, nhz], 
            "south"     :   jnp.s_[..., nhx, nh:nh+1, nhz], 
            "top"       :   jnp.s_[..., nhx, nhy, -nh-1:-nh], 
            "bottom"    :   jnp.s_[..., nhx, nhy, nh:nh+1], 
        }
    }
    return face_slices_retrieve

def get_types_to_location_retrieve(
        LOCATIONS: Tuple[str],
        TYPES: Tuple[str]
        ) -> Dict[str, Dict[str, str]]:
    """Generates a mapping that assigns
    boundary types at the edges/vertices of
    the domain to the locations to
    retrieve the halos from.

    :param LOCATIONS: _description_
    :type LOCATIONS: Tuple
    :param TYPES: _description_
    :type TYPES: Tuple
    :raises NotImplementedError: _description_
    :return: _description_
    :rtype: Dict[str, Dict[str, str]]
    """

    types_to_location_retrieve = {}

    face_to_opposite_face = DomainInformation.face_to_opposite_face

    for edge_location in LOCATIONS:
        temp_dict = {}
        for boundary_type in TYPES:
            location_list = []
            value_list = []
            boundary_type_list = boundary_type.split("_")
            face_location_list = edge_location.split("_")
            for face, b_type in zip(face_location_list, boundary_type_list):
                if b_type == "PERIODIC":
                    location = face_to_opposite_face[face]
                    value = "1"
                elif b_type == "SYMMETRY":
                    location = face
                    value = "1"
                elif b_type == "ANY":
                    location = face
                    value = "0"
                else:
                    raise NotImplementedError
                value_list.append(value)
                location_list.append(location)
            location_retrieve = "_".join(location_list) + "_" + "".join(value_list)
            temp_dict[boundary_type] = location_retrieve
        types_to_location_retrieve[edge_location] = temp_dict

    return types_to_location_retrieve


def get_flip_slices_symmetry(
        LOCATIONS: Tuple[str],
        TYPES: Tuple[str]
        ) -> Dict[str, Dict[str, Tuple]]:
    """Generates a mapping that assigns
    SYMMETRY boundary types at edges/vertices
    to flip slices.

    :param LOCATIONS: _description_
    :type LOCATIONS: Tuple[str]
    :param TYPES: _description_
    :type TYPES: Tuple[str]
    :return: _description_
    :rtype: Dict[str, Dict[str, Tuple]]
    """
    
    flip_slices = {}
    face_location_to_axis_index = DomainInformation.face_location_to_axis_index
    for location in LOCATIONS:
        temp_dict = {}
        for boundary_types in TYPES:
            if not "SYMMETRY" in boundary_types:
                continue
            location_list = location.split("_")
            boundary_types_list = boundary_types.split("_")
            s_ = (...,)
            for i in range(3):
                flag = True
                for face, b_type in zip(location_list, boundary_types_list):
                    axis_index = face_location_to_axis_index[face]
                    if axis_index == i and b_type == "SYMMETRY":
                        flag = False
                        s_ += (jnp.s_[::-1],)
                        break
                if flag:
                    s_ += (jnp.s_[:],)
            temp_dict[boundary_types] = s_
        flip_slices[location] = temp_dict

    return flip_slices

def get_signs_symmetry(
        no_primes: int,
        equation_type: str,
        vel_indices: Tuple[int]
        ) -> Tuple[Dict[str, Array],
                   Dict[str, Dict[str, Array]],
                   Dict[str, Dict[str, Array]]]:
    """Generates a mapping that assigns 
    boundary locations to sign change
    indices for SYMMETRY boundary types.

    :param no_primes: _description_
    :type no_primes: int
    :param equation_type: _description_
    :type equation_type: str
    :param vel_indices: _description_
    :type vel_indices: Tuple[int]
    :return: _description_
    :rtype: Tuple[Dict[str, Array], Dict[str, Dict[str, Array]], Dict[str, Dict[str, Array]]]
    """

    face_location_to_axis_index = DomainInformation.face_location_to_axis_index

    face_signs = {}
    for face_location in FACE_LOCATIONS:
        axis_index = face_location_to_axis_index[face_location]
        vel_id = vel_indices[axis_index]
        signs = jnp.ones(no_primes)
        signs = signs.at[vel_id].mul(-1.0)
        if equation_type == "TWO-PHASE-LS":
            signs = jnp.reshape(signs, (-1,1,1,1,1))
        else:
            signs = jnp.reshape(signs, (-1,1,1,1))
        face_signs[face_location] = signs

    def get_signs(
            LOCATIONS: Tuple[str],
            TYPES: Tuple[str]
            ) -> Dict[str, Dict[str,Array]]:
        signs_dict = {}
        for location in LOCATIONS:
            temp_dict = {}
            for boundary_type in TYPES:
                if not "SYMMETRY" in boundary_type:
                    continue
                location_list = location.split("_")
                boundary_types_list = boundary_type.split("_")
                signs = jnp.ones(no_primes)
                for b_type, face in zip(boundary_types_list, location_list):
                    if b_type == "SYMMETRY":
                        axis_index = face_location_to_axis_index[face]
                        vel_id = vel_indices[axis_index]
                        signs = signs.at[vel_id].mul(-1.0)
                if equation_type == "TWO-PHASE-LS":
                    signs = jnp.reshape(signs, (-1,1,1,1,1))
                else:
                    signs = jnp.reshape(signs, (-1,1,1,1))
                temp_dict[boundary_type] = signs
            signs_dict[location] = temp_dict
        return signs_dict
    
    edge_signs = get_signs(EDGE_LOCATIONS, EDGE_TYPES)
    vertex_signs = get_signs(VERTEX_LOCATIONS, VERTEX_TYPES)

    return face_signs, edge_signs, vertex_signs
