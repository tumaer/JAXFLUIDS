from functools import partial
from typing import List, Tuple, Dict
import types

import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt

from jaxfluids.domain.domain_information import DomainInformation
from jaxfluids.equation_manager import EquationManager
from jaxfluids.halos.inner.halo_communication import HaloCommunication
from jaxfluids.data_types.case_setup.boundary_conditions import BoundaryConditionsField, BoundaryConditionsFace
from jaxfluids.domain import FACE_LOCATIONS, EDGE_LOCATIONS, VERTEX_LOCATIONS

Array = jax.Array

class HaloCommunicationMesh(HaloCommunication):
    def __init__(
            self,
            domain_information: DomainInformation,
            boundary_conditions: BoundaryConditionsField
            ) -> None:
        super().__init__(domain_information, boundary_conditions)
        self.domain_information = domain_information
        self.boundary_conditions = boundary_conditions

    def face_halo_update(
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
    
    def edge_halo_update(self):
        pass

    def vertex_halo_update(self):
        pass