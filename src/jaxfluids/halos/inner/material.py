from functools import partial
from typing import List, Tuple, Dict
import types

import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt

from jaxfluids.domain.domain_information import DomainInformation
from jaxfluids.equation_manager import EquationManager
from jaxfluids.halos.halo_slices import HaloSlices
from jaxfluids.halos.inner.halo_communication import HaloCommunication
from jaxfluids.data_types.case_setup.boundary_conditions import BoundaryConditionsField, BoundaryConditionsFace
from jaxfluids.domain import FACE_LOCATIONS, EDGE_LOCATIONS, VERTEX_LOCATIONS

Array = jax.Array

class HaloCommunicationMaterial(HaloCommunication):
    def __init__(
            self,
            domain_information: DomainInformation,
            equation_manager: EquationManager,
            boundary_conditions: BoundaryConditionsField
            ) -> None:
        
        super().__init__(domain_information, boundary_conditions)
        self.equation_manager = equation_manager
    
    def face_halo_update(
            self,
            buffer: Array,
            conservatives: Array = None,
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

        edge_slices_retrieve = self.edge_slices_retrieve_conservatives
        edge_slices_fill = self.halo_slices.edge_slices_conservatives

        split_factors = self.domain_information.split_factors
        face_locations_to_axis_index = self.domain_information.face_location_to_axis_index
        active_face_locations = self.domain_information.active_face_locations
        active_edge_locations = self.domain_information.active_edge_locations

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


        vertex_slices_retrieve = self.vertex_slices_retrieve_conservatives
        vertex_slices_fill = self.halo_slices.vertex_slices_conservatives

        split_factors = self.domain_information.split_factors
        face_locations_to_axis_index = self.domain_information.face_location_to_axis_index
        
        # TODO dont loop over vertices multiple times
        for face_location in vertex_slices_retrieve.keys():

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