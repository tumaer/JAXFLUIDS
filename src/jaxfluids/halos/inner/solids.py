from functools import partial
from typing import List, Tuple, Dict
import types

import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt

from jaxfluids.domain.domain_information import DomainInformation
from jaxfluids.halos.inner.halo_communication import HaloCommunication
from jaxfluids.levelset.fluid_solid.solid_properties_manager import SolidPropertiesManager
from jaxfluids.data_types.case_setup.boundary_conditions import BoundaryConditionsField
from jaxfluids.domain import FACE_LOCATIONS

Array = jax.Array

class HaloCommunicationSolids(HaloCommunication):
    def __init__(
            self,
            domain_information: DomainInformation,
            solid_properties_manager: SolidPropertiesManager,
            boundary_conditions: BoundaryConditionsField
            ) -> None:
        
        super().__init__(domain_information, boundary_conditions)

        self.solid_properties_manager = solid_properties_manager
    
    def face_halo_update(
            self,
            buffer: Array,
            solid_energy: Array = None,
            ) -> Tuple[Array, Array]:

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
            if solid_energy != None:
                solid_energy_halos_retrieve = \
                self.solid_properties_manager.compute_internal_energy(
                    buffer_halos_retrieve)
                solid_energy = solid_energy.at[slice_fill].mul(1.0 - mask_value)
                solid_energy = solid_energy.at[slice_fill].add(solid_energy_halos_retrieve * mask_value)

        if solid_energy != None:
            return buffer, solid_energy
        else:
            return buffer

    def edge_halo_update(
            self,
            buffer: Array,
            solid_energy: Array = None,
            ) -> Tuple[Array, Array]:

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
                if solid_energy != None:
                    solid_energy_halos_retrieve = \
                    self.solid_properties_manager.compute_internal_energy(
                            buffer_halos_retrieve)
                    solid_energy = solid_energy.at[slice_fill].mul(1.0 - mask_value)
                    solid_energy = solid_energy.at[slice_fill].add(solid_energy_halos_retrieve * mask_value)
        
        if solid_energy != None:
            return buffer, solid_energy
        else:
            return buffer

    def vertex_halo_update(
            self,
            buffer: Array,
            solid_energy: Array = None,
            ) -> Tuple[Array, Array]:

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
                if solid_energy != None:
                    solid_energy_halos_retrieve = \
                    self.solid_properties_manager.compute_internal_energy(
                            buffer_halos_retrieve)
                    solid_energy = solid_energy.at[slice_fill].mul(1.0 - mask_value)
                    solid_energy = solid_energy.at[slice_fill].add(solid_energy_halos_retrieve * mask_value)
        
        if solid_energy != None:
            return buffer, solid_energy
        else:
            return buffer