from typing import Dict, Tuple

import jax
import jax.numpy as jnp
from jax import Array
import numpy as np

from jaxfluids.halos.outer.boundary_condition import BoundaryCondition
from jaxfluids.domain.domain_information import DomainInformation
from jaxfluids.data_types.case_setup.boundary_conditions import BoundaryConditionsField, BoundaryConditionsFace
from jaxfluids.domain import EDGE_LOCATIONS
from jaxfluids.domain import VERTEX_LOCATIONS

class BoundaryConditionFlux(BoundaryCondition):

    def __init__(
            self,
            domain_information: DomainInformation,
            boundary_conditions: BoundaryConditionsField
            ) -> None:

        super().__init__(domain_information, boundary_conditions)
        
        self.axis_index_to_face_locations = (
            ("east", "west"),
            ("north", "south"),
            ("top", "bottom"))

        self.flux_slices = {
            "east"  : jnp.s_[..., -1,  :,  :],
            "west"  : jnp.s_[...,  0,  :,  :],
            "north" : jnp.s_[...,  :, -1,  :],
            "south" : jnp.s_[...,  :,  0,  :],
            "top"   : jnp.s_[...,  :,  :, -1],
            "bottom": jnp.s_[...,  :,  :,  0],
        }
        
        self.slices_retrieve_flux = {
            "east"  : jnp.s_[..., -2,  :,  :],
            "west"  : jnp.s_[...,  1,  :,  :],
            "north" : jnp.s_[...,  :, -2,  :],
            "south" : jnp.s_[...,  :,  1,  :],
            "top"   : jnp.s_[...,  :,  :, -2],
            "bottom": jnp.s_[...,  :,  :,  1],
        }
        
    def face_flux_update(
            self,
            flux_xi: Array,
            axis: int,
            ) -> Array:
        """Fills the curvature buffer halo cells.

        :param curvature: _description_
        :type curvature: Array
        :return: _description_
        :rtype: Array
        """
        is_parallel = self.domain_information.is_parallel
        face_locations = self.axis_index_to_face_locations[axis]
        for face_location in face_locations:
            
            boundary_conditions_face_tuple: Tuple[BoundaryConditionsFace] = \
                getattr(self.boundary_conditions, face_location)
                        
            if len(boundary_conditions_face_tuple) > 1:
                multiple_types_at_face = True
            else:
                multiple_types_at_face = False

            for boundary_conditions_face in boundary_conditions_face_tuple:

                boundary_type = boundary_conditions_face.boundary_type
                
                if boundary_type in ("ZEROGRADIENT", "DIRICHLET", "NEUMANN"):

                    slice_retrieve = self.slices_retrieve_flux[face_location]
                    halos = flux_xi[slice_retrieve]

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

                    slice_fill = self.flux_slices[face_location]
                    flux_xi = flux_xi.at[slice_fill].mul(1 - mask)
                    flux_xi = flux_xi.at[slice_fill].add(halos * mask)
                
                elif boundary_type in ("PERIODIC", "SYMMETRY") \
                    or "WALL" in boundary_type:
                    # TODO what should happen with WALL???
                    pass
                
                else:
                    raise NotImplementedError

        return flux_xi

    def face_halo_update(self) -> None:
        pass

    def edge_halo_update(self) -> None: 
        pass
    
    def vertex_halo_update(self) -> None:
        pass
