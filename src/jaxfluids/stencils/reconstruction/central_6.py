from typing import List

import jax.numpy as jnp
from jax import Array

from jaxfluids.stencils.spatial_reconstruction import SpatialReconstruction

class CentralSixthOrderReconstruction(SpatialReconstruction):
    """CentralSixthOrderReconstruction 

    6th order stencil for reconstruction at the cell face
                       x
    |      |     |     |     |     |     |
    | i-2  | i-1 |  i  | i+1 | i+2 | i+3 | 
    |      |     |     |     |     |     |
    """

    required_halos = 3
    
    def __init__(
            self, 
            nh: int, 
            inactive_axes: List, 
            offset: int = 0, 
            **kwargs
            ) -> None:
        super(CentralSixthOrderReconstruction, self).__init__(nh=nh, 
                                                              inactive_axes=inactive_axes,
                                                              offset=offset)

        self.array_slices([range(-3, 3, 1)])

    def reconstruct_xi(
            self,
            buffer: Array,
            axis: int,
            **kwargs
        ) -> Array:
        s1_ = self.s_[axis]
        cell_state_xi = (1.0 / 256.0) * (
            3.0 * buffer[s1_[0]] \
            - 25.0 * buffer[s1_[1]] \
            + 150.0 * buffer[s1_[2]] \
            + 150.0 * buffer[s1_[3]] \
            - 25.0 * buffer[s1_[4]] \
            + 3.0 * buffer[s1_[5]])
        return cell_state_xi