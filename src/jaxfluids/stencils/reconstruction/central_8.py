from typing import List

import jax.numpy as jnp
from jax import Array

from jaxfluids.stencils.spatial_reconstruction import SpatialReconstruction

class CentralEighthOrderReconstruction(SpatialReconstruction):
    """CentralEighthOrderReconstruction 

    8th order stencil for reconstruction at the cell face
                             x
    |     |      |     |     |     |     |     |     |
    | i-3 | i-2  | i-1 |  i  | i+1 | i+2 | i+3 | i+4 | 
    |     |      |     |     |     |     |     |     |
    """

    required_halos = 4
    
    def __init__(
            self, 
            nh: int, 
            inactive_axes: List, 
            offset: int = 0, 
            **kwargs
            ) -> None:
        super(CentralEighthOrderReconstruction, self).__init__(nh=nh,
                                                               inactive_axes=inactive_axes,
                                                               offset=offset)

        self.array_slices([range(-4, 4, 1)])

    def reconstruct_xi(
            self,
            buffer: Array,
            axis: int,
            **kwargs
        ) -> Array:
        s1_ = self.s_[axis]
        cell_state_xi = (1.0 / 2048.0) * (
            - 5.0 * buffer[s1_[0]] \
            + 49.0 * buffer[s1_[1]] \
            - 245.0 * buffer[s1_[2]] \
            + 1225.0 * buffer[s1_[3]] \
            + 1225.0 * buffer[s1_[4]] \
            - 245.0 * buffer[s1_[5]] \
            + 49.0 * buffer[s1_[6]] \
            - 5.0 * buffer[s1_[7]])
        return cell_state_xi