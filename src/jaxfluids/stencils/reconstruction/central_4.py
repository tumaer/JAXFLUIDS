from typing import List

import jax.numpy as jnp
from jax import Array

from jaxfluids.stencils.spatial_reconstruction import SpatialReconstruction

class CentralFourthOrderReconstruction(SpatialReconstruction):
    """CentralFourthOrderReconstruction 

    4th order stencil for reconstruction at the cell face
                x
    |     |     |     |     |
    | i-1 |  i  | i+1 | i+2 |
    |     |     |     |     |
    """

    required_halos = 2
    
    def __init__(
            self, 
            nh: int, 
            inactive_axes: List, 
            offset: int = 0, 
            **kwargs
        ) -> None:
        super(CentralFourthOrderReconstruction, self).__init__(nh=nh,
                                                               inactive_axes=inactive_axes,
                                                               offset=offset)

        self.array_slices([range(-2, 2, 1)])

    def reconstruct_xi(
            self,
            buffer: Array,
            axis: int,
            **kwargs
        ) -> Array:
        s1_ = self.s_[axis]
        cell_state_xi = (1.0 / 16.0) * (-buffer[s1_[0]] + 9.0 * buffer[s1_[1]] + 9.0 * buffer[s1_[2]] - buffer[s1_[3]])
        return cell_state_xi