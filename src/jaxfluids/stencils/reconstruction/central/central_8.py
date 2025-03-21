from typing import List

import jax
import jax.numpy as jnp

from jaxfluids.stencils.spatial_reconstruction import SpatialReconstruction

Array = jax.Array

class CentralEighthOrderReconstruction(SpatialReconstruction):
    """CentralEighthOrderReconstruction 

    8th order stencil for reconstruction at the cell face

    du / dxi = 1 / 2048 * (
        -5 * u_{i-3} + 49 * u_{i-2} - 245 * u_{i-1} + 1225 * u_{i}
        + 1225 * u_{i+1} - 245 * u_{i+2} + 49 * u_{i+3} - 5 * u_{i+4})
    
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
        self.coeffs = jnp.array([-5.0, 49.0, -245.0, 1225.0]) / 2048.0

    def reconstruct_xi(
            self,
            buffer: Array,
            axis: int,
            **kwargs
        ) -> Array:
        s1_ = self.s_[axis]

        cell_state_xi = self.coeffs[0] * (buffer[s1_[0]] + buffer[s1_[7]]) \
                      + self.coeffs[1] * (buffer[s1_[1]] + buffer[s1_[6]]) \
                      + self.coeffs[2] * (buffer[s1_[2]] + buffer[s1_[5]]) \
                      + self.coeffs[3] * (buffer[s1_[3]] + buffer[s1_[4]]) 
        return cell_state_xi