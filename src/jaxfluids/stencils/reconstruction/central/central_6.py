from typing import List

import jax
import jax.numpy as jnp

from jaxfluids.stencils.spatial_reconstruction import SpatialReconstruction

Array = jax.Array

class CentralSixthOrderReconstruction(SpatialReconstruction):
    """CentralSixthOrderReconstruction 

    6th order stencil for reconstruction at the cell face

    du / dxi = 1 / 256 * (
        3 * u_{i-2} - 25 * u_{i-1} + 150 * u_{i}
        + 150 * u_{i+1} - 25 * u_{i+2} + 3 * u_{i+3})

    
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
        self.coeffs = jnp.array([3.0, -25.0, 150.0]) / 256.0

    def reconstruct_xi(
            self,
            buffer: Array,
            axis: int,
            **kwargs
        ) -> Array:
        s1_ = self.s_[axis]

        cell_state_xi = self.coeffs[0] * (buffer[s1_[0]] + buffer[s1_[5]]) \
                      + self.coeffs[1] * (buffer[s1_[1]] + buffer[s1_[4]]) \
                      + self.coeffs[2] * (buffer[s1_[2]] + buffer[s1_[3]])
              
        return cell_state_xi