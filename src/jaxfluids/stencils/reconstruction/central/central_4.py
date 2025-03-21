from typing import List

import jax
import jax.numpy as jnp

from jaxfluids.stencils.spatial_reconstruction import SpatialReconstruction

Array = jax.Array

class CentralFourthOrderReconstruction(SpatialReconstruction):
    """CentralFourthOrderReconstruction 

    4th order stencil for reconstruction at the cell face

    du / dx = 1 / 16) * (-u_{i-1} + 9 * u_{i} + 9 * u_{i+1} - u_{i+2})

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
        self.coeffs = jnp.array([-1.0, 9.0]) / 16.0

    def reconstruct_xi(
            self,
            buffer: Array,
            axis: int,
            **kwargs
        ) -> Array:
        s1_ = self.s_[axis]

        cell_state_xi = self.coeffs[0] * (buffer[s1_[0]] + buffer[s1_[3]]) \
                      + self.coeffs[1] * (buffer[s1_[1]] + buffer[s1_[2]])

        return cell_state_xi