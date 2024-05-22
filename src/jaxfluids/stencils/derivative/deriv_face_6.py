from typing import List

import jax.numpy as jnp
from jax import Array

from jaxfluids.stencils.spatial_derivative import SpatialDerivative

class DerivativeSixthOrderFace(SpatialDerivative):
    ''' 
    6th order stencil for 1st derivative at the cell face
                      x
    |     |     |     |     |     |     |
    | i-2 | i-1 |  i  | i+1 | i+2 | i+3 |
    |     |     |     |     |     |     |
    '''
    required_halos = 3
    
    def __init__(
            self,
            nh: int,
            inactive_axes: List,
            offset: int = 0,
            **kwargs
            ) -> None:
        super(DerivativeSixthOrderFace, self).__init__(nh=nh,
                                                       inactive_axes=inactive_axes,
                                                       offset=offset)

        self.array_slices([(-3, -2, -1, 0, 1, 2)])

    def derivative_xi(
            self,
            buffer: Array,
            dxi: Array,
            axis: int,
            **kwargs
            ) -> Array:
        s1_ = self.s_[axis]
        deriv_xi = 1.0 / (1920.0 * dxi) * (
            - 9.0 * buffer[s1_[0]] \
            + 125.0 * buffer[s1_[1]] \
            - 2250.0 * buffer[s1_[2]] \
            + 2250.0 * buffer[s1_[3]] \
            - 125.0 * buffer[s1_[4]] \
            + 9.0 * buffer[s1_[5]])
        return deriv_xi
