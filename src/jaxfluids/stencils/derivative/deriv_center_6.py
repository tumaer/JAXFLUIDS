from typing import List

import jax.numpy as jnp
from jax import Array

from jaxfluids.stencils.spatial_derivative import SpatialDerivative

class DerivativeSixthOrderCenter(SpatialDerivative):
    ''' 
    6th order stencil for 1st derivative at the cell center
                         x   
    |     |     |     |     |     |     |     |
    | i-3 | i-2 | i-1 |  i  | i+1 | i+2 | i+3 |
    |     |     |     |     |     |     |     |
    '''
    required_halos = 3
    
    def __init__(
            self,
            nh: int,
            inactive_axes: List,
            offset: int = 0,
            **kwargs
            ) -> None:
        super(DerivativeSixthOrderCenter, self).__init__(nh=nh, inactive_axes=inactive_axes, offset=offset)

        self.array_slices([ (-3, -2, -1, 1, 2, 3) ], at_cell_center=True)

    def derivative_xi(
            self,
            buffer: Array,
            dxi: Array,
            axis: int,
            **kwargs
            ) -> Array:
        s1_ = self.s_[axis]
        deriv_xi = 1.0 / (60.0 * dxi) * (
            - buffer[s1_[0]] \
            + 9.0 * buffer[s1_[1]] \
            - 45.0 * buffer[s1_[2]] \
            + 45.0 * buffer[s1_[3]] \
            - 9.0 * buffer[s1_[4]] \
            + buffer[s1_[5]])
        return deriv_xi