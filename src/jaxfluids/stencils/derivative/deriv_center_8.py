from typing import List

import jax.numpy as jnp
from jax import Array

from jaxfluids.stencils.spatial_derivative import SpatialDerivative

class DerivativeEighthOrderCenter(SpatialDerivative):
    ''' 
    8th order stencil for 1st derivative at the cell center
                               x   
    |     |     |     |     |     |     |     |     |     |
    | i-4 | i-3 | i-2 | i-1 |  i  | i+1 | i+2 | i+3 | i+4 |
    |     |     |     |     |     |     |     |     |     |
    '''
    required_halos = 4
    
    def __init__(
            self,
            nh: int,
            inactive_axes: List,
            offset: int = 0,
            **kwargs
            ) -> None:
        super(DerivativeEighthOrderCenter, self).__init__(nh=nh, inactive_axes=inactive_axes, offset=offset)

        self.array_slices([ (-4, -3, -2, -1, 1, 2, 3, 4) ], at_cell_center=True)

    def derivative_xi(
            self,
            buffer: Array,
            dxi: Array,
            axis: int,
            **kwargs
            ) -> Array:
        s1_ = self.s_[axis]
        deriv_xi = 1.0 / (840.0 * dxi) * (
            3.0 * buffer[s1_[0]] \
            - 32.0 * buffer[s1_[1]] \
            + 168.0 * buffer[s1_[2]] \
            - 672.0 * buffer[s1_[3]] \
            + 672.0 * buffer[s1_[4]] \
            - 168.0 * buffer[s1_[5]] \
            + 32.0 * buffer[s1_[6]] \
            - 3.0 * buffer[s1_[7]])
        return deriv_xi