from typing import List

import jax.numpy as jnp
from jax import Array

from jaxfluids.stencils.spatial_derivative import SpatialDerivative

class DerivativeFourthOrderCenter(SpatialDerivative):
    ''' 
    4th order stencil for 1st derivative at the cell center
                  x  
    |     |     |   |     |     |
    | i-2 | i-1 | i | i+1 | i+2 |
    |     |     |   |     |     |
    '''
    required_halos = 2
    
    def __init__(
            self,
            nh: int,
            inactive_axes: List,
            offset: int = 0,
            **kwargs
        ) -> None:
        super(DerivativeFourthOrderCenter, self).__init__(nh=nh,
                                                          inactive_axes=inactive_axes,
                                                          offset=offset)

        self.array_slices([(-2,-1,1,2)], at_cell_center=True)

    def derivative_xi(
            self,
            buffer: Array,
            dxi: Array,
            axis: int,
            **kwargs
            ) -> Array:
        s1_ = self.s_[axis]
        deriv_xi = (1.0 / 12.0 / dxi) * (buffer[s1_[0]] - 8.0 * buffer[s1_[1]] + 8.0 * buffer[s1_[2]] - buffer[s1_[3]])
        return deriv_xi