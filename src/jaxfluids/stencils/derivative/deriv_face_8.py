from typing import List

import jax.numpy as jnp
from jax import Array

from jaxfluids.stencils.spatial_derivative import SpatialDerivative

class DerivativeEighthOrderFace(SpatialDerivative):
    ''' 
    8th order stencil for 1st derivative at the cell face
                            x
    |     |     |     |     |     |     |     |     |
    | i-3 | i-2 | i-1 |  i  | i+1 | i+2 | i+3 | i+4 |
    |     |     |     |     |     |     |     |     |
    '''
    required_halos = 4
    
    def __init__(
            self,
            nh: int,
            inactive_axes: List,
            offset: int = 0,
            **kwargs
            ) -> None:
        super(DerivativeEighthOrderFace, self).__init__(nh=nh,
                                                        inactive_axes=inactive_axes,
                                                        offset=offset)

        self.array_slices([(-4, -3, -2, -1, 0, 1, 2, 3)])

    def derivative_xi(
            self,
            buffer: Array,
            dxi: Array,
            axis: int,
            **kwargs
            ) -> Array:
        s1_ = self.s_[axis]
        deriv_xi = 1.0 / (107520.0 * dxi) * (
            75.0 * buffer[s1_[0]] \
            - 1029.0 * buffer[s1_[1]] \
            + 8575.0 * buffer[s1_[2]] \
            - 128625.0 * buffer[s1_[3]] \
            + 128625.0 * buffer[s1_[4]] \
            - 8575.0 * buffer[s1_[5]] \
            + 1029.0 * buffer[s1_[6]] \
            - 75.0 * buffer[s1_[7]])
        return deriv_xi