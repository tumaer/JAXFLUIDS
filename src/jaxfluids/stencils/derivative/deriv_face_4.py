from typing import List

import jax
import jax.numpy as jnp

from jaxfluids.stencils.spatial_derivative import SpatialDerivative

Array = jax.Array

class DerivativeFourthOrderFace(SpatialDerivative):
    ''' 
    4th order stencil for 1st derivative at the cell face

    du / dx = 1 / (24 * dxi) * (u_{i-1} - 27 * u_{i} + 27 * u_{i+1} - u_{i+2})

              x
    |     |   |     |     |
    | i-1 | i | i+1 | i+2 |
    |     |   |     |     |
    '''
    required_halos = 2
    
    def __init__(
            self,
            nh: int,
            inactive_axes: List,
            offset: int = 0,
            **kwargs
        ) -> None:
        super(DerivativeFourthOrderFace, self).__init__(nh=nh,
                                                        inactive_axes=inactive_axes,
                                                        offset=offset)

        self.array_slices([(-2, -1, 0, 1)])
        self.coeffs = jnp.array([1.0, 27.0]) / 24.0

    def derivative_xi(
            self,
            buffer: Array,
            dxi: Array,
            axis: int,
            **kwargs
        ) -> Array:
        s1_ = self.s_[axis]

        deriv_xi = 1.0 / dxi * (
            self.coeffs[0] * (buffer[s1_[0]] - buffer[s1_[3]]) \
          + self.coeffs[1] * (buffer[s1_[2]] - buffer[s1_[1]]))

        return deriv_xi
