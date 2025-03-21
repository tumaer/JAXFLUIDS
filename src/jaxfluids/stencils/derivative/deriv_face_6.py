from typing import List

import jax
import jax.numpy as jnp

from jaxfluids.stencils.spatial_derivative import SpatialDerivative

Array = jax.Array

class DerivativeSixthOrderFace(SpatialDerivative):
    ''' 
    6th order stencil for 1st derivative at the cell face

    du /dx = 1 / (1920 * dxi) * (
        - 9 * u_{i-2} + 125 * u_{i-1} - 2250 * u_{i}
        + 2250 * u_{i+1} - 125 * u_{i+2} + 9 * u_{i+3})

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
        self.coeffs = jnp.array([9.0, 125.0, 2250.0]) / 1920.0

    def derivative_xi(
            self,
            buffer: Array,
            dxi: Array,
            axis: int,
            **kwargs
            ) -> Array:
        s1_ = self.s_[axis]

        deriv_xi = 1.0 / dxi * (
            self.coeffs[0] * (buffer[s1_[5]] - buffer[s1_[0]]) \
          + self.coeffs[1] * (buffer[s1_[1]] - buffer[s1_[4]]) \
          + self.coeffs[2] * (buffer[s1_[3]] - buffer[s1_[2]]))

        return deriv_xi
