from typing import List

import jax
import jax.numpy as jnp

from jaxfluids.stencils.spatial_derivative import SpatialDerivative

Array = jax.Array

class DerivativeEighthOrderFace(SpatialDerivative):
    ''' 
    8th order stencil for 1st derivative at the cell face

    du / dxi = 1 / (107520 * dxi) * (
        75 * u_{i-3} - 1029 * u_{i-2} + 8575 * u_{i-1} - 128625 * u_{i}
        + 128625 * u_{i+1} - 8575 * u_{i+2} + 1029 * u_{i+3} - 75 * u_{i+4})

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
        self.coeffs = jnp.array([75.0, 1029.0, 8575.0, 128625.0]) / 107520.0

    def derivative_xi(
            self,
            buffer: Array,
            dxi: Array,
            axis: int,
            **kwargs
            ) -> Array:
        s1_ = self.s_[axis]

        deriv_xi = 1.0 / dxi * (
            self.coeffs[0] * (buffer[s1_[0]] - buffer[s1_[7]]) \
          + self.coeffs[1] * (buffer[s1_[6]] - buffer[s1_[1]]) \
          + self.coeffs[2] * (buffer[s1_[2]] - buffer[s1_[5]]) \
          + self.coeffs[3] * (buffer[s1_[4]] - buffer[s1_[3]]))
    
        return deriv_xi