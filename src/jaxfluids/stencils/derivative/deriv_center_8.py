from typing import List

import jax
import jax.numpy as jnp

from jaxfluids.stencils.spatial_derivative import SpatialDerivative

Array = jax.Array

class DerivativeEighthOrderCenter(SpatialDerivative):
    ''' 
    8th order stencil for 1st derivative at the cell center

    du / dx = 1 / (840 * dxi) * (
        3 * u_{i-4} - 32 * u_{i-3} + 168 * u_{i-2} - 672 * u_{i-1}
        + 672 * u_{i+1} - 168 * u_{i+2} + 32 * u_{i+3} - 3 * u_{i+4})
    
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
        self.coeffs = jnp.array([3.0, 32.0, 168.0, 672.0]) / 840.0

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