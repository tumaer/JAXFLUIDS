from typing import List

import jax
import jax.numpy as jnp

from jaxfluids.stencils.spatial_derivative import SpatialDerivative

Array = jax.Array

class DerivativeSixthOrderCenter(SpatialDerivative):
    ''' 
    6th order stencil for 1st derivative at the cell center

    du / dx = 1 / (60 * dx) * (
        - u_{i-3} + 9 * u_{i-2} - 45 * u_{i-1} 
        + 45 * u_{i+1} - 9 * u_{i+2} + u_{i+3})

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
        self.coeffs = jnp.array([1.0, 9.0, 45.0]) / 60.0

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