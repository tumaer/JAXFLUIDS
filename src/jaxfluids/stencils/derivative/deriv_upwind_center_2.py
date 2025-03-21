from typing import List

import jax
import jax.numpy as jnp

from jaxfluids.stencils.spatial_derivative import SpatialDerivative

Array = jax.Array

class DerivativeSecondOrderUpwindCenter(SpatialDerivative):
    """2nd order upwind stencil for 1st derivative at the cell center
                   x
    |     |     |     |     
    | i-2 | i-1 |  i  |     
    |     |     |     |     
    """
    
    required_halos = 2
    
    def __init__(
            self,
            nh: int,
            inactive_axes: List[str],
            offset: int = 0,
            **kwargs
        ) -> None:
        super(DerivativeSecondOrderUpwindCenter, self).__init__(
            nh=nh, inactive_axes=inactive_axes, offset=offset)

        self.array_slices([(-2,-1,0)], at_cell_center=True)
        self.coeffs = 1.0 / 2.0 * jnp.array([1.0, -4.0, 3.0])

    def derivative_xi(
            self,
            buffer: Array,
            dxi: Array,
            axis: int,
            **kwargs
        ) -> Array:
        s1_ = self.s_[axis]
        deriv_xi = 1.0 / dxi * (self.coeffs[0] * buffer[s1_[0]] + self.coeffs[1] * buffer[s1_[1]] + self.coeffs[2] * buffer[s1_[2]])
        return deriv_xi