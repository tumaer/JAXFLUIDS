from typing import List

import jax
import jax.numpy as jnp

from jaxfluids.stencils.spatial_derivative import SpatialDerivative

Array = jax.Array

class DerivativeFirstOrderDownwindCenter(SpatialDerivative):
    """1st order downwind stencil for 1st derivative at the cell center
       x          
    |     |     |     
    |  i  | i+1 |     
    |     |     |     
    """
    
    required_halos = 1
    
    def __init__(
            self,
            nh: int,
            inactive_axes: List[str],
            offset: int = 0,
            **kwargs
        ) -> None:
        super(DerivativeFirstOrderDownwindCenter, self).__init__(
            nh=nh, inactive_axes=inactive_axes, offset=offset)

        self.array_slices([(0,1)], at_cell_center=True)
        self.coeffs = jnp.array([-1.0, 1.0])

    def derivative_xi(
            self, buffer: Array,
            dxi: Array, axis: int,
            **kwargs
        ) -> Array:
        s1_ = self.s_[axis]
        deriv_xi = 1.0 / dxi * (self.coeffs[0] * buffer[s1_[0]] + self.coeffs[1] * buffer[s1_[1]])
        return deriv_xi