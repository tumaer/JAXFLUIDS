from typing import List

import jax
import jax.numpy as jnp

from jaxfluids.stencils.spatial_derivative import SpatialDerivative

Array = jax.Array

class HOUC5(SpatialDerivative):
    
    required_halos = 3

    def __init__(self, nh: int, inactive_axes: List, offset: int = 0):
        super(HOUC5, self).__init__(nh=nh, inactive_axes=inactive_axes, offset=offset)
        
        self.array_slices([range(-3, 3, 1), range(3, -3, -1)], True)
        self.coeff = jnp.array([-2.0, 15.0, -60.0, 20.0, 30.0, -3.0]) / 60.0
        self.sign = (1, -1)

    def derivative_xi(
            self,
            buffer: Array,
            dxi:float,
            axis: int,
            j: int,
            *args
            ) -> Array:
        s_ = self.s_[j][axis]
        sign = self.sign[j]
        deriv_xi =  sign * sum([buffer[s_[k]]*self.coeff[k] for
                                k in range(len(self.coeff))]) / dxi
        return deriv_xi