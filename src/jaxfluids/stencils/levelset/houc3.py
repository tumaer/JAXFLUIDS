from typing import List

import jax
import jax.numpy as jnp

from jaxfluids.stencils.spatial_derivative import SpatialDerivative

Array = jax.Array

class HOUC3(SpatialDerivative):
    
    required_halos = 2

    def __init__(self, nh: int, inactive_axes: List, offset: int = 0):
        super(HOUC3, self).__init__(nh=nh, inactive_axes=inactive_axes, offset=offset)
    
        self.array_slices([range(-2, 2, 1), range(2, -2, -1)], True)
        self.coeff = (1.0/6.0, -1.0, 1.0/2.0, 1.0/3.0)
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
