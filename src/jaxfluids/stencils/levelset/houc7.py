from typing import List

import jax
import jax.numpy as jnp

from jaxfluids.stencils.spatial_derivative import SpatialDerivative

Array = jax.Array

class HOUC7(SpatialDerivative):
    
    required_halos = 4

    def __init__(self, nh: int, inactive_axes: List, offset: int = 0):
        super(HOUC7, self).__init__(nh=nh, inactive_axes=inactive_axes, offset=offset)
        
        self.coeff = jnp.array([3.0, -28.0, 126.0, -420.0,
                                105.0, 252.0, -42.0, 4.0]) / 420.0
        self.array_slices([range(-4, 4, 1), range(4, -4, -1)], True)
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
