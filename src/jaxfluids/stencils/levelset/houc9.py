from typing import List

import jax
import jax.numpy as jnp

from jaxfluids.stencils.spatial_derivative import SpatialDerivative
Array = jax.Array

class HOUC9(SpatialDerivative):

    required_halos = 5

    def __init__(self, nh: int, inactive_axes: List, offset: int = 0):
        super(SpatialDerivative, self).__init__(nh, inactive_axes, offset)

        self.coeff = jnp.array([-1.0/630.0, 1.0/56.0, -2.0/21.0, 1.0/3.0,
                                -1.0, 1.0/5.0, 2.0/3.0, -1.0/7.0, 1.0/42.0,
                                -1.0/504.0])
        self.array_slices([range(-5, 5, 1), range(5, -5, -1)], True)
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