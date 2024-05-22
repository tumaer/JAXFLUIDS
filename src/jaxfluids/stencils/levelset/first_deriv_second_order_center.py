from typing import List

import jax.numpy as jnp
from jax import Array

from jaxfluids.stencils.spatial_derivative import SpatialDerivative

class FirstDerivativeSecondOrder(SpatialDerivative):

    required_halos = 2

    def __init__(self, nh: int, inactive_axes: List, offset: int = 0):
        super(FirstDerivativeSecondOrder, self).__init__(nh, inactive_axes, offset)

        self.sign = [1, -1]

        self.s_ = [
        [
            [   jnp.s_[..., jnp.s_[self.n-2*j:-self.n-2*j] if -self.n-2*j != 0 else jnp.s_[self.n-2*j:None], self.nhy, self.nhz],
                jnp.s_[..., self.n-1*j:-self.n-1*j, self.nhy, self.nhz],          
                jnp.s_[..., self.n-0*j:-self.n-0*j, self.nhy, self.nhz],   ],

            [   jnp.s_[..., self.nhx, jnp.s_[self.n-2*j:-self.n-2*j] if -self.n-2*j != 0 else jnp.s_[self.n-2*j:None], self.nhz],
                jnp.s_[..., self.nhx, self.n-1*j:-self.n-1*j, self.nhz],          
                jnp.s_[..., self.nhx, self.n-0*j:-self.n-0*j, self.nhz],   ],

            [   jnp.s_[..., self.nhx, self.nhy, jnp.s_[self.n-2*j:-self.n-2*j] if -self.n-2*j != 0 else jnp.s_[self.n-2*j:None]],
                jnp.s_[..., self.nhx, self.nhy, self.n-1*j:-self.n-1*j],          
                jnp.s_[..., self.nhx, self.nhy, self.n-0*j:-self.n-0*j],   ],
                
        ] for j in self.sign ]

    def derivative_xi(self, levelset: Array, dxi: Array, i: int, j: int, *args) -> Array:
        # TODO aaron
        s1_ = self.s_[j][i]
        deriv_xi = (1.0 / dxi) * ( 1./2. * levelset[s1_[0]] - 2 * levelset[s1_[1]] + 3./2. * levelset[s1_[2]] )
        deriv_xi *= self.sign[j]
        return deriv_xi