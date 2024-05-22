from typing import List

import jax.numpy as jnp
from jax import Array

from jaxfluids.stencils.spatial_derivative import SpatialDerivative

class SecondDerivativeSecondOrderCenter(SpatialDerivative):
    ''' 
    2nd order stencil for 1st derivative at the cell center
            x
    |     |   |     |     
    | i-1 | i | i+1 |     
    |     |   |     |     
    '''
    required_halos = 1

    def __init__(
            self,
            nh: int,
            inactive_axes: List,
            offset: int = 0,
            **kwargs
        ) -> None:
        super(SecondDerivativeSecondOrderCenter, self).__init__(nh=nh,
                                                                inactive_axes=inactive_axes,
                                                                offset=offset)

        self.array_slices([ (-1, 0, 1) ], at_cell_center=True)

        # MIXED DERIVATIVE
        self.s__ = [

            [   jnp.s_[..., self.n-1:-self.n-1, self.n-1:-self.n-1, self.nhz],         # i-1,j-1,k 
                jnp.s_[..., self.n+1:-self.n+1, self.n-1:-self.n-1, self.nhz],         # i+1,j-1,k 
                jnp.s_[..., self.n-1:-self.n-1, self.n+1:-self.n+1, self.nhz],         # i-1,j+1,k 
                jnp.s_[..., self.n+1:-self.n+1, self.n+1:-self.n+1, self.nhz], ],      # i+1,j+1,k

            [   jnp.s_[..., self.n-1:-self.n-1, self.nhy, self.n-1:-self.n-1],         # i-1,j,k-1 
                jnp.s_[..., self.n+1:-self.n+1, self.nhy, self.n-1:-self.n-1],         # i+1,j,k-1 
                jnp.s_[..., self.n-1:-self.n-1, self.nhy, self.n+1:-self.n+1],         # i-1,j,k+1 
                jnp.s_[..., self.n+1:-self.n+1, self.nhy, self.n+1:-self.n+1], ],      # i+1,j,k+1

            [   jnp.s_[..., self.nhx, self.n-1:-self.n-1, self.n-1:-self.n-1],         # i,j-1,k-1 
                jnp.s_[..., self.nhx, self.n+1:-self.n+1, self.n-1:-self.n-1],         # i,j+1,k-1 
                jnp.s_[..., self.nhx, self.n-1:-self.n-1, self.n+1:-self.n+1],         # i,j-1,k+1 
                jnp.s_[..., self.nhx, self.n+1:-self.n+1, self.n+1:-self.n+1], ]       # i,j+1,k+1

        ]

        self.index_pair_dict = {"01": 0, "02": 1, "12": 2}

    def derivative_xi(
            self,
            buffer: Array,
            dxi: Array,
            axis: int,
            **kwargs
        ) -> Array:
        s1_ = self.s_[axis]
        deriv_xi = (1.0 / dxi / dxi ) * (buffer[s1_[0]] - 2.0 * buffer[s1_[1]] + buffer[s1_[2]])
        return deriv_xi

    def derivative_xi_xj(
            self,
            buffer: Array,
            dxi: Array,
            dxj: Array,
            i: int,
            j: int
        ) -> Array:
        s1_ = self.s__[self.index_pair_dict[str(i) + (str(j))]]
        deriv_xi_xj = 1.0 / 4.0 / dxi / dxj  * ( buffer[s1_[0]]  - buffer[s1_[1]]  - buffer[s1_[2]] + buffer[s1_[3]] )   
        return deriv_xi_xj
