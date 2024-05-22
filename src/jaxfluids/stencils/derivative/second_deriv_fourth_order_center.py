from typing import List

import jax.numpy as jnp
from jax import Array

from jaxfluids.stencils.spatial_derivative import SpatialDerivative

class SecondDerivativeFourthOrderCenter(SpatialDerivative):
    ''' 
    4th order stencil for 2nd derivative at the cell center
                  x  
    |     |     |   |     |     |
    | i-2 | i-1 | i | i+1 | i+2 |
    |     |     |   |     |     |
    '''
    required_halos = 2
    
    def __init__(
            self,
            nh: int,
            inactive_axes: List,
            offset: int = 0,
            **kwargs
        ) -> None:
        super(SecondDerivativeFourthOrderCenter, self).__init__(nh=nh,
                                                                inactive_axes=inactive_axes,
                                                                offset=offset)

        self.array_slices([ (-2, -1, 0, 1, 2) ], at_cell_center=True)

        # MIXED DERIVATIVE
        self.s__ = [

            [   jnp.s_[..., self.n-2:-self.n-2, self.n-2:-self.n-2, self.nhz],   # i-2,j-2,k
                jnp.s_[..., self.n-2:-self.n-2, self.n-1:-self.n-1, self.nhz],   # i-2,j-1,k   
                jnp.s_[..., self.n-2:-self.n-2, self.n+1:-self.n+1, self.nhz],   # i-2,j+1,k
                jnp.s_[..., self.n-2:-self.n-2, jnp.s_[self.n+2:-self.n+2] if self.n != 2 else jnp.s_[self.n+2:None], self.nhz],   # i-2,j+2,k 

                jnp.s_[..., self.n-1:-self.n-1, self.n-2:-self.n-2, self.nhz],   # i-1,j-2,k 
                jnp.s_[..., self.n-1:-self.n-1, self.n-1:-self.n-1, self.nhz],   # i-1,j-1,k 
                jnp.s_[..., self.n-1:-self.n-1, self.n+1:-self.n+1, self.nhz],   # i-1,j+1,k
                jnp.s_[..., self.n-1:-self.n-1, jnp.s_[self.n+2:-self.n+2] if self.n != 2 else jnp.s_[self.n+2:None], self.nhz],   # i-1,j+2,k 

                jnp.s_[..., self.n+1:-self.n+1, self.n-2:-self.n-2, self.nhz],   # i+1,j-2,k
                jnp.s_[..., self.n+1:-self.n+1, self.n-1:-self.n-1, self.nhz],   # i+1,j-1,k 
                jnp.s_[..., self.n+1:-self.n+1, self.n+1:-self.n+1, self.nhz],   # i+1,j+1,k 
                jnp.s_[..., self.n+1:-self.n+1, jnp.s_[self.n+2:-self.n+2] if self.n != 2 else jnp.s_[self.n+2:None], self.nhz],   # i+1,j+2,k 

                jnp.s_[..., jnp.s_[self.n+2:-self.n+2] if self.n != 2 else jnp.s_[self.n+2:None], self.n-2:-self.n-2, self.nhz],   # i+2,j-2,k 
                jnp.s_[..., jnp.s_[self.n+2:-self.n+2] if self.n != 2 else jnp.s_[self.n+2:None], self.n-1:-self.n-1, self.nhz],   # i+2,j-1,k 
                jnp.s_[..., jnp.s_[self.n+2:-self.n+2] if self.n != 2 else jnp.s_[self.n+2:None], self.n+1:-self.n+1, self.nhz],   # i+2,j+1,k
                jnp.s_[..., jnp.s_[self.n+2:-self.n+2] if self.n != 2 else jnp.s_[self.n+2:None], jnp.s_[self.n+2:-self.n+2] if self.n != 2 else jnp.s_[self.n+2:None], self.nhz]],  # i+2,j+2,k



            [   jnp.s_[..., self.n-2:-self.n-2, self.nhy, self.n-2:-self.n-2],   # i-2,j,k-2
                jnp.s_[..., self.n-2:-self.n-2, self.nhy, self.n-1:-self.n-1],   # i-2,j,k-1   
                jnp.s_[..., self.n-2:-self.n-2, self.nhy, self.n+1:-self.n+1],   # i-2,j,k+1
                jnp.s_[..., self.n-2:-self.n-2, self.nhy, jnp.s_[self.n+2:-self.n+2] if self.n != 2 else jnp.s_[self.n+2:None]],   # i-2,j,k+2 

                jnp.s_[..., self.n-1:-self.n-1, self.nhy, self.n-2:-self.n-2],   # i-1,j,k-2 
                jnp.s_[..., self.n-1:-self.n-1, self.nhy, self.n-1:-self.n-1],   # i-1,j,k-1 
                jnp.s_[..., self.n-1:-self.n-1, self.nhy, self.n+1:-self.n+1],   # i-1,j,k+1
                jnp.s_[..., self.n-1:-self.n-1, self.nhy, jnp.s_[self.n+2:-self.n+2] if self.n != 2 else jnp.s_[self.n+2:None]],   # i-1,j,k+2 

                jnp.s_[..., self.n+1:-self.n+1, self.nhy, self.n-2:-self.n-2],   # i+1,j,k-2
                jnp.s_[..., self.n+1:-self.n+1, self.nhy, self.n-1:-self.n-1],   # i+1,j,k-1 
                jnp.s_[..., self.n+1:-self.n+1, self.nhy, self.n+1:-self.n+1],   # i+1,j,k+1 
                jnp.s_[..., self.n+1:-self.n+1, self.nhy, jnp.s_[self.n+2:-self.n+2] if self.n != 2 else jnp.s_[self.n+2:None]],   # i+1,j,k+2 

                jnp.s_[..., jnp.s_[self.n+2:-self.n+2] if self.n != 2 else jnp.s_[self.n+2:None], self.nhy, self.n-2:-self.n-2],   # i+2,j,k-2, 
                jnp.s_[..., jnp.s_[self.n+2:-self.n+2] if self.n != 2 else jnp.s_[self.n+2:None], self.nhy, self.n-1:-self.n-1],   # i+2,j,k-1, 
                jnp.s_[..., jnp.s_[self.n+2:-self.n+2] if self.n != 2 else jnp.s_[self.n+2:None], self.nhy, self.n+1:-self.n+1],   # i+2,j,k+1,
                jnp.s_[..., jnp.s_[self.n+2:-self.n+2] if self.n != 2 else jnp.s_[self.n+2:None], self.nhy, jnp.s_[self.n+2:-self.n+2] if self.n != 2 else jnp.s_[self.n+2:None]]],  # i+2,j,k+2


            [   jnp.s_[..., self.nhx, self.n-2:-self.n-2, self.n-2:-self.n-2],   # i,j-2,k-2
                jnp.s_[..., self.nhx, self.n-2:-self.n-2, self.n-1:-self.n-1],   # i,j-2,k-1   
                jnp.s_[..., self.nhx, self.n-2:-self.n-2, self.n+1:-self.n+1],   # i,j-2,k+1
                jnp.s_[..., self.nhx, self.n-2:-self.n-2, jnp.s_[self.n+2:-self.n+2] if self.n != 2 else jnp.s_[self.n+2:None]],   # i,j-2,k+2 

                jnp.s_[..., self.nhx, self.n-1:-self.n-1, self.n-2:-self.n-2],   # i,j-1,k-2 
                jnp.s_[..., self.nhx, self.n-1:-self.n-1, self.n-1:-self.n-1],   # i,j-1,k-1 
                jnp.s_[..., self.nhx, self.n-1:-self.n-1, self.n+1:-self.n+1],   # i,j-1,k+1
                jnp.s_[..., self.nhx, self.n-1:-self.n-1, jnp.s_[self.n+2:-self.n+2] if self.n != 2 else jnp.s_[self.n+2:None]],   # i,j-1,j+2 

                jnp.s_[..., self.nhx, self.n+1:-self.n+1, self.n-2:-self.n-2],   # i,j+1,k-2
                jnp.s_[..., self.nhx, self.n+1:-self.n+1, self.n-1:-self.n-1],   # i,j+1,k-1 
                jnp.s_[..., self.nhx, self.n+1:-self.n+1, self.n+1:-self.n+1],   # i,j+1,k+1 
                jnp.s_[..., self.nhx, self.n+1:-self.n+1, jnp.s_[self.n+2:-self.n+2] if self.n != 2 else jnp.s_[self.n+2:None]],   # i,j+1,k+2

                jnp.s_[..., self.nhx, jnp.s_[self.n+2:-self.n+2] if self.n != 2 else jnp.s_[self.n+2:None], self.n-2:-self.n-2],   # i,j+2,k-2 
                jnp.s_[..., self.nhx, jnp.s_[self.n+2:-self.n+2] if self.n != 2 else jnp.s_[self.n+2:None], self.n-1:-self.n-1],   # i,j+2,k-1 
                jnp.s_[..., self.nhx, jnp.s_[self.n+2:-self.n+2] if self.n != 2 else jnp.s_[self.n+2:None], self.n+1:-self.n+1],   # i,j+2,k+1
                jnp.s_[..., self.nhx, jnp.s_[self.n+2:-self.n+2] if self.n != 2 else jnp.s_[self.n+2:None], jnp.s_[self.n+2:-self.n+2] if self.n != 2 else jnp.s_[self.n+2:None]]],  # i,j+2,k+2
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
        deriv_xi = (1.0 / 12.0 / dxi / dxi) * (- buffer[s1_[0]] + 16.0 * buffer[s1_[1]] - 30.0 * buffer[s1_[2]] + 16.0 * buffer[s1_[3]] - buffer[s1_[4]])
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
        deriv_xi_xj = 1.0 / 144.0 / dxi / dxj  * \
                ( + 1 * ( buffer[s1_[0]]  - 8 * buffer[s1_[1]]  + 8 * buffer[s1_[2]]  - buffer[s1_[3]] )  
                  - 8 * ( buffer[s1_[4]]  - 8 * buffer[s1_[5]]  + 8 * buffer[s1_[6]]  - buffer[s1_[7]] )              
                  + 8 * ( buffer[s1_[8]]  - 8 * buffer[s1_[9]]  + 8 * buffer[s1_[10]] - buffer[s1_[11]])              
                  - 1 * ( buffer[s1_[12]] - 8 * buffer[s1_[13]] + 8 * buffer[s1_[14]] - buffer[s1_[15]])   )    
        return deriv_xi_xj