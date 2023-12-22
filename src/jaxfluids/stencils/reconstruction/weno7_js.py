#*------------------------------------------------------------------------------*
#* JAX-FLUIDS -                                                                 *
#*                                                                              *
#* A fully-differentiable CFD solver for compressible two-phase flows.          *
#* Copyright (C) 2022  Deniz A. Bezgin, Aaron B. Buhendwa, Nikolaus A. Adams    *
#*                                                                              *
#* This program is free software: you can redistribute it and/or modify         *
#* it under the terms of the GNU General Public License as published by         *
#* the Free Software Foundation, either version 3 of the License, or            *
#* (at your option) any later version.                                          *
#*                                                                              *
#* This program is distributed in the hope that it will be useful,              *
#* but WITHOUT ANY WARRANTY; without even the implied warranty of               *
#* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the                *
#* GNU General Public License for more details.                                 *
#*                                                                              *
#* You should have received a copy of the GNU General Public License            *
#* along with this program.  If not, see <https://www.gnu.org/licenses/>.       *
#*                                                                              *
#*------------------------------------------------------------------------------*
#*                                                                              *
#* CONTACT                                                                      *
#*                                                                              *
#* deniz.bezgin@tum.de // aaron.buhendwa@tum.de // nikolaus.adams@tum.de        *
#*                                                                              *
#*------------------------------------------------------------------------------*
#*                                                                              *
#* Munich, April 15th, 2022                                                     *
#*                                                                              *
#*------------------------------------------------------------------------------*

from typing import List

import jax.numpy as jnp

from jaxfluids.stencils.spatial_reconstruction import SpatialReconstruction

class WENO7(SpatialReconstruction):
    ''' Balsara & Shu - 2000 - '''
    
    def __init__(self, nh: int, inactive_axis: List) -> None:
        super(WENO7, self).__init__(nh=nh, inactive_axis=inactive_axis)

        self.dr_ = [
            [1/35, 12/35, 18/35, 4/35],
            [4/35, 18/35, 12/35, 1/35],
        ]
        self.cr_ = [
            [[-1/4, 13/12, -23/12, 25/12], [1/12, -5/12, 13/12, 1/4], [-1/12, 7/12, 7/12, -1/12], [1/4, 13/12, -5/12, 1/12]],
            [[1/12, -5/12, 13/12, 1/4], [-1/12, 7/12, 7/12, -1/12], [1/4, 13/12, -5/12, 1/12], [25/12, -23/12, 13/12, -1/4]],
        ]

        self._stencil_size = 8

        self._slices = [
            [
                [   jnp.s_[..., self.n-4+j:-self.n-3+j, self.nhy, self.nhz],
                    jnp.s_[..., self.n-3+j:-self.n-2+j, self.nhy, self.nhz],  
                    jnp.s_[..., self.n-2+j:-self.n-1+j, self.nhy, self.nhz],  
                    jnp.s_[..., self.n-1+j:-self.n+j  , self.nhy, self.nhz],  
                    jnp.s_[..., self.n+j  :-self.n+1+j, self.nhy, self.nhz],  
                    jnp.s_[..., self.n+1+j:-self.n+2+j, self.nhy, self.nhz],  
                    jnp.s_[..., self.n+2+j:-self.n+3+j, self.nhy, self.nhz],   ],

                [   jnp.s_[..., self.nhx, self.n-4+j:-self.n-3+j, self.nhz],
                    jnp.s_[..., self.nhx, self.n-3+j:-self.n-2+j, self.nhz],  
                    jnp.s_[..., self.nhx, self.n-2+j:-self.n-1+j, self.nhz],  
                    jnp.s_[..., self.nhx, self.n-1+j:-self.n+j  , self.nhz],  
                    jnp.s_[..., self.nhx, self.n+j  :-self.n+1+j, self.nhz],  
                    jnp.s_[..., self.nhx, self.n+1+j:-self.n+2+j, self.nhz],  
                    jnp.s_[..., self.nhx, self.n+2+j:-self.n+3+j, self.nhz],   ],  

                [   jnp.s_[..., self.nhx, self.nhy, self.n-4+j:-self.n-3+j],
                    jnp.s_[..., self.nhx, self.nhy, self.n-3+j:-self.n-2+j],  
                    jnp.s_[..., self.nhx, self.nhy, self.n-2+j:-self.n-1+j],  
                    jnp.s_[..., self.nhx, self.nhy, self.n-1+j:-self.n+j  ],  
                    jnp.s_[..., self.nhx, self.nhy, self.n+j  :-self.n+1+j],  
                    jnp.s_[..., self.nhx, self.nhy, self.n+1+j:-self.n+2+j],
                    jnp.s_[..., self.nhx, self.nhy, self.n+2+j:-self.n+3+j],   ]

            ] for j in range(2)]
        
        # check whether upper slicing limit is 0 
        for j in range(2):
            if -self.n + 3 + j == 0:
                    self._slices[j][0][-1] = jnp.s_[..., self.n+2+j:None, self.nhy, self.nhz]
                    self._slices[j][1][-1] = jnp.s_[..., self.nhx, self.n+2+j:None, self.nhz]
                    self._slices[j][2][-1] = jnp.s_[..., self.nhx, self.nhy, self.n+2+j:None]

    def set_slices_stencil(self) -> None:
        self._slices = [
            [
                [   jnp.s_[..., 0+j, None:None, None:None],  
                    jnp.s_[..., 1+j, None:None, None:None],  
                    jnp.s_[..., 2+j, None:None, None:None],  
                    jnp.s_[..., 3+j, None:None, None:None],  
                    jnp.s_[..., 4+j, None:None, None:None],  
                    jnp.s_[..., 5+j, None:None, None:None],  
                    jnp.s_[..., 6+j, None:None, None:None],    ],  

                [   jnp.s_[..., None:None, 0+j, None:None],  
                    jnp.s_[..., None:None, 1+j, None:None],  
                    jnp.s_[..., None:None, 2+j, None:None],  
                    jnp.s_[..., None:None, 3+j, None:None],  
                    jnp.s_[..., None:None, 4+j, None:None],  
                    jnp.s_[..., None:None, 5+j, None:None],  
                    jnp.s_[..., None:None, 6+j, None:None],    ],

                [   jnp.s_[..., None:None, None:None, 0+j],  
                    jnp.s_[..., None:None, None:None, 1+j],  
                    jnp.s_[..., None:None, None:None, 2+j],  
                    jnp.s_[..., None:None, None:None, 3+j],  
                    jnp.s_[..., None:None, None:None, 4+j],  
                    jnp.s_[..., None:None, None:None, 5+j],  
                    jnp.s_[..., None:None, None:None, 6+j],    ],  
        ] for j in range(2)]

    def reconstruct_xi(self, buffer: jnp.ndarray, axis: int, j: int, dx: float = None, **kwargs) -> jnp.ndarray:
        s1_ = self._slices[j][axis]
        
        beta_0 = buffer[s1_[0]] * (547   * buffer[s1_[0]] - 3882  * buffer[s1_[1]] + 4642 * buffer[s1_[2]] - 1854 * buffer[s1_[3]]) \
            +    buffer[s1_[1]] * (7043  * buffer[s1_[1]] - 17246 * buffer[s1_[2]] + 7042 * buffer[s1_[3]]) \
            +    buffer[s1_[2]] * (11003 * buffer[s1_[2]] - 9402  * buffer[s1_[3]]) \
            +    buffer[s1_[3]] * (2107  * buffer[s1_[3]])

        beta_1 = buffer[s1_[1]] * (267   * buffer[s1_[1]] - 1642  * buffer[s1_[2]] + 1602 * buffer[s1_[3]] - 494  * buffer[s1_[4]]) \
            +    buffer[s1_[2]] * (2843  * buffer[s1_[2]] - 5966  * buffer[s1_[3]] + 1922 * buffer[s1_[4]]) \
            +    buffer[s1_[3]] * (3443  * buffer[s1_[3]] - 2522  * buffer[s1_[4]]) \
            +    buffer[s1_[4]] * (547   * buffer[s1_[4]])

        beta_2 = buffer[s1_[2]] * (547   * buffer[s1_[2]] - 2522  * buffer[s1_[3]] + 1922 * buffer[s1_[4]] - 494  * buffer[s1_[5]]) \
            +    buffer[s1_[3]] * (3443  * buffer[s1_[3]] - 5966  * buffer[s1_[4]] + 1602 * buffer[s1_[5]]) \
            +    buffer[s1_[4]] * (2843  * buffer[s1_[4]] - 1642  * buffer[s1_[5]]) \
            +    buffer[s1_[5]] * (267   * buffer[s1_[5]])

        beta_3 = buffer[s1_[3]] * (2107  * buffer[s1_[3]] - 9402  * buffer[s1_[4]] + 7042 * buffer[s1_[5]] - 1854 * buffer[s1_[6]]) \
            +    buffer[s1_[4]] * (11003 * buffer[s1_[4]] - 17246 * buffer[s1_[5]] + 4642 * buffer[s1_[6]]) \
            +    buffer[s1_[5]] * (7043  * buffer[s1_[5]] - 3882  * buffer[s1_[6]]) \
            +    buffer[s1_[6]] * (547   * buffer[s1_[6]])

        one_beta_0_sq = 1.0 / ((self.eps + beta_0) * (self.eps + beta_0)) 
        one_beta_1_sq = 1.0 / ((self.eps + beta_1) * (self.eps + beta_1)) 
        one_beta_2_sq = 1.0 / ((self.eps + beta_2) * (self.eps + beta_2)) 
        one_beta_3_sq = 1.0 / ((self.eps + beta_3) * (self.eps + beta_3)) 

        alpha_0 = self.dr_[j][0] * one_beta_0_sq
        alpha_1 = self.dr_[j][1] * one_beta_1_sq
        alpha_2 = self.dr_[j][2] * one_beta_2_sq
        alpha_3 = self.dr_[j][3] * one_beta_3_sq

        one_alpha = 1.0 / (alpha_0 + alpha_1 + alpha_2 + alpha_3)

        omega_0 = alpha_0 * one_alpha 
        omega_1 = alpha_1 * one_alpha 
        omega_2 = alpha_2 * one_alpha 
        omega_3 = alpha_3 * one_alpha 

        p_0 = self.cr_[j][0][0] * buffer[s1_[0]] + self.cr_[j][0][1] * buffer[s1_[1]] + self.cr_[j][0][2] * buffer[s1_[2]] + self.cr_[j][0][3] * buffer[s1_[3]]
        p_1 = self.cr_[j][1][0] * buffer[s1_[1]] + self.cr_[j][1][1] * buffer[s1_[2]] + self.cr_[j][1][2] * buffer[s1_[3]] + self.cr_[j][1][3] * buffer[s1_[4]]
        p_2 = self.cr_[j][2][0] * buffer[s1_[2]] + self.cr_[j][2][1] * buffer[s1_[3]] + self.cr_[j][2][2] * buffer[s1_[4]] + self.cr_[j][2][3] * buffer[s1_[5]]
        p_3 = self.cr_[j][3][0] * buffer[s1_[3]] + self.cr_[j][3][1] * buffer[s1_[4]] + self.cr_[j][3][2] * buffer[s1_[5]] + self.cr_[j][3][3] * buffer[s1_[6]]

        cell_state_xi_j = omega_0 * p_0 + omega_1 * p_1 + omega_2 * p_2 + omega_3 * p_3

        return cell_state_xi_j