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

class WENO9(SpatialReconstruction):
    ''' Balsara & Shu - 2000 - '''
    def __init__(self, nh: int, inactive_axis: List) -> None:
        super(WENO9, self).__init__(nh=nh, inactive_axis=inactive_axis)

        self.dr_ = [
            [1/126, 10/63, 10/21, 20/63, 5/126],
            [5/126, 20/63, 10/21, 10/63, 1/126],
        ]
        self.cr_ = [
            [[1/5, -21/20, 137/60, -163/60, 137/60], [-1/20, 17/60, -43/60, 77/60, 1/5], [1/30, -13/60, 47/60, 9/20, -1/20], [-1/20, 9/20, 47/60, -13/60, 1/30], [1/5, 77/60, -43/60, 17/60, -1/20]],
            [[-1/20, 17/60, -43/60, 77/60, 1/5]    , [1/30, -13/60, 47/60, 9/20, -1/20], [-1/20, 9/20, 47/60, -13/60, 1/30], [1/5, 77/60, -43/60, 17/60, -1/20], [137/60, -163/60, 137/60, -21/20, 1/5]],
        ]

        self._stencil_size = 10

        self._slices = [
            [
                [   jnp.s_[..., self.n-5+j:-self.n-4+j, self.nhy, self.nhz],
                    jnp.s_[..., self.n-4+j:-self.n-3+j, self.nhy, self.nhz],
                    jnp.s_[..., self.n-3+j:-self.n-2+j, self.nhy, self.nhz],  
                    jnp.s_[..., self.n-2+j:-self.n-1+j, self.nhy, self.nhz],  
                    jnp.s_[..., self.n-1+j:-self.n+j  , self.nhy, self.nhz],  
                    jnp.s_[..., self.n+j  :-self.n+1+j, self.nhy, self.nhz],  
                    jnp.s_[..., self.n+1+j:-self.n+2+j, self.nhy, self.nhz],  
                    jnp.s_[..., self.n+2+j:-self.n+3+j, self.nhy, self.nhz],
                    jnp.s_[..., self.n+3+j:-self.n+4+j, self.nhy, self.nhz],    ],

                [   jnp.s_[..., self.nhx, self.n-5+j:-self.n-4+j, self.nhz],
                    jnp.s_[..., self.nhx, self.n-4+j:-self.n-3+j, self.nhz],
                    jnp.s_[..., self.nhx, self.n-3+j:-self.n-2+j, self.nhz],  
                    jnp.s_[..., self.nhx, self.n-2+j:-self.n-1+j, self.nhz],  
                    jnp.s_[..., self.nhx, self.n-1+j:-self.n+j  , self.nhz],  
                    jnp.s_[..., self.nhx, self.n+j  :-self.n+1+j, self.nhz],  
                    jnp.s_[..., self.nhx, self.n+1+j:-self.n+2+j, self.nhz],  
                    jnp.s_[..., self.nhx, self.n+2+j:-self.n+3+j, self.nhz],
                    jnp.s_[..., self.nhx, self.n+3+j:-self.n+4+j, self.nhz],   ],  

                [   jnp.s_[..., self.nhx, self.nhy, self.n-5+j:-self.n-4+j],
                    jnp.s_[..., self.nhx, self.nhy, self.n-4+j:-self.n-3+j],
                    jnp.s_[..., self.nhx, self.nhy, self.n-3+j:-self.n-2+j],  
                    jnp.s_[..., self.nhx, self.nhy, self.n-2+j:-self.n-1+j],  
                    jnp.s_[..., self.nhx, self.nhy, self.n-1+j:-self.n+j  ],  
                    jnp.s_[..., self.nhx, self.nhy, self.n+j  :-self.n+1+j],  
                    jnp.s_[..., self.nhx, self.nhy, self.n+1+j:-self.n+2+j],
                    jnp.s_[..., self.nhx, self.nhy, self.n+2+j:-self.n+3+j],
                    jnp.s_[..., self.nhx, self.nhy, self.n+3+j:-self.n+4+j],   ]

            ] for j in range(2)]
        
        # check whether upper slicing limit is 0 
        for j in range(2):
            if -self.n + 4 + j == 0:
                    self._slices[j][0][-1] = jnp.s_[..., self.n+3+j:None, self.nhy, self.nhz]
                    self._slices[j][1][-1] = jnp.s_[..., self.nhx, self.n+3+j:None, self.nhz]
                    self._slices[j][2][-1] = jnp.s_[..., self.nhx, self.nhy, self.n+3+j:None]

    def set_slices_stencil(self) -> None:
        self._slices = [
            [
                [   jnp.s_[..., 0+j, None:None, None:None],  
                    jnp.s_[..., 1+j, None:None, None:None],  
                    jnp.s_[..., 2+j, None:None, None:None],  
                    jnp.s_[..., 3+j, None:None, None:None],  
                    jnp.s_[..., 4+j, None:None, None:None],  
                    jnp.s_[..., 5+j, None:None, None:None],  
                    jnp.s_[..., 6+j, None:None, None:None],
                    jnp.s_[..., 7+j, None:None, None:None],
                    jnp.s_[..., 8+j, None:None, None:None],    ],  

                [   jnp.s_[..., None:None, 0+j, None:None],  
                    jnp.s_[..., None:None, 1+j, None:None],  
                    jnp.s_[..., None:None, 2+j, None:None],  
                    jnp.s_[..., None:None, 3+j, None:None],  
                    jnp.s_[..., None:None, 4+j, None:None],  
                    jnp.s_[..., None:None, 5+j, None:None],  
                    jnp.s_[..., None:None, 6+j, None:None],
                    jnp.s_[..., None:None, 7+j, None:None],
                    jnp.s_[..., None:None, 8+j, None:None],    ],

                [   jnp.s_[..., None:None, None:None, 0+j],  
                    jnp.s_[..., None:None, None:None, 1+j],  
                    jnp.s_[..., None:None, None:None, 2+j],  
                    jnp.s_[..., None:None, None:None, 3+j],  
                    jnp.s_[..., None:None, None:None, 4+j],  
                    jnp.s_[..., None:None, None:None, 5+j],  
                    jnp.s_[..., None:None, None:None, 6+j],
                    jnp.s_[..., None:None, None:None, 7+j],
                    jnp.s_[..., None:None, None:None, 8+j],    ],  
        ] for j in range(2)]

    def reconstruct_xi(self, buffer: jnp.ndarray, axis: int, j: int, dx: float = None, **kwargs) -> jnp.ndarray:
        s1_ = self._slices[j][axis]
        
        beta_0 = buffer[s1_[0]] * (22658   * buffer[s1_[0]] - 208501  * buffer[s1_[1]] + 364863  * buffer[s1_[2]] - 288007 * buffer[s1_[3]] + 86329 * buffer[s1_[4]]) \
            +    buffer[s1_[1]] * (482963  * buffer[s1_[1]] - 1704396 * buffer[s1_[2]] + 1358458 * buffer[s1_[3]] - 411487 * buffer[s1_[4]]) \
            +    buffer[s1_[2]] * (1521393 * buffer[s1_[2]] - 2462076 * buffer[s1_[3]] + 758823  * buffer[s1_[4]]) \
            +    buffer[s1_[3]] * (1020563 * buffer[s1_[3]] - 649501  * buffer[s1_[4]]) \
            +    buffer[s1_[4]] * (107918  * buffer[s1_[4]])

        beta_1 = buffer[s1_[1]] * (6908    * buffer[s1_[1]] - 60871   * buffer[s1_[2]] + 99213   * buffer[s1_[3]] - 70237  * buffer[s1_[4]] + 18079 * buffer[s1_[5]]) \
            +    buffer[s1_[2]] * (138563  * buffer[s1_[2]] - 464976  * buffer[s1_[3]] + 337018  * buffer[s1_[4]] - 88297  * buffer[s1_[5]]) \
            +    buffer[s1_[3]] * (406293  * buffer[s1_[3]] - 611976  * buffer[s1_[4]] + 165153  * buffer[s1_[5]]) \
            +    buffer[s1_[4]] * (242723  * buffer[s1_[4]] - 140251  * buffer[s1_[5]]) \
            +    buffer[s1_[5]] * (22658   * buffer[s1_[5]])

        beta_2 = buffer[s1_[2]] * (6908    * buffer[s1_[2]] - 51001  * buffer[s1_[3]] + 67923  * buffer[s1_[4]] - 38947 * buffer[s1_[5]] + 8209 * buffer[s1_[6]]) \
            +    buffer[s1_[3]] * (104963  * buffer[s1_[3]] - 299076 * buffer[s1_[4]] + 179098 * buffer[s1_[5]] - 38947 * buffer[s1_[6]]) \
            +    buffer[s1_[4]] * (231153  * buffer[s1_[4]] - 299076 * buffer[s1_[5]] + 67923  * buffer[s1_[6]]) \
            +    buffer[s1_[5]] * (104963  * buffer[s1_[5]] - 51001  * buffer[s1_[6]]) \
            +    buffer[s1_[6]] * (6908    * buffer[s1_[6]])

        beta_3 = buffer[s1_[3]] * (22658  * buffer[s1_[3]] - 140251 * buffer[s1_[4]] + 165153 * buffer[s1_[5]] - 88297 * buffer[s1_[6]] + 18079 * buffer[s1_[7]]) \
            +    buffer[s1_[4]] * (242723 * buffer[s1_[4]] - 611976 * buffer[s1_[5]] + 337018 * buffer[s1_[6]] - 70237 * buffer[s1_[7]]) \
            +    buffer[s1_[5]] * (406293 * buffer[s1_[5]] - 464976 * buffer[s1_[6]] + 99213  * buffer[s1_[7]]) \
            +    buffer[s1_[6]] * (138563 * buffer[s1_[6]] - 60871  * buffer[s1_[7]]) \
            +    buffer[s1_[7]] * (6908   * buffer[s1_[7]])

        beta_4 = buffer[s1_[4]] * (107918  * buffer[s1_[4]] - 649501  * buffer[s1_[5]] + 758823  * buffer[s1_[6]] - 411487 * buffer[s1_[7]] + 86329 * buffer[s1_[8]]) \
            +    buffer[s1_[5]] * (1020563 * buffer[s1_[5]] - 2462076 * buffer[s1_[6]] + 1358458 * buffer[s1_[7]] - 288007 * buffer[s1_[8]]) \
            +    buffer[s1_[6]] * (1521393 * buffer[s1_[6]] - 1704396 * buffer[s1_[7]] + 364863  * buffer[s1_[8]]) \
            +    buffer[s1_[7]] * (482963  * buffer[s1_[7]] - 208501  * buffer[s1_[8]]) \
            +    buffer[s1_[8]] * (22658   * buffer[s1_[8]])

        one_beta_0_sq = 1.0 / ((self.eps + beta_0) * (self.eps + beta_0)) 
        one_beta_1_sq = 1.0 / ((self.eps + beta_1) * (self.eps + beta_1)) 
        one_beta_2_sq = 1.0 / ((self.eps + beta_2) * (self.eps + beta_2)) 
        one_beta_3_sq = 1.0 / ((self.eps + beta_3) * (self.eps + beta_3)) 
        one_beta_4_sq = 1.0 / ((self.eps + beta_4) * (self.eps + beta_4)) 

        alpha_0 = self.dr_[j][0] * one_beta_0_sq
        alpha_1 = self.dr_[j][1] * one_beta_1_sq
        alpha_2 = self.dr_[j][2] * one_beta_2_sq
        alpha_3 = self.dr_[j][3] * one_beta_3_sq
        alpha_4 = self.dr_[j][4] * one_beta_4_sq

        one_alpha = 1.0 / (alpha_0 + alpha_1 + alpha_2 + alpha_3 + alpha_4)

        omega_0 = alpha_0 * one_alpha 
        omega_1 = alpha_1 * one_alpha 
        omega_2 = alpha_2 * one_alpha 
        omega_3 = alpha_3 * one_alpha 
        omega_4 = alpha_4 * one_alpha 

        p_0 = self.cr_[j][0][0] * buffer[s1_[0]] + self.cr_[j][0][1] * buffer[s1_[1]] + self.cr_[j][0][2] * buffer[s1_[2]] + self.cr_[j][0][3] * buffer[s1_[3]] + self.cr_[j][0][4] * buffer[s1_[4]]
        p_1 = self.cr_[j][1][0] * buffer[s1_[1]] + self.cr_[j][1][1] * buffer[s1_[2]] + self.cr_[j][1][2] * buffer[s1_[3]] + self.cr_[j][1][3] * buffer[s1_[4]] + self.cr_[j][1][4] * buffer[s1_[5]]
        p_2 = self.cr_[j][2][0] * buffer[s1_[2]] + self.cr_[j][2][1] * buffer[s1_[3]] + self.cr_[j][2][2] * buffer[s1_[4]] + self.cr_[j][2][3] * buffer[s1_[5]] + self.cr_[j][2][4] * buffer[s1_[6]]
        p_3 = self.cr_[j][3][0] * buffer[s1_[3]] + self.cr_[j][3][1] * buffer[s1_[4]] + self.cr_[j][3][2] * buffer[s1_[5]] + self.cr_[j][3][3] * buffer[s1_[6]] + self.cr_[j][3][4] * buffer[s1_[7]]
        p_4 = self.cr_[j][4][0] * buffer[s1_[4]] + self.cr_[j][4][1] * buffer[s1_[5]] + self.cr_[j][4][2] * buffer[s1_[6]] + self.cr_[j][4][3] * buffer[s1_[7]] + self.cr_[j][4][4] * buffer[s1_[8]]

        cell_state_xi_j = omega_0 * p_0 + omega_1 * p_1 + omega_2 * p_2 + omega_3 * p_3 + omega_4 * p_4

        return cell_state_xi_j
