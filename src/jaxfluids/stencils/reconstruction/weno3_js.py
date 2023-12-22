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

class WENO3(SpatialReconstruction):
    """WENO3-JS 
    """
    
    def __init__(self, nh: int, inactive_axis: List) -> None:
        super(WENO3, self).__init__(nh=nh, inactive_axis=inactive_axis)
        
        self.dr = [1/3, 2/3]

        self.dr_ = [
            [1/3, 2/3],
            [2/3, 1/3],
        ]
        self.cr_ = [
            [[-0.5, 1.5], [0.5, 0.5]],
            [[0.5, 0.5], [1.5, -0.5]],
        ]

        self._stencil_size = 4

        self._slices = [
            [
                [   jnp.s_[..., self.n-2+j:-self.n-1+j, self.nhy, self.nhz],  
                    jnp.s_[..., self.n-1+j:-self.n+j,   self.nhy, self.nhz],  
                    jnp.s_[..., self.n+j:-self.n+1+j,   self.nhy, self.nhz], ],  

                [   jnp.s_[..., self.nhx, self.n-2+j:-self.n-1+j, self.nhz],  
                    jnp.s_[..., self.nhx, self.n-1+j:-self.n+j,   self.nhz],  
                    jnp.s_[..., self.nhx, self.n+j:-self.n+1+j,   self.nhz], ],   

                [   jnp.s_[..., self.nhx, self.nhy, self.n-2+j:-self.n-1+j,],  
                    jnp.s_[..., self.nhx, self.nhy, self.n-1+j:-self.n+j,  ],  
                    jnp.s_[..., self.nhx, self.nhy, self.n+j:-self.n+1+j,  ], ],

            ] for j in range(2)]

    def set_slices_stencil(self) -> None:
        self._slices = [
            [
                [   jnp.s_[..., 0+j, None:None, None:None],  
                    jnp.s_[..., 1+j, None:None, None:None],  
                    jnp.s_[..., 2+j, None:None, None:None], ],   

                [   jnp.s_[..., None:None, 0+j, None:None],  
                    jnp.s_[..., None:None, 1+j, None:None],  
                    jnp.s_[..., None:None, 2+j, None:None], ],

                [   jnp.s_[..., None:None, None:None, 0+j],  
                    jnp.s_[..., None:None, None:None, 1+j],  
                    jnp.s_[..., None:None, None:None, 2+j], ],

        ] for j in range(2)]

    def reconstruct_xi(self, buffer: jnp.ndarray, axis: int, j: int, dx: float = None, **kwargs) -> jnp.ndarray:
        s1_ = self._slices[j][axis]

        beta_0 = (buffer[s1_[1]] - buffer[s1_[0]]) * (buffer[s1_[1]] - buffer[s1_[0]])
        beta_1 = (buffer[s1_[2]] - buffer[s1_[1]]) * (buffer[s1_[2]] - buffer[s1_[1]])

        one_beta_0_sq = 1.0 / ((self.eps + beta_0) * (self.eps + beta_0))
        one_beta_1_sq = 1.0 / ((self.eps + beta_1) * (self.eps + beta_1))

        alpha_0 = self.dr_[j][0] * one_beta_0_sq
        alpha_1 = self.dr_[j][1] * one_beta_1_sq

        one_alpha = 1.0 / (alpha_0 + alpha_1)

        omega_0 = alpha_0 * one_alpha
        omega_1 = alpha_1 * one_alpha

        p_0 = self.cr_[j][0][0] * buffer[s1_[0]] + self.cr_[j][0][1] * buffer[s1_[1]] 
        p_1 = self.cr_[j][1][0] * buffer[s1_[1]] + self.cr_[j][1][1] * buffer[s1_[2]]

        cell_state_xi_j = omega_0 * p_0 + omega_1 * p_1

        return cell_state_xi_j