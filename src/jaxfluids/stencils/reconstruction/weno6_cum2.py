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

class WENO6CUM2(SpatialReconstruction):
    ''' Hu et al. - 2011 - Scale separation for implicit large eddy simulation '''    
    
    def __init__(self, nh: int, inactive_axis: List) -> None:
        super(WENO6CUM2, self).__init__(nh=nh, inactive_axis=inactive_axis)

        self.dr_ = [
            [1/20, 9/20, 9/20, 1/20],
            [1/20, 9/20, 9/20, 1/20],
        ]
        self.cr_ = [
            [[1/3, -7/6, 11/6], [-1/6, 5/6, 1/3], [1/3, 5/6, -1/6], [11/6, -7/6, 1/3]],
            [[1/3, -7/6, 11/6], [-1/6, 5/6, 1/3], [1/3, 5/6, -1/6], [11/6, -7/6, 1/3]],
        ]
        self.Cq_ = 1000
        self.q_  = 4
        self.eps = 1e-8
        self.chi = 1.0 / self.eps

        self._stencil_size = 6

        self._slices = [
            [
                [   jnp.s_[..., self.n-3+j:-self.n-2+j, self.nhy, self.nhz],  
                    jnp.s_[..., self.n-2+j:-self.n-1+j, self.nhy, self.nhz],  
                    jnp.s_[..., self.n-1+j:-self.n+j  , self.nhy, self.nhz],  
                    jnp.s_[..., self.n+j  :-self.n+1+j, self.nhy, self.nhz],  
                    jnp.s_[..., self.n+1+j:-self.n+2+j, self.nhy, self.nhz],  
                    jnp.s_[..., self.n+2+j:-self.n+3+j, self.nhy, self.nhz],   ],

                [   jnp.s_[..., self.nhx, self.n-3+j:-self.n-2+j, self.nhz],  
                    jnp.s_[..., self.nhx, self.n-2+j:-self.n-1+j, self.nhz],  
                    jnp.s_[..., self.nhx, self.n-1+j:-self.n+j  , self.nhz],  
                    jnp.s_[..., self.nhx, self.n+j  :-self.n+1+j, self.nhz],  
                    jnp.s_[..., self.nhx, self.n+1+j:-self.n+2+j, self.nhz],  
                    jnp.s_[..., self.nhx, self.n+2+j:-self.n+3+j, self.nhz],   ],  

                [   jnp.s_[..., self.nhx, self.nhy, self.n-3+j:-self.n-2+j],  
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
                [   jnp.s_[..., 0, None:None, None:None],  
                    jnp.s_[..., 1, None:None, None:None],  
                    jnp.s_[..., 2, None:None, None:None],  
                    jnp.s_[..., 3, None:None, None:None],  
                    jnp.s_[..., 4, None:None, None:None],  
                    jnp.s_[..., 5, None:None, None:None],  ],  

                [   jnp.s_[..., None:None, 0, None:None],  
                    jnp.s_[..., None:None, 1, None:None],  
                    jnp.s_[..., None:None, 2, None:None],  
                    jnp.s_[..., None:None, 3, None:None],  
                    jnp.s_[..., None:None, 4, None:None],  
                    jnp.s_[..., None:None, 5, None:None],  ],

                [   jnp.s_[..., None:None, None:None, 0],  
                    jnp.s_[..., None:None, None:None, 1],  
                    jnp.s_[..., None:None, None:None, 2],  
                    jnp.s_[..., None:None, None:None, 3],  
                    jnp.s_[..., None:None, None:None, 4],  
                    jnp.s_[..., None:None, None:None, 5],  ],  
            ],
            [
                [   jnp.s_[..., 5, None:None, None:None],  
                    jnp.s_[..., 4, None:None, None:None],  
                    jnp.s_[..., 3, None:None, None:None],  
                    jnp.s_[..., 2, None:None, None:None],  
                    jnp.s_[..., 1, None:None, None:None],  
                    jnp.s_[..., 0, None:None, None:None],  ],  

                [   jnp.s_[..., None:None, 5, None:None],  
                    jnp.s_[..., None:None, 4, None:None],  
                    jnp.s_[..., None:None, 3, None:None],  
                    jnp.s_[..., None:None, 2, None:None],  
                    jnp.s_[..., None:None, 1, None:None],  
                    jnp.s_[..., None:None, 0, None:None],  ],

                [   jnp.s_[..., None:None, None:None, 5],  
                    jnp.s_[..., None:None, None:None, 4],  
                    jnp.s_[..., None:None, None:None, 3],  
                    jnp.s_[..., None:None, None:None, 2],  
                    jnp.s_[..., None:None, None:None, 1],  
                    jnp.s_[..., None:None, None:None, 0],  ],  
            ],
            
            ]

    def reconstruct_xi(self, buffer: jnp.ndarray, axis: int, j: int, dx: float, **kwargs) -> jnp.ndarray:
        s1_ = self._slices[j][axis]

        beta_0 = 13.0 / 12.0 * (buffer[s1_[0]] - 2 * buffer[s1_[1]] + buffer[s1_[2]]) * (buffer[s1_[0]] - 2 * buffer[s1_[1]] + buffer[s1_[2]]) \
            + 1.0 / 4.0 * (buffer[s1_[0]] - 4 * buffer[s1_[1]] + 3 * buffer[s1_[2]]) * (buffer[s1_[0]] - 4 * buffer[s1_[1]] + 3 * buffer[s1_[2]])
        beta_1 = 13.0 / 12.0 * (buffer[s1_[1]] - 2 * buffer[s1_[2]] + buffer[s1_[3]]) * (buffer[s1_[1]] - 2 * buffer[s1_[2]] + buffer[s1_[3]]) \
            + 1.0 / 4.0 * (buffer[s1_[1]] - buffer[s1_[3]]) * (buffer[s1_[1]] - buffer[s1_[3]])
        beta_2 = 13.0 / 12.0 * (buffer[s1_[2]] - 2 * buffer[s1_[3]] + buffer[s1_[4]]) * (buffer[s1_[2]] - 2 * buffer[s1_[3]] + buffer[s1_[4]]) \
            + 1.0 / 4.0 * (3 * buffer[s1_[2]] - 4 * buffer[s1_[3]] + buffer[s1_[4]]) * (3 * buffer[s1_[2]] - 4 * buffer[s1_[3]] + buffer[ s1_[4]])

        # # Eq. 25 from Hu et al. 
        # beta_3 = 1.0 / 10080 * (
        #     271779 * buffer[s1_[0]] * buffer[s1_[0]] + \
        #     buffer[s1_[0]] * (2380800  * buffer[s1_[1]] + 4086352  * buffer[s1_[2]]  - 3462252  * buffer[s1_[3]] + 1458762 * buffer[s1_[4]]  - 245620  * buffer[s1_[5]]) + \
        #     buffer[s1_[1]] * (5653317  * buffer[s1_[1]] - 20427884 * buffer[s1_[2]]  + 17905032 * buffer[s1_[3]] - 7727988 * buffer[s1_[4]]  + 1325006 * buffer[s1_[5]]) + \
        #     buffer[s1_[2]] * (19510972 * buffer[s1_[2]] - 35817664 * buffer[s1_[3]]  + 15929912 * buffer[s1_[4]] - 2792660 * buffer[s1_[5]]) + \
        #     buffer[s1_[3]] * (17195652 * buffer[s1_[3]] - 15880404 * buffer[s1_[4]]  + 2863984  * buffer[s1_[5]]) + \
        #     buffer[s1_[4]] * (3824847  * buffer[s1_[4]] - 1429976  * buffer[s1_[5]]) + \
        #     139633 * buffer[s1_[5]] * buffer[s1_[5]]
        #     )

        # # Corrected version
        beta_3 = 1.0 / 10080 / 12 * (
            271779 * buffer[s1_[0]] * buffer[s1_[0]] + \
            buffer[s1_[0]] * (-2380800 * buffer[s1_[1]] + 4086352  * buffer[s1_[2]]  - 3462252  * buffer[s1_[3]] + 1458762 * buffer[s1_[4]]  - 245620  * buffer[s1_[5]]) + \
            buffer[s1_[1]] * (5653317  * buffer[s1_[1]] - 20427884 * buffer[s1_[2]]  + 17905032 * buffer[s1_[3]] - 7727988 * buffer[s1_[4]]  + 1325006 * buffer[s1_[5]]) + \
            buffer[s1_[2]] * (19510972 * buffer[s1_[2]] - 35817664 * buffer[s1_[3]]  + 15929912 * buffer[s1_[4]] - 2792660 * buffer[s1_[5]]) + \
            buffer[s1_[3]] * (17195652 * buffer[s1_[3]] - 15880404 * buffer[s1_[4]]  + 2863984  * buffer[s1_[5]]) + \
            buffer[s1_[4]] * (3824847  * buffer[s1_[4]] - 1429976  * buffer[s1_[5]]) + \
            139633 * buffer[s1_[5]] * buffer[s1_[5]]
            )

        beta_ave = 1/6 * (beta_0 + beta_2 + 4*beta_1)
        tau_6    = beta_3 - beta_ave

        dx2 = dx * dx

        alpha_0 = self.dr_[j][0] * jnp.power( ( self.Cq_ + tau_6 / (beta_0 + self.eps * dx2) * (beta_ave + self.chi * dx2) / (beta_0 + self.chi * dx2) ), self.q_ )
        alpha_1 = self.dr_[j][1] * jnp.power( ( self.Cq_ + tau_6 / (beta_1 + self.eps * dx2) * (beta_ave + self.chi * dx2) / (beta_1 + self.chi * dx2) ), self.q_ )
        alpha_2 = self.dr_[j][2] * jnp.power( ( self.Cq_ + tau_6 / (beta_2 + self.eps * dx2) * (beta_ave + self.chi * dx2) / (beta_2 + self.chi * dx2) ), self.q_ )
        alpha_3 = self.dr_[j][3] * jnp.power( ( self.Cq_ + tau_6 / (beta_3 + self.eps * dx2) * (beta_ave + self.chi * dx2) / (beta_3 + self.chi * dx2) ), self.q_ )

        one_alpha = 1.0 / (alpha_0 + alpha_1 + alpha_2 + alpha_3)

        omega_0 = alpha_0 * one_alpha 
        omega_1 = alpha_1 * one_alpha 
        omega_2 = alpha_2 * one_alpha 
        omega_3 = alpha_3 * one_alpha 

        p_0 = self.cr_[j][0][0] * buffer[s1_[0]] + self.cr_[j][0][1] * buffer[s1_[1]] + self.cr_[j][0][2] * buffer[s1_[2]]
        p_1 = self.cr_[j][1][0] * buffer[s1_[1]] + self.cr_[j][1][1] * buffer[s1_[2]] + self.cr_[j][1][2] * buffer[s1_[3]]
        p_2 = self.cr_[j][2][0] * buffer[s1_[2]] + self.cr_[j][2][1] * buffer[s1_[3]] + self.cr_[j][2][2] * buffer[s1_[4]]
        p_3 = self.cr_[j][3][0] * buffer[s1_[3]] + self.cr_[j][3][1] * buffer[s1_[4]] + self.cr_[j][3][2] * buffer[s1_[5]]

        cell_state_xi_j = omega_0 * p_0 + omega_1 * p_1 + omega_2 * p_2 + omega_3 * p_3

        return cell_state_xi_j