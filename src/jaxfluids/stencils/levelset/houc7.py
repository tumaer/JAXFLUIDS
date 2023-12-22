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

from jaxfluids.stencils.spatial_derivative import SpatialDerivative

class HOUC7(SpatialDerivative):
    
    def __init__(self, nh: int, inactive_axis: List):
        super(HOUC7, self).__init__(nh=nh, inactive_axis=inactive_axis)
        
        self.coeff = [3.0, -28.0, 126.0, -420.0, 105.0, 252.0, -42.0, 4.0]

        self.sign = [1, -1]
        
        self._slices = [
            [
                [   
                    jnp.s_[..., jnp.s_[self.n-4*j:-self.n-4*j] if -self.n-4*j != 0 else jnp.s_[self.n-4*j:None], self.nhy, self.nhz],  
                    jnp.s_[..., self.n-3*j:-self.n-3*j, self.nhy, self.nhz],  
                    jnp.s_[..., self.n-2*j:-self.n-2*j, self.nhy, self.nhz],  
                    jnp.s_[..., self.n-1*j:-self.n-1*j, self.nhy, self.nhz],  
                    jnp.s_[..., self.n+0*j:-self.n+0*j, self.nhy, self.nhz],
                    jnp.s_[..., self.n+1*j:-self.n+1*j, self.nhy, self.nhz],
                    jnp.s_[..., self.n+2*j:-self.n+2*j, self.nhy, self.nhz],   
                    jnp.s_[..., self.n+3*j:-self.n+3*j, self.nhy, self.nhz]   
                                                                                    ],  

                [   
                    jnp.s_[..., self.nhx, jnp.s_[self.n-4*j:-self.n-4*j] if -self.n-4*j != 0 else jnp.s_[self.n-4*j:None], self.nhz],
                    jnp.s_[..., self.nhx, self.n-3*j:-self.n-3*j, self.nhz],
                    jnp.s_[..., self.nhx, self.n-2*j:-self.n-2*j, self.nhz],
                    jnp.s_[..., self.nhx, self.n-1*j:-self.n-1*j, self.nhz],     
                    jnp.s_[..., self.nhx, self.n+0*j:-self.n+0*j, self.nhz],     
                    jnp.s_[..., self.nhx, self.n+1*j:-self.n+1*j, self.nhz],
                    jnp.s_[..., self.nhx, self.n+2*j:-self.n+2*j, self.nhz],   
                    jnp.s_[..., self.nhx, self.n+3*j:-self.n+3*j, self.nhz]   
                                                                                    ],

                [   
                    jnp.s_[..., self.nhx, self.nhy, jnp.s_[self.n-4*j:-self.n-4*j] if -self.n-4*j != 0 else jnp.s_[self.n-4*j:None]],
                    jnp.s_[..., self.nhx, self.nhy, self.n-3*j:-self.n-3*j],
                    jnp.s_[..., self.nhx, self.nhy, self.n-2*j:-self.n-2*j],
                    jnp.s_[..., self.nhx, self.nhy, self.n-1*j:-self.n-1*j],     
                    jnp.s_[..., self.nhx, self.nhy, self.n+0*j:-self.n+0*j],     
                    jnp.s_[..., self.nhx, self.nhy, self.n+1*j:-self.n+1*j],
                    jnp.s_[..., self.nhx, self.nhy, self.n+2*j:-self.n+2*j],   
                    jnp.s_[..., self.nhx, self.nhy, self.n+3*j:-self.n+3*j]   
                                                                                    ]

            ] for j in self.sign ]

    def derivative_xi(self, levelset: jnp.ndarray, dxi:float, i: int, j: int, *args) -> jnp.ndarray:
        s1_ = self._slices[j][i]

        cell_state_xi_j = sum(levelset[s1_[k]]*self.coeff[k] for k in range(len(self.coeff)))
        cell_state_xi_j *= self.sign[j] * 1.0 / 420.0
        cell_state_xi_j *= (1.0 / dxi)

        return cell_state_xi_j
