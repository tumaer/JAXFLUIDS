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

class DerivativeFirstOrderSided(SpatialDerivative):
    
    def __init__(self, nh: int, inactive_axis: List, offset: int = 0):
        super(DerivativeFirstOrderSided, self).__init__(nh, inactive_axis, offset)

        self.s_ = [
        [
            [   jnp.s_[..., self.n-1+j:-self.n-1+j, self.nhy, self.nhz],        
                jnp.s_[..., jnp.s_[self.n-0+j:-self.n-0+j] if -self.n-0+j != 0 else jnp.s_[self.n-0+j:None], self.nhy, self.nhz],   ],

            [   jnp.s_[..., self.nhx, self.n-1+j:-self.n-1+j, self.nhz],     
                jnp.s_[..., self.nhx, jnp.s_[self.n-0+j:-self.n-0+j] if -self.n-0+j != 0 else jnp.s_[self.n-0+j:None], self.nhz],   ],

            [   jnp.s_[..., self.nhx, self.nhy, self.n-1+j:-self.n-1+j],     
                jnp.s_[..., self.nhx, self.nhy, jnp.s_[self.n-0+j:-self.n-0+j] if -self.n-0+j != 0 else jnp.s_[self.n-0+j:None]],   ],
                
        ] for j in [0, 1] ]

    def derivative_xi(self, levelset: jnp.ndarray, dxi: jnp.ndarray, i: int, j: int, *args) -> jnp.ndarray:
        s1_ = self.s_[j][i]
        deriv_xi = (1.0 / dxi) * (-levelset[s1_[0]] + levelset[s1_[1]])
        return deriv_xi