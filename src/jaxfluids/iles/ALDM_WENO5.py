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

from functools import partial
from typing import List, Tuple

import jax
import jax.numpy as jnp

from jaxfluids.stencils.spatial_reconstruction import SpatialReconstruction

class ALDM_WENO5(SpatialReconstruction):
    """ALDM_WENO5 

    Implementation details provided in parent class.
    """
    
    def __init__(self, nh: int, inactive_axis: List):
        super(ALDM_WENO5, self).__init__(nh=nh, inactive_axis=inactive_axis)

        self.dr_adlm_ = [
            [0.89548, 0.08550, 0.01902],
            [0.01902, 0.08550, 0.89548],
            # [0.01902, 0.08550, 0.89548],
            # [0.89548, 0.08550, 0.01902],
        ]

        self.dr_ = [
            [0.1, 0.6, 0.3],
            [0.3, 0.6, 0.1],
        ]

        self.cr_ = [
            [[1/3, -7/6, 11/6], [-1/6, 5/6, 1/3], [1/3, 5/6, -1/6]],
            [[-1/6, 5/6, 1/3], [1/3, 5/6, -1/6], [11/6, -7/6, 1/3]],
        ]

        self._stencil_size = 6

        self._slices = [
            [
                [   jnp.s_[..., self.n-3+j:-self.n-2+j, self.nhy, self.nhz],  
                    jnp.s_[..., self.n-2+j:-self.n-1+j, self.nhy, self.nhz],  
                    jnp.s_[..., self.n-1+j:-self.n+j  , self.nhy, self.nhz],  
                    jnp.s_[..., self.n+j  :-self.n+1+j, self.nhy, self.nhz],  
                    jnp.s_[..., self.n+1+j:-self.n+2+j, self.nhy, self.nhz],   ],

                [   jnp.s_[..., self.nhx, self.n-3+j:-self.n-2+j, self.nhz],  
                    jnp.s_[..., self.nhx, self.n-2+j:-self.n-1+j, self.nhz],  
                    jnp.s_[..., self.nhx, self.n-1+j:-self.n+j  , self.nhz],  
                    jnp.s_[..., self.nhx, self.n+j  :-self.n+1+j, self.nhz],  
                    jnp.s_[..., self.nhx, self.n+1+j:-self.n+2+j, self.nhz],   ],  

                [   jnp.s_[..., self.nhx, self.nhy, self.n-3+j:-self.n-2+j],  
                    jnp.s_[..., self.nhx, self.nhy, self.n-2+j:-self.n-1+j],  
                    jnp.s_[..., self.nhx, self.nhy, self.n-1+j:-self.n+j  ],  
                    jnp.s_[..., self.nhx, self.nhy, self.n+j  :-self.n+1+j],  
                    jnp.s_[..., self.nhx, self.nhy, self.n+1+j:-self.n+2+j],   ]

            ] for j in range(2)]

    def get_adaptive_ideal_weights(self, j: int, fs: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        d0 = self.dr_adlm_[j][0] + fs * (self.dr_[j][0] - self.dr_adlm_[j][0])
        d1 = self.dr_adlm_[j][1] + fs * (self.dr_[j][1] - self.dr_adlm_[j][1])
        d2 = self.dr_adlm_[j][2] + fs * (self.dr_[j][2] - self.dr_adlm_[j][2])
        return d0, d1, d2

    def reconstruct_xi(self, primes: jnp.ndarray, axis: int, j: int, dx: float = None, fs: jnp.ndarray = 0) -> jnp.ndarray:
        s1_ = self._slices[j][axis]
        
        beta_0 = (primes[s1_[1]] - primes[s1_[0]]) * (primes[s1_[1]] - primes[s1_[0]]) \
            +    (primes[s1_[2]] - primes[s1_[1]]) * (primes[s1_[2]] - primes[s1_[1]])
        beta_1 = (primes[s1_[2]] - primes[s1_[1]]) * (primes[s1_[2]] - primes[s1_[1]]) \
            +    (primes[s1_[3]] - primes[s1_[2]]) * (primes[s1_[3]] - primes[s1_[2]])
        beta_2 = (primes[s1_[3]] - primes[s1_[2]]) * (primes[s1_[3]] - primes[s1_[2]]) \
            +    (primes[s1_[4]] - primes[s1_[3]]) * (primes[s1_[4]] - primes[s1_[3]])

        one_beta_0_sq = 1.0 / ((self.eps + beta_0) * (self.eps + beta_0)) 
        one_beta_1_sq = 1.0 / ((self.eps + beta_1) * (self.eps + beta_1)) 
        one_beta_2_sq = 1.0 / ((self.eps + beta_2) * (self.eps + beta_2)) 

        d0, d1, d2 = self.get_adaptive_ideal_weights(j, fs)

        alpha_0 = d0 * one_beta_0_sq
        alpha_1 = d1 * one_beta_1_sq
        alpha_2 = d2 * one_beta_2_sq

        one_alpha = 1.0 / (alpha_0 + alpha_1 + alpha_2)

        omega_0 = alpha_0 * one_alpha 
        omega_1 = alpha_1 * one_alpha 
        omega_2 = alpha_2 * one_alpha 

        p_0 = self.cr_[j][0][0] * primes[s1_[0]] + self.cr_[j][0][1] * primes[s1_[1]] + self.cr_[j][0][2] * primes[s1_[2]]
        p_1 = self.cr_[j][1][0] * primes[s1_[1]] + self.cr_[j][1][1] * primes[s1_[2]] + self.cr_[j][1][2] * primes[s1_[3]]
        p_2 = self.cr_[j][2][0] * primes[s1_[2]] + self.cr_[j][2][1] * primes[s1_[3]] + self.cr_[j][2][2] * primes[s1_[4]]

        cell_state_xi_j = omega_0 * p_0 + omega_1 * p_1 + omega_2 * p_2

        return cell_state_xi_j