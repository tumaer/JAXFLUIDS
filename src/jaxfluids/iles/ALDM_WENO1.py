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
from typing import List

import jax
import jax.numpy as jnp

from jaxfluids.stencils.spatial_reconstruction import SpatialReconstruction

class ALDM_WENO1(SpatialReconstruction):
    """ALDM_WENO1 

    Implementation details provided in parent class.
    """

    def __init__(self, nh: int, inactive_axis: List) -> None:
        super(ALDM_WENO1, self).__init__(nh=nh, inactive_axis=inactive_axis)

        self._stencil_size = 6

        self._slices = [
            [
                [jnp.s_[..., self.n-1+j:-self.n+j, self.nhy, self.nhz], ],

                [jnp.s_[..., self.nhx, self.n-1+j:-self.n+j, self.nhz], ],

                [jnp.s_[..., self.nhx, self.nhy, self.n-1+j:-self.n+j], ],

            ] for j in range(2)]

    def reconstruct_xi(self, primes: jnp.ndarray, axis: int, j: int, dx: float = None, fs=0) -> jnp.ndarray:
        s1_ = self._slices[j][axis]

        cell_state_xi_j = primes[s1_[0]]
        
        return cell_state_xi_j