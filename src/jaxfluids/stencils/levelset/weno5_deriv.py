from typing import List

import jax.numpy as jnp
from jax import Array

from jaxfluids.stencils.spatial_derivative import SpatialDerivative
from jaxfluids.stencils.levelset.first_deriv_first_order_center import FirstDerivativeFirstOrderCenter
from jaxfluids.stencils.levelset.first_deriv_second_order_center import FirstDerivativeSecondOrder
from jaxfluids.stencils.reconstruction.weno5_js import WENO5

class WENO5DERIV(SpatialDerivative):
    
    required_halos = 5

    def __init__(self, nh: int, inactive_axes: List, offset: int = 0):
        fixed_offset = 3
        super(WENO5DERIV, self).__init__(nh, inactive_axes, offset)
        self.derivative_stencil  = FirstDerivativeFirstOrderCenter(nh, inactive_axes, offset + fixed_offset)
        self.reconstruction_stencil = WENO5(offset + fixed_offset, inactive_axes, offset)
        self.slices = [
            [jnp.s_[...,1:,:,:], jnp.s_[...,:,1:,:], jnp.s_[...,:,:,1:]],
            [jnp.s_[...,:-1,:,:], jnp.s_[...,:,:-1,:], jnp.s_[...,:,:,:-1]]
        ]

    def derivative_xi(self, levelset: Array, dx:int, i: int, j: int, *args) -> Array:
        levelset = self.derivative_stencil.derivative_xi(levelset, dx, i, j)
        cell_state_xi_j = self.reconstruction_stencil.reconstruct_xi(levelset, i, j)
        s_ = self.slices[j][i]
        return cell_state_xi_j[s_]