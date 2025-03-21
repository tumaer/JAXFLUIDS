from typing import List

import jax
import jax.numpy as jnp

from jaxfluids.stencils.spatial_derivative import SpatialDerivative
from jaxfluids.stencils.reconstruction.shock_capturing.weno.weno3_js import WENO3JS
from jaxfluids.stencils.derivative.deriv_face_2 import DerivativeSecondOrderFace

Array = jax.Array

class WENO3HJ(SpatialDerivative):

    required_halos = 2

    def __init__(self, nh: int, inactive_axes: List, offset: int = 0):
        super(SpatialDerivative, self).__init__(nh, inactive_axes, offset)
        self.derivative_stencil = DerivativeSecondOrderFace(nh, inactive_axes, offset + self.required_halos - 1)
        self.reconstruction_stencil = WENO3JS(offset + self.required_halos, inactive_axes, offset)

        self.sd_ = (
            jnp.s_[:,self.nhy,self.nhz],
            jnp.s_[self.nhx,:,self.nhz],
            jnp.s_[self.nhx,self.nhy,:],
        )

    def derivative_xi(self, buffer: Array, dx: float, axis: int, j: int, *args) -> Array:
        # NOTE we use mesh slices here to prevent slicing in non present axis directions.
        # Otherwise, offset of derivative stencil must be equal to nh of weno stencil,
        # which would increase number of required halos of full stencil by 1.
        buffer = self.derivative_stencil.derivative_xi(buffer, dx, axis, is_use_s_mesh=True)
        buffer = self.reconstruction_stencil.reconstruct_xi(buffer, axis, j, is_use_s_mesh=True)
        s_ = self.sd_[axis]
        buffer = buffer[s_]
        return buffer
