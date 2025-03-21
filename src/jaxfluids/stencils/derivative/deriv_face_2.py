from typing import List

import jax
import jax.numpy as jnp

from jaxfluids.stencils.spatial_derivative import SpatialDerivative

Array = jax.Array

class DerivativeSecondOrderFace(SpatialDerivative):
    """2nd order stencil for 1st derivative at the cell face
        x
    |   |     |     
    | i | i+1 |     
    |   |     |     
    """

    required_halos = 1
    
    def __init__(
            self,
            nh: int,
            inactive_axes: List,
            offset: int = 0,
            **kwargs
        ) -> None:
        super(DerivativeSecondOrderFace, self).__init__(nh=nh,
                                                        inactive_axes=inactive_axes,
                                                        offset=offset)

        self.array_slices([(-1, 0)])

    def derivative_xi(
            self,
            buffer: Array,
            dxi: Array,
            axis: int,
            is_use_s_mesh: bool = False,
            **kwargs
        ) -> Array:

        if is_use_s_mesh:
            s_ = self.s_mesh[axis]
        else:
            s_ = self.s_[axis]

        deriv_xi = (1.0 / dxi) * (buffer[s_[1]] - buffer[s_[0]])
        return deriv_xi