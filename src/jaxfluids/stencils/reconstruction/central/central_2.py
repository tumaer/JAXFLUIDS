from typing import List

import jax
import jax.numpy as jnp

from jaxfluids.stencils.spatial_reconstruction import SpatialReconstruction

Array = jax.Array

class CentralSecondOrderReconstruction(SpatialReconstruction):
    """CentralSecondOrderReconstruction 

    2nd order stencil for reconstruction at the cell face
          x
    |     |     |
    |  i  | i+1 |
    |     |     |
    """

    required_halos = 1

    def __init__(
            self, 
            nh: int, 
            inactive_axes: List, 
            offset: int = 0, 
            **kwargs
        ) -> None:
        super(CentralSecondOrderReconstruction, self).__init__(nh=nh,
                                                               inactive_axes=inactive_axes,
                                                               offset=offset)

        self._stencil_size = 2
        self.array_slices([range(-1, 1, 1)])
        self.stencil_slices([range(0, 2, 1)])
        self.coeffs = jnp.array([1.0 / 2.0])

    def reconstruct_xi(
            self,
            buffer: Array,
            axis: int,
            j: int = None,
            **kwargs
        ) -> Array:
        s1_ = self.s_[axis]
        cell_state_xi = self.coeffs[0] * (buffer[s1_[0]] + buffer[s1_[1]])

        return cell_state_xi
