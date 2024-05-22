from typing import List

import jax.numpy as jnp
from jax import Array

from jaxfluids.stencils.spatial_reconstruction import SpatialReconstruction

class WENO1(SpatialReconstruction):

    is_for_adaptive_mesh = True

    def __init__(self, 
            nh: int, 
            inactive_axes: List, 
            offset: int = 0, 
            **kwargs) -> None:
        super(WENO1, self).__init__(nh=nh, inactive_axes=inactive_axes, offset=offset)

        self._stencil_size = 2
        self.array_slices([range(-1, 0, 1), range(0, -1, -1)])
        self.stencil_slices([range(0, 1, 1), range(1, 0, -1)])

    def reconstruct_xi(self, buffer: Array, axis: int, j: int, dx = None, **kwargs) -> Array:
        s1_ = self.s_[j][axis]

        cell_state_xi_j = buffer[s1_[0]]
        
        return cell_state_xi_j