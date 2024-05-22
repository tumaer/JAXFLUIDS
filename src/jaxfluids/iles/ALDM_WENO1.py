from functools import partial
from typing import List

import jax
import jax.numpy as jnp
from jax import Array

from jaxfluids.stencils.spatial_reconstruction import SpatialReconstruction

class ALDM_WENO1(SpatialReconstruction):
    """ALDM_WENO1 

    Implementation details provided in parent class.
    """

    is_for_adaptive_mesh = True

    def __init__(
            self, 
            nh: int, 
            inactive_axes: List,
            is_mesh_stretching: List = None,
            cell_sizes: List = None,
            smoothness_measure: str = "TV",
        ) -> None:
        super(ALDM_WENO1, self).__init__(nh=nh, inactive_axes=inactive_axes)

        self.smoothness_measure = smoothness_measure
        self._stencil_size = 6
        self.array_slices([range(-1, 0, 1), range(0, -1, -1)])

    def reconstruct_xi(
            self, 
            buffer: Array, 
            axis: int,
            j: int, 
            dx: float = None, 
            fs=0
        ) -> Array:
        s1_ = self.s_[j][axis]
        cell_state_xi_j = buffer[s1_[0]]

        return cell_state_xi_j