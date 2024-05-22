from typing import List

import jax.numpy as jnp
from jax import Array

from jaxfluids.stencils.spatial_reconstruction import SpatialReconstruction

class WENO3N(SpatialReconstruction):
    """WENO3N [summary]

    Xiaoshuai et al. - 2015 - A high-resolution hybrid scheme for hyperbolic conservation laws
    """
    
    def __init__(self, 
            nh: int, 
            inactive_axes: List, 
            offset: int = 0,
            **kwargs) -> None:
        super(WENO3N, self).__init__(nh=nh, inactive_axes=inactive_axes, offset=offset)
        
        self.dr_ = [1/3, 2/3]

        self.cr_ = [
            [-0.5, 1.5], 
            [0.5, 0.5]
        ]

        self._stencil_size = 4
        self.array_slices([range(-2, 1, 1), range(1, -2, -1)])
        self.stencil_slices([range(0, 3, 1), range(3, 0, -1)])

    def reconstruct_xi(self, buffer: Array, axis: int, j: int, dx: float = None, **kwargs) -> Array:
        s1_ = self.s_[j][axis]

        beta_0 = (buffer[s1_[1]] - buffer[s1_[0]]) * (buffer[s1_[1]] - buffer[s1_[0]])
        
        beta_1 = (buffer[s1_[2]] - buffer[s1_[1]]) * (buffer[s1_[2]] - buffer[s1_[1]])
        
        beta_3 = 13/12 * (buffer[s1_[0]] - 2*buffer[s1_[1]] + buffer[s1_[2]]) * (buffer[s1_[0]] - 2*buffer[s1_[1]] + buffer[s1_[2]]) \
            + 1/4 * (buffer[s1_[0]] - buffer[s1_[2]]) * (buffer[s1_[0]] - buffer[s1_[2]])

        tau_3 = jnp.abs( 0.5* (beta_0 + beta_1) - beta_3 )

        alpha_z_0 = self.dr_[0] * (1.0 + tau_3 / (beta_0 + self.eps) )
        alpha_z_1 = self.dr_[1] * (1.0 + tau_3 / (beta_1 + self.eps) )

        one_alpha_z = 1.0 / (alpha_z_0 + alpha_z_1)

        omega_z_0 = alpha_z_0 * one_alpha_z
        omega_z_1 = alpha_z_1 * one_alpha_z

        p_0 = self.cr_[0][0] * buffer[s1_[0]] + self.cr_[0][1] * buffer[s1_[1]] 
        p_1 = self.cr_[1][0] * buffer[s1_[1]] + self.cr_[1][1] * buffer[s1_[2]]

        cell_state_xi_j = omega_z_0 * p_0 + omega_z_1 * p_1

        return cell_state_xi_j