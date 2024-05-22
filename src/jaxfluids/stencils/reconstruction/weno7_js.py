from typing import List

import jax.numpy as jnp
from jax import Array

from jaxfluids.stencils.spatial_reconstruction import SpatialReconstruction

class WENO7(SpatialReconstruction):
    ''' Balsara & Shu - 2000 - '''
    
    def __init__(self, 
            nh: int, 
            inactive_axes: List,
            offset: int = 0,
            **kwargs) -> None:
        super(WENO7, self).__init__(nh=nh, inactive_axes=inactive_axes, offset=offset)

        self.dr_ = [1/35, 12/35, 18/35, 4/35]
        
        self.cr_ = [
            [-1/4, 13/12, -23/12, 25/12], 
            [1/12, -5/12, 13/12, 1/4], 
            [-1/12, 7/12, 7/12, -1/12], 
            [1/4, 13/12, -5/12, 1/12]
        ]

        self._stencil_size = 8
        self.array_slices([range(-4, 3, 1), range(3, -4, -1)])
        self.stencil_slices([range(0, 7, 1), range(7, 0, -1)])

    def reconstruct_xi(self, buffer: Array, axis: int, j: int, dx: float = None, **kwargs) -> Array:
        s1_ = self.s_[j][axis]
        
        beta_0 = buffer[s1_[0]] * (547   * buffer[s1_[0]] - 3882  * buffer[s1_[1]] + 4642 * buffer[s1_[2]] - 1854 * buffer[s1_[3]]) \
            +    buffer[s1_[1]] * (7043  * buffer[s1_[1]] - 17246 * buffer[s1_[2]] + 7042 * buffer[s1_[3]]) \
            +    buffer[s1_[2]] * (11003 * buffer[s1_[2]] - 9402  * buffer[s1_[3]]) \
            +    buffer[s1_[3]] * (2107  * buffer[s1_[3]])

        beta_1 = buffer[s1_[1]] * (267   * buffer[s1_[1]] - 1642  * buffer[s1_[2]] + 1602 * buffer[s1_[3]] - 494  * buffer[s1_[4]]) \
            +    buffer[s1_[2]] * (2843  * buffer[s1_[2]] - 5966  * buffer[s1_[3]] + 1922 * buffer[s1_[4]]) \
            +    buffer[s1_[3]] * (3443  * buffer[s1_[3]] - 2522  * buffer[s1_[4]]) \
            +    buffer[s1_[4]] * (547   * buffer[s1_[4]])

        beta_2 = buffer[s1_[2]] * (547   * buffer[s1_[2]] - 2522  * buffer[s1_[3]] + 1922 * buffer[s1_[4]] - 494  * buffer[s1_[5]]) \
            +    buffer[s1_[3]] * (3443  * buffer[s1_[3]] - 5966  * buffer[s1_[4]] + 1602 * buffer[s1_[5]]) \
            +    buffer[s1_[4]] * (2843  * buffer[s1_[4]] - 1642  * buffer[s1_[5]]) \
            +    buffer[s1_[5]] * (267   * buffer[s1_[5]])

        beta_3 = buffer[s1_[3]] * (2107  * buffer[s1_[3]] - 9402  * buffer[s1_[4]] + 7042 * buffer[s1_[5]] - 1854 * buffer[s1_[6]]) \
            +    buffer[s1_[4]] * (11003 * buffer[s1_[4]] - 17246 * buffer[s1_[5]] + 4642 * buffer[s1_[6]]) \
            +    buffer[s1_[5]] * (7043  * buffer[s1_[5]] - 3882  * buffer[s1_[6]]) \
            +    buffer[s1_[6]] * (547   * buffer[s1_[6]])

        one_beta_0_sq = 1.0 / (beta_0 * beta_0 + self.eps)
        one_beta_1_sq = 1.0 / (beta_1 * beta_1 + self.eps)
        one_beta_2_sq = 1.0 / (beta_2 * beta_2 + self.eps)
        one_beta_3_sq = 1.0 / (beta_3 * beta_3 + self.eps)

        alpha_0 = self.dr_[0] * one_beta_0_sq
        alpha_1 = self.dr_[1] * one_beta_1_sq
        alpha_2 = self.dr_[2] * one_beta_2_sq
        alpha_3 = self.dr_[3] * one_beta_3_sq

        one_alpha = 1.0 / (alpha_0 + alpha_1 + alpha_2 + alpha_3)

        omega_0 = alpha_0 * one_alpha
        omega_1 = alpha_1 * one_alpha
        omega_2 = alpha_2 * one_alpha
        omega_3 = alpha_3 * one_alpha

        p_0 = self.cr_[0][0] * buffer[s1_[0]] + self.cr_[0][1] * buffer[s1_[1]] \
            + self.cr_[0][2] * buffer[s1_[2]] + self.cr_[0][3] * buffer[s1_[3]]
        p_1 = self.cr_[1][0] * buffer[s1_[1]] + self.cr_[1][1] * buffer[s1_[2]] \
            + self.cr_[1][2] * buffer[s1_[3]] + self.cr_[1][3] * buffer[s1_[4]]
        p_2 = self.cr_[2][0] * buffer[s1_[2]] + self.cr_[2][1] * buffer[s1_[3]] \
            + self.cr_[2][2] * buffer[s1_[4]] + self.cr_[2][3] * buffer[s1_[5]]
        p_3 = self.cr_[3][0] * buffer[s1_[3]] + self.cr_[3][1] * buffer[s1_[4]] \
            + self.cr_[3][2] * buffer[s1_[5]] + self.cr_[3][3] * buffer[s1_[6]]

        cell_state_xi_j = omega_0 * p_0 + omega_1 * p_1 + omega_2 * p_2 + omega_3 * p_3

        return cell_state_xi_j
