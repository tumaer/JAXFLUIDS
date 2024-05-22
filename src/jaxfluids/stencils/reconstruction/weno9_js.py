from typing import List

import jax.numpy as jnp
from jax import Array

from jaxfluids.stencils.spatial_reconstruction import SpatialReconstruction

class WENO9(SpatialReconstruction):
    ''' Balsara & Shu - 2000 - '''
    def __init__(self,
            nh: int,
            inactive_axes: List,
            offset: int = 0,
            **kwargs) -> None:
        super(WENO9, self).__init__(nh=nh, inactive_axes=inactive_axes, offset=offset)

        self.dr_ = [1/126, 10/63, 10/21, 20/63, 5/126]

        self.cr_ = [
            [1/5  , -21/20, 137/60, -163/60, 137/60],
            [-1/20, 17/60 , -43/60, 77/60  , 1/5   ],
            [1/30 , -13/60, 47/60 , 9/20   , -1/20 ],
            [-1/20, 9/20  , 47/60 , -13/60 , 1/30  ],
            [1/5  , 77/60 , -43/60, 17/60  , -1/20 ],
        ]

        self._stencil_size = 10
        self.array_slices([range(-5, 4, 1), range(4, -5, -1)])
        self.stencil_slices([range(0, 9, 1), range(9, 0, -1)])

    def reconstruct_xi(self, buffer: Array, axis: int, j: int, dx: float = None, **kwargs) -> Array:
        s1_ = self.s_[j][axis]
        
        beta_0 = buffer[s1_[0]] * (22658   * buffer[s1_[0]] - 208501  * buffer[s1_[1]] + 364863  * buffer[s1_[2]] - 288007 * buffer[s1_[3]] + 86329 * buffer[s1_[4]]) \
            +    buffer[s1_[1]] * (482963  * buffer[s1_[1]] - 1704396 * buffer[s1_[2]] + 1358458 * buffer[s1_[3]] - 411487 * buffer[s1_[4]]) \
            +    buffer[s1_[2]] * (1521393 * buffer[s1_[2]] - 2462076 * buffer[s1_[3]] + 758823  * buffer[s1_[4]]) \
            +    buffer[s1_[3]] * (1020563 * buffer[s1_[3]] - 649501  * buffer[s1_[4]]) \
            +    buffer[s1_[4]] * (107918  * buffer[s1_[4]])

        beta_1 = buffer[s1_[1]] * (6908    * buffer[s1_[1]] - 60871   * buffer[s1_[2]] + 99213   * buffer[s1_[3]] - 70237  * buffer[s1_[4]] + 18079 * buffer[s1_[5]]) \
            +    buffer[s1_[2]] * (138563  * buffer[s1_[2]] - 464976  * buffer[s1_[3]] + 337018  * buffer[s1_[4]] - 88297  * buffer[s1_[5]]) \
            +    buffer[s1_[3]] * (406293  * buffer[s1_[3]] - 611976  * buffer[s1_[4]] + 165153  * buffer[s1_[5]]) \
            +    buffer[s1_[4]] * (242723  * buffer[s1_[4]] - 140251  * buffer[s1_[5]]) \
            +    buffer[s1_[5]] * (22658   * buffer[s1_[5]])

        beta_2 = buffer[s1_[2]] * (6908    * buffer[s1_[2]] - 51001  * buffer[s1_[3]] + 67923  * buffer[s1_[4]] - 38947 * buffer[s1_[5]] + 8209 * buffer[s1_[6]]) \
            +    buffer[s1_[3]] * (104963  * buffer[s1_[3]] - 299076 * buffer[s1_[4]] + 179098 * buffer[s1_[5]] - 38947 * buffer[s1_[6]]) \
            +    buffer[s1_[4]] * (231153  * buffer[s1_[4]] - 299076 * buffer[s1_[5]] + 67923  * buffer[s1_[6]]) \
            +    buffer[s1_[5]] * (104963  * buffer[s1_[5]] - 51001  * buffer[s1_[6]]) \
            +    buffer[s1_[6]] * (6908    * buffer[s1_[6]])

        beta_3 = buffer[s1_[3]] * (22658  * buffer[s1_[3]] - 140251 * buffer[s1_[4]] + 165153 * buffer[s1_[5]] - 88297 * buffer[s1_[6]] + 18079 * buffer[s1_[7]]) \
            +    buffer[s1_[4]] * (242723 * buffer[s1_[4]] - 611976 * buffer[s1_[5]] + 337018 * buffer[s1_[6]] - 70237 * buffer[s1_[7]]) \
            +    buffer[s1_[5]] * (406293 * buffer[s1_[5]] - 464976 * buffer[s1_[6]] + 99213  * buffer[s1_[7]]) \
            +    buffer[s1_[6]] * (138563 * buffer[s1_[6]] - 60871  * buffer[s1_[7]]) \
            +    buffer[s1_[7]] * (6908   * buffer[s1_[7]])

        beta_4 = buffer[s1_[4]] * (107918  * buffer[s1_[4]] - 649501  * buffer[s1_[5]] + 758823  * buffer[s1_[6]] - 411487 * buffer[s1_[7]] + 86329 * buffer[s1_[8]]) \
            +    buffer[s1_[5]] * (1020563 * buffer[s1_[5]] - 2462076 * buffer[s1_[6]] + 1358458 * buffer[s1_[7]] - 288007 * buffer[s1_[8]]) \
            +    buffer[s1_[6]] * (1521393 * buffer[s1_[6]] - 1704396 * buffer[s1_[7]] + 364863  * buffer[s1_[8]]) \
            +    buffer[s1_[7]] * (482963  * buffer[s1_[7]] - 208501  * buffer[s1_[8]]) \
            +    buffer[s1_[8]] * (22658   * buffer[s1_[8]])

        one_beta_0_sq = 1.0 / (beta_0 * beta_0 + self.eps)
        one_beta_1_sq = 1.0 / (beta_1 * beta_1 + self.eps)
        one_beta_2_sq = 1.0 / (beta_2 * beta_2 + self.eps)
        one_beta_3_sq = 1.0 / (beta_3 * beta_3 + self.eps)
        one_beta_4_sq = 1.0 / (beta_4 * beta_4 + self.eps)

        alpha_0 = self.dr_[0] * one_beta_0_sq
        alpha_1 = self.dr_[1] * one_beta_1_sq
        alpha_2 = self.dr_[2] * one_beta_2_sq
        alpha_3 = self.dr_[3] * one_beta_3_sq
        alpha_4 = self.dr_[4] * one_beta_4_sq

        one_alpha = 1.0 / (alpha_0 + alpha_1 + alpha_2 + alpha_3 + alpha_4)

        omega_0 = alpha_0 * one_alpha
        omega_1 = alpha_1 * one_alpha
        omega_2 = alpha_2 * one_alpha
        omega_3 = alpha_3 * one_alpha
        omega_4 = alpha_4 * one_alpha

        p_0 = self.cr_[0][0] * buffer[s1_[0]] + self.cr_[0][1] * buffer[s1_[1]] \
            + self.cr_[0][2] * buffer[s1_[2]] + self.cr_[0][3] * buffer[s1_[3]] \
            + self.cr_[0][4] * buffer[s1_[4]]
        p_1 = self.cr_[1][0] * buffer[s1_[1]] + self.cr_[1][1] * buffer[s1_[2]] \
            + self.cr_[1][2] * buffer[s1_[3]] + self.cr_[1][3] * buffer[s1_[4]] \
            + self.cr_[1][4] * buffer[s1_[5]]
        p_2 = self.cr_[2][0] * buffer[s1_[2]] + self.cr_[2][1] * buffer[s1_[3]] \
            + self.cr_[2][2] * buffer[s1_[4]] + self.cr_[2][3] * buffer[s1_[5]] \
            + self.cr_[2][4] * buffer[s1_[6]]
        p_3 = self.cr_[3][0] * buffer[s1_[3]] + self.cr_[3][1] * buffer[s1_[4]] \
            + self.cr_[3][2] * buffer[s1_[5]] + self.cr_[3][3] * buffer[s1_[6]] \
            + self.cr_[3][4] * buffer[s1_[7]]
        p_4 = self.cr_[4][0] * buffer[s1_[4]] + self.cr_[4][1] * buffer[s1_[5]] \
            + self.cr_[4][2] * buffer[s1_[6]] + self.cr_[4][3] * buffer[s1_[7]] \
            + self.cr_[4][4] * buffer[s1_[8]]

        cell_state_xi_j = omega_0 * p_0 + omega_1 * p_1 + omega_2 * p_2 + omega_3 * p_3 + omega_4 * p_4

        return cell_state_xi_j
