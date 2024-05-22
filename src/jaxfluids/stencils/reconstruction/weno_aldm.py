from typing import List

import jax.numpy as jnp
from jax import Array

from jaxfluids.stencils.spatial_reconstruction import SpatialReconstruction

class WENOALDM(SpatialReconstruction):
    
    def __init__(self, 
            nh: int, 
            inactive_axes: List,
            offset: int = 0,
            **kwargs) -> None:
        super(WENOALDM, self).__init__(nh=nh, inactive_axes=inactive_axes, offset=offset)

        self.dr_aldm_3 = [
            [0.0, 1.0],
            [1.0, 0.0],
        ]

        self.dr_adlm_5 = [
            [0.89548, 0.08550, 0.01902],
            [0.01902, 0.08550, 0.89548]
        ]

        self.cr_3 = [
            [[-0.5, 1.5], [0.5, 0.5]],
            [[0.5, 0.5], [1.5, -0.5]],
        ]

        self.cr_5 = [
            [[1/3, -7/6, 11/6], [-1/6, 5/6, 1/3], [1/3, 5/6, -1/6]],
            [[-1/6, 5/6, 1/3], [1/3, 5/6, -1/6], [11/6, -7/6, 1/3]],
        ]

        self._stencil_size = 6
        self.array_slices([range(-3, 2, 1), range(2, -3, -1)])
        self.stencil_slices([range(0, 5, 1), range(5, 0, -1)])

    def set_slices_stencil(self) -> None:
        self._slices = [
            [
                [   jnp.s_[..., 0+j, None:None, None:None],  
                    jnp.s_[..., 1+j, None:None, None:None],  
                    jnp.s_[..., 2+j, None:None, None:None],  
                    jnp.s_[..., 3+j, None:None, None:None],  
                    jnp.s_[..., 4+j, None:None, None:None], ],  

                [   jnp.s_[..., None:None, 0+j, None:None],  
                    jnp.s_[..., None:None, 1+j, None:None],  
                    jnp.s_[..., None:None, 2+j, None:None],  
                    jnp.s_[..., None:None, 3+j, None:None],  
                    jnp.s_[..., None:None, 4+j, None:None], ],

                [   jnp.s_[..., None:None, None:None, 0+j],  
                    jnp.s_[..., None:None, None:None, 1+j],  
                    jnp.s_[..., None:None, None:None, 2+j],  
                    jnp.s_[..., None:None, None:None, 3+j],  
                    jnp.s_[..., None:None, None:None, 4+j], ],

        ] for j in range(2) ]

    def reconstruct_xi(self, buffer: Array, axis: int, j: int, dx: float = None, **kwargs) -> Array:
        s1_ = self._slices[j][axis]

        # SMOOTHNES WENO3
        beta_02 = (buffer[s1_[2]] - buffer[s1_[1]]) * (buffer[s1_[2]] - buffer[s1_[1]])
        beta_12 = (buffer[s1_[3]] - buffer[s1_[2]]) * (buffer[s1_[3]] - buffer[s1_[2]])

        one_beta_02_sq = 1.0 / ((self.eps + beta_02) * (self.eps + beta_02))
        one_beta_12_sq = 1.0 / ((self.eps + beta_12) * (self.eps + beta_12))

        alpha_02 = self.dr_aldm_3[j][0] * one_beta_02_sq
        alpha_12 = self.dr_aldm_3[j][1] * one_beta_12_sq

        one_alpha = 1.0 / (alpha_02 + alpha_12)

        omega_02 = alpha_02 * one_alpha
        omega_12 = alpha_12 * one_alpha

        # SMOOTHNESS WENO5
        beta_03 = (buffer[s1_[1]] - buffer[s1_[0]]) * (buffer[s1_[1]] - buffer[s1_[0]]) \
            +    (buffer[s1_[2]] - buffer[s1_[1]]) * (buffer[s1_[2]] - buffer[s1_[1]])
        beta_13 = (buffer[s1_[2]] - buffer[s1_[1]]) * (buffer[s1_[2]] - buffer[s1_[1]]) \
            +    (buffer[s1_[3]] - buffer[s1_[2]]) * (buffer[s1_[3]] - buffer[s1_[2]])
        beta_23 = (buffer[s1_[3]] - buffer[s1_[2]]) * (buffer[s1_[3]] - buffer[s1_[2]]) \
            +    (buffer[s1_[4]] - buffer[s1_[3]]) * (buffer[s1_[4]] - buffer[s1_[3]])

        one_beta_03_sq = 1.0 / ((self.eps + beta_03) * (self.eps + beta_03)) 
        one_beta_13_sq = 1.0 / ((self.eps + beta_13) * (self.eps + beta_13)) 
        one_beta_23_sq = 1.0 / ((self.eps + beta_23) * (self.eps + beta_23)) 

        # d0, d1, d2 = self.get_adaptive_ideal_weights(j, fs)

        alpha_03 = self.dr_adlm_5[j][0] * one_beta_03_sq
        alpha_13 = self.dr_adlm_5[j][1] * one_beta_13_sq
        alpha_23 = self.dr_adlm_5[j][2] * one_beta_23_sq

        one_alpha = 1.0 / (alpha_03 + alpha_13 + alpha_23)

        omega_03 = alpha_03 * one_alpha 
        omega_13 = alpha_13 * one_alpha 
        omega_23 = alpha_23 * one_alpha

        # WENO1 polynomial
        p_01 = buffer[s1_[2]] 

        # WENO3 polynomial
        p_02 = self.cr_3[j][0][0] * buffer[s1_[1]] + self.cr_3[j][0][1] * buffer[s1_[2]] 
        p_12 = self.cr_3[j][1][0] * buffer[s1_[2]] + self.cr_3[j][1][1] * buffer[s1_[3]]

        # WENO5 polynomial        
        p_03 = self.cr_5[j][0][0] * buffer[s1_[0]] + self.cr_5[j][0][1] * buffer[s1_[1]] + self.cr_5[j][0][2] * buffer[s1_[2]]
        p_13 = self.cr_5[j][1][0] * buffer[s1_[1]] + self.cr_5[j][1][1] * buffer[s1_[2]] + self.cr_5[j][1][2] * buffer[s1_[3]]
        p_23 = self.cr_5[j][2][0] * buffer[s1_[2]] + self.cr_5[j][2][1] * buffer[s1_[3]] + self.cr_5[j][2][2] * buffer[s1_[4]]

        cell_state_xi_j_1 = p_01
        cell_state_xi_j_3 = omega_02 * p_02 + omega_12 * p_12
        cell_state_xi_j_5 = omega_03 * p_03 + omega_13 * p_13 + omega_23 * p_23

        cell_state_xi_j = 1/3 * (cell_state_xi_j_1 + cell_state_xi_j_3 + cell_state_xi_j_5)

        return cell_state_xi_j