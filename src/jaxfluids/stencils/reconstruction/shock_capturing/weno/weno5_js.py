from typing import List

import jax
import jax.numpy as jnp

from jaxfluids.stencils.reconstruction.shock_capturing.weno5_base import WENO5Base

Array = jax.Array

class WENO5JS(WENO5Base):

    def __init__(
            self, 
            nh: int, 
            inactive_axes: List, 
            offset: int = 0,
            **kwargs) -> None:
        super(WENO5JS, self).__init__(nh=nh, inactive_axes=inactive_axes, offset=offset)

        # DEBUG FEATURE
        self.debug = False

    def reconstruct_xi(self, buffer: Array, axis: int, j: int, dx: float = None,
                       is_use_s_mesh: bool = False, **kwargs) -> Array:

        if is_use_s_mesh:
            s1_ = self.s_mesh[j][axis]
        else:
            s1_ = self.s_[j][axis]
            
        u_imm = buffer[s1_[0]]
        u_im  = buffer[s1_[1]]
        u_i   = buffer[s1_[2]]
        u_ip  = buffer[s1_[3]]
        u_ipp = buffer[s1_[4]]

        beta_0, beta_1, beta_2 = self.smoothness(u_imm, u_im, u_i, u_ip, u_ipp)

        one_beta_0_sq = 1.0 / (beta_0 * beta_0 + self.eps)
        one_beta_1_sq = 1.0 / (beta_1 * beta_1 + self.eps)
        one_beta_2_sq = 1.0 / (beta_2 * beta_2 + self.eps)

        # one_beta_0_sq = 1.0 / ((beta_0 + self.eps) * (beta_0 + self.eps))
        # one_beta_1_sq = 1.0 / ((beta_1 + self.eps) * (beta_1 + self.eps))
        # one_beta_2_sq = 1.0 / ((beta_2 + self.eps) * (beta_2 + self.eps))

        alpha_0 = self._dr[0] * one_beta_0_sq
        alpha_1 = self._dr[1] * one_beta_1_sq
        alpha_2 = self._dr[2] * one_beta_2_sq

        one_alpha = 1.0 / (alpha_0 + alpha_1 + alpha_2)

        omega_0 = alpha_0 * one_alpha 
        omega_1 = alpha_1 * one_alpha 
        omega_2 = alpha_2 * one_alpha 

        p_0, p_1, p_2 = self.polynomials(u_imm, u_im, u_i, u_ip, u_ipp)
        cell_state_xi_j = omega_0 * p_0 + omega_1 * p_1 + omega_2 * p_2

        if self.debug:
            omega = jnp.stack([omega_0, omega_1, omega_2], axis=-1)
            debug_out = {"omega": omega}
            return cell_state_xi_j, debug_out

        return cell_state_xi_j