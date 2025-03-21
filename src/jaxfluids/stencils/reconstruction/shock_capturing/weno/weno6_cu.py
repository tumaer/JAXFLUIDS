from typing import List

import jax
import jax.numpy as jnp

from jaxfluids.stencils.reconstruction.shock_capturing.weno6_base import WENO6Base

Array = jax.Array

class WENO6CU(WENO6Base):
    ''' Hu et al. - 2010 - An adaptive central-upwind WENO scheme '''    
    
    def __init__(
            self, 
            nh: int, 
            inactive_axes: List, 
            offset: int = 0,
            **kwargs
            ) -> None:
        super(WENO6CU, self).__init__(nh=nh, inactive_axes=inactive_axes, offset=offset)

        self.C_ = 20

        self.debug = False

    def reconstruct_xi(
            self,
            buffer: Array,
            axis: int,
            j: int,
            dx: float = None,
            **kwargs
            ) -> Array:
        s1_ = self.s_[j][axis]
        u_imm  = buffer[s1_[0]]
        u_im   = buffer[s1_[1]]
        u_i    = buffer[s1_[2]]
        u_ip   = buffer[s1_[3]]
        u_ipp  = buffer[s1_[4]]
        u_ippp = buffer[s1_[5]]

        beta_0, beta_1, beta_2, beta_3 = self.smoothness(
            u_imm, u_im, u_i, u_ip, u_ipp, u_ippp)
        tau_6 = beta_3 - 1/6 * (beta_0 + 4 * beta_1 + beta_2)

        alpha_0 = self._dr[0] * (self.C_ + tau_6 / (beta_0 + self.eps) )
        alpha_1 = self._dr[1] * (self.C_ + tau_6 / (beta_1 + self.eps) )
        alpha_2 = self._dr[2] * (self.C_ + tau_6 / (beta_2 + self.eps) )
        alpha_3 = self._dr[3] * (self.C_ + tau_6 / (beta_3 + self.eps) )

        one_alpha = 1.0 / (alpha_0 + alpha_1 + alpha_2 + alpha_3)

        omega_0 = alpha_0 * one_alpha 
        omega_1 = alpha_1 * one_alpha 
        omega_2 = alpha_2 * one_alpha 
        omega_3 = alpha_3 * one_alpha 

        p_0, p_1, p_2, p_3 = self.polynomials(
            u_imm, u_im, u_i, u_ip, u_ipp, u_ippp)
        cell_state_xi_j = omega_0 * p_0 + omega_1 * p_1 + omega_2 * p_2 + omega_3 * p_3

        if self.debug:
            omega = jnp.stack([omega_0, omega_1, omega_2, omega_3], axis=-1)
            debug_out = {"omega": omega}
            return cell_state_xi_j, debug_out

        return cell_state_xi_j