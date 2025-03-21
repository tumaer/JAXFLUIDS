from typing import List

import jax
import jax.numpy as jnp

from jaxfluids.stencils.reconstruction.shock_capturing.weno5_base import WENO5Base

Array = jax.Array

class WENO5Z(WENO5Base):
    ''' Borges et al. - 2008 - An improved WENO scheme for hyperbolic conservation laws '''    
    
    def __init__(
            self, 
            nh: int, 
            inactive_axes: List,
            offset: int = 0,
            **kwargs
        ) -> None:
        super(WENO5Z, self).__init__(nh=nh, inactive_axes=inactive_axes, offset=offset)

        self.debug = False
        
    def reconstruct_xi(
            self, 
            buffer: Array, 
            axis: int, 
            j: int, 
            dx: float = None, 
            **kwargs) -> Array:
        s1_ = self.s_[j][axis]
        u_imm = buffer[s1_[0]]
        u_im  = buffer[s1_[1]]
        u_i   = buffer[s1_[2]]
        u_ip  = buffer[s1_[3]]
        u_ipp = buffer[s1_[4]]

        beta_0, beta_1, beta_2 = self.smoothness(u_imm, u_im, u_i, u_ip, u_ipp)
        tau_5 = jnp.abs(beta_0 - beta_2)

        alpha_z_0 = self._dr[0] * (1.0 + tau_5 / (beta_0 + self.eps))
        alpha_z_1 = self._dr[1] * (1.0 + tau_5 / (beta_1 + self.eps))
        alpha_z_2 = self._dr[2] * (1.0 + tau_5 / (beta_2 + self.eps))

        one_alpha_z = 1.0 / (alpha_z_0 + alpha_z_1 + alpha_z_2)

        omega_z_0 = alpha_z_0 * one_alpha_z 
        omega_z_1 = alpha_z_1 * one_alpha_z 
        omega_z_2 = alpha_z_2 * one_alpha_z 

        p_0, p_1, p_2 = self.polynomials(u_imm, u_im, u_i, u_ip, u_ipp)
        cell_state_xi_j = omega_z_0 * p_0 + omega_z_1 * p_1 + omega_z_2 * p_2

        if self.debug:
            omega = jnp.stack([omega_z_0, omega_z_1, omega_z_2], axis=-1)
            debug_out = {"omega": omega}
            return cell_state_xi_j, debug_out

        return cell_state_xi_j