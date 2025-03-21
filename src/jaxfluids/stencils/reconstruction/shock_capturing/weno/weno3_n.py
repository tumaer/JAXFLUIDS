from typing import List

import jax
import jax.numpy as jnp

from jaxfluids.stencils.reconstruction.shock_capturing.weno3_base import WENO3Base

Array = jax.Array

class WENO3N(WENO3Base):
    """WENO3N [summary]

    Xiaoshuai et al. - 2015 - A high-resolution hybrid scheme for hyperbolic conservation laws
    """
    
    def __init__(
            self, 
            nh: int, 
            inactive_axes: List, 
            offset: int = 0,
            **kwargs) -> None:
        super(WENO3N, self).__init__(nh=nh, inactive_axes=inactive_axes, offset=offset)
        
    def reconstruct_xi(self, buffer: Array, axis: int, j: int, dx: float = None, **kwargs) -> Array:
        s1_ = self.s_[j][axis]
        u_im = buffer[s1_[0]]
        u_i = buffer[s1_[1]]
        u_ip = buffer[s1_[2]]

        beta_0, beta_1 = self.smoothness(u_im, u_i, u_ip)      
        beta_3 = 13/12 * jnp.square(u_im - 2*u_i + u_ip) \
            + 1/4 * jnp.square(u_im - u_ip)

        tau_3 = jnp.abs(0.5 * (beta_0 + beta_1) - beta_3)

        alpha_z_0 = self._dr[0] * (1.0 + tau_3 / (beta_0 + self.eps))
        alpha_z_1 = self._dr[1] * (1.0 + tau_3 / (beta_1 + self.eps))

        one_alpha_z = 1.0 / (alpha_z_0 + alpha_z_1)

        omega_z_0 = alpha_z_0 * one_alpha_z
        omega_z_1 = alpha_z_1 * one_alpha_z

        p_0, p_1 = self.polynomials(u_im, u_i, u_ip)
        cell_state_xi_j = omega_z_0 * p_0 + omega_z_1 * p_1

        return cell_state_xi_j