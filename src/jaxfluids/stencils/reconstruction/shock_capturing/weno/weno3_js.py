from typing import List

import jax
import jax.numpy as jnp

from jaxfluids.stencils.reconstruction.shock_capturing.weno3_base import WENO3Base

Array = jax.Array

class WENO3JS(WENO3Base):

    def __init__(self, nh: int, inactive_axes: List, offset: int = 0, **kwargs) -> None:
        super(WENO3JS, self).__init__(nh, inactive_axes, offset, **kwargs)

    def reconstruct_xi(self, buffer: Array, axis: int, j: int, dx: float = None,
                       is_use_s_mesh: bool = False, **kwargs) -> Array:

        if is_use_s_mesh:
            s1_ = self.s_mesh[j][axis]
        else:
            s1_ = self.s_[j][axis]

        u_im = buffer[s1_[0]]
        u_i = buffer[s1_[1]]
        u_ip = buffer[s1_[2]]

        beta_0, beta_1 = self.smoothness(u_im, u_i, u_ip)

        one_beta_0_sq = 1.0 / (beta_0 * beta_0 + self.eps)
        one_beta_1_sq = 1.0 / (beta_1 * beta_1 + self.eps)

        alpha_0 = self._dr[0] * one_beta_0_sq
        alpha_1 = self._dr[1] * one_beta_1_sq
        one_alpha = 1.0 / (alpha_0 + alpha_1)

        omega_0 = alpha_0 * one_alpha
        omega_1 = alpha_1 * one_alpha

        p_0, p_1 = self.polynomials(u_im, u_i, u_ip)
        cell_state_xi_j = omega_0 * p_0 + omega_1 * p_1

        return cell_state_xi_j
