from typing import List

import jax.numpy as jnp

import jax
import jax.numpy as jnp

from jaxfluids.stencils.reconstruction.shock_capturing.weno5_base import WENO5Base

Array = jax.Array

class TENO5(WENO5Base):
    ''' Fu et al. - 2016 -  A family of high-order targeted ENO schemes for compressible-fluid simulations'''    
    
    def __init__(
            self, 
            nh: int, 
            inactive_axes: List, 
            offset: int = 0,
            **kwargs) -> None:
        super(TENO5, self).__init__(nh=nh, inactive_axes=inactive_axes, offset=offset)

        # Coefficients for 5-th order convergence
        # self._dr = [1/10, 6/10, 3/10]
        # Coefficients for optimized spectral properties
        self._dr = [0.05, 0.55, 0.40]

        self.C = 1.0
        self.q = 6
        self.CT = 1e-5

    def reconstruct_xi(self, buffer: Array, axis: int, j: int, dx: float = None, **kwargs) -> Array:
        s1_ = self.s_[j][axis]
        u_imm = buffer[s1_[0]]
        u_im  = buffer[s1_[1]]
        u_i   = buffer[s1_[2]]
        u_ip  = buffer[s1_[3]]
        u_ipp = buffer[s1_[4]]

        beta_0, beta_1, beta_2 = self.smoothness(u_imm, u_im, u_i, u_ip, u_ipp)
        tau_5 = jnp.abs(beta_0 - beta_2)

        # SMOOTHNESS MEASURE
        gamma_0 = jnp.power((self.C + tau_5 / (beta_0 + self.eps)), self.q)
        gamma_1 = jnp.power((self.C + tau_5 / (beta_1 + self.eps)), self.q)
        gamma_2 = jnp.power((self.C + tau_5 / (beta_2 + self.eps)), self.q)

        # gamma_0 *= (gamma_0 * gamma_0)
        # gamma_1 *= (gamma_1 * gamma_1)
        # gamma_2 *= (gamma_2 * gamma_2)

        # gamma_0 *= gamma_0
        # gamma_1 *= gamma_1
        # gamma_2 *= gamma_2

        one_gamma_sum = 1.0 / (gamma_0 + gamma_1 + gamma_2)

        # SHARP CUTOFF FUNCTION
        w0 = self._dr[0] * jnp.where(gamma_0 * one_gamma_sum < self.CT, 0, 1)
        w1 = self._dr[1] * jnp.where(gamma_1 * one_gamma_sum < self.CT, 0, 1)
        w2 = self._dr[2] * jnp.where(gamma_2 * one_gamma_sum < self.CT, 0, 1)

        # TODO eps should not be necessary
        one_dk = 1.0 / (w0 + w1 + w2 + self.eps)

        omega_0 = w0 * one_dk 
        omega_1 = w1 * one_dk 
        omega_2 = w2 * one_dk 

        p_0, p_1, p_2 = self.polynomials(u_imm, u_im, u_i, u_ip, u_ipp)
        cell_state_xi_j = omega_0 * p_0 + omega_1 * p_1 + omega_2 * p_2

        return cell_state_xi_j