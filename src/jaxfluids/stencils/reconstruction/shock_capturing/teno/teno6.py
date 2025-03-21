from typing import List

import jax
import jax.numpy as jnp

from jaxfluids.stencils.reconstruction.shock_capturing.teno6_base import TENO6Base

Array = jax.Array

class TENO6(TENO6Base):
    ''' Fu et al. - 2016 -  A family of high-order targeted ENO schemes for compressible-fluid simulations'''    
    
    def __init__(
            self, 
            nh: int, 
            inactive_axes: List,
            offset: int = 0,
            **kwargs
            ) -> None:
        super(TENO6, self).__init__(nh=nh, inactive_axes=inactive_axes, offset=offset)

        # Coefficients for 6-th order convergence
        self._dr = (0.050, 0.450, 0.300, 0.200)
        # # Coefficients for optimized spectral properties
        # self._dr = (0.054, 0.462, 0.300, 0.184) 

        self.C = 1.0
        self.q = 6
        self.CT = 1e-7

    def reconstruct_xi(
            self, 
            buffer: Array, 
            axis: int, 
            j: int, 
            dx: float = None, 
            **kwargs) -> Array:
        s1_ = self.s_[j][axis]
        u_imm  = buffer[s1_[0]]
        u_im   = buffer[s1_[1]]
        u_i    = buffer[s1_[2]]
        u_ip   = buffer[s1_[3]]
        u_ipp  = buffer[s1_[4]]
        u_ippp = buffer[s1_[5]]

        beta_0, beta_1, beta_2, beta_3, beta_6 = \
        self.smoothness(u_imm, u_im, u_i, u_ip, u_ipp, u_ippp)

        tau_6 = jnp.abs(beta_6 - 1/6 * (beta_0 + 4 * beta_1 + beta_2))

        # SMOOTHNESS MEASURE
        gamma_0 = (self.C + tau_6 / (beta_0 + self.eps))**self.q
        gamma_1 = (self.C + tau_6 / (beta_1 + self.eps))**self.q
        gamma_2 = (self.C + tau_6 / (beta_2 + self.eps))**self.q
        gamma_3 = (self.C + tau_6 / (beta_3 + self.eps))**self.q

        one_gamma_sum = 1.0 / (gamma_0 + gamma_1 + gamma_2 + gamma_3)

        # SHARP CUTOFF FUNCTION
        w0 = self._dr[0] * jnp.where(gamma_0 * one_gamma_sum < self.CT, 0, 1)
        w1 = self._dr[1] * jnp.where(gamma_1 * one_gamma_sum < self.CT, 0, 1)
        w2 = self._dr[2] * jnp.where(gamma_2 * one_gamma_sum < self.CT, 0, 1)
        w3 = self._dr[3] * jnp.where(gamma_3 * one_gamma_sum < self.CT, 0, 1)

        # TODO eps should not be necessary
        one_dk = 1.0 / (w0 + w1 + w2 + w3 + self.eps)
        # one_dk = 1.0 / (w0 + w1 + w2 + w3)

        omega_0 = w0 * one_dk 
        omega_1 = w1 * one_dk 
        omega_2 = w2 * one_dk 
        omega_3 = w3 * one_dk 

        p_0, p_1, p_2, p_3 = self.polynomials(
            u_imm, u_im, u_i, u_ip, u_ipp, u_ippp)
        cell_state_xi_j = omega_0 * p_0 + omega_1 * p_1 \
            + omega_2 * p_2 + omega_3 * p_3
        
        return cell_state_xi_j