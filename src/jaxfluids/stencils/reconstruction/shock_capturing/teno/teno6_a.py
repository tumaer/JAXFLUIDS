from typing import List

import jax
import jax.numpy as jnp

from jaxfluids.stencils.reconstruction.shock_capturing.teno6_base import TENO6Base
from jaxfluids.math.power_functions import power6

Array = jax.Array

class TENO6A(TENO6Base):
    ''' Fu et al. - 2018 -  Improved five- and six-point targeted essentially
    nonoscillatory schemes with adaptive dissipation'''    
    
    def __init__(
            self, 
            nh: int, 
            inactive_axes: List,
            offset: int = 0,
            **kwargs
            ) -> None:
        super(TENO6A, self).__init__(nh=nh, inactive_axes=inactive_axes, offset=offset)

        # Coefficients for 6-th order convergence
        # self._dr = (0.050, 0.450, 0.300, 0.200)
        # Coefficients for optimized spectral properties & 4-th order convergence
        self._dr = (
            0.0855682281039113, 0.4294317718960898, 
            0.1727270875843552, 0.3122729124156450
        )

        self.C = 1.0
        self.q = 6
        
        # Parameters for adaptive dissipation control
        self.xi = 1e-3
        self.Cr = 0.17
        self.eps_dissipation = 0.9 * self.Cr / (1 - self.Cr) * self.xi**2
        self.alpha_1 = 10.5
        self.alpha_2 = 4.5

        self.is_positivity_limiter_smoothness = True

        self.is_debug = False

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

        if self.is_positivity_limiter_smoothness:
            beta_3 = jnp.abs(beta_3)
            beta_6 = jnp.abs(beta_6)

        tau_6 = jnp.abs(beta_6 - 1/6 * (beta_0 + 4 * beta_1 + beta_2))

        # SMOOTHNESS MEASURE
        gamma_0 = (self.C + tau_6 / (beta_0 + self.eps))**self.q
        gamma_1 = (self.C + tau_6 / (beta_1 + self.eps))**self.q
        gamma_2 = (self.C + tau_6 / (beta_2 + self.eps))**self.q
        gamma_3 = (self.C + tau_6 / (beta_3 + self.eps))**self.q

        one_gamma_sum = 1.0 / (gamma_0 + gamma_1 + gamma_2 + gamma_3)

        # ADAPTIVE DISSIPATION CONTROL
        CT = self.compute_adaptive_dissipation(
            u_imm, u_im, u_i, u_ip, u_ipp, u_ippp
        )

        # SHARP CUTOFF FUNCTION
        w0 = self._dr[0] * jnp.where(gamma_0 * one_gamma_sum < CT, 0, 1)
        w1 = self._dr[1] * jnp.where(gamma_1 * one_gamma_sum < CT, 0, 1)
        w2 = self._dr[2] * jnp.where(gamma_2 * one_gamma_sum < CT, 0, 1)
        w3 = self._dr[3] * jnp.where(gamma_3 * one_gamma_sum < CT, 0, 1)

        one_dk = 1.0 / (w0 + w1 + w2 + w3)

        omega_0 = w0 * one_dk 
        omega_1 = w1 * one_dk 
        omega_2 = w2 * one_dk 
        omega_3 = w3 * one_dk 

        p_0, p_1, p_2, p_3 = self.polynomials(
            u_imm, u_im, u_i, u_ip, u_ipp, u_ippp)
        cell_state_xi_j = omega_0 * p_0 + omega_1 * p_1 \
            + omega_2 * p_2 + omega_3 * p_3
        
        if self.is_debug:
            return cell_state_xi_j, jnp.array([omega_0, omega_1, omega_2, omega_3])

        return cell_state_xi_j

    def compute_adaptive_dissipation(
            self, u_imm: Array, u_im: Array,
            u_i: Array, u_ip: Array, u_ipp: Array,
            u_ippp: Array) -> Array:
        
        eta = self.compute_eta(u_imm, u_im, u_i,
                               u_ip, u_ipp, u_ippp)
        m = 1 - jnp.minimum(1.0, eta / self.Cr)
        g = jnp.power((1 - m), 4) * (1 + 4 * m)
        beta_bar = self.alpha_1 - self.alpha_2 * (1 - g)
        beta_bar = jnp.ceil(beta_bar) - 1.0
        CT = jnp.power(10, (-beta_bar))

        return CT

    def compute_eta(self, u_imm: Array, u_im: Array,
                    u_i: Array, u_ip: Array, u_ipp: Array,
                    u_ippp: Array) -> Array:
        # TODO can this be optimized?
        diff_0 = u_im   - u_imm
        diff_1 = u_i    - u_im
        diff_2 = u_ip   - u_i
        diff_3 = u_ipp  - u_ip
        diff_4 = u_ippp - u_ipp

        eta_0 = (jnp.abs(2 * diff_1 * diff_0) + self.eps_dissipation) \
            / (diff_1**2 + diff_0**2 + self.eps_dissipation)
        eta = eta_0

        eta_1 = (jnp.abs(2 * diff_2 * diff_1) + self.eps_dissipation) \
            / (diff_2**2 + diff_1**2 + self.eps_dissipation)
        eta = jnp.minimum(eta, eta_1)

        eta_2 = (jnp.abs(2 * diff_3 * diff_2) + self.eps_dissipation) \
            / (diff_3**2 + diff_2**2 + self.eps_dissipation)
        eta = jnp.minimum(eta, eta_2)

        eta_3 = (jnp.abs(2 * diff_4 * diff_3) + self.eps_dissipation) \
            / (diff_4**2 + diff_3**2 + self.eps_dissipation)
        eta = jnp.minimum(eta, eta_3)

        return eta