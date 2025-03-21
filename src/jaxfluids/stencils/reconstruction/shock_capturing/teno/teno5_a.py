from typing import List, Tuple

import jax
import jax.numpy as jnp

from jaxfluids.stencils.reconstruction.shock_capturing.weno5_base import WENO5Base

Array = jax.Array

class TENO5A(WENO5Base):
    ''' Fu et al. - 2018 -  Improved five- and six-point targeted essentially
    nonoscillatory schemes with adaptive dissipation'''    
    
    def __init__(
            self, 
            nh: int, 
            inactive_axes: List, 
            offset: int = 0,
            **kwargs) -> None:
        super(TENO5A, self).__init__(nh=nh, inactive_axes=inactive_axes, offset=offset)

        # Coefficients for optimized spectral properties
        self.dr_ = [0.1235341937, 0.5065006634, 0.3699651429]

        self.C = 1.0
        self.q = 6

        # Parameters for adaptive dissipation control
        self.xi = 1e-3
        self.Cr = 0.24
        self.eps_dissipation = 0.9 * self.Cr / (1 - self.Cr) * self.xi**2
        self.alpha_1 = 10.0
        self.alpha_2 = 5.0

        self.debug = False


    def compute_adaptive_dissipation(self) -> Tuple[Array, Array, Array]:
        pass


    def reconstruct_xi(self, 
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

        # SMOOTHNESS MEASURE
        gamma_0 = jnp.power((self.C + tau_5 / (beta_0 + self.eps)), self.q)
        gamma_1 = jnp.power((self.C + tau_5 / (beta_1 + self.eps)), self.q)
        gamma_2 = jnp.power((self.C + tau_5 / (beta_2 + self.eps)), self.q)

        one_gamma_sum = 1.0 / (gamma_0 + gamma_1 + gamma_2)

        # ADAPTIVE DISSIPATION CONTROL
        eta_0 = (jnp.abs(2 * (buffer[s1_[2]] - buffer[s1_[1]]) * (buffer[s1_[1]] - buffer[s1_[0]])) + self.eps_dissipation) \
            / ((buffer[s1_[2]] - buffer[s1_[1]])**2 + (buffer[s1_[1]] - buffer[s1_[0]])**2 + self.eps_dissipation) 
        eta_1 = (jnp.abs(2 * (buffer[s1_[3]] - buffer[s1_[2]]) * (buffer[s1_[2]] - buffer[s1_[1]])) + self.eps_dissipation) \
            / ((buffer[s1_[3]] - buffer[s1_[2]])**2 + (buffer[s1_[2]] - buffer[s1_[1]])**2 + self.eps_dissipation)
        eta_2 = (jnp.abs(2 * (buffer[s1_[4]] - buffer[s1_[3]]) * (buffer[s1_[3]] - buffer[s1_[2]])) + self.eps_dissipation) \
            / ((buffer[s1_[4]] - buffer[s1_[3]])**2 + (buffer[s1_[3]] - buffer[s1_[2]])**2 + self.eps_dissipation)

        eta = jnp.minimum(eta_0, jnp.minimum(eta_1, eta_2))
        m = 1 - jnp.minimum(1.0, eta / self.Cr)
        g = jnp.power((1 - m), 4) * (1 + 4*m)
        beta_bar = self.alpha_1 - self.alpha_2 * (1 - g)
        beta_bar = jnp.ceil(beta_bar) - 1.0
        CT = jnp.power(10, (-beta_bar))

        # SHARP CUTOFF FUNCTION
        w0 = self._dr[0] * jnp.where(gamma_0 * one_gamma_sum < CT, 0, 1)
        w1 = self._dr[1] * jnp.where(gamma_1 * one_gamma_sum < CT, 0, 1)
        w2 = self._dr[2] * jnp.where(gamma_2 * one_gamma_sum < CT, 0, 1)

        one_dk = 1.0 / (w0 + w1 + w2 + self.eps)

        omega_0 = w0 * one_dk 
        omega_1 = w1 * one_dk 
        omega_2 = w2 * one_dk 

        p_0, p_1, p_2 = self.polynomials(u_imm, u_im, u_i, u_ip, u_ipp)
        cell_state_xi_j = omega_0 * p_0 + omega_1 * p_1 + omega_2 * p_2

        if self.debug:
            omega = jnp.stack([omega_0, omega_1, omega_2], axis=-1)
            debug_out = {"omega": omega}
            return cell_state_xi_j, debug_out

        return cell_state_xi_j