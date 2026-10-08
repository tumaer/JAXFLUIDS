from functools import partial
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

        if self.debug:
            cell_state_xi_j, omega = self.weno5z(u_imm, u_im, u_i, u_ip, u_ipp)
            debug_out = {"omega": jnp.stack(omega, axis=-1)}
            return cell_state_xi_j, debug_out

        cell_state_xi_j = _weno5z_cell_face(self, u_imm, u_im, u_i, u_ip, u_ipp)

        return cell_state_xi_j

    def weno5z(
            self,
            u_imm: Array,
            u_im: Array,
            u_i: Array,
            u_ip: Array,
            u_ipp: Array
            ) -> Array:
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

        return cell_state_xi_j, (omega_z_0, omega_z_1, omega_z_2)


@partial(jax.custom_jvp, nondiff_argnums=(0,))
def _weno5z_cell_face(
        stencil: WENO5Z,
        u_imm: Array,
        u_im: Array,
        u_i: Array,
        u_ip: Array,
        u_ipp: Array
        ) -> Array:
    """WENO5-Z cell face value. The primal is WENO5Z.weno5z, unchanged.
    The custom JVP below only reorders the derivative of the weighted sum.
    """
    return stencil.weno5z(u_imm, u_im, u_i, u_ip, u_ipp)[0]


@_weno5z_cell_face.defjvp
def _weno5z_cell_face_jvp(stencil: WENO5Z, primals, tangents):
    """Tangent of the WENO5-Z cell face value without catastrophic cancellation.

    In (near-)flat stencils the cell values differ by delta, beta_k ~ delta^2 and
    d omega_k ~ 1/delta (beta_k >> eps), which can be huge (~1e13). The
    straightforward tangent sum_k (d omega_k p_k + omega_k d p_k) then cancels
    terms of that size (sum_k d omega_k = 0) and returns rounding noise that is
    not linear in the tangent. Using sum_k omega_k = 1, the cell face value is
    p_1 + omega_0 (p_0 - p_1) + omega_2 (p_2 - p_1), and the tangent is
    d omega_0 (p_0 - p_1) + d omega_2 (p_2 - p_1) + sum_k omega_k d p_k
    with p_0 - p_1 and p_2 - p_1 formed directly from the cell values.

    The weight tangents are also written without cancellation. The direct form
    d omega_k = (d alpha_k - omega_k sum_j d alpha_j) / A (A = sum_j alpha_j)
    cancels terms of size d alpha_k when omega_k -> 1, e.g. a flat stencil next
    to a jump (alpha_k ~ 1/delta^2, d alpha_k ~ 1/delta^3). Instead use
    d omega_k = ((sum_{j!=k} omega_j) d alpha_k - omega_k sum_{j!=k} d alpha_j) / A.
    The tangent is linear in the input tangents (transposable for reverse mode).
    """
    u_imm, u_im, u_i, u_ip, u_ipp = primals
    du_imm, du_im, du_i, du_ip, du_ipp = tangents
    cell_state_xi_j = _weno5z_cell_face(stencil, u_imm, u_im, u_i, u_ip, u_ipp)

    # SMOOTHNESS INDICATORS AND THEIR TANGENTS
    a_0, da_0 = u_imm - 2*u_im + u_i, du_imm - 2*du_im + du_i
    b_0, db_0 = u_imm - 4*u_im + 3*u_i, du_imm - 4*du_im + 3*du_i
    a_1, da_1 = u_im - 2*u_i + u_ip, du_im - 2*du_i + du_ip
    b_1, db_1 = u_im - u_ip, du_im - du_ip
    a_2, da_2 = u_i - 2*u_ip + u_ipp, du_i - 2*du_ip + du_ipp
    b_2, db_2 = 3*u_i - 4*u_ip + u_ipp, 3*du_i - 4*du_ip + du_ipp
    beta_0, beta_1, beta_2 = stencil.smoothness(u_imm, u_im, u_i, u_ip, u_ipp)
    dbeta_0 = 13.0 / 6.0 * a_0 * da_0 + 1.0 / 2.0 * b_0 * db_0
    dbeta_1 = 13.0 / 6.0 * a_1 * da_1 + 1.0 / 2.0 * b_1 * db_1
    dbeta_2 = 13.0 / 6.0 * a_2 * da_2 + 1.0 / 2.0 * b_2 * db_2

    # WENO-Z WEIGHTS AND THEIR TANGENTS
    tau_5 = jnp.abs(beta_0 - beta_2)
    dtau_5 = jnp.sign(beta_0 - beta_2) * (dbeta_0 - dbeta_2)

    one_beta_0 = 1.0 / (beta_0 + stencil.eps)
    one_beta_1 = 1.0 / (beta_1 + stencil.eps)
    one_beta_2 = 1.0 / (beta_2 + stencil.eps)

    alpha_z_0 = stencil._dr[0] * (1.0 + tau_5 * one_beta_0)
    alpha_z_1 = stencil._dr[1] * (1.0 + tau_5 * one_beta_1)
    alpha_z_2 = stencil._dr[2] * (1.0 + tau_5 * one_beta_2)
    dalpha_z_0 = stencil._dr[0] * (dtau_5 - tau_5 * dbeta_0 * one_beta_0) * one_beta_0
    dalpha_z_1 = stencil._dr[1] * (dtau_5 - tau_5 * dbeta_1 * one_beta_1) * one_beta_1
    dalpha_z_2 = stencil._dr[2] * (dtau_5 - tau_5 * dbeta_2 * one_beta_2) * one_beta_2

    one_alpha_z = 1.0 / (alpha_z_0 + alpha_z_1 + alpha_z_2)

    omega_z_0 = alpha_z_0 * one_alpha_z
    omega_z_1 = alpha_z_1 * one_alpha_z
    omega_z_2 = alpha_z_2 * one_alpha_z
    # (1 - omega_k) formed as the sum of the other weights: no cancellation as omega_k -> 1
    domega_z_0 = ((omega_z_1 + omega_z_2) * dalpha_z_0
                  - omega_z_0 * (dalpha_z_1 + dalpha_z_2)) * one_alpha_z
    domega_z_2 = ((omega_z_0 + omega_z_1) * dalpha_z_2
                  - omega_z_2 * (dalpha_z_0 + dalpha_z_1)) * one_alpha_z

    # CANDIDATE POLYNOMIALS: DIFFERENCES FROM CELL VALUES, TANGENTS
    p_0_m_p_1 = (u_imm - u_ip) / 3.0 - (u_im - u_i)
    p_2_m_p_1 = (u_im - u_ipp) / 6.0 - (u_i - u_ip) / 2.0
    dp_0, dp_1, dp_2 = stencil.polynomials(du_imm, du_im, du_i, du_ip, du_ipp)

    dcell_state_xi_j = domega_z_0 * p_0_m_p_1 + domega_z_2 * p_2_m_p_1 \
        + (omega_z_0 * dp_0 + omega_z_1 * dp_1 + omega_z_2 * dp_2)

    return cell_state_xi_j, dcell_state_xi_j
