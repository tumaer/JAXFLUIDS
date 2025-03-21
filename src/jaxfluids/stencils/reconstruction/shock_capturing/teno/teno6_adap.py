from typing import List, Tuple

import jax
import jax.numpy as jnp

from jaxfluids.stencils.spatial_reconstruction import SpatialReconstruction
from jaxfluids.stencils.helper_functions import compute_coefficients_stretched_mesh_teno6

Array = jax.Array

class TENO6ADAP(SpatialReconstruction):
    ''' Fu et al. - 2016 -  A family of high-order targeted ENO schemes for compressible-fluid simulations'''    

    is_for_adaptive_mesh = True

    def __init__(
            self, 
            nh: int, 
            inactive_axes: List,
            is_mesh_stretching: List[bool] = None,
            cell_sizes: Tuple[Array] = None,
            offset: int = 0,
            **kwargs
            ) -> None:
        super(TENO6ADAP, self).__init__(nh=nh, inactive_axes=inactive_axes, offset=offset)

        # Coefficients for 6-th order convergence
        self.dr_uniform = [0.050, 0.450, 0.300, 0.200]
        # Coefficients for optimized spectral properties
        # TODO make adaptive coefficients go to the 
        # optimized spectral properties
        # self.dr_uniform = [0.054, 0.462, 0.300, 0.184] 

        self.cr_uniform = [
            [1/3, -7/6, 11/6], 
            [-1/6, 5/6, 1/3], 
            [1/3, 5/6, -1/6], 
            [3/12, 13/12, -5/12, 1/12]
        ]

        self.C = 1.0
        self.q = 6
        self.CT = 1e-7

        self._stencil_size = 6
        self.array_slices([range(-3, 3, 1), range(2, -4, -1)])
        self.stencil_slices([range(0, 6, 1), range(5, -1, -1)])
        self.is_mesh_stretching = is_mesh_stretching

        self.cr_stretched, self.c4_stretched, self.betar_stretched, \
        self.dr_stretched, self.ci_stretched, self.cicj_stretched, \
        self.ci_beta_stretched, self.cicj_beta_stretched \
        = compute_coefficients_stretched_mesh_teno6(
            is_mesh_stretching, cell_sizes,
            self.s_mesh, self.s_nh_xi)

    def reconstruct_xi(
            self, 
            buffer: Array, 
            axis: int, 
            j: int, 
            dx: float = None, 
            **kwargs) -> Array:

        s_ = self.s_[j][axis]
        is_mesh_stretching = self.is_mesh_stretching[axis]

        if is_mesh_stretching:
            cr = self.cr_stretched[j][axis]
            c4 = self.c4_stretched[j][axis]
            betar = self.betar_stretched[j][axis]
            dr = self.dr_stretched[j][axis]
            ci = self.ci_stretched[j][axis]
            cicj = self.cicj_stretched[j][axis]
            ci_beta = self.ci_beta_stretched[j][axis]
            cicj_beta = self.cicj_beta_stretched[j][axis]

            # NOTE Slice arrays for mesh-stretching + parallel
            if dr.ndim == 5:
                device_id = jax.lax.axis_index(axis_name="i")
                cr = cr[device_id]
                c4 = c4[device_id]
                betar = betar[device_id]
                dr = dr[device_id]
                ci = ci[device_id]
                cicj = cicj[device_id]
                ci_beta = ci_beta[device_id]
                cicj_beta = cicj_beta[device_id]
        
        else:
            cr = self.cr_uniform[:3]
            c4 = self.cr_uniform[3]
            dr = self.dr_uniform

        if is_mesh_stretching:
            # TODO
            beta_0 = buffer[s_[0]] * (betar[0][0] * buffer[s_[0]] + betar[0][1] * buffer[s_[1]] + betar[0][2] * buffer[s_[2]]) \
                   + buffer[s_[1]] * (betar[0][3] * buffer[s_[1]] + betar[0][4] * buffer[s_[2]]) \
                   + buffer[s_[2]] * (betar[0][5] * buffer[s_[2]])

            beta_1 = buffer[s_[1]] * (betar[1][0] * buffer[s_[1]] + betar[1][1] * buffer[s_[2]] + betar[1][2] * buffer[s_[3]]) \
                   + buffer[s_[2]] * (betar[1][3] * buffer[s_[2]] + betar[1][4] * buffer[s_[3]]) \
                   + buffer[s_[3]] * (betar[1][5] * buffer[s_[3]])

            beta_2 = buffer[s_[2]] * (betar[2][0] * buffer[s_[2]] + betar[2][1] * buffer[s_[3]] + betar[2][2] * buffer[s_[4]]) \
                   + buffer[s_[3]] * (betar[2][3] * buffer[s_[3]] + betar[2][4] * buffer[s_[4]]) \
                   + buffer[s_[4]] * (betar[2][5] * buffer[s_[4]])

            C0 = ci_beta[0]  * buffer[s_[2]]
            C1 = ci_beta[1]  * buffer[s_[2]] + ci_beta[2]  * buffer[s_[3]]
            C2 = ci_beta[3]  * buffer[s_[2]] + ci_beta[4]  * buffer[s_[3]] + ci_beta[5]  * buffer[s_[4]]
            C3 = ci_beta[6]  * buffer[s_[2]] + ci_beta[7]  * buffer[s_[3]] + ci_beta[8]  * buffer[s_[4]] + ci_beta[9]  * buffer[s_[5]]

            beta_3 = C0 * (cicj_beta[0] * C0 + cicj_beta[1] * C1 + cicj_beta[2] * C2 + cicj_beta[3] * C3) \
                   + C1 * (cicj_beta[4] * C1 + cicj_beta[5] * C2 + cicj_beta[6] * C3) \
                   + C2 * (cicj_beta[7] * C2 + cicj_beta[8] * C3) \
                   + C3 * (cicj_beta[9] * C3)

            C0 = ci[0]  * buffer[s_[0]]
            C1 = ci[1]  * buffer[s_[0]] + ci[2]  * buffer[s_[1]]
            C2 = ci[3]  * buffer[s_[0]] + ci[4]  * buffer[s_[1]] + ci[5]  * buffer[s_[2]]
            C3 = ci[6]  * buffer[s_[0]] + ci[7]  * buffer[s_[1]] + ci[8]  * buffer[s_[2]] + ci[9]  * buffer[s_[3]]
            C4 = ci[10] * buffer[s_[0]] + ci[11] * buffer[s_[1]] + ci[12] * buffer[s_[2]] + ci[13] * buffer[s_[3]] + ci[14] * buffer[s_[4]]
            C5 = ci[15] * buffer[s_[0]] + ci[16] * buffer[s_[1]] + ci[17] * buffer[s_[2]] + ci[18] * buffer[s_[3]] + ci[19] * buffer[s_[4]] + ci[20] * buffer[s_[5]]

            beta_6 = cicj[0] * C0 * C0 \
                + C0 * (cicj[1]  * C1 + cicj[2]  * C2 + cicj[3]  * C3 + cicj[4]  * C4 + cicj[5]  * C5) \
                + C1 * (cicj[6]  * C1 + cicj[7]  * C2 + cicj[8]  * C3 + cicj[9]  * C4 + cicj[10] * C5) \
                + C2 * (cicj[11] * C2 + cicj[12] * C3 + cicj[13] * C4 + cicj[14] * C5) \
                + C3 * (cicj[15] * C3 + cicj[16] * C4 + cicj[17] * C5) \
                + C4 * (cicj[18] * C4 + cicj[19] * C5) \
                + cicj[20] * C5 * C5

        else:
            beta_0 = 13.0 / 12.0 * jnp.square(buffer[s_[0]] - 2 * buffer[s_[1]] + buffer[s_[2]]) \
                + 1.0 / 4.0 * jnp.square(buffer[s_[0]] - 4 * buffer[s_[1]] + 3 * buffer[s_[2]])
            beta_1 = 13.0 / 12.0 * jnp.square(buffer[s_[1]] - 2 * buffer[s_[2]] + buffer[s_[3]]) \
                + 1.0 / 4.0 * jnp.square(buffer[s_[1]] - buffer[s_[3]])
            beta_2 = 13.0 / 12.0 * jnp.square(buffer[s_[2]] - 2 * buffer[s_[3]] + buffer[s_[4]]) \
                + 1.0 / 4.0 * jnp.square(3 * buffer[s_[2]] - 4 * buffer[s_[3]] + buffer[s_[4]])
            beta_3 = 1.0 / 240.0 * (
                buffer[s_[2]]   * (2107  * buffer[s_[2]] - 9402  * buffer[s_[3]] + 7042 * buffer[s_[4]] - 1854 * buffer[s_[5]]) \
                + buffer[s_[3]] * (11003 * buffer[s_[3]] - 17246 * buffer[s_[4]] + 4642 * buffer[s_[5]]) \
                + buffer[s_[4]] * (7043  * buffer[s_[4]] - 3882  * buffer[s_[5]]) \
                + 547 * buffer[s_[5]] * buffer[s_[5]]
            )

            beta_6 = 1.0 / 120960 * (
                271779 * buffer[s_[0]] * buffer[s_[0]] + \
                buffer[s_[0]] * (-2380800 * buffer[s_[1]] + 4086352  * buffer[s_[2]]  - 3462252  * buffer[s_[3]] + 1458762 * buffer[s_[4]]  - 245620  * buffer[s_[5]]) + \
                buffer[s_[1]] * (5653317  * buffer[s_[1]] - 20427884 * buffer[s_[2]]  + 17905032 * buffer[s_[3]] - 7727988 * buffer[s_[4]]  + 1325006 * buffer[s_[5]]) + \
                buffer[s_[2]] * (19510972 * buffer[s_[2]] - 35817664 * buffer[s_[3]]  + 15929912 * buffer[s_[4]] - 2792660 * buffer[s_[5]]) + \
                buffer[s_[3]] * (17195652 * buffer[s_[3]] - 15880404 * buffer[s_[4]]  + 2863984  * buffer[s_[5]]) + \
                buffer[s_[4]] * (3824847  * buffer[s_[4]] - 1429976  * buffer[s_[5]]) + \
                139633 * buffer[s_[5]] * buffer[s_[5]]
                )

        tau_6 = jnp.abs(beta_6 - 1/6 * (beta_0 + 4 * beta_1 + beta_2))

        # SMOOTHNESS MEASURE
        gamma_0 = (self.C + tau_6 / (beta_0 + self.eps))**self.q
        gamma_1 = (self.C + tau_6 / (beta_1 + self.eps))**self.q
        gamma_2 = (self.C + tau_6 / (beta_2 + self.eps))**self.q
        gamma_3 = (self.C + tau_6 / (beta_3 + self.eps))**self.q

        one_gamma_sum = 1.0 / (gamma_0 + gamma_1 + gamma_2 + gamma_3)

        # SHARP CUTOFF FUNCTION
        w0 = dr[0] * jnp.where(gamma_0 * one_gamma_sum < self.CT, 0, 1)
        w1 = dr[1] * jnp.where(gamma_1 * one_gamma_sum < self.CT, 0, 1)
        w2 = dr[2] * jnp.where(gamma_2 * one_gamma_sum < self.CT, 0, 1)
        w3 = dr[3] * jnp.where(gamma_3 * one_gamma_sum < self.CT, 0, 1)

        # TODO eps should not be necessary
        one_dk = 1.0 / (w0 + w1 + w2 + w3 + self.eps)

        omega_0 = w0 * one_dk 
        omega_1 = w1 * one_dk 
        omega_2 = w2 * one_dk 
        omega_3 = w3 * one_dk 

        p_0 = cr[0][0] * buffer[s_[0]] + cr[0][1] * buffer[s_[1]] + cr[0][2] * buffer[s_[2]]
        p_1 = cr[1][0] * buffer[s_[1]] + cr[1][1] * buffer[s_[2]] + cr[1][2] * buffer[s_[3]]
        p_2 = cr[2][0] * buffer[s_[2]] + cr[2][1] * buffer[s_[3]] + cr[2][2] * buffer[s_[4]]
        p_3 = c4[0]    * buffer[s_[2]] + c4[1]    * buffer[s_[3]] + c4[2]    * buffer[s_[4]] + c4[3] * buffer[s_[5]]

        cell_state_xi_j = omega_0 * p_0 + omega_1 * p_1 + omega_2 * p_2 + omega_3 * p_3

        return cell_state_xi_j