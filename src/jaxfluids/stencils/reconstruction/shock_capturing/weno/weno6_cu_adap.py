from typing import List, Tuple

import jax
import jax.numpy as jnp

from jaxfluids.stencils.spatial_reconstruction import SpatialReconstruction
from jaxfluids.stencils.helper_functions import compute_coefficients_stretched_mesh_weno6

Array = jax.Array

class WENO6CUADAP(SpatialReconstruction):
    ''' Hu et al. - 2010 - An adaptive central-upwind WENO scheme '''    

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
        super(WENO6CUADAP, self).__init__(nh=nh, inactive_axes=inactive_axes, offset=offset)

        self.dr_uniform = [1/20, 9/20, 9/20, 1/20]
        self.cr_uniform = [
            [1/3, -7/6, 11/6], 
            [-1/6, 5/6, 1/3], 
            [1/3, 5/6, -1/6], 
            [11/6, -7/6, 1/3]
        ]
        self.C_ = 20

        self._stencil_size = 6
        self.array_slices([range(-3, 3, 1), range(2, -4, -1)])
        self.stencil_slices([range(0, 6, 1), range(5, -1, -1)])
        self.is_mesh_stretching = is_mesh_stretching

        self.cr_stretched, self.betar_stretched, self.dr_stretched, \
        self.ci_stretched, self.cicj_stretched = \
        compute_coefficients_stretched_mesh_weno6(
            is_mesh_stretching=is_mesh_stretching,
            cell_sizes=cell_sizes,
            slices_mesh=self.s_mesh,
            slices_cell_sizes=self.s_nh_xi)

        self.debug = False

    def reconstruct_xi(
            self,
            buffer: Array,
            axis: int,
            j: int,
            dx: float = None,
            **kwargs
            ) -> Array:

        s_ = self.s_[j][axis]
        is_mesh_stretching = self.is_mesh_stretching[axis]

        if is_mesh_stretching:
            cr = self.cr_stretched[j][axis]
            betar = self.betar_stretched[j][axis]
            dr = self.dr_stretched[j][axis]
            ci = self.ci_stretched[j][axis]
            cicj = self.cicj_stretched[j][axis]

            # NOTE Slice arrays for mesh-stretching + parallel
            if dr.ndim == 5:
                device_id = jax.lax.axis_index(axis_name="i")
                cr = cr[device_id]
                betar = betar[device_id]
                dr = dr[device_id]
                ci = ci[device_id]
                cicj = cicj[device_id]
        
        else:
            cr = self.cr_uniform
            dr = self.dr_uniform
        
        if is_mesh_stretching:
            beta_0 = buffer[s_[0]] * (betar[0][0] * buffer[s_[0]] + betar[0][1] * buffer[s_[1]] + betar[0][2] * buffer[s_[2]]) \
                   + buffer[s_[1]] * (betar[0][3] * buffer[s_[1]] + betar[0][4] * buffer[s_[2]]) \
                   + buffer[s_[2]] * (betar[0][5] * buffer[s_[2]])

            beta_1 = buffer[s_[1]] * (betar[1][0] * buffer[s_[1]] + betar[1][1] * buffer[s_[2]] + betar[1][2] * buffer[s_[3]]) \
                   + buffer[s_[2]] * (betar[1][3] * buffer[s_[2]] + betar[1][4] * buffer[s_[3]]) \
                   + buffer[s_[3]] * (betar[1][5] * buffer[s_[3]])

            beta_2 = buffer[s_[2]] * (betar[2][0] * buffer[s_[2]] + betar[2][1] * buffer[s_[3]] + betar[2][2] * buffer[s_[4]]) \
                   + buffer[s_[3]] * (betar[2][3] * buffer[s_[3]] + betar[2][4] * buffer[s_[4]]) \
                   + buffer[s_[4]] * (betar[2][5] * buffer[s_[4]])

            C0 = ci[0]  * buffer[s_[0]]
            C1 = ci[1]  * buffer[s_[0]] + ci[2]  * buffer[s_[1]]
            C2 = ci[3]  * buffer[s_[0]] + ci[4]  * buffer[s_[1]] + ci[5]  * buffer[s_[2]]
            C3 = ci[6]  * buffer[s_[0]] + ci[7]  * buffer[s_[1]] + ci[8]  * buffer[s_[2]] + ci[9]  * buffer[s_[3]]
            C4 = ci[10] * buffer[s_[0]] + ci[11] * buffer[s_[1]] + ci[12] * buffer[s_[2]] + ci[13] * buffer[s_[3]] + ci[14] * buffer[s_[4]]
            C5 = ci[15] * buffer[s_[0]] + ci[16] * buffer[s_[1]] + ci[17] * buffer[s_[2]] + ci[18] * buffer[s_[3]] + ci[19] * buffer[s_[4]] + ci[20] * buffer[s_[5]]

            beta_3 = cicj[0] * C0 * C0 \
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

            # # Eq. 25 from Hu et al. 
            # beta_3 = 1.0 / 10080 * (
            #     271779 * buffer[s_[0]] * buffer[s_[0]] + \
            #     buffer[s_[0]] * (2380800  * buffer[s_[1]] + 4086352  * buffer[s_[2]]  - 3462252  * buffer[s_[3]] + 1458762 * buffer[s_[4]]  - 245620  * buffer[s_[5]]) + \
            #     buffer[s_[1]] * (5653317  * buffer[s_[1]] - 20427884 * buffer[s_[2]]  + 17905032 * buffer[s_[3]] - 7727988 * buffer[s_[4]]  + 1325006 * buffer[s_[5]]) + \
            #     buffer[s_[2]] * (19510972 * buffer[s_[2]] - 35817664 * buffer[s_[3]]  + 15929912 * buffer[s_[4]] - 2792660 * buffer[s_[5]]) + \
            #     buffer[s_[3]] * (17195652 * buffer[s_[3]] - 15880404 * buffer[s_[4]]  + 2863984  * buffer[s_[5]]) + \
            #     buffer[s_[4]] * (3824847  * buffer[s_[4]] - 1429976  * buffer[s_[5]]) + \
            #     139633 * buffer[s_[5]] * buffer[s_[5]]
            #     )

            # Corrected version
            beta_3 = 1.0 / 120960 * (
                271779 * buffer[s_[0]] * buffer[s_[0]] + \
                buffer[s_[0]] * (-2380800 * buffer[s_[1]] + 4086352  * buffer[s_[2]]  - 3462252  * buffer[s_[3]] + 1458762 * buffer[s_[4]]  - 245620  * buffer[s_[5]]) + \
                buffer[s_[1]] * (5653317  * buffer[s_[1]] - 20427884 * buffer[s_[2]]  + 17905032 * buffer[s_[3]] - 7727988 * buffer[s_[4]]  + 1325006 * buffer[s_[5]]) + \
                buffer[s_[2]] * (19510972 * buffer[s_[2]] - 35817664 * buffer[s_[3]]  + 15929912 * buffer[s_[4]] - 2792660 * buffer[s_[5]]) + \
                buffer[s_[3]] * (17195652 * buffer[s_[3]] - 15880404 * buffer[s_[4]]  + 2863984  * buffer[s_[5]]) + \
                buffer[s_[4]] * (3824847  * buffer[s_[4]] - 1429976  * buffer[s_[5]]) + \
                139633 * buffer[s_[5]] * buffer[s_[5]]
                )

        tau_6 = beta_3 - 1/6 * (beta_0 + beta_2 + 4*beta_1)

        alpha_0 = dr[0] * (self.C_ + tau_6 / (beta_0 + self.eps))
        alpha_1 = dr[1] * (self.C_ + tau_6 / (beta_1 + self.eps))
        alpha_2 = dr[2] * (self.C_ + tau_6 / (beta_2 + self.eps))
        alpha_3 = dr[3] * (self.C_ + tau_6 / (beta_3 + self.eps))

        one_alpha = 1.0 / (alpha_0 + alpha_1 + alpha_2 + alpha_3)

        omega_0 = alpha_0 * one_alpha 
        omega_1 = alpha_1 * one_alpha 
        omega_2 = alpha_2 * one_alpha 
        omega_3 = alpha_3 * one_alpha 

        p_0 = cr[0][0] * buffer[s_[0]] + cr[0][1] * buffer[s_[1]] + cr[0][2] * buffer[s_[2]]
        p_1 = cr[1][0] * buffer[s_[1]] + cr[1][1] * buffer[s_[2]] + cr[1][2] * buffer[s_[3]]
        p_2 = cr[2][0] * buffer[s_[2]] + cr[2][1] * buffer[s_[3]] + cr[2][2] * buffer[s_[4]]
        p_3 = cr[3][0] * buffer[s_[3]] + cr[3][1] * buffer[s_[4]] + cr[3][2] * buffer[s_[5]]

        cell_state_xi_j = omega_0 * p_0 + omega_1 * p_1 + omega_2 * p_2 + omega_3 * p_3

        if self.debug:
            omega = jnp.stack([omega_0, omega_1, omega_2, omega_3], axis=-1)
            debug_out = {"omega": omega}
            return cell_state_xi_j, debug_out

        return cell_state_xi_j