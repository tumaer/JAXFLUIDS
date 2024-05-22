from typing import List

import jax
import jax.numpy as jnp
from jax import Array

from jaxfluids.stencils.spatial_reconstruction import SpatialReconstruction
from jaxfluids.stencils.helper_functions import compute_coefficients_stretched_mesh_weno6

class WENO6CUADAP(SpatialReconstruction):
    ''' Hu et al. - 2010 - An adaptive central-upwind WENO scheme '''    

    is_for_adaptive_mesh = True

    def __init__(
            self, 
            nh: int, 
            inactive_axes: List,
            is_mesh_stretching: List = None,
            cell_sizes: List = None,
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

        self.betar_uniform = [
            [4/3, -19/3, 11/3, 25/3, -31/3, 10/3],
            [4/3, -13/3, 5/3, 13/3, -13/3, 4/3],
            [10/3, -31/3, 11/3, 25/3, -19/3, 4/3],
            [22/3, -73/3, 29/3, 61/3, -49/3, 10/3],
        ]

        self.C_ = 20

        self._stencil_size = 6
        self.array_slices([range(-3, 3, 1), range(2, -4, -1)])
        self.stencil_slices([range(0, 6, 1), range(5, -1, -1)])
        self.is_mesh_stretching = is_mesh_stretching

        self.cr_, self.betar_, self.dr_, self.ci_, self.cicj_ = compute_coefficients_stretched_mesh_weno6(
            cr_uniform=self.cr_uniform,
            betar_uniform=self.betar_uniform,
            dr_uniform=self.dr_uniform,
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
        s1_ = self.s_[j][axis]
        is_mesh_stretching = self.is_mesh_stretching[axis]

        if self.cr_[j][axis][0][0].ndim == 4:
            device_id = jax.lax.axis_index(axis_name="i")

            cr_ = []
            for m in range(len(self.cr_[j][axis])):
                cr_temp = []
                for n in range(len(self.cr_[j][axis][m])):
                    cr_temp.append(self.cr_[j][axis][m][n][device_id])
                cr_.append(cr_temp)

            betar_ = []
            for m in range(len(self.betar_[j][axis])):
                betar_temp = []
                for n in range(len(self.betar_[j][axis][m])):
                    betar_temp.append(self.betar_[j][axis][m][n][device_id])
                betar_.append(betar_temp)

            dr_ = []
            for m in range(len(self.dr_[j][axis])):
                dr_.append(self.dr_[j][axis][m][device_id])

            if is_mesh_stretching:
                ci_ = []
                cicj_ = []
                for m in range(len(self.ci_[j][axis])):
                    ci_.append(self.ci_[j][axis][m][device_id])
                for m in range(len(self.cicj_[j][axis])):
                    cicj_.append(self.cicj_[j][axis][m][device_id])

        else:
            cr_ = self.cr_[j][axis]
            betar_ = self.betar_[j][axis]
            dr_ = self.dr_[j][axis]
            if is_mesh_stretching:
                ci_ = self.ci_[j][axis]
                cicj_ = self.cicj_[j][axis]
        
        if is_mesh_stretching:
            beta_0 = betar_[0][0] * buffer[s1_[0]] * buffer[s1_[0]] + \
                betar_[0][1] * buffer[s1_[0]] * buffer[s1_[1]] + \
                betar_[0][2] * buffer[s1_[0]] * buffer[s1_[2]] + \
                betar_[0][3] * buffer[s1_[1]] * buffer[s1_[1]] + \
                betar_[0][4] * buffer[s1_[1]] * buffer[s1_[2]] + \
                betar_[0][5] * buffer[s1_[2]] * buffer[s1_[2]]

            beta_1 = betar_[1][0] * buffer[s1_[1]] * buffer[s1_[1]] + \
                betar_[1][1] * buffer[s1_[1]] * buffer[s1_[2]] + \
                betar_[1][2] * buffer[s1_[1]] * buffer[s1_[3]] + \
                betar_[1][3] * buffer[s1_[2]] * buffer[s1_[2]] + \
                betar_[1][4] * buffer[s1_[2]] * buffer[s1_[3]] + \
                betar_[1][5] * buffer[s1_[3]] * buffer[s1_[3]]

            beta_2 = betar_[2][0] * buffer[s1_[2]] * buffer[s1_[2]] + \
                betar_[2][1] * buffer[s1_[2]] * buffer[s1_[3]] + \
                betar_[2][2] * buffer[s1_[2]] * buffer[s1_[4]] + \
                betar_[2][3] * buffer[s1_[3]] * buffer[s1_[3]] + \
                betar_[2][4] * buffer[s1_[3]] * buffer[s1_[4]] + \
                betar_[2][5] * buffer[s1_[4]] * buffer[s1_[4]]

            C0 = ci_[0]  * buffer[s1_[0]]
            C1 = ci_[1]  * buffer[s1_[0]] + ci_[2]  * buffer[s1_[1]]
            C2 = ci_[3]  * buffer[s1_[0]] + ci_[4]  * buffer[s1_[1]] + ci_[5]  * buffer[s1_[2]]
            C3 = ci_[6]  * buffer[s1_[0]] + ci_[7]  * buffer[s1_[1]] + ci_[8]  * buffer[s1_[2]] + ci_[9]  * buffer[s1_[3]]
            C4 = ci_[10] * buffer[s1_[0]] + ci_[11] * buffer[s1_[1]] + ci_[12] * buffer[s1_[2]] + ci_[13] * buffer[s1_[3]] + ci_[14] * buffer[s1_[4]]
            C5 = ci_[15] * buffer[s1_[0]] + ci_[16] * buffer[s1_[1]] + ci_[17] * buffer[s1_[2]] + ci_[18] * buffer[s1_[3]] + ci_[19] * buffer[s1_[4]] + ci_[20] * buffer[s1_[5]]

            beta_3 = cicj_[0] * C0 * C0 \
                + C0 * (cicj_[1]  * C1 + cicj_[2]  * C2 + cicj_[3]  * C3 + cicj_[4]  * C4 + cicj_[5]  * C5) \
                + C1 * (cicj_[6]  * C1 + cicj_[7]  * C2 + cicj_[8]  * C3 + cicj_[9]  * C4 + cicj_[10] * C5) \
                + C2 * (cicj_[11] * C2 + cicj_[12] * C3 + cicj_[13] * C4 + cicj_[14] * C5) \
                + C3 * (cicj_[15] * C3 + cicj_[16] * C4 + cicj_[17] * C5) \
                + C4 * (cicj_[18] * C4 + cicj_[19] * C5) \
                + cicj_[20] * C5 * C5
        
        else:
            beta_0 = 13.0 / 12.0 * (buffer[s1_[0]] - 2 * buffer[s1_[1]] + buffer[s1_[2]]) \
              * (buffer[s1_[0]] - 2 * buffer[s1_[1]] + buffer[s1_[2]]) \
              + 1.0 / 4.0 * (buffer[s1_[0]] - 4 * buffer[s1_[1]] + 3 * buffer[s1_[2]]) \
              * (buffer[s1_[0]] - 4 * buffer[s1_[1]] + 3 * buffer[s1_[2]])
            
            beta_1 = 13.0 / 12.0 * (buffer[s1_[1]] - 2 * buffer[s1_[2]] + buffer[s1_[3]]) \
              * (buffer[s1_[1]] - 2 * buffer[s1_[2]] + buffer[s1_[3]]) \
              + 1.0 / 4.0 * (buffer[s1_[1]] - buffer[s1_[3]]) * (buffer[s1_[1]] - buffer[s1_[3]])
            
            beta_2 = 13.0 / 12.0 * (buffer[s1_[2]] - 2 * buffer[s1_[3]] + buffer[s1_[4]]) \
              * (buffer[s1_[2]] - 2 * buffer[s1_[3]] + buffer[s1_[4]]) \
              + 1.0 / 4.0 * (3 * buffer[s1_[2]] - 4 * buffer[s1_[3]] + buffer[s1_[4]]) \
              * (3 * buffer[s1_[2]] - 4 * buffer[s1_[3]] + buffer[ s1_[4]])

            # # Eq. 25 from Hu et al. 
            # beta_3 = 1.0 / 10080 * (
            #     271779 * buffer[s1_[0]] * buffer[s1_[0]] + \
            #     buffer[s1_[0]] * (2380800  * buffer[s1_[1]] + 4086352  * buffer[s1_[2]]  - 3462252  * buffer[s1_[3]] + 1458762 * buffer[s1_[4]]  - 245620  * buffer[s1_[5]]) + \
            #     buffer[s1_[1]] * (5653317  * buffer[s1_[1]] - 20427884 * buffer[s1_[2]]  + 17905032 * buffer[s1_[3]] - 7727988 * buffer[s1_[4]]  + 1325006 * buffer[s1_[5]]) + \
            #     buffer[s1_[2]] * (19510972 * buffer[s1_[2]] - 35817664 * buffer[s1_[3]]  + 15929912 * buffer[s1_[4]] - 2792660 * buffer[s1_[5]]) + \
            #     buffer[s1_[3]] * (17195652 * buffer[s1_[3]] - 15880404 * buffer[s1_[4]]  + 2863984  * buffer[s1_[5]]) + \
            #     buffer[s1_[4]] * (3824847  * buffer[s1_[4]] - 1429976  * buffer[s1_[5]]) + \
            #     139633 * buffer[s1_[5]] * buffer[s1_[5]]
            #     )

            # Corrected version
            beta_3 = 1.0 / 10080 / 12 * (
                271779 * buffer[s1_[0]] * buffer[s1_[0]] + \
                buffer[s1_[0]] * (-2380800 * buffer[s1_[1]] + 4086352  * buffer[s1_[2]]  - 3462252  * buffer[s1_[3]] + 1458762 * buffer[s1_[4]]  - 245620  * buffer[s1_[5]]) + \
                buffer[s1_[1]] * (5653317  * buffer[s1_[1]] - 20427884 * buffer[s1_[2]]  + 17905032 * buffer[s1_[3]] - 7727988 * buffer[s1_[4]]  + 1325006 * buffer[s1_[5]]) + \
                buffer[s1_[2]] * (19510972 * buffer[s1_[2]] - 35817664 * buffer[s1_[3]]  + 15929912 * buffer[s1_[4]] - 2792660 * buffer[s1_[5]]) + \
                buffer[s1_[3]] * (17195652 * buffer[s1_[3]] - 15880404 * buffer[s1_[4]]  + 2863984  * buffer[s1_[5]]) + \
                buffer[s1_[4]] * (3824847  * buffer[s1_[4]] - 1429976  * buffer[s1_[5]]) + \
                139633 * buffer[s1_[5]] * buffer[s1_[5]]
                )

        tau_6 = beta_3 - 1/6 * (beta_0 + beta_2 + 4*beta_1)

        alpha_0 = dr_[0] * (self.C_ + tau_6 / (beta_0 + self.eps))
        alpha_1 = dr_[1] * (self.C_ + tau_6 / (beta_1 + self.eps))
        alpha_2 = dr_[2] * (self.C_ + tau_6 / (beta_2 + self.eps))
        alpha_3 = dr_[3] * (self.C_ + tau_6 / (beta_3 + self.eps))

        one_alpha = 1.0 / (alpha_0 + alpha_1 + alpha_2 + alpha_3)

        omega_0 = alpha_0 * one_alpha 
        omega_1 = alpha_1 * one_alpha 
        omega_2 = alpha_2 * one_alpha 
        omega_3 = alpha_3 * one_alpha 

        p_0 = cr_[0][0] * buffer[s1_[0]] + cr_[0][1] * buffer[s1_[1]] + cr_[0][2] * buffer[s1_[2]]
        p_1 = cr_[1][0] * buffer[s1_[1]] + cr_[1][1] * buffer[s1_[2]] + cr_[1][2] * buffer[s1_[3]]
        p_2 = cr_[2][0] * buffer[s1_[2]] + cr_[2][1] * buffer[s1_[3]] + cr_[2][2] * buffer[s1_[4]]
        p_3 = cr_[3][0] * buffer[s1_[3]] + cr_[3][1] * buffer[s1_[4]] + cr_[3][2] * buffer[s1_[5]]

        cell_state_xi_j = omega_0 * p_0 + omega_1 * p_1 + omega_2 * p_2 + omega_3 * p_3

        if self.debug:
            omega = jnp.stack([omega_0, omega_1, omega_2, omega_3], axis=-1)
            debug_out = {"omega": omega}
            return cell_state_xi_j, debug_out

        return cell_state_xi_j