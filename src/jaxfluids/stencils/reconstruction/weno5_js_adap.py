from typing import List

import jax
import jax.numpy as jnp
from jax import Array

from jaxfluids.stencils.spatial_reconstruction import SpatialReconstruction
from jaxfluids.stencils.helper_functions import compute_coefficients_stretched_mesh_weno5

class WENO5ADAP(SpatialReconstruction):

    required_halos = 3
    is_for_adaptive_mesh = True

    def __init__(self,
            nh: int,
            inactive_axes: List,
            is_mesh_stretching: List = None,
            cell_sizes: List = None,
            offset: int = 0,
            **kwargs
        ) -> None:
        super(WENO5ADAP, self).__init__(nh=nh, inactive_axes=inactive_axes, offset=offset)

        self.dr_uniform = [1/10, 6/10, 3/10]

        self.cr_uniform = [
            [1/3, -7/6, 11/6],
            [-1/6, 5/6, 1/3],
            [1/3, 5/6, -1/6],
        ]

        self.betar_uniform = [
            [4/3, -19/3, 11/3, 25/3, -31/3, 10/3],
            [4/3, -13/3, 5/3, 13/3, -13/3, 4/3],
            [10/3, -31/3, 11/3, 25/3, -19/3, 4/3],
        ]

        self._stencil_size = 6
        self.array_slices([range(-3, 2, 1), range(2, -3, -1)])
        self.stencil_slices([range(0, 5, 1), range(5, 0, -1)])

        self.cr_, self.betar_, self.dr_ = compute_coefficients_stretched_mesh_weno5(
            cr_uniform=self.cr_uniform,
            betar_uniform=self.betar_uniform,
            dr_uniform=self.dr_uniform,
            is_mesh_stretching=is_mesh_stretching,
            cell_sizes=cell_sizes,
            slices_mesh=self.s_mesh,
            slices_cell_sizes=self.s_nh_xi)

        # DEBUG FEATURE
        self.debug = False

    def reconstruct_xi(self,
        buffer: Array,
        axis: int,
        j: int,
        dx: float = None,
        **kwargs
        ) -> Array:

        s1_ = self.s_[j][axis]

        if self.cr_[j][axis][0][0].ndim == 4:
            cr_ = [[], [], []]
            betar_ = [[], [], []]
            dr_ = []
            device_id = jax.lax.axis_index(axis_name="i")
            for m in range(3):
                for n in range(3):
                    cr_[n].append(self.cr_[j][axis][n][m][device_id])
            for m in range(6):
                for n in range(3):
                   betar_[n].append(self.betar_[j][axis][n][m][device_id])
            for m in range(3):
                dr_.append(self.dr_[j][axis][m][device_id])
        else:
            cr_ = self.cr_[j][axis]
            betar_ = self.betar_[j][axis]
            dr_ = self.dr_[j][axis]

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

        one_beta_0_sq = 1.0 / (beta_0 * beta_0 + self.eps)
        one_beta_1_sq = 1.0 / (beta_1 * beta_1 + self.eps)
        one_beta_2_sq = 1.0 / (beta_2 * beta_2 + self.eps)

        alpha_0 = dr_[0] * one_beta_0_sq
        alpha_1 = dr_[1] * one_beta_1_sq
        alpha_2 = dr_[2] * one_beta_2_sq

        one_alpha = 1.0 / (alpha_0 + alpha_1 + alpha_2)

        omega_0 = alpha_0 * one_alpha
        omega_1 = alpha_1 * one_alpha
        omega_2 = alpha_2 * one_alpha

        p_0 = cr_[0][0] * buffer[s1_[0]] + cr_[0][1] * buffer[s1_[1]] + cr_[0][2] * buffer[s1_[2]]
        p_1 = cr_[1][0] * buffer[s1_[1]] + cr_[1][1] * buffer[s1_[2]] + cr_[1][2] * buffer[s1_[3]]
        p_2 = cr_[2][0] * buffer[s1_[2]] + cr_[2][1] * buffer[s1_[3]] + cr_[2][2] * buffer[s1_[4]]

        cell_state_xi_j = omega_0 * p_0 + omega_1 * p_1 + omega_2 * p_2

        if self.debug:
            return cell_state_xi_j, (omega_0, omega_1, omega_2)

        return cell_state_xi_j