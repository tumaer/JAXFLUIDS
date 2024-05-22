from functools import partial
from typing import List, Tuple

import jax
import jax.numpy as jnp
from jax import Array

from jaxfluids.stencils.spatial_reconstruction import SpatialReconstruction
from jaxfluids.stencils.helper_functions import compute_coefficients_stretched_mesh_weno5

class ALDM_WENO5(SpatialReconstruction):
    """ALDM_WENO5 

    Implementation details provided in parent class.
    """

    is_for_adaptive_mesh = True

    def __init__(
            self, 
            nh: int, 
            inactive_axes: List,
            is_mesh_stretching: List = None,
            cell_sizes: List = None,
            smoothness_measure: str = "TV",
            **kwargs
        ) -> None:
        super(ALDM_WENO5, self).__init__(nh=nh, inactive_axes=inactive_axes)

        self.smoothness_measure = smoothness_measure
        self.dr_adlm_ = [0.89548, 0.08550, 0.01902]
        dr_uniform = [0.1, 0.6, 0.3]
        cr_uniform = [
            [1/3, -7/6, 11/6], 
            [-1/6, 5/6, 1/3], 
            [1/3, 5/6, -1/6]
        ]
        betar_uniform = [
            [4/3, -19/3, 11/3, 25/3, -31/3, 10/3],
            [4/3, -13/3, 5/3, 13/3, -13/3, 4/3],
            [10/3, -31/3, 11/3, 25/3, -19/3, 4/3],
        ]
        self._stencil_size = 6
        self.array_slices([range(-3, 2, 1), range(2, -3, -1)])

        self.cr_, self.betar_, self.dr_ = compute_coefficients_stretched_mesh_weno5(
            cr_uniform=cr_uniform,
            betar_uniform=betar_uniform,
            dr_uniform=dr_uniform,
            is_mesh_stretching=is_mesh_stretching,
            cell_sizes=cell_sizes,
            slices_mesh=self.s_mesh,
            slices_cell_sizes=self.s_nh_xi)

    def get_adaptive_ideal_weights(
            self,
            dr: List,
            j: int, 
            fs: Array
        ) -> Tuple[Array, Array, Array]:
        """Adapts the ideal reconstruction weights based on the shock sensor 
        function fs. If the shock sensor is INACTIVE, optimized ALDM 
        reconstruction weights are used. If the shock sensor is active,
        standard WENO5 ideal weights are used. 

        :param dr: _description_
        :type dr: List
        :param j: _description_
        :type j: int
        :param fs: _description_
        :type fs: Array
        :return: _description_
        :rtype: Tuple[Array, Array, Array]
        """

        d0 = self.dr_adlm_[0] + fs * (dr[0] - self.dr_adlm_[0])
        d1 = self.dr_adlm_[1] + fs * (dr[1] - self.dr_adlm_[1])
        d2 = self.dr_adlm_[2] + fs * (dr[2] - self.dr_adlm_[2])
        return d0, d1, d2

    def reconstruct_xi(
            self, 
            buffer: Array, 
            axis: int, 
            j: int, 
            dx: float = None, 
            fs: Array = 0
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
        
        if self.smoothness_measure == "TV":
            # Total variation smoothness measure
            beta_0 = (buffer[s1_[1]] - buffer[s1_[0]]) * (buffer[s1_[1]] - buffer[s1_[0]]) \
                +    (buffer[s1_[2]] - buffer[s1_[1]]) * (buffer[s1_[2]] - buffer[s1_[1]])
            beta_1 = (buffer[s1_[2]] - buffer[s1_[1]]) * (buffer[s1_[2]] - buffer[s1_[1]]) \
                +    (buffer[s1_[3]] - buffer[s1_[2]]) * (buffer[s1_[3]] - buffer[s1_[2]])
            beta_2 = (buffer[s1_[3]] - buffer[s1_[2]]) * (buffer[s1_[3]] - buffer[s1_[2]]) \
                +    (buffer[s1_[4]] - buffer[s1_[3]]) * (buffer[s1_[4]] - buffer[s1_[3]])

        if self.smoothness_measure == "WENO":
            # WENO smoothness measure
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

        one_beta_0_sq = 1.0 / (self.eps + beta_0 * beta_0) 
        one_beta_1_sq = 1.0 / (self.eps + beta_1 * beta_1) 
        one_beta_2_sq = 1.0 / (self.eps + beta_2 * beta_2) 

        d0, d1, d2 = self.get_adaptive_ideal_weights(dr_, j, fs)

        alpha_0 = d0 * one_beta_0_sq
        alpha_1 = d1 * one_beta_1_sq
        alpha_2 = d2 * one_beta_2_sq

        one_alpha = 1.0 / (alpha_0 + alpha_1 + alpha_2)

        omega_0 = alpha_0 * one_alpha 
        omega_1 = alpha_1 * one_alpha 
        omega_2 = alpha_2 * one_alpha 

        p_0 = cr_[0][0] * buffer[s1_[0]] + cr_[0][1] * buffer[s1_[1]] + cr_[0][2] * buffer[s1_[2]]
        p_1 = cr_[1][0] * buffer[s1_[1]] + cr_[1][1] * buffer[s1_[2]] + cr_[1][2] * buffer[s1_[3]]
        p_2 = cr_[2][0] * buffer[s1_[2]] + cr_[2][1] * buffer[s1_[3]] + cr_[2][2] * buffer[s1_[4]]

        cell_state_xi_j = omega_0 * p_0 + omega_1 * p_1 + omega_2 * p_2

        return cell_state_xi_j