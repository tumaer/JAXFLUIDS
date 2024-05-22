from typing import List

import jax.numpy as jnp
from jax import Array

from jaxfluids.stencils.spatial_reconstruction import SpatialReconstruction

class TENO8(SpatialReconstruction):
    ''' Fu et al. - 2016 -  A family of high-order targeted ENO schemes for compressible-fluid simulations
    Fu et al. - 2019 - A Targeted ENO Scheme as Implicit Model for Turbulent and Genuine Subgrid Scales
    '''    
    
    def __init__(self, 
            nh: int, 
            inactive_axes: List,
            offset: int = 0,
            **kwargs) -> None:
        super(TENO8, self).__init__(nh=nh, inactive_axes=inactive_axes, offset=offset)

        # Coefficients for optimized spectral properties
        self.dr_ = [
            0.06459052081827904, 0.4130855804023061, 0.2193140179474727,
            0.02746675592227204, 0.1439236310125986, 0.1316194938970745]

        self.cr_ = [
            [2/6, -7/6, 11/6], 
            [-1/6, 5/6, 2/6], 
            [2/6, 5/6, -1/6], 
            [-3/12, 13/12, -23/12, 25/12],
            [3/12, 13/12, -5/12, 1/12],
            [12/60, 77/60, -43/60, 17/60, -3/60]
        ]

        self.C = 1.0
        self.q = 6
        self.CT = 1e-7

        self._stencil_size = 8
        self.array_slices([range(-4, 4, 1), range(3, -5, -1)])
        self.stencil_slices([range(0, 8, 1), range(7, -1, -1)])

    def reconstruct_xi(self, 
            buffer: Array, 
            axis: int, 
            j: int, 
            dx: float = None, 
            **kwargs) -> Array:
        s1_ = self.s_[j][axis]

        # (u_{i-2}, u_{i-1}, u_{i})
        beta_0 = 13.0 / 12.0 * (buffer[s1_[1]] - 2 * buffer[s1_[2]] + buffer[s1_[3]]) \
            * (buffer[s1_[1]] - 2 * buffer[s1_[2]] + buffer[s1_[3]]) \
            + 1.0 / 4.0 * (buffer[s1_[1]] - 4 * buffer[s1_[2]] + 3 * buffer[s1_[3]]) \
            * (buffer[s1_[1]] - 4 * buffer[s1_[2]] + 3 * buffer[s1_[3]])
        # (u_{i-1}, u_{i}, u_{i+1})
        beta_1 = 13.0 / 12.0 * (buffer[s1_[2]] - 2 * buffer[s1_[3]] + buffer[s1_[4]]) \
            * (buffer[s1_[2]] - 2 * buffer[s1_[3]] + buffer[s1_[4]]) \
            + 1.0 / 4.0 * (buffer[s1_[2]] - buffer[s1_[4]]) * (buffer[s1_[2]] - buffer[s1_[4]])
        # (u_{i}, u_{i+1}, u_{i+2})
        beta_2 = 13.0 / 12.0 * (buffer[s1_[3]] - 2 * buffer[s1_[4]] + buffer[s1_[5]]) \
            * (buffer[s1_[3]] - 2 * buffer[s1_[4]] + buffer[s1_[5]]) \
            + 1.0 / 4.0 * (3 * buffer[s1_[3]] - 4 * buffer[s1_[4]] + buffer[s1_[5]]) \
            * (3 * buffer[s1_[3]] - 4 * buffer[s1_[4]] + buffer[s1_[5]])
        # (u_{i-3}, u_{i-2}, u_{i-1}, u_{i})
        beta_3 = 1.0 / 240.0 * (
            buffer[s1_[0]]   * (547.0   * buffer[s1_[0]] - 3882.0  * buffer[s1_[1]] + 4642.0 * buffer[s1_[2]] - 1854.0 * buffer[s1_[3]]) \
            + buffer[s1_[1]] * (7043.0  * buffer[s1_[1]] - 17246.0 * buffer[s1_[2]] + 7042.0 * buffer[s1_[3]]) \
            + buffer[s1_[2]] * (11003.0 * buffer[s1_[2]] - 9402.0   * buffer[s1_[3]]) \
            + 2107.0 * buffer[s1_[3]] * buffer[s1_[3]])
        # (u_{i}, u_{i+1}, u_{i+2}, u_{i+3})
        beta_4 = 1.0 / 240.0 * (
            buffer[s1_[3]]   * (2107.0  * buffer[s1_[3]] - 9402.0  * buffer[s1_[4]] + 7042.0 * buffer[s1_[5]] - 1854.0 * buffer[s1_[6]]) \
            + buffer[s1_[4]] * (11003.0 * buffer[s1_[4]] - 17246.0 * buffer[s1_[5]] + 4642.0 * buffer[s1_[6]]) \
            + buffer[s1_[5]] * (7043.0  * buffer[s1_[5]] - 3882.0  * buffer[s1_[6]]) \
            + 547.0 * buffer[s1_[6]] * buffer[s1_[6]])
        # (u_{i}, u_{i+1}, u_{i+2}, u_{i+3})
        beta_5 = 1.0 / 5040.0 * (
            buffer[s1_[3]] * (107918.0  * buffer[s1_[3]] - 649501.0  * buffer[s1_[4]] + 758823.0  * buffer[s1_[5]] \
                            - 411487.0  * buffer[s1_[6]] + 86329.0 * buffer[s1_[7]]) + \

            buffer[s1_[4]] * (1020563.0 * buffer[s1_[4]] - 2462076.0 * buffer[s1_[5]] + 1358458.0 * buffer[s1_[6]] \
                            - 288007.0  * buffer[s1_[7]]) + \

            buffer[s1_[5]] * (1521393.0 * buffer[s1_[5]] - 1704396.0 * buffer[s1_[6]] + 364863.0  * buffer[s1_[7]]) + \

            buffer[s1_[6]] * (482963.0  * buffer[s1_[6]] - 208501.0  * buffer[s1_[7]]) + \
                
            22658.0 * buffer[s1_[7]] * buffer[s1_[7]])

        # (u_{i-3}, u_{i-2}, u_{i-1}, u_{i}, u_{i+1}, u_{i+2}, u_{i+3})
        tau_8 = 1.0 / 62270208000.0 * jnp.abs(
            buffer[s1_[7]] * (75349098471.0     * buffer[s1_[7]] - 1078504915264.0  * buffer[s1_[6]]  + 3263178215782.0   * buffer[s1_[5]] \
                            - 5401061230160.0   * buffer[s1_[4]] + 5274436892970.0  * buffer[s1_[3]]  - 3038037798592.0   * buffer[s1_[2]] \
                            + 956371298594.0    * buffer[s1_[1]] - 127080660272.0   * buffer[s1_[0]]) + \

            buffer[s1_[6]] * (3944861897609.0   * buffer[s1_[6]] - 24347015748304.0 * buffer[s1_[5]]  + 41008808432890.0  * buffer[s1_[4]] \
                            - 40666174667520.0  * buffer[s1_[3]] + 23740865961334.0 * buffer[s1_[2]]  - 7563868580208.0   * buffer[s1_[1]] \
                            + 1016165721854.0   * buffer[s1_[0]]) + \

            buffer[s1_[5]] * (38329064547231.0  * buffer[s1_[5]] - 131672853704480.0 * buffer[s1_[4]] + 132979856899250.0 * buffer[s1_[3]] \
                            - 78915800051952.0  * buffer[s1_[2]] + 25505661974314.0  * buffer[s1_[1]] - 3471156679072.0   * buffer[s1_[0]]) + \

            buffer[s1_[4]] * (115451981835025.0 * buffer[s1_[4]] - 238079153652400.0 * buffer[s1_[3]] + 144094750348910.0 * buffer[s1_[2]] \
                            - 47407534412640.0  * buffer[s1_[1]] + 6553080547830.0   * buffer[s1_[0]]) + \

            buffer[s1_[3]] * (125494539510175.0 * buffer[s1_[3]] - 155373333547520.0 * buffer[s1_[2]] + 52241614797670.0  * buffer[s1_[1]] \
                            - 7366325742800.0   * buffer[s1_[0]]) + \

            buffer[s1_[2]] * (49287325751121.0  * buffer[s1_[2]] - 33999931981264.0  * buffer[s1_[1]] + 4916835566842.0   * buffer[s1_[0]]) + \

            buffer[s1_[1]] * (6033767706599.0   * buffer[s1_[1]] - 1799848509664.0   * buffer[s1_[0]]) + \

            139164877641.0 * buffer[s1_[0]] * buffer[s1_[0]])

        # SMOOTHNESS MEASURE
        gamma_0 = (self.C + tau_8 / (beta_0 + self.eps))**self.q
        gamma_1 = (self.C + tau_8 / (beta_1 + self.eps))**self.q
        gamma_2 = (self.C + tau_8 / (beta_2 + self.eps))**self.q
        gamma_3 = (self.C + tau_8 / (beta_3 + self.eps))**self.q
        gamma_4 = (self.C + tau_8 / (beta_4 + self.eps))**self.q
        gamma_5 = (self.C + tau_8 / (beta_5 + self.eps))**self.q

        one_gamma_sum = 1.0 / (gamma_0 + gamma_1 + gamma_2 + gamma_3 + gamma_4 + gamma_5)

        # SHARP CUTOFF FUNCTION
        delta_0 = jnp.where(gamma_0 * one_gamma_sum < self.CT, 0, 1)
        delta_1 = jnp.where(gamma_1 * one_gamma_sum < self.CT, 0, 1)
        delta_2 = jnp.where(gamma_2 * one_gamma_sum < self.CT, 0, 1)
        delta_3 = jnp.where(gamma_3 * one_gamma_sum < self.CT, 0, 1)
        delta_4 = jnp.where(gamma_4 * one_gamma_sum < self.CT, 0, 1)
        delta_5 = jnp.where(gamma_5 * one_gamma_sum < self.CT, 0, 1)

        w0 = delta_0 * self.dr_[0]
        w1 = delta_1 * self.dr_[1]
        w2 = delta_2 * self.dr_[2]
        w3 = delta_3 * self.dr_[3]
        w4 = delta_4 * self.dr_[4]
        w5 = delta_5 * self.dr_[5]

        one_dk = 1.0 / (w0 + w1 + w2 + w3 + w4 + w5 + self.eps)

        omega_0 = w0 * one_dk 
        omega_1 = w1 * one_dk 
        omega_2 = w2 * one_dk 
        omega_3 = w3 * one_dk 
        omega_4 = w4 * one_dk 
        omega_5 = w5 * one_dk 

        p_0 = self.cr_[0][0] * buffer[s1_[1]] + self.cr_[0][1] * buffer[s1_[2]] + self.cr_[0][2] * buffer[s1_[3]]
        p_1 = self.cr_[1][0] * buffer[s1_[2]] + self.cr_[1][1] * buffer[s1_[3]] + self.cr_[1][2] * buffer[s1_[4]]
        p_2 = self.cr_[2][0] * buffer[s1_[3]] + self.cr_[2][1] * buffer[s1_[4]] + self.cr_[2][2] * buffer[s1_[5]]
        p_3 = self.cr_[3][0] * buffer[s1_[0]] + self.cr_[3][1] * buffer[s1_[1]] + self.cr_[3][2] * buffer[s1_[2]] \
            + self.cr_[3][3] * buffer[s1_[3]]
        p_4 = self.cr_[4][0] * buffer[s1_[3]] + self.cr_[4][1] * buffer[s1_[4]] + self.cr_[4][2] * buffer[s1_[5]] \
            + self.cr_[4][3] * buffer[s1_[6]]
        p_5 = self.cr_[5][0] * buffer[s1_[3]] + self.cr_[5][1] * buffer[s1_[4]] + self.cr_[5][2] * buffer[s1_[5]] \
            + self.cr_[5][3] * buffer[s1_[6]] + self.cr_[5][4] * buffer[s1_[7]]

        cell_state_xi_j = omega_0 * p_0 + omega_1 * p_1 + omega_2 * p_2 \
            + omega_3 * p_3 + omega_4 * p_4 + omega_5 * p_5
        return cell_state_xi_j
