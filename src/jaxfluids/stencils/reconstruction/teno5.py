from typing import List

import jax.numpy as jnp
from jax import Array

from jaxfluids.stencils.spatial_reconstruction import SpatialReconstruction

class TENO5(SpatialReconstruction):
    ''' Fu et al. - 2016 -  A family of high-order targeted ENO schemes for compressible-fluid simulations'''    
    
    def __init__(self, 
            nh: int, 
            inactive_axes: List, 
            offset: int = 0,
            **kwargs) -> None:
        super(TENO5, self).__init__(nh=nh, inactive_axes=inactive_axes, offset=offset)

        # Coefficients for 5-th order convergence
        # self.dr_ = [1/10, 6/10, 3/10]
        # Coefficients for optimized spectral properties
        self.dr_ = [0.05, 0.55, 0.40]

        self.cr_ = [
            [1/3, -7/6, 11/6], 
            [-1/6, 5/6, 1/3], 
            [1/3, 5/6, -1/6]
        ]

        self.C = 1.0
        self.q = 6
        self.CT = 1e-5

        self._stencil_size = 6
        self.array_slices([range(-3, 2, 1), range(2, -3, -1)])
        self.stencil_slices([range(0, 5, 1), range(5, 0, -1)])

    def reconstruct_xi(self, 
            buffer: Array, 
            axis: int, 
            j: int, 
            dx: float = None, 
            **kwargs) -> Array:
        s1_ = self.s_[j][axis]

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
            * (3 * buffer[s1_[2]] - 4 * buffer[s1_[3]] + buffer[s1_[4]])

        tau_5 = jnp.abs(beta_0 - beta_2)

        # SMOOTHNESS MEASURE
        gamma_0 = (self.C + tau_5 / (beta_0 + self.eps))**self.q
        gamma_1 = (self.C + tau_5 / (beta_1 + self.eps))**self.q
        gamma_2 = (self.C + tau_5 / (beta_2 + self.eps))**self.q

        # gamma_0 *= (gamma_0 * gamma_0)
        # gamma_1 *= (gamma_1 * gamma_1)
        # gamma_2 *= (gamma_2 * gamma_2)

        # gamma_0 *= gamma_0
        # gamma_1 *= gamma_1
        # gamma_2 *= gamma_2

        one_gamma_sum = 1.0 / (gamma_0 + gamma_1 + gamma_2)

        # SHARP CUTOFF FUNCTION
        delta_0 = jnp.where(gamma_0 * one_gamma_sum < self.CT, 0, 1)
        delta_1 = jnp.where(gamma_1 * one_gamma_sum < self.CT, 0, 1)
        delta_2 = jnp.where(gamma_2 * one_gamma_sum < self.CT, 0, 1)

        w0 = delta_0 * self.dr_[0]
        w1 = delta_1 * self.dr_[1]
        w2 = delta_2 * self.dr_[2]

        # TODO eps should not be necessary
        one_dk = 1.0 / (w0 + w1 + w2 + self.eps)

        omega_0 = w0 * one_dk 
        omega_1 = w1 * one_dk 
        omega_2 = w2 * one_dk 

        p_0 = self.cr_[0][0] * buffer[s1_[0]] + self.cr_[0][1] * buffer[s1_[1]] + self.cr_[0][2] * buffer[s1_[2]]
        p_1 = self.cr_[1][0] * buffer[s1_[1]] + self.cr_[1][1] * buffer[s1_[2]] + self.cr_[1][2] * buffer[s1_[3]]
        p_2 = self.cr_[2][0] * buffer[s1_[2]] + self.cr_[2][1] * buffer[s1_[3]] + self.cr_[2][2] * buffer[s1_[4]]

        cell_state_xi_j = omega_0 * p_0 + omega_1 * p_1 + omega_2 * p_2
        return cell_state_xi_j