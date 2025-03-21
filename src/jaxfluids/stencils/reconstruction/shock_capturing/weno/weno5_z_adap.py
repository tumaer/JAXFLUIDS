from typing import List, Tuple

import jax
import jax.numpy as jnp

from jaxfluids.stencils.spatial_reconstruction import SpatialReconstruction
from jaxfluids.stencils.helper_functions import compute_coefficients_stretched_mesh_weno5

Array = jax.Array

class WENO5ZADAP(SpatialReconstruction):
    ''' Borges et al. - 2008 - An improved WENO scheme for hyperbolic conservation laws '''    
    
    required_halos = 3
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
        super(WENO5ZADAP, self).__init__(nh=nh, inactive_axes=inactive_axes, offset=offset)

        self.dr_uniform = [1/10, 6/10, 3/10]
        self.cr_uniform = [
            [1/3, -7/6, 11/6], 
            [-1/6, 5/6, 1/3], 
            [1/3, 5/6, -1/6]
        ]

        self._stencil_size = 6
        self.array_slices([range(-3, 2, 1), range(2, -3, -1)])
        self.stencil_slices([range(0, 5, 1), range(5, 0, -1)])
        self.is_mesh_stretching = is_mesh_stretching

        self.cr_stretched, self.betar_streched, self.dr_stretched \
        = compute_coefficients_stretched_mesh_weno5(
            is_mesh_stretching, cell_sizes,
            self.s_mesh, self.s_nh_xi)

        self.is_positivity_limiter_smoothness = True
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
            betar = self.betar_streched[j][axis]
            dr = self.dr_stretched[j][axis]

            # NOTE Slice arrays for mesh-stretching + parallel
            if dr.ndim == 5:
                device_id = jax.lax.axis_index(axis_name="i")
                cr = cr[device_id]
                betar = betar[device_id]
                dr = dr[device_id]
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

            if self.is_positivity_limiter_smoothness:
                # NOTE Beta's might be negative due to machine precision
                beta_0 = jnp.abs(beta_0)
                beta_1 = jnp.abs(beta_1)
                beta_2 = jnp.abs(beta_2)
        
        else:
            beta_0 = 13.0 / 12.0 * jnp.square(buffer[s_[0]] - 2 * buffer[s_[1]] + buffer[s_[2]]) \
                + 1.0 / 4.0 * jnp.square(buffer[s_[0]] - 4 * buffer[s_[1]] + 3 * buffer[s_[2]])
            beta_1 = 13.0 / 12.0 * jnp.square(buffer[s_[1]] - 2 * buffer[s_[2]] + buffer[s_[3]]) \
                + 1.0 / 4.0 * jnp.square(buffer[s_[1]] - buffer[s_[3]])
            beta_2 = 13.0 / 12.0 * jnp.square(buffer[s_[2]] - 2 * buffer[s_[3]] + buffer[s_[4]]) \
                + 1.0 / 4.0 * jnp.square(3 * buffer[s_[2]] - 4 * buffer[s_[3]] + buffer[s_[4]])

        tau_5 = jnp.abs(beta_0 - beta_2)

        alpha_z_0 = dr[0] * (1.0 + tau_5 / (beta_0 + self.eps))
        alpha_z_1 = dr[1] * (1.0 + tau_5 / (beta_1 + self.eps))
        alpha_z_2 = dr[2] * (1.0 + tau_5 / (beta_2 + self.eps))

        one_alpha_z = 1.0 / (alpha_z_0 + alpha_z_1 + alpha_z_2)

        omega_z_0 = alpha_z_0 * one_alpha_z
        omega_z_1 = alpha_z_1 * one_alpha_z
        omega_z_2 = alpha_z_2 * one_alpha_z

        p_0 = cr[0][0] * buffer[s_[0]] + cr[0][1] * buffer[s_[1]] + cr[0][2] * buffer[s_[2]]
        p_1 = cr[1][0] * buffer[s_[1]] + cr[1][1] * buffer[s_[2]] + cr[1][2] * buffer[s_[3]]
        p_2 = cr[2][0] * buffer[s_[2]] + cr[2][1] * buffer[s_[3]] + cr[2][2] * buffer[s_[4]]

        cell_state_xi_j = omega_z_0 * p_0 + omega_z_1 * p_1 + omega_z_2 * p_2

        if self.debug:
            omega = jnp.stack([omega_z_0, omega_z_1, omega_z_2], axis=-1)
            debug_out = {"omega": omega}
            return cell_state_xi_j, debug_out

        return cell_state_xi_j