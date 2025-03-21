from typing import List, Tuple

import jax
import jax.numpy as jnp

from jaxfluids.stencils.spatial_reconstruction import SpatialReconstruction
from jaxfluids.stencils.helper_functions import compute_coefficients_stretched_mesh_weno3

Array = jax.Array

class WENO3ADAP(SpatialReconstruction):
    """WENO3-JS 
    """

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
        super(WENO3ADAP, self).__init__(nh=nh, inactive_axes=inactive_axes, offset=offset)
        
        self.dr_uniform = [1/3, 2/3]
        self.cr_uniform = [
            [-0.5, 1.5], 
            [0.5, 0.5],
        ]

        self._stencil_size = 4
        self.array_slices([range(-2, 1, 1), range(1, -2, -1)])
        self.stencil_slices([range(0, 3, 1), range(3, 0, -1)])
        self.is_mesh_stretching = is_mesh_stretching

        self.cr_stretched, self.betar_streched, self.dr_stretched \
        = compute_coefficients_stretched_mesh_weno3(
            is_mesh_stretching, cell_sizes,
            self.s_mesh, self.s_nh_xi)

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
            beta_0 = jnp.square(betar[0][0] * buffer[s_[0]] + betar[0][1] * buffer[s_[1]])
            beta_1 = jnp.square(betar[1][0] * buffer[s_[1]] + betar[1][1] * buffer[s_[2]])
        
        else:
            beta_0 = jnp.square(buffer[s_[1]] - buffer[s_[0]])
            beta_1 = jnp.square(buffer[s_[2]] - buffer[s_[1]])       

        one_beta_0_sq = 1.0 / (beta_0 * beta_0 + self.eps)
        one_beta_1_sq = 1.0 / (beta_1 * beta_1 + self.eps)

        alpha_0 = dr[0] * one_beta_0_sq
        alpha_1 = dr[1] * one_beta_1_sq

        one_alpha = 1.0 / (alpha_0 + alpha_1)

        omega_0 = alpha_0 * one_alpha
        omega_1 = alpha_1 * one_alpha

        p_0 = cr[0][0] * buffer[s_[0]] + cr[0][1] * buffer[s_[1]] 
        p_1 = cr[1][0] * buffer[s_[1]] + cr[1][1] * buffer[s_[2]]

        cell_state_xi_j = omega_0 * p_0 + omega_1 * p_1

        return cell_state_xi_j