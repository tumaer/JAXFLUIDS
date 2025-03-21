from typing import List, Tuple

import jax
import jax.numpy as jnp

from jaxfluids.stencils.spatial_reconstruction import SpatialReconstruction
from jaxfluids.stencils.helper_functions import compute_coefficients_stretched_mesh_muscl3

Array = jax.Array

class MINMODADADAP(SpatialReconstruction):
    """MUSCL-type reconstruction with different limiters.

    psi_{i+1/2}^L = psi_i     + 0.5 * phi(r_L) * (psi_{i} - psi_{i-1})
    psi_{i+1/2}^R = psi_{i+1} - 0.5 * phi(r_R) * (psi_{i+2} - psi_{i+1})

    r_L = (phi_{i+1} - phi_{i}) / (phi_{i} - phi_{i-1})
    r_R = (phi_{i+1} - phi_{i}) / (phi_{i+2} - phi_{i+1})

    """
    
    required_halos = 2
    is_for_adaptive_mesh = True
    
    def __init__(self, 
            nh: int, 
            inactive_axes: List, 
            is_mesh_stretching: List[bool] = None,
            cell_sizes: Tuple[Array] = None,
            offset: int = 0,
            **kwargs) -> None:
        super(MINMODADADAP, self).__init__(nh=nh, inactive_axes=inactive_axes, offset=offset)
        
        self._stencil_size = 4
        self.array_slices([range(-2, 1, 1), range(1, -2, -1)])
        self.stencil_slices([range(0, 3, 1), range(3, 0, -1)])
        self.is_mesh_stretching = is_mesh_stretching

        self.c_upwind_uniform = 0.5
        self.c_ratio_uniform = 1.0

        self.c_upwind_stretched, self.c_ratio_stretched \
        = compute_coefficients_stretched_mesh_muscl3(
            is_mesh_stretching, cell_sizes,
            self.s_mesh, self.s_nh_xi)

    def reconstruct_xi(self, 
            buffer: Array, 
            axis: int, 
            j: int, 
            dx: float = None, 
            **kwargs
        ) -> Array:
        s_ = self.s_[j][axis]
        is_mesh_stretching = self.is_mesh_stretching[axis]

        if is_mesh_stretching:
            c_upwind = self.c_upwind_stretched[j][axis]
            c_ratio = self.c_ratio_stretched[j][axis]

            # NOTE Slice arrays for mesh-stretching + parallel
            if c_upwind.ndim == 4:
                device_id = jax.lax.axis_index(axis_name="i")
                c_upwind = c_upwind[device_id]
                c_ratio = c_ratio[device_id]
        
        else:
            c_upwind = self.c_upwind_uniform
            c_ratio = self.c_ratio_uniform

        eps_ad = 1e-100
        if j == 0:
            delta_central = buffer[s_[2]] - buffer[s_[1]]
            delta_upwind = buffer[s_[1]] - buffer[s_[0]]
        if j == 1:
            delta_central = buffer[s_[1]] - buffer[s_[2]]
            delta_upwind = buffer[s_[0]] - buffer[s_[1]]

        r = jnp.where(
            delta_upwind >= self.eps, 
            delta_central / (delta_upwind + eps_ad), 
            (delta_central + self.eps) / (delta_upwind + self.eps))
        r *= c_ratio

        cell_state_xi_j = jnp.where(
            r < 0, buffer[s_[1]],
            jnp.where(
                r < 1, 
                buffer[s_[1]] + c_upwind * c_ratio * (buffer[s_[2]] - buffer[s_[1]]),
                buffer[s_[1]] + c_upwind * (buffer[s_[1]] - buffer[s_[0]])
            )                        
        )

        return cell_state_xi_j
