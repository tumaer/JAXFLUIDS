from typing import List

import jax
import jax.numpy as jnp

from jaxfluids.stencils.spatial_reconstruction import SpatialReconstruction

Array = jax.Array

class MINMODAD(SpatialReconstruction):
    """MUSCL-type reconstruction with different limiters.

    psi_{i+1/2}^L = psi_i     + 0.5 * phi(r_L) * (psi_{i} - psi_{i-1})
    psi_{i+1/2}^R = psi_{i+1} - 0.5 * phi(r_R) * (psi_{i+2} - psi_{i+1})

    r_L = (phi_{i+1} - phi_{i}) / (phi_{i} - phi_{i-1})
    r_R = (phi_{i+1} - phi_{i}) / (phi_{i+2} - phi_{i+1})

    """
    
    required_halos = 2
    
    def __init__(self, 
            nh: int, 
            inactive_axes: List, 
            offset: int = 0,
            **kwargs) -> None:
        super(MINMODAD, self).__init__(nh=nh, inactive_axes=inactive_axes, offset=offset)
        
        self._stencil_size = 4

        self.array_slices([range(-2, 1, 1), range(1, -2, -1)])
        self.stencil_slices([range(0, 3, 1), range(3, 0, -1)])

    def reconstruct_xi(self, 
            buffer: Array, 
            axis: int, 
            j: int, 
            dx: float = None, 
            **kwargs
        ) -> Array:
        s1_ = self.s_[j][axis]

        eps_ad = 1e-100
        if j == 0:
            delta_central = buffer[s1_[2]] - buffer[s1_[1]]
            delta_upwind = buffer[s1_[1]] - buffer[s1_[0]]
        if j == 1:
            delta_central = buffer[s1_[1]] - buffer[s1_[2]]
            delta_upwind = buffer[s1_[0]] - buffer[s1_[1]]

        r = jnp.where(
            delta_upwind >= self.eps, 
            delta_central / (delta_upwind + eps_ad), 
            (delta_central + self.eps) / (delta_upwind + self.eps))

        cell_state_xi_j = jnp.where(
            r < 0, buffer[s1_[1]],
            jnp.where(
                r < 1, 
                0.5 * (buffer[s1_[1]] + buffer[s1_[2]]),
                1.5 * buffer[s1_[1]] - 0.5 * buffer[s1_[0]]
            )                        
        )

        return cell_state_xi_j
