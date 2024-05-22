from typing import List

import jax.numpy as jnp
from jax import Array

from jaxfluids.stencils.spatial_reconstruction import SpatialReconstruction
from jaxfluids.stencils.limiter import LIMITER_DICT

class MUSCL3(SpatialReconstruction):
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
            limiter: str, 
            offset: int = 0,
            **kwargs) -> None:
        super(MUSCL3, self).__init__(nh=nh, inactive_axes=inactive_axes, offset=offset)
        
        self._stencil_size = 4

        self.array_slices([range(-2, 1, 1), range(1, -2, -1)])
        self.stencil_slices([range(0, 3, 1), range(3, 0, -1)])

        self.limiter = LIMITER_DICT[limiter]

    def reconstruct_xi(self, 
            buffer: Array, 
            axis: int, 
            j: int, 
            dx: float = None, 
            **kwargs
        ) -> Array:
        s1_ = self.s_[j][axis]

        # r = (buffer[s1_[1]] - buffer[s1_[0]]) / (self.eps + (buffer[s1_[2]] - buffer[s1_[1]])) #2.28
        # limiter = self.limiter(r)
        # cell_state_xi_j = buffer[s1_[1]] + 0.5 * limiter * (buffer[s1_[2]] - buffer[s1_[1]]) #2.31

        # TODO
        # 1e-100 added to prevent getting nan gradients
        eps_ad = 1e-10
        if j == 0:
            delta_central = buffer[s1_[2]] - buffer[s1_[1]]
            delta_upwind = buffer[s1_[1]] - buffer[s1_[0]]
            r = jnp.where(
                delta_upwind >= self.eps,
                delta_central / (delta_upwind + eps_ad), 
                (delta_central + self.eps) / (delta_upwind + self.eps))
            limiter = self.limiter(r)
            cell_state_xi_j = buffer[s1_[1]] + 0.5 * limiter * (buffer[s1_[1]] - buffer[s1_[0]]) #2.31
        if j == 1:
            delta_central = buffer[s1_[1]] - buffer[s1_[2]]
            delta_upwind = buffer[s1_[0]] - buffer[s1_[1]]
            r = jnp.where(
                delta_upwind >= self.eps, 
                delta_central / (delta_upwind + eps_ad), 
                (delta_central + self.eps) / (delta_upwind + self.eps))
            limiter = self.limiter(r)
            cell_state_xi_j = buffer[s1_[1]] - 0.5 * limiter * (buffer[s1_[0]] - buffer[s1_[1]]) #2.31


        return cell_state_xi_j

class KOREN(MUSCL3):

    def __init__(self, nh: int, inactive_axes: List, **kwargs) -> None:
        super().__init__(nh, inactive_axes, limiter="KOREN", **kwargs)

class MC(MUSCL3):

    def __init__(self, nh: int, inactive_axes: List, **kwargs) -> None:
        super().__init__(nh, inactive_axes, limiter="MC", **kwargs)

class MINMOD(MUSCL3):

    def __init__(self, nh: int, inactive_axes: List, **kwargs) -> None:
        super().__init__(nh, inactive_axes, limiter="MINMOD", **kwargs)

class SUPERBEE(MUSCL3):

    def __init__(self, nh: int, inactive_axes: List, **kwargs) -> None:
        super().__init__(nh, inactive_axes, limiter="SUPERBEE", **kwargs)

class VANALBADA(MUSCL3):

    def __init__(self, nh: int, inactive_axes: List, **kwargs) -> None:
        super().__init__(nh, inactive_axes, limiter="VANALBADA", **kwargs)

class VANLEER(MUSCL3):

    def __init__(self, nh: int, inactive_axes: List, **kwargs) -> None:
        super().__init__(nh, inactive_axes, limiter="VANLEER", **kwargs)
