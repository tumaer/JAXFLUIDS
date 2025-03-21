from typing import List, Tuple

import jax
import jax.numpy as jnp

from jaxfluids.stencils.spatial_reconstruction import SpatialReconstruction

Array = jax.Array

class WENO3Base(SpatialReconstruction):
    """WENO3Base implements basic functionality
    for WENO3-type stencils.

    :param SpatialReconstruction: _description_
    :type SpatialReconstruction: _type_
    """
    
    required_halos = 2

    def __init__(self, nh: int, inactive_axes: List, offset: int = 0, **kwargs) -> None:
        super(WENO3Base, self).__init__(nh, inactive_axes, offset)

        self.required_halos = 2
        self._dr = (1/3, 2/3)
        self._cr = ((-0.5, 1.5), (0.5, 0.5))

        self._stencil_size = 4
        self.array_slices([range(-2, 1, 1), range(1, -2, -1)])
        self.stencil_slices([range(0, 3, 1), range(3, 0, -1)])


    def smoothness(
            self, u_im: Array, u_i: Array, 
            u_ip: Array) -> Tuple[Array, Array]:
        beta_0 = jnp.square(u_i - u_im)
        beta_1 = jnp.square(u_ip - u_i)
        return beta_0, beta_1


    def polynomials(
            self, u_im: Array, u_i: Array, 
            u_ip: Array) -> Tuple[Array, Array]:
        p_0 = self._cr[0][0] * u_im + self._cr[0][1] * u_i 
        p_1 = self._cr[1][0] * u_i  + self._cr[1][1] * u_ip
        return p_0, p_1