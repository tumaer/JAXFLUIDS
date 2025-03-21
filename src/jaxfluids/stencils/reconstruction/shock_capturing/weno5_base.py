from typing import List, Tuple

import jax
import jax.numpy as jnp

from jaxfluids.stencils.spatial_reconstruction import SpatialReconstruction

Array = jax.Array

class WENO5Base(SpatialReconstruction):
    """WENO5Base implements basic functionality
    for WENO5-type stencils.

    :param SpatialReconstruction: _description_
    :type SpatialReconstruction: _type_
    """

    required_halos = 3

    def __init__(self, nh: int, inactive_axes: List, offset: int = 0, **kwargs) -> None:
        super(WENO5Base, self).__init__(nh, inactive_axes, offset)


        self._dr = (1/10, 6/10, 3/10)
        self._cr = ((1/3, -7/6, 11/6), 
                    (-1/6, 5/6, 1/3),
                    (1/3, 5/6, -1/6))

        self._stencil_size = 6
        self.array_slices([range(-3, 2, 1), range(2, -3, -1)])
        self.stencil_slices([range(0, 5, 1), range(5, 0, -1)])


    def smoothness(self, u_imm: Array, u_im: Array, 
                   u_i: Array, u_ip: Array, u_ipp: Array
                   ) -> Tuple[Array, Array, Array]:
        beta_0 = 13.0 / 12.0 * jnp.square(u_imm - 2*u_im + u_i) + 1.0 / 4.0 * jnp.square(u_imm - 4*u_im + 3*u_i)
        beta_1 = 13.0 / 12.0 * jnp.square(u_im - 2*u_i + u_ip)  + 1.0 / 4.0 * jnp.square(u_im - u_ip)
        beta_2 = 13.0 / 12.0 * jnp.square(u_i - 2*u_ip + u_ipp) + 1.0 / 4.0 * jnp.square(3*u_i - 4*u_ip + u_ipp)
        
        return beta_0, beta_1, beta_2


    def polynomials(self, u_imm: Array, u_im: Array, 
                   u_i: Array, u_ip: Array, u_ipp: Array
                   ) -> Tuple[Array, Array, Array]:
        p_0 = self._cr[0][0] * u_imm + self._cr[0][1] * u_im + self._cr[0][2] * u_i
        p_1 = self._cr[1][0] * u_im  + self._cr[1][1] * u_i  + self._cr[1][2] * u_ip
        p_2 = self._cr[2][0] * u_i   + self._cr[2][1] * u_ip + self._cr[2][2] * u_ipp

        return p_0, p_1, p_2