from typing import List, Tuple

import jax
import jax.numpy as jnp

from jaxfluids.stencils.spatial_reconstruction import SpatialReconstruction

Array = jax.Array

class WENO6Base(SpatialReconstruction):
    """WENO6Base implements basic functionality
    for WENO6-type stencils.

    :param SpatialReconstruction: _description_
    :type SpatialReconstruction: _type_
    """
    required_halos = 3

    def __init__(self, nh: int, inactive_axes: List, offset: int = 0, **kwargs) -> None:
        super(WENO6Base, self).__init__(nh, inactive_axes, offset)

        self._dr = (1/20, 9/20, 9/20, 1/20)
        self._cr = ((1/3, -7/6, 11/6), 
                    (-1/6, 5/6, 1/3),
                    (1/3, 5/6, -1/6),
                    (11/6, -7/6, 1/3))

        self._stencil_size = 6
        self.array_slices([range(-3, 3, 1), range(2, -4, -1)])
        self.stencil_slices([range(0, 6, 1), range(5, -1, -1)])


    def smoothness(self, u_imm: Array, u_im: Array, 
                   u_i: Array, u_ip: Array, u_ipp: Array,
                   u_ippp: Array) -> Tuple[Array, Array, Array, Array]:
        beta_0 = 13.0 / 12.0 * jnp.square(u_imm - 2*u_im + u_i  ) + 1.0 / 4.0 * jnp.square(u_imm - 4*u_im + 3*u_i)
        beta_1 = 13.0 / 12.0 * jnp.square(u_im  - 2*u_i  + u_ip ) + 1.0 / 4.0 * jnp.square(u_im  - u_ip)
        beta_2 = 13.0 / 12.0 * jnp.square(u_i   - 2*u_ip + u_ipp) + 1.0 / 4.0 * jnp.square(3*u_i - 4*u_ip + u_ipp)
        beta_3 = 1.0 / 10080 / 12 * (
            271779 * u_imm * u_imm + \
            u_imm  * (-2380800 * u_im  + 4086352  * u_i     - 3462252  * u_ip    + 1458762 * u_ipp   - 245620  * u_ippp) + \
            u_im   * (5653317  * u_im  - 20427884 * u_i     + 17905032 * u_ip    - 7727988 * u_ipp   + 1325006 * u_ippp) + \
            u_i    * (19510972 * u_i   - 35817664 * u_ip    + 15929912 * u_ipp   - 2792660 * u_ippp) + \
            u_ip   * (17195652 * u_ip  - 15880404 * u_ipp   + 2863984  * u_ippp) + \
            u_ipp  * (3824847  * u_ipp - 1429976  * u_ippp) + \
            139633 * u_ippp * u_ippp
            )
        
        return beta_0, beta_1, beta_2, beta_3


    def polynomials(self, u_imm: Array, u_im: Array, 
                   u_i: Array, u_ip: Array, u_ipp: Array,
                   u_ippp: Array) -> Tuple[Array, Array, Array, Array]:
        p_0 = self._cr[0][0] * u_imm + self._cr[0][1] * u_im  + self._cr[0][2] * u_i
        p_1 = self._cr[1][0] * u_im  + self._cr[1][1] * u_i   + self._cr[1][2] * u_ip
        p_2 = self._cr[2][0] * u_i   + self._cr[2][1] * u_ip  + self._cr[2][2] * u_ipp
        p_3 = self._cr[3][0] * u_ip  + self._cr[3][1] * u_ipp + self._cr[3][2] * u_ippp

        return p_0, p_1, p_2, p_3