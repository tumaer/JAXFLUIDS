from typing import Tuple

import jax.numpy as jnp
from jax import Array

from jaxfluids.config import precision

def constant_interpolation_scattered(
        interpolation_position: Array,
        field_buffer: Array,
        cell_centers: Tuple[Array],
        is_nearest_cell_id: bool = False
        ) -> Array:
    """Interpolates a field buffer at scattered
    interpolation positions. The interpolation
    position must have shape (N,dim), where
    N is the number of points and dim the
    the spatial coordinates. The field buffer
    must have the shape (...,Nx+2*Nh,Ny+2*Nh,Nz+2*Nh)
    where Nx,Ny and Nz describe the amount
    of cells in x,y and z direction, respectively.
    Nh is the number of halo cells. The cell
    centers must include halo cells, e.g., the
    cell center in x direction must have 
    shape Nx+2*Nh.

    :param interpolation_position: _description_
    :type interpolation_position: Array
    :param field_buffer: _description_
    :type field_buffer: Array
    :param cell_centers: _description_
    :type cell_centers: Tuple[Array]
    :raises NotImplementedError: _description_
    :raises NotImplementedError: _description_
    :return: _description_
    :rtype: Array
    """
    # TODO


def constant_interpolation_griddata(
        cell_centers: Tuple[Array],
        points: Array,
        values: Array
        ):
    pass