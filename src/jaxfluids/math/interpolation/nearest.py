from typing import Tuple

import jax
import jax.numpy as jnp

from jaxfluids.config import precision

Array = jax.Array

def nearest_interpolation_scattered(
        interpolation_position: Array,
        field_buffer: Array,
        cell_centers: Tuple[Array],
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

    raise NotImplementedError

    cell_centers = [xi.flatten() for xi in cell_centers]
    dim = interpolation_position.shape[-1]
    spatial_dims = field_buffer.shape[-3:]
    active_axes_indices = []

    for i in range(3):
        cells = spatial_dims[i]
        if cells > 1:
            active_axes_indices.append(i)
        if cell_centers[i].size != cells:
            raise RuntimeError
    
    if dim != len(active_axes_indices):
        raise RuntimeError
    
    cell_centers = [cell_centers[i] for i in active_axes_indices]

    # NEAREST CELL ID
    nearest_cell_id = []
    for axis_index in range(dim):
        cell_centers_xi = cell_centers[axis_index]
        cell_centers_xi = jnp.expand_dims(cell_centers_xi, axis=-1)
        interpolation_position_xi = interpolation_position[:,axis_index]
        distance = jnp.abs(cell_centers_xi - interpolation_position_xi)
        nearest_cell_id_xi = jnp.argmin(distance, axis=0)
        nearest_cell_id.append(nearest_cell_id_xi)
    nearest_cell_id = jnp.stack(nearest_cell_id, axis=-1)

    field_buffer = jnp.squeeze(field_buffer)

    s_ = (...,)
    for i in range(dim):
        s_ += (nearest_cell_id[:,i],)

    print(s_)
    exit()

    print(nearest_cell_id.shape)
    print(interpolation_position.shape)
    print(field_buffer.shape)
    exit()




def nearest_interpolation_griddata(
        cell_centers: Tuple[Array],
        points: Array,
        values: Array
        ):
    pass