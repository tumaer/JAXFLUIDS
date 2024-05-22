import jax.numpy as jnp
from jax import Array

from typing import Tuple

from jaxfluids.config import precision

def linear_interpolation_scattered(
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

    # IDENTIFY LEFT AND RIGHT NEAREST CELL CENTERS
    nearest_cell_centers = []
    for i in range(dim):
        cell_centers_xi = cell_centers[i]
        indices_xi = nearest_cell_id[:,i]
        nearest_cell_centers_xi = cell_centers_xi[indices_xi]
        nearest_cell_centers.append(nearest_cell_centers_xi)
    nearest_cell_centers = jnp.stack(nearest_cell_centers, axis=1)

    mask_minus = jnp.where(nearest_cell_centers - interpolation_position < 0, 1, 0)
    mask_plus = 1 - mask_minus

    nearest_cell_id_minus = nearest_cell_id * mask_minus + \
                            (nearest_cell_id - 1) * mask_plus
    nearest_cell_id_plus = nearest_cell_id * mask_plus + \
                            (nearest_cell_id + 1) * mask_minus

    nearest_cell_center_minus = []
    nearest_cell_center_plus = []
    for i in range(dim):
        cell_centers_xi = cell_centers[i]
        indices_minus_xi = nearest_cell_id_minus[:,i]
        indices_plus_xi = nearest_cell_id_plus[:,i]
        nearest_cell_centers_minus_xi = cell_centers_xi[indices_minus_xi]
        nearest_cell_centers_plus_xi = cell_centers_xi[indices_plus_xi]
        nearest_cell_center_minus.append(nearest_cell_centers_minus_xi)
        nearest_cell_center_plus.append(nearest_cell_centers_plus_xi)
    nearest_cell_center_minus = jnp.stack(nearest_cell_center_minus, axis=1)
    nearest_cell_center_plus = jnp.stack(nearest_cell_center_plus, axis=1)
    
    if dim == 1:
        position_tuple = ("L", "R")
    elif dim == 2:
        position_tuple = ("LL", "RL", "LR", "RR")
    elif dim == 3:
        position_tuple = ("LLL", "RLL", "LRL", "RRL",
                          "LLR", "RLR", "LRR", "RRR")
    else:
        raise NotImplementedError

    nearest_cell_id_dict = {
        "L": nearest_cell_id_minus,
        "R": nearest_cell_id_plus}

    field_buffer = jnp.squeeze(field_buffer)
    field_buffer_tuple = []
    for position in position_tuple:
        s_ = (...,)
        for i, side in enumerate(position):
            cell_id = nearest_cell_id_dict[side]
            s_ += (cell_id[:,i],)
        field_buffer_tuple.append(field_buffer[s_])
    field_buffer_tuple = tuple(field_buffer_tuple)

    eps = 1e-30

    if dim == 1:
        field_buffer_L, field_buffer_R = field_buffer_tuple
        x_interp = interpolation_position[:,1]
        x_L, x_R = nearest_cell_center_minus[:,1], nearest_cell_center_plus[:,1]
        dx = x_R - x_L + eps
        interpolation_field_buffer = (x_R - x_interp)/dx * field_buffer_L \
            + (x_interp - x_L)/dx * field_buffer_R # TODO USE NEAREST WHEN OUT OF BOUNDS

    elif dim == 2:
        field_buffer_LL, field_buffer_RL, \
        field_buffer_LR, field_buffer_RR = field_buffer_tuple

        # X INTERPOLATION
        x_interp = interpolation_position[:,0]
        x_L, x_R = nearest_cell_center_minus[:,0], nearest_cell_center_plus[:,0]
        dx = x_R - x_L + eps

        LL = (x_R - x_interp)/dx * field_buffer_LL
        RL = (x_interp - x_L)/dx * field_buffer_RL
        L = LL + RL
        LR = (x_R - x_interp)/dx * field_buffer_LR
        RR = (x_interp - x_L)/dx * field_buffer_RR
        R = LR + RR

        # Y INTERPOLATION
        y_interp = interpolation_position[:,1]
        y_L, y_R = nearest_cell_center_minus[:,1], nearest_cell_center_plus[:,1]
        dy = y_R - y_L + eps

        interpolation_field_buffer = (y_R - y_interp)/dy * L + (y_interp - y_L)/dy * R

    elif dim == 3:

        field_buffer_LLL, field_buffer_RLL, \
        field_buffer_LRL, field_buffer_RRL, \
        field_buffer_LLR, field_buffer_RLR, \
        field_buffer_LRR, field_buffer_RRR = field_buffer_tuple

        # X INTERPOLATION
        x_interp = interpolation_position[:,0]
        x_L, x_R = nearest_cell_center_minus[:,0], nearest_cell_center_plus[:,0]
        dx = x_R - x_L + eps

        LLL = (x_R - x_interp)/dx * field_buffer_LLL
        RLL = (x_interp - x_L)/dx * field_buffer_RLL
        LL = LLL + RLL
        LRL = (x_R - x_interp)/dx * field_buffer_LRL
        RRL = (x_interp - x_L)/dx * field_buffer_RRL
        RL = LRL + RRL
        LLR = (x_R - x_interp)/dx * field_buffer_LLR
        RLR = (x_interp - x_L)/dx * field_buffer_RLR
        LR = LLR + RLR
        LRR = (x_R - x_interp)/dx * field_buffer_LRR
        RRR = (x_interp - x_L)/dx * field_buffer_RRR
        RR = LRR + RRR

        # Y INTERPOLATION
        y_interp = interpolation_position[:,1]
        y_L, y_R = nearest_cell_center_minus[:,1], nearest_cell_center_plus[:,1]
        dy = y_R - y_L + eps

        LL = (y_R - y_interp)/dy * LL
        RL = (y_interp - y_L)/dy * RL
        L = LL + RL
        LR = (y_R - y_interp)/dy * LR
        RR = (y_interp - y_L)/dy * RR
        R = LR + RR

        # Z INTERPOLATION
        z_interp = interpolation_position[:,2]
        z_L, z_R = nearest_cell_center_minus[:,2], nearest_cell_center_plus[:,2]
        dz = z_R - z_L + eps

        interpolation_field_buffer = (z_R - z_interp)/dz * L + (z_interp - z_L)/dz * R

    else:
        raise NotImplementedError

    if is_nearest_cell_id:
        return interpolation_field_buffer, nearest_cell_id
    else:
        return interpolation_field_buffer


def linear_interpolation_griddata() -> Array:
    pass # TODO INTERPOLATE SCATTERED DATA AT GRID