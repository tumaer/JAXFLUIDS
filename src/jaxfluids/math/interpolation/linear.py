from typing import Tuple

import jax
import jax.numpy as jnp
import numpy as np

from jaxfluids.config import precision

Array = jax.Array

def linear_interpolation_scattered(
        interpolation_position: Array,
        field_buffer: Array,
        cell_centers: Tuple[Array],
        is_nearest_cell_id: bool = False,
        method: str = "REPEATED",
        nearest_out_of_bounds: bool = True,
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

    method_tuple = ("REPEATED", "POLYNOMIAL", "WEIGHTED")
    if method not in method_tuple:
        raise RuntimeError

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

        itemsize = field_buffer.itemsize
        nxi = spatial_dims[axis_index]

        DISTANCE_BUFFER_MEM = 0.05e9 # TODO hard coded, move to numerical setup?
        BATCH_SIZE = int(DISTANCE_BUFFER_MEM/itemsize/nxi) 
        NO_POINTS = interpolation_position.shape[0]
        NO_BATCHES = int(np.ceil(NO_POINTS/BATCH_SIZE))

        indices = jnp.stack([jnp.arange(i*BATCH_SIZE,(i+1)*BATCH_SIZE, dtype=int) for i in range(NO_BATCHES)],axis=0)
        nearest_cell_id_xi = jnp.zeros(NO_POINTS, dtype=int)
        def _body_func(
                i: int,
                args: Array
                ) -> Array:
            nearest_cell_id_xi, indices = args
            interpolation_position_xi_batch = interpolation_position_xi[indices[i]]
            distance = jnp.abs(cell_centers_xi - interpolation_position_xi_batch)
            nearest_cell_id_xi_batch = jnp.argmin(distance, axis=0)
            nearest_cell_id_xi = nearest_cell_id_xi.at[indices[i]].set(nearest_cell_id_xi_batch)
            return nearest_cell_id_xi, indices
        args = (nearest_cell_id_xi, indices)
        args = jax.lax.fori_loop(0, NO_BATCHES, _body_func, args)
        nearest_cell_id_xi, indices = args

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

    mask_minus = jnp.where(nearest_cell_centers - interpolation_position <= 0, 1, 0)
    mask_plus = 1 - mask_minus

    nearest_cell_id_minus = nearest_cell_id*mask_minus + \
                            (nearest_cell_id - 1)*mask_plus
    nearest_cell_id_plus = nearest_cell_id*mask_plus + \
                            (nearest_cell_id + 1)*mask_minus

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
        "R": nearest_cell_id_plus
        }

    field_buffer = jnp.squeeze(field_buffer)
    field_buffer_tuple = []
    for position in position_tuple:
        s_ = (...,)
        for i, side in enumerate(position):
            cell_id = nearest_cell_id_dict[side]
            s_ += (cell_id[:,i],)
        field_buffer_tuple.append(field_buffer[s_])
    field_buffer_tuple = tuple(field_buffer_tuple)

    eps = 1e-50

    if dim == 1:
        field_buffer_L, field_buffer_R = field_buffer_tuple

        x_P = interpolation_position[:,1]

        x_L, x_R = nearest_cell_center_minus[:,1], nearest_cell_center_plus[:,1]

        dx = x_R - x_L + eps

        interpolation_field_buffer = (x_R - x_P)/dx*field_buffer_L \
            + (x_P - x_L)/dx*field_buffer_R
    
        if nearest_out_of_bounds:
            mask_R = x_P >= x_R
            mask_L = x_P <= x_L
            mask = mask_R | mask_L
            interpolation_field_buffer = interpolation_field_buffer * (1 - mask) + mask_R * field_buffer_R + mask_L * field_buffer_L

    elif dim == 2:

        field_buffer_LL, field_buffer_RL, \
        field_buffer_LR, field_buffer_RR = field_buffer_tuple

        x_P = interpolation_position[:,0]
        y_P = interpolation_position[:,1]

        x_L, x_R = nearest_cell_center_minus[:,0], nearest_cell_center_plus[:,0]
        y_L, y_R = nearest_cell_center_minus[:,1], nearest_cell_center_plus[:,1]
        
        if method == "REPEATED":

            dx = x_R - x_L + eps
            dy = y_R - y_L + eps

            L = (x_R - x_P)/dx*field_buffer_LL + (x_P - x_L)/dx*field_buffer_RL
            R = (x_R - x_P)/dx*field_buffer_LR + (x_P - x_L)/dx*field_buffer_RR

            if nearest_out_of_bounds: # NOTE use nearest for out of bounds interpolation points
                mask_R = x_P >= x_R
                mask_L = x_P <= x_L
                mask = mask_R | mask_L
                L = L * (1 - mask) + mask_R * field_buffer_RL + mask_L * field_buffer_LL
                R = R * (1 - mask) + mask_R * field_buffer_RR + mask_L * field_buffer_LR

            interpolation_field_buffer = (y_R - y_P)/dy*L + (y_P - y_L)/dy*R

            if nearest_out_of_bounds:
                mask_R = y_P >= y_R
                mask_L = y_P <= y_L
                mask = mask_R | mask_L
                interpolation_field_buffer = interpolation_field_buffer * (1 - mask) + mask_R * R + mask_L * L

        elif method == "POLYNOMIAL":

            ones = jnp.ones_like(x_L)
            mat = jnp.array([
                [ones, x_L, y_L, x_L*y_L],
                [ones, x_L, y_R, x_L*y_R],
                [ones, x_R, y_L, x_R*y_L],
                [ones, x_R, y_R, x_R*y_R]
            ])
            mat = jnp.moveaxis(mat, -1, 0)

            vec = jnp.stack([field_buffer_LL, field_buffer_LR,
                             field_buffer_RL, field_buffer_RR],
                             axis=-1)
            
            coeff = jnp.linalg.solve(mat, vec)

            interpolation_field_buffer = coeff[:,0] + coeff[:,1]*x_P + coeff[:,2]*y_P + coeff[:,3]*x_P*y_P


        elif method == "WEIGHTED":

            ones = jnp.ones_like(x_L)
            mat = jnp.array([
                [ones,      ones,       ones,       ones],
                [x_L,       x_L,        x_R,        x_R],
                [y_L,       y_R,        y_L,        y_R],
                [x_L*y_L,   x_L*y_R,    x_R*y_L,    x_R*y_R],
            ])
            mat = jnp.moveaxis(mat, -1, 0)

            vec = jnp.stack([ones, x_P, y_P, x_P*y_P], axis=-1)
            
            coeff = jnp.linalg.solve(mat, vec)

            interpolation_field_buffer = coeff[:,0]*field_buffer_LL + coeff[:,1]*field_buffer_LR + \
                coeff[:,2]*field_buffer_RL + coeff[:,3]*field_buffer_RR


    elif dim == 3:

        field_buffer_LLL, field_buffer_RLL, \
        field_buffer_LRL, field_buffer_RRL, \
        field_buffer_LLR, field_buffer_RLR, \
        field_buffer_LRR, field_buffer_RRR = field_buffer_tuple

        x_P = interpolation_position[:,0]
        y_P = interpolation_position[:,1]
        z_P = interpolation_position[:,2]

        x_L, x_R = nearest_cell_center_minus[:,0], nearest_cell_center_plus[:,0]
        y_L, y_R = nearest_cell_center_minus[:,1], nearest_cell_center_plus[:,1]
        z_L, z_R = nearest_cell_center_minus[:,2], nearest_cell_center_plus[:,2]

        if method == "REPEATED":
                
            dx = x_R - x_L + eps
            dy = y_R - y_L + eps
            dz = z_R - z_L + eps

            LL = (x_R - x_P)/dx*field_buffer_LLL + (x_P - x_L)/dx*field_buffer_RLL
            RL = (x_R - x_P)/dx*field_buffer_LRL + (x_P - x_L)/dx*field_buffer_RRL
            LR = (x_R - x_P)/dx*field_buffer_LLR + (x_P - x_L)/dx*field_buffer_RLR
            RR = (x_R - x_P)/dx*field_buffer_LRR + (x_P - x_L)/dx*field_buffer_RRR

            if nearest_out_of_bounds: # NOTE use nearest for out of bounds interpolation points
                mask_R = x_P >= x_R
                mask_L = x_P <= x_L
                mask = mask_R | mask_L
                LL = LL * (1 - mask) + mask_R * field_buffer_RLL + mask_L * field_buffer_LLL
                RL = RL * (1 - mask) + mask_R * field_buffer_LRL + mask_L * field_buffer_RRL
                LR = LR * (1 - mask) + mask_R * field_buffer_LLR + mask_L * field_buffer_RLR
                RR = RR * (1 - mask) + mask_R * field_buffer_LRR + mask_L * field_buffer_RRR

            L = (y_R - y_P)/dy*LL + (y_P - y_L)/dy*RL
            R = (y_R - y_P)/dy*LR +  (y_P - y_L)/dy*RR

            if nearest_out_of_bounds: # NOTE use nearest for out of bounds interpolation points
                mask_R = y_P >= y_R
                mask_L = y_P <= y_L
                mask = mask_R | mask_L
                L = L * (1 - mask) + mask_R * RL + mask_L * LL
                R = R * (1 - mask) + mask_R * RR + mask_L * LR

            interpolation_field_buffer = (z_R - z_P)/dz*L + (z_P - z_L)/dz*R

            if nearest_out_of_bounds: # NOTE use nearest for out of bounds interpolation points
                mask_R = z_P >= z_R
                mask_L = z_P <= z_L
                mask = mask_R | mask_L
                interpolation_field_buffer =  interpolation_field_buffer * (1 - mask) + mask_R * R + mask_L * L

        elif method == "POLYNOMIAL":

            ones = jnp.ones_like(x_L)
            mat = jnp.array([
                [ones, x_L, y_L, z_L, x_L*y_L, x_L*z_L, y_L*z_L, x_L*y_L*z_L],
                [ones, x_R, y_L, z_L, x_R*y_L, x_R*z_L, y_L*z_L, x_R*y_L*z_L],
                [ones, x_L, y_R, z_L, x_L*y_R, x_L*z_L, y_R*z_L, x_L*y_R*z_L],
                [ones, x_R, y_R, z_L, x_R*y_R, x_R*z_L, y_R*z_L, x_R*y_R*z_L],
                [ones, x_L, y_L, z_R, x_L*y_L, x_L*z_R, y_L*z_R, x_L*y_L*z_R],
                [ones, x_R, y_L, z_R, x_R*y_L, x_R*z_R, y_L*z_R, x_R*y_L*z_R],
                [ones, x_L, y_R, z_R, x_L*y_R, x_L*z_R, y_R*z_R, x_L*y_R*z_R],
                [ones, x_R, y_R, z_R, x_R*y_R, x_R*z_R, y_R*z_R, x_R*y_R*z_R]
            ])
            mat = jnp.moveaxis(mat, -1, 0)

            vec = jnp.stack([field_buffer_LLL, field_buffer_RLL, field_buffer_LRL, field_buffer_RRL,
                             field_buffer_LLR, field_buffer_RLR, field_buffer_LRR, field_buffer_RRR],
                             axis=-1)
            
            coeff = jnp.linalg.solve(mat, vec)

            interpolation_field_buffer = coeff[:,0] + coeff[:,1]*x_P + coeff[:,2]*y_P + coeff[:,3]*z_P +\
                coeff[:,4]*x_P*y_P + coeff[:,5]*x_P*z_P + coeff[:,6]*y_P*z_P + \
                coeff[:,7]*x_P*y_P*z_P


        elif method == "WEIGHTED":
            raise NotImplementedError

    if is_nearest_cell_id:
        return interpolation_field_buffer, nearest_cell_id
    else:
        return interpolation_field_buffer


def linear_interpolation_griddata() -> Array:
    pass # TODO INTERPOLATE SCATTERED DATA AT GRID