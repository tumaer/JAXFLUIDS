from typing import Tuple

import jax
import jax.numpy as jnp
import numpy as np

from jaxfluids.config import precision

Array = jax.Array

NEIGHBOR_PATTERN: dict[int, Array] = {
    1: ("L", "R"),
    2: ("LL", "LR", "RL", "RR"),
    3: ("LLL", "LLR", "LRL", "LRR",
        "RLL", "RLR", "RRL", "RRR"),
}

def linear_interpolation_scattered(
        interpolation_position: Array,
        field_buffer: Array,
        cell_centers: Tuple[Array],
        method: str = "REPEATED",
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

    method_tuple = ("REPEATED", "POLYNOMIAL")
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


    nearest_cell_id = compute_nearest_cell_id(interpolation_position, cell_centers)

    (
        nearest_cell_id_left,
        nearest_cell_id_right
    ) = get_cell_id_neighbors(
        interpolation_position,
        nearest_cell_id,
        cell_centers
        )
    
    field_buffer_ip = interp_repeated_unrolled(
        field_buffer,
        interpolation_position,
        nearest_cell_id_left,
        nearest_cell_id_right,
        cell_centers
    )

    return field_buffer_ip


def compute_nearest_cell_id(
        ip_location: Array,
        cell_centers: tuple[Array]
        ) -> tuple[Array, Array]:

    dim = ip_location.shape[-1]
    nearest_ids = []

    for d in range(dim):
        xi = cell_centers[d]  # shape (n,)
        xP = ip_location[:, d]  # shape (N,)

        # searchsorted gives index where xP would be inserted
        idx = jnp.searchsorted(xi, xP, side="left", method="scan_unrolled")

        # clamp to valid range [1, n-1]
        idx = jnp.clip(idx, 1, xi.shape[0] - 1)

        left = idx - 1
        right = idx

        # compare distances to left vs right cell
        left_center = xi[left]
        right_center = xi[right]

        choose_right = (jnp.abs(right_center - xP) <
                        jnp.abs(left_center - xP))

        nearest = jnp.where(choose_right, right, left)
        nearest_ids.append(nearest)

    return jnp.stack(nearest_ids, axis=1)


def get_cell_id_neighbors(
        ip_location: Array,
        nearest_cell_id: Array,
        cell_centers: tuple[Array]
        ) -> tuple[Array,Array]:

    # Gather nearest cell center positions
    # shape (N, dim)
    nearest_centers = jnp.stack(
        [xi[nearest_cell_id[:, d]] for d, xi in enumerate(cell_centers)],
        axis=1
    )

    mask_left = (ip_location >= nearest_centers).astype(int)
    mask_right = 1 - mask_left

    left_ids = nearest_cell_id * mask_left + (nearest_cell_id - 1) * mask_right
    right_ids = nearest_cell_id * mask_right + (nearest_cell_id + 1) * mask_left

    # clip neighbor indices per dimension
    for d, xi in enumerate(cell_centers):
        max_idx = xi.shape[0] - 1

        # clip left
        left_ids = left_ids.at[:, d].set(
            jnp.clip(left_ids[:, d], 0, max_idx)
        )

        # clip right
        right_ids = right_ids.at[:, d].set(
            jnp.clip(right_ids[:, d], 0, max_idx)
        )

    return left_ids, right_ids


def get_cell_centers_from_cell_id(
        cell_centers: tuple[Array],
        left_ids: Array,
        right_ids: Array
        ) -> tuple[Array, Array]:

    left_centers = jnp.stack(
        [xi[left_ids[:, d]] for d, xi in enumerate(cell_centers)],
        axis=1
    )

    right_centers = jnp.stack(
        [xi[right_ids[:, d]] for d, xi in enumerate(cell_centers)],
        axis=1
    )

    return left_centers, right_centers


def interp_repeated_unrolled(
        buffer: Array, # (...,Nx,Ny,Nz)
        xP: Array, # (N, dim)
        idL: Array, # (N, dim)
        idR: Array, # (N, dim)
        cell_centers: tuple[Array] 
    ):
    dim = xP.shape[1]

    xL, xR = get_cell_centers_from_cell_id(
        cell_centers,
        idL, idR
        )
    
    # Compute normalized interpolation coordinates
    eps = 1e-50
    dx = xR - xL + eps
    t = jnp.clip((xP - xL) / dx, 0., 1.)   # (N, dim)

    N = xP.shape[0]

    buffer = jnp.squeeze(buffer)

    corner_patterns = NEIGHBOR_PATTERN[dim]   # e.g. ("LL", "LR", "RL", "RR")

    result = 0.0

    for pat in corner_patterns:

        # Build corner indices
        # For pattern "LRL", pick idL[:,0], idR[:,1], idL[:,2]
        gather_ids = []
        for d, c in enumerate(pat):
            gather_ids.append(jnp.where(c == "L", idL[:, d], idR[:, d]))

        # Compute weight for this corner
        w = jnp.ones(N)
        for d, c in enumerate(pat):
            w = w * (t[:, d] if c == "R" else (1 - t[:, d]))

        # Gather the correct neighbor values
        if dim == 1:
            vals = buffer[..., gather_ids[0]]
        elif dim == 2:
            vals = buffer[..., gather_ids[0], gather_ids[1]]
        elif dim == 3:
            vals = buffer[..., gather_ids[0], gather_ids[1], gather_ids[2]]

        # Accumulate weighted contribution
        result = result + vals * w

    return result