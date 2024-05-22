from typing import Dict, List, Tuple

import jax
import jax.numpy as jnp
from jax import Array
import numpy as np

from jaxfluids.domain.helper_functions import split_cell_centers_xi

def real_wavenumber_grid(N: int) -> Tuple[Array, Array, Array]:
    """Initializes wavenumber grid and wavenumber vector. For given
    resolution (Nx,Ny,Nz) in physical space, the corresponding wave-
    number resolution is (Nf,Ny,Nz).

    :param N: Resolution.
    :type N: int
    :return: Wavenumber grid and wavenumber vector.
    :rtype: Tuple[Array, Array]
    """

    Nf = N//2 + 1
    k = jnp.fft.fftfreq(N, 1./N) # for y and z direction
    kx = k[:Nf]
    kx = kx.at[-1].mul(-1)
    k_field = jnp.array(jnp.meshgrid(kx, k, k, indexing="ij"), dtype=int)
    k_vec = jnp.arange(N)

    fact = 2 * (k_field[0] > 0) * (k_field[0] < N//2) + 1 * (k_field[0] == 0) + 1 * (k_field[0] == N//2)
    
    return k_field, k_vec, fact


def wavenumber_grid_parallel(
        number_of_cells: Tuple[int],
        split_factors: Tuple[int]
        ) -> Array:
    """_summary_

    :param number_of_cells: _description_
    :type number_of_cells: Tuple[int]
    :param split_factors: _description_
    :type split_factors: Tuple[int]
    :return: _description_
    :rtype: Array
    """
    k_split = []
    for i, N in enumerate(number_of_cells):
        ki = jnp.fft.fftfreq(N, 1./N)
        ki_split = split_cell_centers_xi(ki, split_factors, i)
        device_id = jax.lax.axis_index(axis_name="i")
        ki_device = ki_split[device_id]
        k_split.append(ki_device)
    k_field = jnp.array(jnp.meshgrid(*k_split, indexing="ij"), dtype=jnp.int32)
    return k_field
    