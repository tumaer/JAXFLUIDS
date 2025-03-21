from typing import Dict, List, Tuple

import jax
import jax.numpy as jnp
import numpy as np

from jaxfluids.domain.helper_functions import split_cell_centers_xi

Array = jax.Array

def real_wavenumber_grid(N: int) -> Array:
    """Creates the grid of real wave number
    vectors where the last axis is the real
    wave number axis, i.e., the shape of the
    returned array is (3,N,N,Nf+1).

    :param N: Number of cells
    :type N: int
    :return: Real wavenumber vector grid, Shape = (3,N,N,Nf+1)
    :rtype: Array
    """
    Nf = N//2 + 1
    k = jnp.fft.fftfreq(N, 1/N).astype(int)
    k_real = jnp.arange(Nf).astype(int)
    k_field = jnp.array(jnp.meshgrid(k,k,k_real, indexing="ij"))
    return k_field

def real_wavenumber_grid_np(N: int) -> np.ndarray:
    """Numpy version of real_wavenumber_grid.
    Creates the grid of real wave number
    vectors where the last axis is the real
    wave number axis, i.e., the shape of the
    returned array is (3,N,N,Nf+1).

    :param N: Number of cells
    :type N: int
    :return: Real wavenumber vector grid, Shape = (3,N,N,Nf+1)
    :rtype: np.ndarray
    """
    Nf = N//2 + 1
    k = np.fft.fftfreq(N, 1/N).astype(int)
    k_real = np.arange(Nf, dtype=int)
    k_field = np.array(np.meshgrid(k,k,k_real, indexing="ij"))
    return k_field

def real_wavenumber_grid_parallel(
        number_of_cells: Tuple[int],
        split_factors: Tuple[int],
        is_number_of_cells_real: bool = False
    ) -> Array:
    """Creates the grid of real wave number
    vectors in parallel. The last axis is the real
    wave number axis, i.e., the shape of the
    returned array is (3,N_split_1,N_split_2,Nf+1).
    The last axis must not be split.

    :param N: Number of cells
    :type N: int
    :return: Real wavenumber vector grid, 
        Shape = (3,N_split_1,N_split_2,Nf+1)
    :rtype: Array
    """
    split_axis = np.argmax(np.array(split_factors))
    if split_axis == 2:
        raise RuntimeError

    if is_number_of_cells_real:
        Nx, Ny, Nf = number_of_cells
    else:
        Nx, Ny, Nz = number_of_cells
        Nf = Nz//2 + 1

    k_real = jnp.arange(Nf).astype(int)

    k_split = []
    for i, N in enumerate((Nx, Ny,)):
        ki = jnp.moveaxis(jnp.fft.fftfreq(N, 1./N).astype(int).reshape(-1,1,1), 0, i)
        ki_split = split_cell_centers_xi(ki, split_factors, i)
        device_id = jax.lax.axis_index(axis_name="i")
        ki_device = jnp.squeeze(ki_split[device_id])
        k_split.append(ki_device)

    k_field = jnp.array(jnp.meshgrid(*k_split,k_real, indexing="ij"))
    return k_field

def wavenumber_grid(N: int) -> Array:
    """Creates the grid of wave number
    vectors, i.e., the shape of the
    returned array is (3,N,N,N).

    :param N: _description_
    :type N: int
    :return: _description_
    :rtype: Array
    """
    k = jnp.fft.fftfreq(N, 1/N).astype(int)
    k_field = jnp.array(jnp.meshgrid(k,k,k, indexing="ij"))
    return k_field

def wavenumber_grid_parallel(
        number_of_cells: Tuple[int],
        split_factors: Tuple[int]
        ) -> Array:
    """Initializes wavenumber grid in parallel.

    :param number_of_cells: _description_
    :type number_of_cells: Tuple[int]
    :param split_factors: _description_
    :type split_factors: Tuple[int]
    :return: _description_
    :rtype: Array
    """
    k_split = []
    for i, N in enumerate(number_of_cells):
        ki = jnp.moveaxis(jnp.fft.fftfreq(N, 1./N).astype(int).reshape(-1,1,1), 0, i)
        ki_split = split_cell_centers_xi(ki, split_factors, i)
        device_id = jax.lax.axis_index(axis_name="i")
        ki_device = jnp.squeeze(ki_split[device_id])
        k_split.append(ki_device)
    k_field = jnp.array(jnp.meshgrid(*k_split, indexing="ij"))
    return k_field

def wavenumber_vec(N: int) -> Array:
    return jnp.arange(N)

def factor_real(k_field: Array) -> Array:
    _, Nx, Ny, Nf = k_field.shape
    N_nyquist = Nx // 2 
    fact = 2 * (k_field[2] > 0) * (k_field[2] < N_nyquist) + 1 * (k_field[2] == 0) + 1 * (k_field[2] == N_nyquist)
    return fact

# def compute_real_wavenumber_grid(N: int) ->