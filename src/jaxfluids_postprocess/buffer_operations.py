from typing import List, Tuple
from functools import partial

import jax
import numpy as np, jax.numpy as jnp

def reassemble_buffer(
        buffer: np.ndarray,
        split_factors: Tuple,
        jax_numpy = False,
        keep_transpose = False,
        )-> np.ndarray:
    """Reassembles a decomposed buffer.
    jax_numpy specifies whether jax or
    numpy operators are used.

    :param buffer: Buffer with shape (Ni,Nz+2*Nh,Ny+2*Nh,Nx+2*Nh,...)
        where Ni is the number of subdomains, Nx, Ny, Nz, are
        the number of cells, and Nh are the number of halo
        cells
    :type buffer: Array
    :param split_factors: Specifies the domain decomposition, defaults to None
    :type split_factors: Tuple, optional
    :param nh: Number of halo cells, defaults to None
    :type nh: int, optional
    :return: _description_
    :rtype: Array
    """
    if jax_numpy:
        return reassemble_buffer_jnp(buffer, split_factors, keep_transpose)
    else:
        return reassemble_buffer_np(buffer, split_factors, keep_transpose)

@partial(jax.jit, static_argnums=(1,2))
def reassemble_buffer_jnp(
        buffer: Array,
        split_factors: Tuple,
        keep_transpose: bool
        ) -> Array:
    shape = buffer.shape
    reshape = tuple(split_factors) + shape[1:]
    buffer = jnp.reshape(buffer, reshape)
    buffer = jnp.concatenate([buffer[i] for i in range(split_factors[0])], axis=4)
    buffer = jnp.concatenate([buffer[i] for i in range(split_factors[1])], axis=2)
    buffer = jnp.concatenate([buffer[i] for i in range(split_factors[2])], axis=0)
    if not keep_transpose:
        buffer = jnp.transpose(buffer)
    return buffer

def reassemble_buffer_np(
        buffer: np.ndarray,
        split_factors: Tuple,
        keep_transpose: bool
        ) -> np.ndarray:
    shape = buffer.shape
    reshape = tuple(split_factors) + shape[1:]
    buffer = np.reshape(buffer, reshape)
    buffer = np.concatenate([buffer[i] for i in range(split_factors[0])], axis=4)
    buffer = np.concatenate([buffer[i] for i in range(split_factors[1])], axis=2)
    buffer = np.concatenate([buffer[i] for i in range(split_factors[2])], axis=0)
    if not keep_transpose:
        buffer = np.transpose(buffer)
    return buffer

def split_subdomain_dimensions(
    buffer: np.ndarray,
    split_factors: Tuple
    ) -> np.ndarray:
    """Splits up the subdomain dimensions of the buffer.

    :param buffer: _description_
    :type buffer: np.ndarray
    :return: _description_
    :rtype: np.ndarray
    """
    shape = buffer.shape
    reshape = tuple(split_factors) + shape[1:]
    buffer = buffer.reshape(reshape)
    return buffer

def flatten_subdomain_dimensions(
    buffer: np.ndarray
    ) -> np.ndarray:
    """Flattens the subdomain dimensions of the buffer.

    :param buffer: _description_
    :type buffer: np.ndarray
    :return: _description_
    :rtype: List
    """
    shape = buffer.shape
    reshape = (-1,) + shape[3:]
    buffer = buffer.reshape(reshape)
    return buffer