from typing import List, Tuple, Dict, Callable, Sequence, Union

import jax
import jax.numpy as jnp
from jax import Array
import numpy as np
import h5py

from jaxfluids.config import precision
from jaxfluids.domain.helper_functions import split_buffer_np, reassemble_buffer_np

def create_field_buffer(
        nh: int,
        number_of_cells: Tuple[int],
        dtype: type = jnp.float32,
        leading_dim: Tuple = None,
        ) -> Array:
    """Creates a jax.ndarray field buffer
    with halo cells and specified leading dimension.
    The shape of the buffer is
    (N_l, N_x + 2*N_h, N_y + 2*N_h, N_z + 2*N_h),
    where N_l, N_x, N_y N_z, denote the leading
    dimension and the number of cells in x, y
    and z direction, respectively. If leading_dim
    is None, then the created buffer has
    no leading dimension. The buffer is initialized
    with jaxfluids epsilon.

    :param nh: _description_
    :type nh: int
    :param number_of_cells: _description_
    :type number_of_cells: Tuple[int]
    :param dtype: _description_
    :type dtype: type
    :param leading_dim: _description_, defaults to None
    :type leading_dim: Tuple, optional
    :return: _description_
    :rtype: Array
    """
    shape = tuple([n + 2*nh if n > 1 else 1 for n in number_of_cells])

    if leading_dim == None:
        buffer = jnp.ones(shape, dtype=dtype) * precision.get_eps()
    else:
        if type(leading_dim) == int:
            lead = (leading_dim,)
        else:
            lead = leading_dim
        buffer = jnp.ones(lead + shape, dtype=dtype) * precision.get_eps()

    return buffer

def create_field_buffer_np(
        nh: int,
        number_of_cells: Tuple[int],
        dtype: type = np.float64,
        leading_dim: Tuple = None,
        ) -> np.ndarray:
    """Numpy version of
    create_field_buffer.

    :param nh: _description_
    :type nh: int
    :param number_of_cells: _description_
    :type number_of_cells: Tuple[int]
    :param dtype: _description_
    :type dtype: type
    :param leading_dim: _description_, defaults to None
    :type leading_dim: Tuple, optional
    :return: _description_
    :rtype: np.ndarray
    """
    shape = tuple([n + 2*nh if n > 1 else 1 for n in number_of_cells])

    if leading_dim == None:
        buffer = np.ones(shape, dtype=dtype) * precision.get_eps()
    else:
        if type(leading_dim) == int:
            lead = (leading_dim,)
        else:
            lead = leading_dim
        buffer = np.ones(lead + shape, dtype=dtype) * precision.get_eps()

    return buffer

def load_and_reassemble_buffer(
        quantity: str,
        h5file: Union[h5py.File, Sequence[h5py.File]],
        split_factors_restart: Tuple[int],
        is_vector_buffer: bool = False,
        **kwargs
        ) -> np.ndarray:
    """Loads a decomposed buffer from h5 and
    reassembles it to initialize the simulation.

    :param h5file: _description_
    :type h5file: h5py.File
    :param quantity: _description_
    :type quantity: str
    :param split_factors: _description_
    :type split_factors: Tuple[int]
    :param device_number_of_cells: _description_
    :type device_number_of_cells: Tuple[int]
    :param split_factors_restart: _description_
    :type split_factors_restart: Tuple[int]
    :param dtype: _description_
    :type dtype: type
    :param is_velocity: _description_, defaults to False
    :type is_velocity: bool, optional
    :param active_axes_indices: _description_, defaults to None
    :type active_axes_indices: Tuple[int], optional
    :raises RuntimeError: _description_
    :return: _description_
    :rtype: np.ndarray
    """
    buffer = load_buffer(quantity, h5file, is_vector_buffer)
    buffer = reassemble_buffer_np(buffer, split_factors_restart)
    return buffer

def load_and_redecompose_buffer(
        quantity: str,
        h5file: Union[h5py.File, Sequence[h5py.File]],
        split_factors: Tuple[int],
        split_factors_restart: Tuple[int],
        is_vector_buffer: bool = False,
        **kwargs
        ) -> np.ndarray:
    """Loads a decomposed buffer from h5,
    reassembles and decomposes the buffer.

    :param h5file: _description_
    :type h5file: h5py.File
    :param quantity: _description_
    :type quantity: str
    :param split_factors: _description_
    :type split_factors: Tuple[int]
    :param device_number_of_cells: _description_
    :type device_number_of_cells: Tuple[int]
    :param split_factors_restart: _description_
    :type split_factors_restart: Tuple[int]
    :param dtype: _description_
    :type dtype: type
    :param is_velocity: _description_, defaults to False
    :type is_velocity: bool, optional
    :param active_axes_indices: _description_, defaults to None
    :type active_axes_indices: Tuple[int], optional
    :return: _description_
    :rtype: np.ndarray
    """
    buffer = load_and_reassemble_buffer(quantity, h5file, split_factors_restart, is_vector_buffer)
    buffer = split_buffer_np(buffer, split_factors)
    return buffer

def load_and_decompose_buffer(
        quantity: str,
        h5file: h5py.File,
        split_factors: Tuple[int],
        is_vector_buffer: bool = False,
        **kwargs
        ) -> np.ndarray:
    """Loads a non decomposed buffer from h5
    and performs a domain decomposition.

    :param h5file: _description_
    :type h5file: h5py.File
    :param quantity: _description_
    :type quantity: str
    :param split_factors: _description_
    :type split_factors: Tuple[int]
    :param device_number_of_cells: _description_
    :type device_number_of_cells: Tuple[int]
    :param split_factors_restart: _description_
    :type split_factors_restart: Tuple[int]
    :param dtype: _description_
    :type dtype: type
    :param is_velocity: _description_, defaults to False
    :type is_velocity: bool, optional
    :param active_axes_indices: _description_, defaults to None
    :type active_axes_indices: Tuple[int], optional
    :raises RuntimeError: _description_
    :return: _description_
    :rtype: np.ndarray
    """
    buffer = load_buffer(quantity, h5file, is_vector_buffer)
    buffer = split_buffer_np(buffer, split_factors)
    return buffer

def load_buffer(
        quantity: str,
        h5file: Tuple[h5py.File],
        is_vector_buffer: bool = False,
        **kwargs
        ) -> np.ndarray:
    """Loads a buffer from h5 and maintains
    its domain decomposition.

    :param quantity: _description_
    :type quantity: str
    :param h5file: _description_
    :type h5file: h5py.File
    :param is_vector_buffer: _description_, defaults to False
    :type is_vector_buffer: bool, optional
    :return: _description_
    :rtype: np.ndarray
    """

    host_count = len(h5file)
    buffer_list = []
    for i in range(host_count):
        buffer = h5file[i][quantity][:]
        if is_vector_buffer:
            buffer = np.moveaxis(buffer, -1, -4)
        buffer = np.swapaxes(buffer, -1, -3)
        buffer_list.append(buffer)
    buffer = np.concatenate(buffer_list, axis=0)

    return buffer


def get_load_function(
        is_parallel: bool,
        is_parallel_restart: bool,
        split_factors: np.ndarray,
        split_factors_restart: np.ndarray,
        ) -> Tuple[Callable, str]:
    """Identifies if the h5 restart file buffers
    must be reassembled, decomposed, redecomposed
    or kept in terms of spatial dimensions.

    :param is_parallel: _description_
    :type is_parallel: bool
    :param is_parallel_restart: _description_
    :type is_parallel_restart: bool
    :param split_factors: _description_
    :type split_factors: np.ndarray
    :param split_factors_restart: _description_
    :type split_factors_restart: np.ndarray
    :return: _description_
    :rtype: Callable
    """
    if is_parallel and is_parallel_restart:
        if (np.array(split_factors) == split_factors_restart).all():
            load_type = "keep"
        else:
            load_type = "redecompose"
    elif is_parallel and not is_parallel_restart:
        load_type = "decompose"
    elif not is_parallel and is_parallel_restart:
        load_type = "reassemble"
    else:
        load_type = "keep"
    load_function_mapping: Dict[str, Callable] = {
        "redecompose": load_and_redecompose_buffer,
        "decompose": load_and_decompose_buffer,
        "reassemble": load_and_reassemble_buffer,
        "keep": load_buffer
        }
    load_function = load_function_mapping[load_type]
    return load_function


def expand_buffers(*buffers: List[np.ndarray], axis: int) -> List[np.ndarray]:
    """Expanding the dimensions of the input buffers.

    :param axis: _description_
    :type axis: int
    :return: _description_
    :rtype: List[np.ndarray]
    """
    buffer_expand = []
    for buffer_i in buffers:
        buffer_expand.append(np.expand_dims(buffer_i, axis=axis))
    return buffer_expand