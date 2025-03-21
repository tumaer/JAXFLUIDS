from typing import List, Tuple, Dict, Callable, Sequence, Union
import os
import warnings

import h5py
import jax
import jax.numpy as jnp
import json
import numpy as np


from jaxfluids.config import precision
from jaxfluids.domain.domain_information import DomainInformation
from jaxfluids.equation_information import EquationInformation
from jaxfluids.halos.halo_manager import HaloManager
from jaxfluids.domain.helper_functions import split_buffer_np, reassemble_buffer_np
from jaxfluids.math.interpolation.linear import linear_interpolation_scattered

Array = jax.Array

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
    if is_parallel and is_parallel_restart: # TODO what happens here if same split different number of hosts ? 
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

def interpolate(
        field_type: str,
        cell_centers_interp: Tuple[Array],
        buffer_sample: Array,
        time_sample: float,
        domain_information: DomainInformation,
        equation_information: EquationInformation,
        halo_manager: HaloManager,
        dtype: np.float32 | np.float64,
    ) -> Array:

    active_axes_indices = domain_information.active_axes_indices
    nh_conservatives = domain_information.nh_conservatives
    device_number_of_cells = domain_information.device_number_of_cells
    nhx,nhy,nhz = domain_information.domain_slices_conservatives

    if field_type == "MATERIAL":
        no_primes = equation_information.no_primes
        ids_velocity = equation_information.ids_velocity
        inactive_axes_indices = domain_information.inactive_axes_indices
        inactive_velocity_indices = [ids_velocity[i] for i in range(3) if i in inactive_axes_indices]
        prime_indices = []
        for i in range(no_primes):
            if i not in inactive_velocity_indices:
                prime_indices.append(i)
        prime_indices = tuple(prime_indices)

        buffer = create_field_buffer(nh_conservatives, device_number_of_cells,
                                     dtype, no_primes)
        buffer = buffer.at[prime_indices,...,nhx,nhy,nhz].set(buffer_sample)
        buffer = halo_manager.perform_halo_update_material(
            buffer, time_sample, True, True)
        buffer = buffer[prime_indices,...]
    
    elif field_type == "SOLIDS":
        buffer = create_field_buffer(nh_conservatives, device_number_of_cells,
                                     dtype, buffer_sample.shape[0])
        buffer = buffer.at[...,nhx,nhy,nhz].set(buffer_sample)
        buffer = halo_manager.perform_halo_update_solids(
            buffer, time_sample, True, True)

    else:
        raise NotImplementedError

    cell_centers_halos = domain_information.get_device_cell_centers_halos()

    number_of_cells_interp = tuple([xi.size for xi in cell_centers_interp])
    shape = (-1,) + number_of_cells_interp
    cell_centers_interp = [xi.flatten() for i, xi in enumerate(cell_centers_interp) if i in active_axes_indices]
    mesh_grid = jnp.meshgrid(*cell_centers_interp, indexing="ij")
    ip_position = jnp.stack([xi.flatten() for xi in mesh_grid], axis=-1)

    buffer_interp = linear_interpolation_scattered(
        interpolation_position=ip_position,
        field_buffer=buffer,
        cell_centers=cell_centers_halos)
    buffer_interp = buffer_interp.reshape(*shape)

    return buffer_interp

def get_h5file_list(
        restart_file_path: bool,
        process_id: int,
        is_equal_decomposition_multihost: bool
        ) -> Tuple[h5py.File]:
    """Assigns the required restart files to the hosts

    :param restart_file_path: _description_
    :type restart_file_path: bool
    :param process_id: _description_
    :type process_id: int
    :param is_equal_decomposition_multihost: _description_
    :type is_equal_decomposition_multihost: bool
    :return: _description_
    :rtype: Tuple[h5py.File]
    """
    
    h5file_basename = os.path.split(restart_file_path)[-1]
    if "proc" in h5file_basename:
        h5file_path = os.path.split(restart_file_path)[0]
        time_string = h5file_basename.split("_")[-1]
        if is_equal_decomposition_multihost:
            # NOTE for multihost with equal decomposition, each host only loads its respective .h5 file
            file_name = f"data_proc{process_id:d}_{time_string:s}"
            file_path = os.path.join(h5file_path, file_name)
            h5file = h5py.File(file_path, "r")
            h5file_list = [h5file]
        else:
            # NOTE for multihost with different decomposition, each host must load the entire
            # domain and redecompose it, this could become a CPU memory bottleneck for large domains
            h5file_names = [file for file in os.listdir(h5file_path) if time_string in file]
            h5file_list = []
            for i in range(len(h5file_names)):
                file_name = f"data_proc{i:d}_{time_string:s}"
                file_path = os.path.join(h5file_path, file_name)
                h5file = h5py.File(file_path, "r")
                h5file_list.append(h5file)
    else:
        h5file = h5py.File(restart_file_path, "r")
        h5file_list = [h5file]
    
    return tuple(h5file_list)


def parse_restart_files(
        restart_file_path: str
        ):
    """Parses the file path to find restart .h5 file.
    If restart_file_path is a directory,
    the latest time snapshot is chosen.
    For multihost restart, proc0 is chosen.

    :param restart_file_path: _description_
    :type restart_file_path: str
    :return: _description_
    :rtype: _type_
    """

    if os.path.isfile(restart_file_path):
        assert_string = (f"{restart_file_path} is not a valid h5-file. "
                            "For restarting a JAX-Fluids simulation, restart_file_path "
                            "must either point to a valid h5-file or to an output folder "
                            "containing a valid h5-file.")
        assert restart_file_path.endswith(".h5"), assert_string

    elif os.path.isdir(restart_file_path):
        # IF restart_file_path is a folder, try to find last data_*.h5 checkpoint
        warning_string = (f"Restart file path {restart_file_path} points "
        "to a folder and not to a file. By default, the simulation is "
        "restarted from the latest existing checkpoint file in the given folder.")
        warnings.warn(warning_string, RuntimeWarning)

        files = []
        is_multihost_restart = []
        for file in os.listdir(restart_file_path):
            if file.endswith("h5"):
                if "nan" in file:
                    assert_string = (
                        f"Trying to restart from given folder {restart_file_path}. "
                        "However, a nan file was found. Aborting default restart.")
                    assert False, assert_string

                if file.startswith("data_proc"):
                    if file.startswith("data_proc0"):
                        files.append(file)
                elif file.startswith("data_"):
                    files.append(file)
                else:
                    assert_string = (
                        f"Trying to restart from given folder {restart_file_path}. "
                        "However, no data_*.h5 or data_proc*.h5 files found. "
                        "Aborting default restart.")
                    assert False, assert_string
                is_multihost_restart.append(file.startswith("data_proc"))

        assert_string = (f"Trying to restart from given folder {restart_file_path}. "
                            "However, no suitable h5 files were found. "
                            "Aborting default restart.")
        assert len(files) > 0, assert_string

        is_multihost_restart = all(is_multihost_restart)
        if is_multihost_restart:
            times = [float(os.path.splitext(file)[0][11:]) for file in files]
        else:
            times = [float(os.path.splitext(file)[0][5:]) for file in files]

        indices = np.argsort(np.array(times))
        last_file = np.array(files)[indices][-1]
        restart_file_path = os.path.join(restart_file_path, last_file)

    else:
        assert_string = (
            "restart_file_path must be an existing regular file or an existing directory. "
            f"However, {restart_file_path} is neither of the above.")
        assert False, assert_string

    return restart_file_path

