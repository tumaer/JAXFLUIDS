from functools import partial
from typing import List, Tuple

import jax.numpy as jnp
from jax import Array
import numpy as np
import jax

def split_buffer(
        buffer: Array,
        split_factors: Tuple,
        is_tranpose: bool = False
        ) -> Array:
    """Splits a buffer according to the
    domain decomposition specified by
    the split factors. The subdomain 
    dimensions are flattened.

    :param buffer: _description_
    :type buffer: Array
    :param split_factors: _description_
    :type split_factors: Tuple
    :return: _description_
    :rtype: _type_
    """
    if not is_tranpose:
        buffer = jnp.stack(jnp.split(buffer, split_factors[0], axis=-3), axis=0)
        buffer = jnp.stack(jnp.split(buffer, split_factors[1], axis=-2), axis=1)
        buffer = jnp.stack(jnp.split(buffer, split_factors[2], axis=-1), axis=2)
    else:
        buffer = jnp.stack(jnp.split(buffer, split_factors[0], axis=2), axis=0)
        buffer = jnp.stack(jnp.split(buffer, split_factors[1], axis=2), axis=1)
        buffer = jnp.stack(jnp.split(buffer, split_factors[2], axis=2), axis=2)
    buffer = flatten_subdomain_dimensions(buffer)
    return buffer

def split_and_shard_buffer(
        buffer: Array,
        split_factors: Tuple
        ) -> Array:
    """Splits a buffer according to the
    domain decomposition specified by
    the split factors. Subsequently shards 
    the subdomains across availabe
    XLA devices and returns 
    a sharded device array.

    :param buffer: _description_
    :type buffer: Array
    :param split_factors: _description_
    :type split_factors: Tuple
    :return: _description_
    :rtype: Array
    """
    buffer = split_buffer(buffer, split_factors)
    no_subdomains = buffer.shape[0]
    buffer = jax.device_put_sharded([buffer[i] for i in range(no_subdomains)], devices=jax.devices()[:no_subdomains])
    return buffer

def split_buffer_np(
        buffer: np.ndarray,
        split_factors: Tuple
        ) -> np.ndarray:
    """Numpy version of split_buffer()

    :param buffer: _description_
    :type buffer: np.ndarray
    :param split_factors: _description_
    :type split_factors: Tuple
    :return: _description_
    :rtype: np.ndarray
    """
    buffer = np.stack(np.split(buffer, split_factors[0], axis=-3), axis=0)
    buffer = np.stack(np.split(buffer, split_factors[1], axis=-2), axis=1)
    buffer = np.stack(np.split(buffer, split_factors[2], axis=-1), axis=2)
    buffer = flatten_subdomain_dimensions(buffer)
    return buffer

def split_and_shard_buffer_np(
        buffer: np.ndarray,
        split_factors: Tuple
        ) -> np.ndarray:
    """Numpy version of split_and_shard_buffer()

    :param buffer: _description_
    :type buffer: np.ndarray
    :param split_factors: _description_
    :type split_factors: Tuple
    :return: _description_
    :rtype: np.ndarray
    """
    buffer = split_buffer_np(buffer, split_factors)
    no_subdomains = buffer.shape[0]
    buffer = jax.device_put_sharded([buffer[i] for i in range(no_subdomains)], devices=jax.devices()[:no_subdomains])
    return buffer

def split_cell_centers_xi(
        cell_centers_xi: np.ndarray,
        split_factors: Tuple[int],
        axis: int,
        ) -> np.ndarray:
    """Splits the cell centers
    according to the domain 
    decomposition.

    :param cell_centers_xi: _description_
    :type cell_centers_xi: Tuple
    :param split_factors: _description_
    :type split_factors: Tuple
    :return: _description_
    :rtype: Tuple
    """
    shape = (1,1,1) + cell_centers_xi.shape
    cell_centers_xi = cell_centers_xi.reshape(shape)
    for i in range(3):
        split_xi = split_factors[i]
        if i == axis:
            cell_centers_xi = jnp.concatenate(jnp.split(cell_centers_xi, split_xi, -3+axis), axis=axis)
        else:
            cell_centers_xi = jnp.repeat(cell_centers_xi, split_xi, i)
    cell_centers_xi = flatten_subdomain_dimensions(cell_centers_xi)
    return cell_centers_xi

def split_cell_sizes_xi(
        cell_sizes_xi: np.ndarray,
        split_factors: Tuple[int],
        is_mesh_stretching: Tuple[bool],
        axis: Tuple
        ) -> Tuple:
    """Splits the cell sizes
    according to the domain 
    decomposition

    :param cell_centers_xi: _description_
    :type cell_centers_xi: Tuple
    :param split_factors: _description_
    :type split_factors: Tuple
    :return: _description_
    :rtype: Tuple
    """
    shape = (1,1,1) + cell_sizes_xi.shape
    cell_sizes_xi = cell_sizes_xi.reshape(shape)
    for i in range(3):
        split_xi = split_factors[i]
        if i == axis and is_mesh_stretching[i]:
            cell_sizes_xi = jnp.concatenate(jnp.split(cell_sizes_xi, split_xi, -3 + axis), axis=axis)
        else:
            cell_sizes_xi = jnp.repeat(cell_sizes_xi, split_xi, i)
    cell_sizes_xi = flatten_subdomain_dimensions(cell_sizes_xi)
    return cell_sizes_xi

def split_subdomain_dimensions(
        buffer: Array,
        split_factors: Tuple
        ) -> Array:
    """Splits up the subdomain dimensions of the buffer.

    :param buffer: _description_
    :type buffer: Array
    :return: _description_
    :rtype: Array
    """
    shape = buffer.shape
    reshape = tuple(split_factors) + shape[1:]
    buffer = buffer.reshape(reshape)
    return buffer

def flatten_subdomain_dimensions(
        buffer: Array
        ) -> Array:
    """Flattens the subdomain dimensions of the buffer.

    :param buffer: _description_
    :type buffer: Array
    :return: _description_
    :rtype: List
    """
    shape = buffer.shape
    reshape = (-1,) + shape[3:]
    buffer = buffer.reshape(reshape)
    return buffer

def reassemble_buffer(
        buffer: Array,
        split_factors: Tuple,
        nh: int = None,
        is_transpose: bool = False,
        keep_transpose: bool = False
        ) -> Array:
    """Reassembles a decomposed buffer.
    If nh is not None, then the buffer 
    is expected to include nh halo cells.
    Buffer shape must be (Ni,...,Nx+2*Nh,Ny+2*Nh,Nz+2*Nh)
    if is_transpose is False or
    (Ni,Nz+2*Nh,Ny+2*Nh,Nx+2*Nh,...) if
    is_transpose is True.

    :param buffer: _description_
    :type buffer: Array
    :param split_factors: _description_
    :type split_factors: Tuple
    :param nh: _description_, defaults to None
    :type nh: int, optional
    :param is_transpose: _description_, defaults to False
    :type is_transpose: bool, optional
    :return: _description_
    :rtype: Array
    """
    shape = buffer.shape
    reshape = tuple(split_factors) + shape[1:]
    buffer = jnp.reshape(buffer, reshape)
    resolution = buffer.shape[-3:]
    if nh != None:
        domain_slices = jnp.s_[[jnp.s_[nh:-nh] if res > 1 else jnp.s_[:] for res in resolution]]
        buffer  = buffer[...,domain_slices[0],domain_slices[1],domain_slices[2]]
    if is_transpose:
        buffer = jnp.concatenate([buffer[i] for i in range(split_factors[0])], axis=4)
        buffer = jnp.concatenate([buffer[i] for i in range(split_factors[1])], axis=2)
        buffer = jnp.concatenate([buffer[i] for i in range(split_factors[2])], axis=0)
        if not keep_transpose:
            buffer = jnp.transpose(buffer)
    else:
        buffer = jnp.concatenate([buffer[i] for i in range(split_factors[0])], axis=-3)
        buffer = jnp.concatenate([buffer[i] for i in range(split_factors[1])], axis=-2)
        buffer = jnp.concatenate([buffer[i] for i in range(split_factors[2])], axis=-1)
    return buffer


def reassemble_buffer_np(
        buffer: Array,
        split_factors: Tuple,
        nh: int = None,
        is_transpose: bool = False,
        keep_transpose: bool = False
        ) -> Array:
    """Numpy version of reassemble_buffer().

    :param buffer: _description_
    :type buffer: Array
    :param split_factors: _description_
    :type split_factors: Tuple
    :param nh: _description_, defaults to None
    :type nh: int, optional
    :param is_transpose: _description_, defaults to False
    :type is_transpose: bool, optional
    :return: _description_
    :rtype: Array
    """
    shape = buffer.shape
    reshape = tuple(split_factors) + shape[1:]
    buffer = np.reshape(buffer, reshape)
    resolution = buffer.shape[-3:]
    if nh != None:
        domain_slices = np.s_[[np.s_[nh:-nh] if res > 1 else np.s_[:] for res in resolution]]
        buffer  = buffer[...,domain_slices[0],domain_slices[1],domain_slices[2]]
    if is_transpose:
        buffer = np.concatenate([buffer[i] for i in range(split_factors[0])], axis=4)
        buffer = np.concatenate([buffer[i] for i in range(split_factors[1])], axis=2)
        buffer = np.concatenate([buffer[i] for i in range(split_factors[2])], axis=0)
        if not keep_transpose:
            buffer = np.transpose(buffer)
    else:
        buffer = np.concatenate([buffer[i] for i in range(split_factors[0])], axis=-3)
        buffer = np.concatenate([buffer[i] for i in range(split_factors[1])], axis=-2)
        buffer = np.concatenate([buffer[i] for i in range(split_factors[2])], axis=-1)
    return buffer

def reassemble_cell_centers(
        cell_centers: Tuple[Array],
        split_factors: Tuple
        ) -> Tuple[Array]:
    """Reassembles cell centers.

    :param cell_centers: _description_
    :type cell_centers: Tuple[Array]
    :param split_factors: _description_
    :type split_factors: Tuple
    :return: _description_
    :rtype: Array
    """
    cell_centers_xi = []

    for i in range(3):
        xi = cell_centers[i]

        if xi.ndim == 1:
            # NOTE inactive axis squeezed
            xi = split_subdomain_dimensions(xi, split_factors)
            xi = xi[0,0,0]

        else:
            if xi.ndim == 2:
                # NOTE active axis squeezed
                concat_axis = -1
            elif xi.ndim == 4:
                # NOTE active axis non squeezed
                concat_axis = -3+i
            else:
                assert_string = "Reassemble cell centers failed due to wrong shape."
                assert False, assert_string
            
            xi = split_subdomain_dimensions(xi, split_factors)
            if split_factors[i] == 1:
                xi = xi[0,0,0]
            else:
                xi = jnp.concatenate([
                    xi[tuple(np.roll([k,0,0], i))] for k in range(split_factors[i])
                ], axis=concat_axis)

        cell_centers_xi.append(xi)

    return cell_centers_xi

def reassemble_cell_sizes(
        cell_sizes: Tuple[Array],
        split_factors: Tuple
        ) -> Tuple[Array]:
    """Reassembles cell sizes.

    :param cell_centers: _description_
    :type cell_centers: Tuple[Array]
    :param split_factors: _description_
    :type split_factors: Tuple
    :return: _description_
    :rtype: Array
    """
    cell_sizes_xi = []
    for i in range(3):
        dxi = cell_sizes[i]

        if dxi.ndim == 1:
            # NOTE inactive axis or active axis squeezed no mesh stretching
            dxi = split_subdomain_dimensions(dxi, split_factors)
            dxi = dxi[0,0,0]

        elif dxi.ndim == 2:

            # NOTE active axis squeezed mesh stretching
            dxi = split_subdomain_dimensions(dxi, split_factors)
            dxi = jnp.concatenate([
                dxi[tuple(np.roll([k,0,0], i))] for k in range(split_factors[i])
            ], axis=-1)

        elif dxi.ndim == 4:
            dxi = split_subdomain_dimensions(dxi, split_factors)
            if split_factors[i] > 1 and dxi.shape[-3+i] > 1:
                # NOTE active axis non squeezed mesh stretching
                dxi = jnp.concatenate([
                    dxi[tuple(np.roll([k,0,0], i))] for k in range(split_factors[i])
                ], axis=-3+i)
            else:
                # NOTE active axis non squeezed no mesh stretching
                dxi = dxi[0,0,0]

        else:
            assert_string = "Reassemble cell sizes failed due to wrong shape."
            assert False, assert_string
        
        cell_sizes_xi.append(dxi)

    return cell_sizes_xi

def reassemble_cell_faces(
        cell_faces: Tuple[Array],
        split_factors: Tuple
        ) -> Tuple[Array]:
    cell_faces_xi = []
    for i in range(3):
        fxi = cell_faces[i]

        if fxi.ndim == 2:
            # NOTE inactive axis or active axis squeezed
            fxi = split_subdomain_dimensions(fxi, split_factors)
            if split_factors[i] == 1:
                fxi = fxi[0,0,0]
            else:
                fxi_list = []
                for k in range(split_factors[i]):
                    s_1 = tuple(np.roll([k,0,0], i))
                    if k == split_factors[i]-1:
                        s_2 = np.s_[:,]
                    else:
                        s_2 = np.s_[:-1,]
                    s_ = s_1 + s_2
                    fxi_list.append(fxi[s_])
                fxi = jnp.concatenate(fxi_list, axis=-1)

        elif fxi.ndim == 4:
            # NOTE active axis non squeezed
            fxi = split_subdomain_dimensions(fxi, split_factors)
            if split_factors[i] == 1:
                fxi = fxi[0,0,0]
            else:
                fxi_list = []
                for k in range(split_factors[i]):
                    s_1 = tuple(np.roll([k,0,0], i))
                    if k == split_factors[i]-1:
                        s_2 = tuple(np.roll(np.s_[:,:,:], i))
                    else:
                        s_2 = tuple(np.roll(np.s_[:-1,:,:], i))
                    s_ = s_1 + s_2
                    fxi_list.append(fxi[s_])
                fxi = jnp.concatenate(fxi_list, axis=-3+i)

        else:
            assert_string = "Reassemble cell faces failed due to wrong shape."
            assert False, assert_string

        cell_faces_xi.append(fxi)
    return cell_faces_xi