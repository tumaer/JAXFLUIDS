from typing import Tuple

import jax
import jax.numpy as jnp

Array = jax.Array

def get_slices_ii(
        nh: int,
        axis: int,
        directions: Tuple[int],
        active_axes_indices: Tuple[int],
        offset: int
        ) -> Tuple:
    slices_tuple = []
    for j in directions:
        slices_ = (...,)
        for i in range(3):
            if i == axis:
                start = nh-j-offset
                stop = -nh-j+offset if -nh-j+offset != 0 else None
                slices_ += (jnp.s_[start:stop],)
            else:
                if i in active_axes_indices:
                    start = nh-offset
                    stop = -nh+offset if -nh+offset != 0 else None
                    slices_ += (jnp.s_[start:stop],)
                else:
                    slices_ += (jnp.s_[:],)
        slices_tuple.append(slices_)
    return tuple(slices_tuple)

def get_slices_ij(
        nh: int,
        axis_i: int,
        axis_j: int,
        directions: Tuple[int],
        active_axes_indices: Tuple[int],
        offset: int
        ) -> Tuple:
    slices_tuple = []
    for dir_i, dir_j in directions:
        slices_ = (...,)
        for i in range(3):
            if i == axis_i:
                start = nh-dir_i-offset
                stop = -nh-dir_i+offset if -nh-dir_i+offset != 0 else None
                slices_ += (jnp.s_[start:stop],)
            elif i == axis_j:
                start = nh-dir_j-offset
                stop = -nh-dir_j+offset if -nh-dir_j+offset != 0 else None
                slices_ += (jnp.s_[start:stop],)
            else:
                if i in active_axes_indices:
                    start = nh-offset
                    stop = -nh+offset if -nh+offset != 0 else None
                    slices_ += (jnp.s_[start:stop],)
                else:
                    slices_ += (jnp.s_[:],)
        slices_tuple.append(slices_)
    return tuple(slices_tuple)

def get_slices_ijk(
        nh: int,
        directions: Tuple[int],
        offset: int
        ) -> Tuple:
    slices_tuple = []
    for directions_i in directions:
        slices_ = (...,)
        for i in range(3):
            dir_i = directions_i[i]
            start = nh-dir_i-offset
            stop = -nh-dir_i+offset if -nh-dir_i+offset != 0 else None
            slices_ += (jnp.s_[start:stop],)
        slices_tuple.append(slices_)
    return tuple(slices_tuple)


def move_source_to_target_ii(
        source_array: Array,
        normal_sign: Array,
        axis: int,
        nh: int,
        active_axes_indices: Tuple[int],
        offset: int = 0
        ) -> Array:
    """Moves the source array in positive
    normal direction in axis i direction.

    :param source_array: Source array buffer
    :type source_array: Array
    :param normal_sign: Normal sign buffer
    :type normal_sign: Array
    :param axis: axis i
    :type axis: int
    :return: Moved source array in ii plane
    :rtype: Array
    """

    if source_array.shape[-3:] != normal_sign.shape[-3:]:
        raise RuntimeError

    array = 0.0
    directions = (
        1, -1
    )
    slices_ = get_slices_ii(nh, axis, directions,
                            active_axes_indices, offset=offset)
    for s_, i in zip(slices_, directions):
        normal_sign_shift_ii = normal_sign[axis][s_]
        array_shift_ii = source_array[s_]
        array += array_shift_ii * (normal_sign_shift_ii*i > 0)

    return array

def move_source_to_target_ij(
        source_array: Array,
        normal_sign: Array,
        axis_i: int,
        axis_j: int,
        nh: int,
        active_axes_indices: Tuple[int],
        offset: int = 0
        ) -> Array:

    if source_array.shape[-3:] != normal_sign.shape[-3:]:
        raise RuntimeError

    directions = [
        ( 1, 1),
        ( 1,-1),
        (-1, 1),
        (-1,-1),
    ]
    
    array = 0.0
    slices_ = get_slices_ij(nh, axis_i, axis_j, directions, active_axes_indices, offset=offset)
    for s_, (i,j) in zip(slices_, directions):
        normal_sign_shift_ij = normal_sign[s_]
        array_shift_ij = source_array[s_]
        mask = (normal_sign_shift_ij[axis_i]*i > 0) & (normal_sign_shift_ij[axis_j]*j > 0)
        array += array_shift_ij * mask
    return array

def move_source_to_target_ijk(
        source_array: Array,
        normal_sign: Array,
        nh: int,
        offset: int = 0 
        ) -> Array:
    
    if source_array.shape[-3:] != normal_sign.shape[-3:]:
        raise RuntimeError
    
    directions = [
        ( 1, 1, 1),
        ( 1,-1, 1),
        (-1, 1, 1),
        (-1,-1, 1),
        ( 1, 1,-1),
        ( 1,-1,-1),
        (-1, 1,-1),
        (-1,-1,-1),
    ]

    array = 0.0
    slices_ = get_slices_ijk(nh, directions, offset=offset)
    for s_, (i,j,k) in zip(slices_, directions):
        normal_sign_shift_ijk = normal_sign[s_]
        array_shift_ijk = source_array[s_]
        mask = (normal_sign_shift_ijk[0]*i > 0) & (normal_sign_shift_ijk[1]*j > 0) & (normal_sign_shift_ijk[2]*k > 0)
        array += array_shift_ijk * mask

    return array

def move_target_to_source_ii(
        target_array: Array,
        normal_sign: Array,
        axis: int,
        nh: int,
        active_axes_indices: Tuple[int],
        offset: int = 1
        ) -> Array:
    """Moves the target array in negative normal
    in axis i direction.

    :param target_array: Target array buffer
    :type target_array: Array
    :param normal_sign: Normal sign buffer
    :type normal_sign: Array
    :param axis: axis i
    :type axis: int
    :return: Moved target array in ii plane
    :rtype: Array
    """

    resolution = target_array.shape[-3:]
    resolution = tuple([
        resolution[0]-2*nh+2*offset if 0 in active_axes_indices else 1,
        resolution[1]-2*nh+2*offset if 1 in active_axes_indices else 1,
        resolution[2]-2*nh+2*offset if 2 in active_axes_indices else 1,
        ])
    
    if resolution != normal_sign.shape[-3:]:
        raise RuntimeError

    array = 0.0
    directions = (
        1, -1
    )
    slices_ = get_slices_ii(nh, axis, directions, active_axes_indices, offset)
    for s_, i in zip(slices_, directions):
        target_shift_ii = target_array[s_]
        array += target_shift_ii * (normal_sign[axis]*i < 0)
    return array

def move_target_to_source_ij(
        target_array: Array,
        normal_sign: Array,
        axis_i: int,
        axis_j: int,
        nh: int,
        active_axes_indices: Tuple[int],
        offset: int = 1
        ) -> Array:
    
    resolution = target_array.shape[-3:]
    resolution = tuple([
        resolution[0]-2*nh+2*offset if 0 in active_axes_indices else 1,
        resolution[1]-2*nh+2*offset if 1 in active_axes_indices else 1,
        resolution[2]-2*nh+2*offset if 2 in active_axes_indices else 1,
        ])
    
    if resolution != normal_sign.shape[-3:]:
        raise RuntimeError
    
    directions = [
        ( 1, 1),
        ( 1,-1),
        (-1, 1),
        (-1,-1),
    ]
    
    array = 0.0
    slices_ = get_slices_ij(nh, axis_i, axis_j, directions, active_axes_indices, offset)
    for s_, (i,j) in zip(slices_, directions):
        target_shift_ij = target_array[s_]
        mask = (normal_sign[axis_i]*i < 0) & (normal_sign[axis_j]*j < 0)
        array += target_shift_ij * mask
    
    return array

def move_target_to_source_ijk(
        target_array: Array,
        normal_sign: Array,
        nh: int,
        offset: int = 1
        ) -> Array:

    resolution = target_array.shape[-3:]
    resolution = tuple([
        resolution[0]-2*nh+2*offset,
        resolution[1]-2*nh+2*offset,
        resolution[2]-2*nh+2*offset,
        ])
    
    if resolution != normal_sign.shape[-3:]:
        raise RuntimeError
        
    directions = [
        ( 1, 1, 1),
        ( 1,-1, 1),
        (-1, 1, 1),
        (-1,-1, 1),
        ( 1, 1,-1),
        ( 1,-1,-1),
        (-1, 1,-1),
        (-1,-1,-1),
    ]

    array = 0.0
    slices_ = get_slices_ijk(nh, directions, offset)
    for s_, (i,j,k) in zip(slices_, directions):
        target_shift_ijk = target_array[s_]
        mask = ((normal_sign[0]*i < 0)) & ((normal_sign[1]*j < 0)) & ((normal_sign[2]*k < 0))
        array += target_shift_ijk * mask

    return array