import jax.numpy as jnp
from jax import Array

def move_source_to_target_ii(
        source_array: Array,
        normal_sign: Array,
        axis: int
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

    array = 0.0
    directions = (
        1, -1
    )

    for i in directions:
        normal_sign_shift_ii = jnp.roll(normal_sign[axis],  i, -3 + axis)
        array_shift_ii = jnp.roll(source_array,  i, -3 + axis)
        array += array_shift_ii * (normal_sign_shift_ii*i > 0)

    return array

def move_source_to_target_ij(
        source_array: Array,
        normal_sign: Array,
        axis_i: int,
        axis_j: int
        ) -> Array:

    directions = [
        ( 1, 1),
        ( 1,-1),
        (-1, 1),
        (-1,-1),
    ]
    
    array = 0.0
    for (i,j) in directions:
        normal_sign_shift_ij = jnp.roll(jnp.roll(normal_sign,  i, -3 + axis_i), j , -3 + axis_j)
        array_shift_ij = jnp.roll(jnp.roll(source_array,  i, -3 + axis_i), j, -3 + axis_j)
        mask = (normal_sign_shift_ij[axis_i]*i > 0) & (normal_sign_shift_ij[axis_j]*j > 0)
        array += array_shift_ij * mask

    return array

def move_source_to_target_ijk(
        source_array: Array,
        normal_sign: Array
        ) -> Array:
    
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
    for (i,j,k) in directions:
        normal_sign_shift_ijk = jnp.roll(jnp.roll(
            jnp.roll(normal_sign,  i, -3),
            j, -2), k, -1)
        array_shift_ijk = jnp.roll(jnp.roll(
            jnp.roll(source_array,  i, -3),
            j, -2), k, -1)
        mask = (normal_sign_shift_ijk[0]*i > 0) & (normal_sign_shift_ijk[1]*j > 0) & (normal_sign_shift_ijk[2]*k > 0)
        array += array_shift_ijk * mask

    return array

def move_target_to_source_ii(
        target_array: Array,
        normal_sign: Array,
        axis: int
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

    array = 0.0
    directions = [
        1, -1
    ]

    for i in directions:
        target_shift_ii = jnp.roll(target_array,  i, -3 + axis)
        array += target_shift_ii * (normal_sign[axis]*i < 0)

    return array

def move_target_to_source_ij(
        target_array: Array,
        normal_sign: Array,
        axis_i: int,
        axis_j: int
        ) -> Array:
    
    directions = [
        ( 1, 1),
        ( 1,-1),
        (-1, 1),
        (-1,-1),
    ]
    
    array = 0.0
    for (i,j) in directions:
        target_shift_ij = jnp.roll(jnp.roll(
            target_array,  i, -3 + axis_i), j, -3 + axis_j)
        mask = (normal_sign[axis_i]*i < 0) & (normal_sign[axis_j]*j < 0)
        array += target_shift_ij * mask
    
    return array

def move_target_to_source_ijk(
        target_array: Array,
        normal_sign: Array
        ) -> Array:

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
    for (i,j,k) in directions:
        target_shift_ijk = jnp.roll(jnp.roll(
            jnp.roll(target_array,  i, -3),
            j, -2), k, -1)
        mask = ((normal_sign[0]*i < 0)) & ((normal_sign[1]*j < 0)) & ((normal_sign[2]*k < 0))
        array += target_shift_ijk * mask

    return array


