#*------------------------------------------------------------------------------*
#* JAX-FLUIDS -                                                                 *
#*                                                                              *
#* A fully-differentiable CFD solver for compressible two-phase flows.          *
#* Copyright (C) 2022  Deniz A. Bezgin, Aaron B. Buhendwa, Nikolaus A. Adams    *
#*                                                                              *
#* This program is free software: you can redistribute it and/or modify         *
#* it under the terms of the GNU General Public License as published by         *
#* the Free Software Foundation, either version 3 of the License, or            *
#* (at your option) any later version.                                          *
#*                                                                              *
#* This program is distributed in the hope that it will be useful,              *
#* but WITHOUT ANY WARRANTY; without even the implied warranty of               *
#* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the                *
#* GNU General Public License for more details.                                 *
#*                                                                              *
#* You should have received a copy of the GNU General Public License            *
#* along with this program.  If not, see <https://www.gnu.org/licenses/>.       *
#*                                                                              *
#*------------------------------------------------------------------------------*
#*                                                                              *
#* CONTACT                                                                      *
#*                                                                              *
#* deniz.bezgin@tum.de // aaron.buhendwa@tum.de // nikolaus.adams@tum.de        *
#*                                                                              *
#*------------------------------------------------------------------------------*
#*                                                                              *
#* Munich, April 15th, 2022                                                     *
#*                                                                              *
#*------------------------------------------------------------------------------*

import jax.numpy as jnp

# TODO MEMORY FOOTPRINT OF THESE FUNCTIONS IS HIGH - LOOPS ?

def move_source_to_target_ii(source_array: jnp.ndarray, normal_sign: jnp.ndarray, axis: int) -> jnp.ndarray:
    """Moves the source array in positive normal direction within the ii plane.

    :param source_array: Source array buffer
    :type source_array: jnp.ndarray
    :param normal_sign: Normal sign buffer
    :type normal_sign: jnp.ndarray
    :param axis: axis i
    :type axis: int
    :return: Moved source array in ii plane
    :rtype: jnp.ndarray
    """
    array_plus  = jnp.roll(source_array,  1, -3 + axis) * jnp.where(jnp.roll(normal_sign[axis],  1, -3 + axis) > 0, 1, 0)
    array_minus = jnp.roll(source_array, -1, -3 + axis) * jnp.where(jnp.roll(normal_sign[axis], -1, -3 + axis) < 0, 1, 0)
    array = array_plus + array_minus
    return array

def move_source_to_target_ij(source_array: jnp.ndarray, normal_sign: jnp.ndarray, axis_i: int, axis_j: int) -> jnp.ndarray:
    normal_sign_i_plus_j_plus   = jnp.roll(jnp.roll(normal_sign,  1, -3 + axis_i),  1, -3 + axis_j)
    normal_sign_i_plus_j_minus  = jnp.roll(jnp.roll(normal_sign,  1, -3 + axis_i), -1, -3 + axis_j)
    normal_sign_i_minus_j_plus  = jnp.roll(jnp.roll(normal_sign, -1, -3 + axis_i),  1, -3 + axis_j)
    normal_sign_i_minus_j_minus = jnp.roll(jnp.roll(normal_sign, -1, -3 + axis_i), -1, -3 + axis_j)
    array_i_plus_j_plus         = jnp.roll(jnp.roll(source_array,  1, -3 + axis_i),  1, -3 + axis_j) * jnp.where((normal_sign_i_plus_j_plus[axis_i]   > 0) & (normal_sign_i_plus_j_plus[axis_j]   > 0), 1, 0)
    array_i_plus_j_minus        = jnp.roll(jnp.roll(source_array,  1, -3 + axis_i), -1, -3 + axis_j) * jnp.where((normal_sign_i_plus_j_minus[axis_i]  > 0) & (normal_sign_i_plus_j_minus[axis_j]  < 0), 1, 0)
    array_i_minus_j_plus        = jnp.roll(jnp.roll(source_array, -1, -3 + axis_i),  1, -3 + axis_j) * jnp.where((normal_sign_i_minus_j_plus[axis_i]  < 0) & (normal_sign_i_minus_j_plus[axis_j]  > 0), 1, 0)
    array_i_minus_j_minus       = jnp.roll(jnp.roll(source_array, -1, -3 + axis_i), -1, -3 + axis_j) * jnp.where((normal_sign_i_minus_j_minus[axis_i] < 0) & (normal_sign_i_minus_j_minus[axis_j] < 0), 1, 0)
    array = array_i_plus_j_plus + array_i_plus_j_minus + array_i_minus_j_plus + array_i_minus_j_minus
    return array

def move_source_to_target_ijk(source_array: jnp.ndarray, normal_sign: jnp.ndarray) -> jnp.ndarray:
    normal_sign_i_plus_j_plus_k_plus      = jnp.roll(jnp.roll(jnp.roll(normal_sign,  1, -3),  1, -2),  1, -1)
    normal_sign_i_plus_j_minus_k_plus     = jnp.roll(jnp.roll(jnp.roll(normal_sign,  1, -3), -1, -2),  1, -1)
    normal_sign_i_minus_j_plus_k_plus     = jnp.roll(jnp.roll(jnp.roll(normal_sign, -1, -3),  1, -2),  1, -1)
    normal_sign_i_minus_j_minus_k_plus    = jnp.roll(jnp.roll(jnp.roll(normal_sign, -1, -3), -1, -2),  1, -1)
    normal_sign_i_plus_j_plus_k_minus     = jnp.roll(jnp.roll(jnp.roll(normal_sign,  1, -3),  1, -2), -1, -1)
    normal_sign_i_plus_j_minus_k_minus    = jnp.roll(jnp.roll(jnp.roll(normal_sign,  1, -3), -1, -2), -1, -1)
    normal_sign_i_minus_j_plus_k_minus    = jnp.roll(jnp.roll(jnp.roll(normal_sign, -1, -3),  1, -2), -1, -1)
    normal_sign_i_minus_j_minus_k_minus   = jnp.roll(jnp.roll(jnp.roll(normal_sign, -1, -3), -1, -2), -1, -1)
    array_i_plus_j_plus_k_plus      = jnp.roll(jnp.roll(jnp.roll(source_array,  1, -3),  1, -2),  1, -1) * jnp.where((normal_sign_i_plus_j_plus_k_plus[0]    > 0) & (normal_sign_i_plus_j_plus_k_plus[1]    > 0) & (normal_sign_i_plus_j_plus_k_plus[2]    > 0), 1, 0)    
    array_i_plus_j_minus_k_plus     = jnp.roll(jnp.roll(jnp.roll(source_array,  1, -3), -1, -2),  1, -1) * jnp.where((normal_sign_i_plus_j_minus_k_plus[0]   > 0) & (normal_sign_i_plus_j_minus_k_plus[1]   < 0) & (normal_sign_i_plus_j_minus_k_plus[2]   > 0), 1, 0)
    array_i_minus_j_plus_k_plus     = jnp.roll(jnp.roll(jnp.roll(source_array, -1, -3),  1, -2),  1, -1) * jnp.where((normal_sign_i_minus_j_plus_k_plus[0]   < 0) & (normal_sign_i_minus_j_plus_k_plus[1]   > 0) & (normal_sign_i_minus_j_plus_k_plus[2]   > 0), 1, 0)
    array_i_minus_j_minus_k_plus    = jnp.roll(jnp.roll(jnp.roll(source_array, -1, -3), -1, -2),  1, -1) * jnp.where((normal_sign_i_minus_j_minus_k_plus[0]  < 0) & (normal_sign_i_minus_j_minus_k_plus[1]  < 0) & (normal_sign_i_minus_j_minus_k_plus[2]  > 0), 1, 0)
    array_i_plus_j_plus_k_minus     = jnp.roll(jnp.roll(jnp.roll(source_array,  1, -3),  1, -2), -1, -1) * jnp.where((normal_sign_i_plus_j_plus_k_minus[0]   > 0) & (normal_sign_i_plus_j_plus_k_minus[1]   > 0) & (normal_sign_i_plus_j_plus_k_minus[2]   < 0), 1, 0)
    array_i_plus_j_minus_k_minus    = jnp.roll(jnp.roll(jnp.roll(source_array,  1, -3), -1, -2), -1, -1) * jnp.where((normal_sign_i_plus_j_minus_k_minus[0]  > 0) & (normal_sign_i_plus_j_minus_k_minus[1]  < 0) & (normal_sign_i_plus_j_minus_k_minus[2]  < 0), 1, 0)
    array_i_minus_j_plus_k_minus    = jnp.roll(jnp.roll(jnp.roll(source_array, -1, -3),  1, -2), -1, -1) * jnp.where((normal_sign_i_minus_j_plus_k_minus[0]  < 0) & (normal_sign_i_minus_j_plus_k_minus[1]  > 0) & (normal_sign_i_minus_j_plus_k_minus[2]  < 0), 1, 0)
    array_i_minus_j_minus_k_minus   = jnp.roll(jnp.roll(jnp.roll(source_array, -1, -3), -1, -2), -1, -1) * jnp.where((normal_sign_i_minus_j_minus_k_minus[0] < 0) & (normal_sign_i_minus_j_minus_k_minus[1] < 0) & (normal_sign_i_minus_j_minus_k_minus[2] < 0), 1, 0)
    array = array_i_plus_j_plus_k_plus + array_i_plus_j_minus_k_plus + array_i_minus_j_plus_k_plus + array_i_minus_j_minus_k_plus + \
            array_i_plus_j_plus_k_minus + array_i_plus_j_minus_k_minus + array_i_minus_j_plus_k_minus + array_i_minus_j_minus_k_minus
    return array

def move_target_to_source_ii(target_array: jnp.ndarray, normal_sign: jnp.ndarray, axis: int) -> jnp.ndarray:
    """Moves the target array in negative normal direction in the ii plane.

    :param target_array: Target array buffer
    :type target_array: jnp.ndarray
    :param normal_sign: Normal sign buffer
    :type normal_sign: jnp.ndarray
    :param axis: axis i
    :type axis: int
    :return: Moved target array in ii plane
    :rtype: jnp.ndarray
    """
    array_plus  = jnp.roll(target_array,  1, -3 + axis) * jnp.where(normal_sign[axis] < 0, 1, 0)
    array_minus = jnp.roll(target_array, -1, -3 + axis) * jnp.where(normal_sign[axis] > 0, 1, 0)
    array = array_plus + array_minus
    return array

def move_target_to_source_ij(target_array: jnp.ndarray, normal_sign: jnp.ndarray, axis_i: int, axis_j: int) -> jnp.ndarray:
    array_i_plus_j_plus     = jnp.roll(jnp.roll(target_array, 1, -3 + axis_i), 1, -3 + axis_j) * jnp.where((normal_sign[axis_i]   < 0) & (normal_sign[axis_j] < 0), 1, 0)
    array_i_plus_j_minus    = jnp.roll(jnp.roll(target_array, 1, -3 + axis_i), -1, -3 + axis_j) * jnp.where((normal_sign[axis_i]  < 0) & (normal_sign[axis_j] > 0), 1, 0)
    array_i_minus_j_plus    = jnp.roll(jnp.roll(target_array, -1, -3 + axis_i), 1, -3 + axis_j) * jnp.where((normal_sign[axis_i]  > 0) & (normal_sign[axis_j] < 0), 1, 0)
    array_i_minus_j_minus   = jnp.roll(jnp.roll(target_array, -1, -3 + axis_i), -1, -3 + axis_j) * jnp.where((normal_sign[axis_i] > 0) & (normal_sign[axis_j] > 0), 1, 0)
    array = array_i_plus_j_plus + array_i_plus_j_minus + array_i_minus_j_plus + array_i_minus_j_minus
    return array

def move_target_to_source_ijk(target_array: jnp.ndarray, normal_sign: jnp.ndarray) -> jnp.ndarray:
    array_i_plus_j_plus_k_plus      = jnp.roll(jnp.roll(jnp.roll(target_array,  1, -3),  1, -2),  1, -1) * jnp.where((normal_sign[0] < 0) & (normal_sign[1] < 0) & (normal_sign[2] < 0), 1, 0)
    array_i_plus_j_minus_k_plus     = jnp.roll(jnp.roll(jnp.roll(target_array,  1, -3), -1, -2),  1, -1) * jnp.where((normal_sign[0] < 0) & (normal_sign[1] > 0) & (normal_sign[2] < 0), 1, 0)
    array_i_minus_j_plus_k_plus     = jnp.roll(jnp.roll(jnp.roll(target_array, -1, -3),  1, -2),  1, -1) * jnp.where((normal_sign[0] > 0) & (normal_sign[1] < 0) & (normal_sign[2] < 0), 1, 0)
    array_i_minus_j_minus_k_plus    = jnp.roll(jnp.roll(jnp.roll(target_array, -1, -3), -1, -2),  1, -1) * jnp.where((normal_sign[0] > 0) & (normal_sign[1] > 0) & (normal_sign[2] < 0), 1, 0)
    array_i_plus_j_plus_k_minus     = jnp.roll(jnp.roll(jnp.roll(target_array,  1, -3),  1, -2), -1, -1) * jnp.where((normal_sign[0] < 0) & (normal_sign[1] < 0) & (normal_sign[2] > 0), 1, 0)
    array_i_plus_j_minus_k_minus    = jnp.roll(jnp.roll(jnp.roll(target_array,  1, -3), -1, -2), -1, -1) * jnp.where((normal_sign[0] < 0) & (normal_sign[1] > 0) & (normal_sign[2] > 0), 1, 0)
    array_i_minus_j_plus_k_minus    = jnp.roll(jnp.roll(jnp.roll(target_array, -1, -3),  1, -2), -1, -1) * jnp.where((normal_sign[0] > 0) & (normal_sign[1] < 0) & (normal_sign[2] > 0), 1, 0)
    array_i_minus_j_minus_k_minus   = jnp.roll(jnp.roll(jnp.roll(target_array, -1, -3), -1, -2), -1, -1) * jnp.where((normal_sign[0] > 0) & (normal_sign[1] > 0) & (normal_sign[2] > 0), 1, 0)
    array = array_i_plus_j_plus_k_plus + array_i_plus_j_minus_k_plus + array_i_minus_j_plus_k_plus + array_i_minus_j_minus_k_plus + \
            array_i_plus_j_plus_k_minus + array_i_plus_j_minus_k_minus + array_i_minus_j_plus_k_minus + array_i_minus_j_minus_k_minus
    return array
