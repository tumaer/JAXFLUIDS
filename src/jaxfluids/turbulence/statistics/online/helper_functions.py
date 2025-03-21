from typing import Sequence

import jax
import jax.numpy as jnp

Array = jax.Array

def parallel_mean(
        buffer: Array, is_parallel: bool,
        axis: int | Sequence[int] = None,
        keepdims: bool = False) -> Array:

    buffer = jnp.mean(buffer, axis=axis, keepdims=keepdims)
    if is_parallel:
        buffer = jax.lax.pmean(buffer, axis_name="i")
    return buffer

def parallel_sum(
        buffer: Array, is_parallel: bool,
        axis: int | Sequence[int] = None,
        keepdims: bool = False) -> Array:

    buffer= jnp.sum(buffer, axis=axis, keepdims=keepdims)
    if is_parallel:
        buffer = jax.lax.psum(buffer, axis_name="i")
    return buffer

def update_mean(mean_agg: Array, N_agg: int, mean_new: Array, N_new: int) -> Array:
    mean = mean_agg + (mean_new - mean_agg) * N_new / (N_agg + N_new)
    return mean

def update_sum_square(
        sum_of_squares_agg: Array, mean_agg: Array, N_agg: int,
        sum_of_squares_new: Array, mean_new: Array, N_new: int,
        ) -> Array:
    # Chan et al. - 1979 - 
    # UPDATING FORMULAE AND A PAIRWISE ALGORITHM FOR COMPUTING SAMPLE VARIANCES
    # Eq. (2.1b)
    delta = mean_agg - mean_new
    factor = N_agg * N_new / (N_agg + N_new + 1e-100)
    sum_of_squares = sum_of_squares_agg + sum_of_squares_new + factor * delta * delta 
    return sum_of_squares

def update_sum_square_cov(
        sum_of_squares_agg: Array, mean_x_agg: Array, mean_y_agg: Array, N_agg: int,
        sum_of_squares_new: Array, mean_x_new: Array, mean_y_new: Array, N_new: int,
    ) -> Array:
    # Chan et al. - 1979 - 
    # UPDATING FORMULAE AND A PAIRWISE ALGORITHM FOR COMPUTING SAMPLE VARIANCES
    # Eq. (5.3)
    delta_x = mean_x_agg - mean_x_new
    delta_y = mean_y_agg - mean_y_new
    factor = N_agg * N_new / (N_agg + N_new + 1e-100)
    sum_of_squares = sum_of_squares_agg + sum_of_squares_new + factor * delta_x * delta_y
    return sum_of_squares