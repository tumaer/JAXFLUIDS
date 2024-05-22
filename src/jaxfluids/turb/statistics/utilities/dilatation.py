from typing import Tuple

import jax
import jax.numpy as jnp
from jax import Array

from jaxfluids.math.parallel_fft import parallel_ifft
from jaxfluids.turb.statistics.utilities.rfft import rfft3D, irfft3D


def calculate_dilatation_spectral(
        velocity_hat: Array,
        k_field: Array
        ) -> Array:
    """Calculates the dilatation of the given velocity field in spectral space.

    :param velocity_hat: Buffer of velocities in spectral space.
    :type velocity_hat: Array
    :return: Buffer of the dilatational field.
    :rtype: Array
    """
    # dilatation_hat = 1j * (k_field[0] * velocity_hat[0] \
    #     + k_field[1] * velocity_hat[1] \
    #     + k_field[2] * velocity_hat[2])
    # dilatation = jnp.fft.irfftn(dilatation_hat, axes=(2,1,0))

    dilatation_hat = 1j * jnp.sum(k_field * velocity_hat, axis=-4)
    dilatation = irfft3D(dilatation_hat)
    return dilatation

def calculate_dilatation_spectral_parallel(
        velocity_hat: Array,
        k_field: Array,
        split_factors: Tuple[int],
        split_axis_out: int
        ) -> Array:
    """Calculates the dilatation of the given velocity field in spectral space.

    :param velocity_hat: Buffer of velocities in spectral space.
    :type velocity_hat: Array
    :return: Buffer of the dilatational field.
    :rtype: Array
    """
    # dilatation_hat = 1j * (k_field[0] * velocity_hat[0] \
    #     + k_field[1] * velocity_hat[1] \
    #     + k_field[2] * velocity_hat[2])
    # dilatation = jnp.fft.irfftn(dilatation_hat, axes=(2,1,0))

    dilatation_hat = 1j * jnp.sum(k_field * velocity_hat, axis=-4)
    dilatation = parallel_ifft(dilatation_hat, split_factors, split_axis_out)
    return dilatation

def _calculate_dilatation_spectral(
        velocity_hat: Array,
        k_field: Array
        ) -> Array:
    if velocity_hat.ndim == 4:
        return calculate_dilatation_spectral(velocity_hat, k_field)
    
    if velocity_hat.ndim == 5:
        return jax.vmap(calculate_dilatation_spectral, in_axes=(0,None),
            out_axes=0)(velocity_hat, k_field)

def calculate_dilatation_physical(
        velocity: Array, 
        k_field: Array
        ) -> Array:
    """_summary_

    Calculates dilatation in spectral space 

    velocity: (3, Nx, Ny, Nz) array

    dilatation: (Nx, Ny, Nz) array
    dilatation = du1/dx1 + du2/dx2 + du3/dx3

    :param velocity: Buffer of velocities in physical space.
    :type velocity: Array
    :return: Buffer of dilatational field.
    :rtype: Array
    """
    
    velocity_hat = rfft3D(velocity)
    return calculate_dilatation_spectral(velocity_hat, k_field)

