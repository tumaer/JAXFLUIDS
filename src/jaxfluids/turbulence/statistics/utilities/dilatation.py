from typing import Tuple

import jax
import jax.numpy as jnp
import numpy as np

from jaxfluids.math.fft import parallel_rfft, parallel_irfft, rfft3D, irfft3D

Array = jax.Array

def calculate_dilatation_spectral(
        velocity_hat: Array,
        k_field: Array
    ) -> Array:
    """Calculates the dilatation of the given velocity field
    in spectral space.

    dilatation = du1/dx1 + du2/dx2 + du3/dx3
    FFT(du_i / dx_i) = 1j * k_i u_hat_i

    :param velocity_hat: Buffer of velocities in spectral space,
        shape = (3,N,N,Nf)
    :type velocity_hat: Array
    :return: Buffer of the dilatational field, shape =(N,N,N)
    :rtype: Array
    """
    dilatation = irfft3D(1j * jnp.sum(k_field * velocity_hat, axis=-4))
    return dilatation

def calculate_dilatation_spectral_parallel(
        velocity_hat: Array,
        k_field: Array,
        split_factors: Tuple[int],
        split_axis_out: int
    ) -> Array:
    """Calculates the dilatation of the given velocity field
    in spectral space.

    FFT(du_i / dx_i) = 1j * k_i u_hat_i

    :param velocity_hat: _description_
    :type velocity_hat: Array
    :param k_field: _description_
    :type k_field: Array
    :param split_factors: _description_
    :type split_factors: Tuple[int]
    :param split_axis_out: _description_
    :type split_axis_out: int
    :return: _description_
    :rtype: Array
    """
    dilatation_hat = 1j * jnp.sum(k_field * velocity_hat, axis=-4)
    dilatation = parallel_irfft(dilatation_hat, split_factors, split_axis_out)
    return dilatation

def calculate_dilatation(
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

def calculate_dilatation_parallel(
        velocity: Array,
        k_field: Array,
        split_factors: Tuple[int],
    ) -> Array:

    split_axis_in = np.argmax(np.array(split_factors))
    split_axis_out = np.roll(np.array([0,1]),-1)[split_axis_in]
    split_factors_out = tuple([split_factors[split_axis_in] if i == split_axis_out else 1 for i in range(3)])

    velocity_hat = parallel_rfft(velocity, split_factors, split_axis_out)
    return calculate_dilatation_spectral_parallel(
        velocity_hat, k_field,
        split_factors_out, split_axis_in)
