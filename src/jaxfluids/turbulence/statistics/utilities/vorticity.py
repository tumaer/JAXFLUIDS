from typing import Tuple

import numpy as np
import jax
import jax.numpy as jnp

from jaxfluids.math.fft import parallel_rfft, parallel_irfft, rfft3D, irfft3D

Array = jax.Array

def calculate_vorticity_spectral(
        velocity_hat: Array,
        k_field: Array
    ) -> Array:
    """Calculates the vortiticy of the input velocity field.
    Calculation done in spectral space.

    omega = (du3/dx2 - du2/dx3, du1/dx3 - du3/dx1, du2/dx1 - du1/dx2)

    :param velocity_hat: Buffer of velocities in spectral space.
    :type velocity_hat: Array
    :return: Vorticity vector in physical space.
    :rtype: Array
    """
    kx, ky, kz = k_field
    omega_0 = irfft3D(1j * (ky * velocity_hat[...,2,:,:,:] - kz * velocity_hat[...,1,:,:,:]))
    omega_1 = irfft3D(1j * (kz * velocity_hat[...,0,:,:,:] - kx * velocity_hat[...,2,:,:,:]))
    omega_2 = irfft3D(1j * (kx * velocity_hat[...,1,:,:,:] - ky * velocity_hat[...,0,:,:,:]))
    omega = jnp.stack([omega_0, omega_1, omega_2], axis=-4)
    return omega

def calculate_vorticity_spectral_parallel(
        velocity_hat: Array,
        k_field: Array,
        split_factors: Tuple[int],
        split_axis_out: int
    ) -> Array:
    """Calculates the vortiticy of the input velocity field.
    Calculation done in spectral space.

    omega = (du3/dx2 - du2/dx3, du1/dx3 - du3/dx1, du2/dx1 - du1/dx2)

    :param velocity_hat: Buffer of velocities in spectral space.
    :type velocity_hat: Array
    :return: Vorticity vector in physical space.
    :rtype: Array
    """
    kx, ky, kz = k_field
    omega_0 = parallel_irfft(
        1j * (ky * velocity_hat[...,2,:,:,:] - kz * velocity_hat[...,1,:,:,:]),
        split_factors, split_axis_out)
    omega_1 = parallel_irfft(
        1j * (kz * velocity_hat[...,0,:,:,:] - kx * velocity_hat[...,2,:,:,:]),
        split_factors, split_axis_out)
    omega_2 = parallel_irfft(
        1j * (kx * velocity_hat[...,1,:,:,:] - ky * velocity_hat[...,0,:,:,:]),
        split_factors, split_axis_out)
    omega = jnp.stack([omega_0, omega_1, omega_2], axis=-4)
    return omega

def calculate_vorticity(
        velocity: Array,
        k_field: Array
    ) -> Array:
    """Calculates the vortiticy of the input velocity field. Calculation is 
    done in spectral space.

    omega = [   du3/dx2 - du2/dx3
                du1/dx3 - du3/dx1
                du2/dx1 - du1/dx2]

    :param velocity: Buffer of velocities in physical space.
    :type velocity: Array
    :return: Vorticity vector in physical space.
    :rtype: Array
    """
    velocity_hat = rfft3D(velocity)
    return calculate_vorticity_spectral(velocity_hat, k_field)

def calculate_vorticity_parallel(
        velocity: Array,
        k_field: Array,
        split_factors: Tuple[int],
    ) -> Array:

    split_axis_in = np.argmax(np.array(split_factors))
    split_axis_out = np.roll(np.array([0,1]),-1)[split_axis_in]
    split_factors_out = tuple([split_factors[split_axis_in] if i == split_axis_out else 1 for i in range(3)])

    velocity_hat = parallel_rfft(velocity, split_factors, split_axis_out)
    return calculate_vorticity_spectral_parallel(
        velocity_hat, k_field,
        split_factors_out, split_axis_in)