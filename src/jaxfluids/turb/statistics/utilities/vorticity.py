from typing import Dict, List, Tuple

import jax
import jax.numpy as jnp
from jax import Array
import numpy as np

from jaxfluids.math.parallel_fft import parallel_ifft
from jaxfluids.turb.statistics.utilities.rfft import irfft3D,rfft3D

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

    omega_0 = irfft3D(1j * (k_field[1] * velocity_hat[...,2,:,:,:] - k_field[2] * velocity_hat[...,1,:,:,:]))
    omega_1 = irfft3D(1j * (k_field[2] * velocity_hat[...,0,:,:,:] - k_field[0] * velocity_hat[...,2,:,:,:]))
    omega_2 = irfft3D(1j * (k_field[0] * velocity_hat[...,1,:,:,:] - k_field[1] * velocity_hat[...,0,:,:,:]))
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
    
    omega_0 = 1j * (k_field[1] * velocity_hat[...,2,:,:,:] - k_field[2] * velocity_hat[...,1,:,:,:])
    omega_1 = 1j * (k_field[2] * velocity_hat[...,0,:,:,:] - k_field[0] * velocity_hat[...,2,:,:,:])
    omega_2 = 1j * (k_field[0] * velocity_hat[...,1,:,:,:] - k_field[1] * velocity_hat[...,0,:,:,:])
    omega_0 = parallel_ifft(omega_0, split_factors, split_axis_out)
    omega_1 = parallel_ifft(omega_1, split_factors, split_axis_out)
    omega_2 = parallel_ifft(omega_2, split_factors, split_axis_out)
    omega = jnp.stack([omega_0, omega_1, omega_2], axis=-4)
    return omega


def _calculate_vorticity_spectral(
        velocity_hat: Array,
        k_field: Array
        ) -> Array:
    """Calculates the vorticity in spectral space given a FFT velocity vector.
    - shape of velocity_hat has to be (..., 3, Nx, Ny, Nz)
    - shape of k_field has to be (3, Nx, Ny, Nz)

    :param velocity_hat: _description_
    :type velocity_hat: Array
    :param k_field: _description_
    :type k_field: Array
    :raises NotImplementedError: _description_
    :return: _description_
    :rtype: Array
    """
    # if velocity_hat.ndim == 4:
    #     return calculate_vorticity_spectral(velocity_hat, k_field)
    
    # elif velocity_hat.ndim == 5:
    #     return jax.vmap(calculate_vorticity_spectral, in_axes=(0,None),
    #         out_axes=0)(velocity_hat, k_field)

    return calculate_vorticity_spectral(velocity_hat, k_field)

# @partial(jax.jit, static_argnums=(0,))
def calculate_vorticity(
        velocity: Array,
        k_field: Array) -> Array:
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

def vmap_calculate_vorticity(velocity: Array,
    k_field: Array) -> Array:
    return jax.vmap(calculate_vorticity, in_axes=(0,None),
        out_axes=0)(velocity, k_field)