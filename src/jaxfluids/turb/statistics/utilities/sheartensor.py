from typing import Tuple

import jax
import jax.numpy as jnp
from jax import Array

from jaxfluids.math.parallel_fft import parallel_ifft
from jaxfluids.turb.statistics.utilities.rfft import rfft3D


def calculate_strain(duidj: Array) -> Array:
    """Calculates the strain given the velocity gradient tensor.

    :param duidj: Buffer of velocity gradient.
    :type duidj: Array
    :return: Buffer of strain tensor.
    :rtype: Array
    """

    S_ij = 0.5 * ( duidj + jnp.transpose(duidj, axes=(1,0,2,3,4)) )
    return S_ij

def calculate_sheartensor_spectral(
        velocity_hat: Array,
        k_field: Array
        ) -> Array:
    """Calculates the shear tensor in spectral space.

    dui/dxj = IFFT ( 1j * k_j * u_i_hat  )

    :param velocity_hat: Buffer of velocities in spectral space.
    :type velocity_hat: Array
    :return: Buffer of the shear tensor.
    :rtype: Array
    """

    duidj = [[], [], []]
    for ii in range(3):
        for jj in range(3):
            duidj[ii].append(jnp.fft.irfftn(1j * k_field[jj] * velocity_hat[ii], axes=(2,1,0)))
    return jnp.array(duidj)

def _calculate_sheartensor_spectral(
        velocity_hat: Array,
        k_field: Array
        ) -> Array:
    if velocity_hat.ndim == 4:
        return calculate_sheartensor_spectral(velocity_hat, k_field)
    
    if velocity_hat.ndim == 5:
        return jax.vmap(calculate_sheartensor_spectral, in_axes=(0,None),
            out_axes=0)(velocity_hat, k_field)

def calculate_sheartensor_spectral_parallel(
        velocity_hat: Array,
        k_field: Array,
        split_factors: Tuple[int],
        split_axis_out: int
        ) -> Array:
    """Calculates the shear tensor in spectral space.

    dui/dxj = IFFT ( 1j * k_j * u_i_hat  )

    :param velocity_hat: Buffer of velocities in spectral space.
    :type velocity_hat: Array
    :return: Buffer of the shear tensor.
    :rtype: Array
    """

    duidj = [[], [], []]
    for ii in range(3):
        for jj in range(3):
            duidj[ii].append(parallel_ifft(1j * k_field[jj] * velocity_hat[ii],
                                           split_factors, split_axis_out))
    return jnp.array(duidj)


def _calculate_sheartensor_spectral_parallel(
        velocity_hat: Array,
        k_field: Array,
        split_factors: Tuple[int],
        split_axis_out: int
        ) -> Array:
    if velocity_hat.ndim == 4:
        return calculate_sheartensor_spectral_parallel(velocity_hat, k_field,
                                                       split_factors, split_axis_out)
    
    if velocity_hat.ndim == 5:
        return jax.vmap(calculate_sheartensor_spectral_parallel, in_axes=(0,None,None,None),
                        out_axes=0)(velocity_hat, k_field, split_factors, split_axis_out)


def calculate_sheartensor(self, velocity: Array,
    k_field: Array) -> Array:
    """Calculates the shear tensor in spectral space. Wrapper around 
    self.calculate_sheartensor_spectral().

    duidj = [
        du1/dx1 du1/dx2 du1/dx3
        du2/dx1 du2/dx2 du2/dx3
        du3/dx1 du3/dx2 du3/dx3
    ]

    :param velocity: Buffer of velocities in physical space.
    :type velocity: Array
    :return: Buffer of the shear tensor.
    :rtype: Array
    """
    velocity_hat = rfft3D(velocity)
    return calculate_sheartensor_spectral(velocity_hat, k_field)
