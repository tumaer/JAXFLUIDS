import jax.numpy as jnp
from jax import Array
import numpy as np
import jax
from numpy.fft import rfftn, irfftn

def rfft3D(velocity: Array) -> Array:
    """Real-valued 3-dimensional FFT. FFT is applied over the
    last three dimensions.

    :param velocity: _description_
    :type velocity: Array
    :return: _description_
    :rtype: Array
    """
    return jnp.fft.rfftn(velocity, axes=(-1,-2,-3))

def irfft3D(buffer: Array) -> Array:
    """Real-valued 3-dimensional inverse FFT. FFT is applied over the
    last three dimensions.

    :param buffer: _description_
    :type buffer: Array
    :return: _description_
    :rtype: Array
    """
    return jnp.fft.irfftn(buffer, axes=(-1,-2,-3))

def rfft3D_np(velocity: np.ndarray) -> np.ndarray:
    return rfftn(velocity, axes=(-1,-2,-3))

def irfft3D_np(buffer: np.ndarray) -> np.ndarray:
    return irfftn(buffer, axes=(-1,-2,-3))