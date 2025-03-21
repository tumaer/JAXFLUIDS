import jax
import jax.numpy as jnp
import numpy as np

Array = jax.Array

def rfft3D(velocity: Array) -> Array:
    """Real-valued 3-dimensional FFT. FFT is applied over the
    last three dimensions.

    :param velocity: _description_
    :type velocity: Array
    :return: _description_
    :rtype: Array
    """
    return jnp.fft.rfftn(velocity, axes=(-3,-2,-1))

def irfft3D(buffer: Array) -> Array:
    """Real-valued 3-dimensional inverse FFT. FFT is applied over the
    last three dimensions.

    :param buffer: _description_
    :type buffer: Array
    :return: _description_
    :rtype: Array
    """
    return jnp.fft.irfftn(buffer, axes=(-3,-2,-1))

def rfft3D_np(velocity: np.ndarray) -> np.ndarray:
    return np.fft.rfftn(velocity, axes=(-3,-2,-1))

def irfft3D_np(buffer: np.ndarray) -> np.ndarray:
    return np.fft.irfftn(buffer, axes=(-3,-2,-1))