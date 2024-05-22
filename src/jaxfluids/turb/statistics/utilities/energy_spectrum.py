from typing import Tuple

import jax
import jax.numpy as jnp
from jax import Array
import numpy as np

from jaxfluids.math.parallel_fft import parallel_fft
from jaxfluids.turb.statistics.utilities.wavenumber import real_wavenumber_grid, wavenumber_grid_parallel
from jaxfluids.turb.statistics.utilities.rfft import rfft3D


def energy_spectrum_1D_spectral(
        buffer_hat: Array,
        multiplicative_factor: float = 1.0,
        is_scalar_field: bool = False,
    ) -> Array:
    """Calculates the one-dimensional spectral energy spectrum of the input velocity.

    Args:
        velocity_hat (Array): Two-dimensional array. Zero dimension
        is over the velocity components, first dimension is over space.

    Returns:
        Array: [description]
    """
    Nx = buffer_hat.shape[1]
    k = jnp.fft.fftfreq(Nx, 1./Nx)
    kmag = jnp.sqrt(k * k)
    ek = jnp.zeros(Nx)
    shell = (kmag + 0.5).astype(int)

    buffer_hat /= Nx
    if not is_scalar_field:
        abs_energy = jnp.sum(jnp.real(buffer_hat * jnp.conj(buffer_hat)), axis=0)
        # abs_energy = jnp.sum(jnp.abs(buffer_hat * buffer_hat), axis=(-4))
    else:
        abs_energy = jnp.real(buffer_hat * jnp.conj(buffer_hat))
    abs_energy *= multiplicative_factor

    ek = ek.at[shell.flatten()].add(abs_energy.flatten())
    return ek

def energy_spectrum_spectral(
        buffer_hat: Array,
        resolution: Tuple,
        multiplicative_factor: float = 1.0,
        is_scalar_field: bool = False,
    ) -> Array:
    """Calculates the three-dimensional spectral energy spectrum
    of the input velocity. Velocity is in spectral space and has
    shape (3,Nf,Ny,Nz) if buffer.

    :param buffer_hat: Velocity vector in spectral space.
    :type buffer_hat: Array
    :return: Spectral energy spectrum.
    :rtype: Array
    """
    Nx, Ny, Nz = resolution
    k_field, k_vec, fact = real_wavenumber_grid(Nx)
    k_mag_field = jnp.linalg.norm(k_field, axis=0, ord=2)
    shell = (k_mag_field + 0.5).astype(int).flatten()

    buffer_hat /= Nx**3
    if not is_scalar_field:
        abs_energy = jnp.sum(jnp.real(buffer_hat * jnp.conj(buffer_hat)), axis=(-4))
        # abs_energy = jnp.sum(jnp.abs(buffer_hat * buffer_hat), axis=(-4))
    else:
        abs_energy = jnp.real(buffer_hat * jnp.conj(buffer_hat))

    abs_energy *= fact * multiplicative_factor

    n_samples = jnp.zeros(Nx)
    n_samples = n_samples.at[shell].add(fact.flatten())

    energy_spec = jnp.zeros(Nx)
    energy_spec = energy_spec.at[shell].add(abs_energy.flatten())
    energy_spec *= 4 * jnp.pi * k_vec**2 / (n_samples + 1e-10)

    return energy_spec

def energy_spectrum_physical(
        buffer: Array,
        multiplicative_factor: float = 1.0,
        is_scalar_field: bool = False,
    ) -> Array:
    """Calculates the three-dimensional spectral energy spectrum of the input velocity.
    Wrapper around energy_spectrum_spectral()

    :param velocity: Velocity vector in physical space.
    :type velocity: Array
    :return: Spectral energy spectrum.
    :rtype: Array
    """
    resolution = buffer.shape[-3:]
    buffer_hat = rfft3D(buffer)
    return energy_spectrum_spectral(buffer_hat, resolution,
        multiplicative_factor, is_scalar_field)

def energy_spectrum_physical_parallel(
        velocity: Array,
        split_factors: Tuple[int],
        multiplicative_factor: float = 1.0,
        is_scalar_field: bool = False,
    ) -> Array:
    """Computes the energy spectrum from the velocity in
    physical space.

    :param velocity: _description_
    :type velocity: Array
    :param split_factors: _description_
    :type split_factors: Tuple[int]
    :param multiplicative_factor: _description_, defaults to 1.0
    :type multiplicative_factor: float, optional
    :param is_scalar_field: _description_, defaults to False
    :type is_scalar_field: bool, optional
    :return: _description_
    :rtype: Array
    """
    split_axis_in = np.argmax(np.array(split_factors))
    split_axis_out = np.roll(np.array([0,1,2]),-1)[split_axis_in]
    split_factors_out = tuple([split_factors[split_axis_in] if i == split_axis_out else 1 for i in range(3)])
    velocity_hat = parallel_fft(velocity, split_factors, split_axis_out)
    energy_spectrum = energy_spectrum_spectral_parallel(velocity_hat, split_factors_out,
                                                        multiplicative_factor, is_scalar_field)
    return energy_spectrum

def energy_spectrum_spectral_parallel(
        velocity_hat: Array,
        split_factors: Tuple[int],
        multiplicative_factor: float = 1.0,
        is_scalar_field: bool = False,
    ) -> Array:
    """Computes the energy spectrum in parallel.
    Velocity is a split buffer decomposed along
    a single axis.

    :param velocity: _description_
    :type velocity: Array
    :param split_factors: _description_
    :type split_factors: Tuple[int]
    :param number_of_cells: _description_
    :type number_of_cells: Tuple[int]
    :param multiplicative_factor: _description_, defaults to 1.0
    :type multiplicative_factor: float, optional
    :param is_scalar_field: _description_, defaults to False
    :type is_scalar_field: bool, optional
    :return: _description_
    :rtype: Array
    """

    number_of_cells_device = velocity_hat.shape[-3:]
    number_of_cells = tuple([int(number_of_cells_device[i]*split_factors[i]) for i in range(3)])
    k_field = wavenumber_grid_parallel(number_of_cells, split_factors)
    k_mag_field = jnp.linalg.norm(k_field, axis=0, ord=2)
    shell = (k_mag_field + 0.5).astype(int).flatten()

    N = number_of_cells[0]
    k_vec = jnp.arange(N)

    velocity_hat /= N**3
    if not is_scalar_field:
        abs_energy = jnp.sum(jnp.real(velocity_hat * jnp.conj(velocity_hat)), axis=(-4))
    else:
        abs_energy = jnp.real(velocity_hat * jnp.conj(velocity_hat))

    abs_energy *= multiplicative_factor

    n_samples = jnp.zeros(N)
    n_samples = n_samples.at[shell].add(1.0)
    n_samples = jax.lax.psum(n_samples, axis_name="i")

    energy_spec = jnp.zeros(N)
    energy_spec = energy_spec.at[shell].add(abs_energy.flatten())
    energy_spec = jax.lax.psum(energy_spec, axis_name="i")
    energy_spec *= 4 * jnp.pi * k_vec**2 / (n_samples + 1e-10)

    return energy_spec

def vmap_energy_spectrum_spectral(
        buffer_hat: Array,
        resolution: Tuple,
        multiplicative_factor: float = 1.0,
        is_scalar_field: bool = False,
    ) -> Array:
    """Wrapper around energy_spectrum_spectral.

    :param buffer_hat: _description_
    :type buffer_hat: Array
    :param resolution: _description_
    :type resolution: Tuple
    :param multiplicative_factor: _description_, defaults to 1.0
    :type multiplicative_factor: float, optional
    :param is_scalar_field: _description_, defaults to False
    :type is_scalar_field: bool, optional
    :return: _description_
    :rtype: Array
    """
    return jax.vmap(
            energy_spectrum_spectral,
            in_axes=(0,None,None,None),
            out_axes=0)(buffer_hat, resolution,
        multiplicative_factor, is_scalar_field)

def vmap_energy_spectrum_physical(
        buffer: Array,
        multiplicative_factor: float = 1.0,
        is_scalar_field: bool = False,
    ) -> Array:
    """Wrapper around energy_spectrum_physical.

    :param buffer: _description_
    :type buffer: Array
    :param multiplicative_factor: _description_, defaults to 1.0
    :type multiplicative_factor: float, optional
    :param is_scalar_field: _description_, defaults to False
    :type is_scalar_field: bool, optional
    :return: _description_
    :rtype: Array
    """
    return jax.vmap(
            energy_spectrum_physical,
            in_axes=(0,None,None),
            out_axes=0)(buffer, multiplicative_factor, is_scalar_field)

def vmap_energy_spectrum_spectral_parallel(
        buffer_hat: Array,
        split_factors: Tuple[int],
        multiplicative_factor: float = 1.0,
        is_scalar_field: bool = False,
    ) -> Array:
    """Wrapper around energy_spectrum_spectral_parallel.

    :param buffer_hat: _description_
    :type buffer_hat: Array
    :param split_factors: _description_
    :type split_factors: Tuple[int]
    :param multiplicative_factor: _description_, defaults to 1.0
    :type multiplicative_factor: float, optional
    :param is_scalar_field: _description_, defaults to False
    :type is_scalar_field: bool, optional
    :return: _description_
    :rtype: Array
    """
    return jax.vmap(
            energy_spectrum_spectral_parallel,
            in_axes=(0,None,None,None),
            out_axes=0)(buffer_hat, split_factors,
                        multiplicative_factor, is_scalar_field)

def vmap_energy_spectrum_physical_parallel(
        buffer: Array,
        split_factors: Tuple[int],
        multiplicative_factor: float = 1.0,
        is_scalar_field: bool = False,
    ) -> Array:
    """Wrapper around energy_spectrum_physical_parallel.

    :param buffer: _description_
    :type buffer: Array
    :param split_factors: _description_
    :type split_factors: Tuple[int]
    :param multiplicative_factor: _description_, defaults to 1.0
    :type multiplicative_factor: float, optional
    :param is_scalar_field: _description_, defaults to False
    :type is_scalar_field: bool, optional
    :return: _description_
    :rtype: Array
    """
    return jax.vmap(
            energy_spectrum_physical_parallel,
            in_axes=(0,None,None,None),
            out_axes=0)(buffer, split_factors,
                        multiplicative_factor, is_scalar_field)

