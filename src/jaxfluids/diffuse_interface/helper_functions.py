import jax.numpy as jnp
from jax import Array

def smoothed_interface_function(
        volume_fraction: Array, 
        vf_power: float,
    ) -> Array:
    """Computes a smoothed interface function based on the volume fraction
    field. See Shukla et al. 2010.

    psi = phi**vf_power / (phi**vf_power + (1 - phi)**vf_power)

    :param volume_fraction: Buffer of volume fraction
    :type volume_fraction: Array
    :param vf_power: Scalar parameter
    :type vf_power: float
    :return: Smoothed interface function
    :rtype: Array
    """
    vf_power_alpha = jnp.power(volume_fraction, vf_power)
    return vf_power_alpha / (vf_power_alpha + jnp.power(1.0 - volume_fraction, vf_power))

def heaviside(
        volume_fraction: Array, 
        const_heaviside: float,
    ) -> Array:
    """Heaviside function approximating the interface region

    H = \tanh( (phi_l * (1 - phi_l) / 0.01)**2 )

    :param volume_fraction: _description_
    :type volume_fraction: Array
    :param const_heaviside: _description_
    :type const_heaviside: float
    :return: _description_
    :rtype: Array
    """
    x = (volume_fraction * (1.0 - volume_fraction)) / const_heaviside
    return jnp.tanh(x * x)
