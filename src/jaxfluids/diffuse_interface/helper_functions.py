import jax
import jax.numpy as jnp

Array = jax.Array

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

def compute_interface_center(volume_fraction: Array, beta_xi: Array) -> Array:
    
    @jax.custom_vjp
    def _compute_interface_center(
            volume_fraction: Array,
            beta_xi: Array
        ):
        tmp = 2.0 * beta_xi
        A = jnp.exp(tmp)
        B = jnp.exp(tmp * volume_fraction)
        xc = 1.0 / tmp * jnp.log((B - 1.0) / (A - B))
        return xc

    def f_fwd(volume_fraction, beta_xi):
        # Returns primal output and residuals to be used in backward pass by f_bwd.
        return _compute_interface_center(volume_fraction, beta_xi), (volume_fraction, beta_xi)

    def f_bwd(res, g):
        volume_fraction, beta_xi = res # Gets residuals computed in f_fwd
        tmp = 2.0 * beta_xi
        A = jnp.exp(tmp)
        B = jnp.exp(tmp * volume_fraction)
        one_beta_xi = 1.0 / beta_xi
        return (
            ((1.0 - A) * B / ((1.0 - B) * (A - B))) * g,
            (
                one_beta_xi * volume_fraction * B / (B - 1.0) \
                + one_beta_xi * (volume_fraction * B - A) / (A - B) \
                - 0.5 * one_beta_xi * one_beta_xi * jnp.log((B - 1.0) / (A - B))
            ) * g
        )

    _compute_interface_center.defvjp(f_fwd, f_bwd)

    return _compute_interface_center(volume_fraction, beta_xi)