import jax.numpy as jnp
from jax import Array

def turb_init_TGV(
        X: Array, 
        gamma: float, 
        Ma_ref: float, 
        rho_ref: float, 
        V_ref: float, 
        L_ref: float
        ) -> Array:
    """Implements initial conditions for compressible Taylor-Green vortex (TGV).

    density = 1.0
    velocityX = V_ref * sin(x/L_ref) * cos(y/L_ref) * cos(z/L_ref)
    velocityY = -V_ref * cos(x/L_ref) * sin(y/L_ref) * cos(z/L_ref)
    velocityZ = 0.0
    pressure = rho_ref * V_ref**2 * (1/gamma/Ma_ref**2 
        + 1/16 * ((cos(2*x/L_ref) + cos(2*y/L_ref)) * (cos(2*z/L_ref) + 2)) )

    :param X: Buffer of cell center coordinats.
    :type X: np.ndarray
    :param gamma: Ratio of specific heats.
    :type gamma: float
    :param Ma_ref: Mach number of the flow.
    :type Ma_ref: float
    :param rho_ref: Reference density scale.
    :type rho_ref: float
    :param V_ref: Reference velocity scale..
    :type V_ref: float
    :param L: Reference length scale.
    :type L: float
    :return: Buffer with TGV initial conditions in terms of primitive variables.
    :rtype: np.ndarray
    """
    
    density   =  rho_ref * jnp.ones_like(X[0])
    velocityX =  V_ref * jnp.sin(X[0]/L_ref) * jnp.cos(X[1]/L_ref) * jnp.cos(X[2]/L_ref)
    velocityY = -V_ref * jnp.cos(X[0]/L_ref) * jnp.sin(X[1]/L_ref) * jnp.cos(X[2]/L_ref)
    velocityZ =  jnp.zeros_like(X[0])
    pressure  =  rho_ref * V_ref**2 * (
        1 / gamma / Ma_ref**2 \
        + 1/16.0 * ((jnp.cos(2*X[0]/L_ref) + jnp.cos(2*X[1]/L_ref)) * (jnp.cos(2*X[2]/L_ref) + 2)))
    
    primitives_init = jnp.stack([
        density, velocityX, velocityY, velocityZ, pressure
    ], axis=0)
    
    return primitives_init