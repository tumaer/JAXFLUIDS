import jax
import jax.numpy as jnp

Array = jax.Array

def initialize_tgv(
        mesh_grid: Array, 
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
    mesh_grid /= L_ref

    density = rho_ref * jnp.ones_like(mesh_grid[0])
    velocityX = V_ref * jnp.sin(mesh_grid[0]) * jnp.cos(mesh_grid[1]) * jnp.cos(mesh_grid[2])
    velocityY = -V_ref * jnp.cos(mesh_grid[0]) * jnp.sin(mesh_grid[1]) * jnp.cos(mesh_grid[2])
    velocityZ = jnp.zeros_like(mesh_grid[0])
    pressure = rho_ref * V_ref**2 * (
        1 / gamma / Ma_ref**2 \
        + 1/16.0 * ((jnp.cos(2*mesh_grid[0]) + jnp.cos(2*mesh_grid[1])) * (jnp.cos(2*mesh_grid[2]) + 2)))
    
    primitives_init = jnp.stack([
        density, velocityX, velocityY,
        velocityZ, pressure], axis=0)
    
    return primitives_init