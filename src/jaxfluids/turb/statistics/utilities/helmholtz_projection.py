import jax.numpy as jnp
from jax import Array

def _helmholtz_projection(
        velocity_hat: Array,
        k_field: Array
        ) -> Array:
    """_summary_

    :param velocity_hat: Velocity vector in spectral space,
    shape (...,3,Nx,Ny,Nz)
    :type velocity_hat: Array
    :param k_field: Wavenumber field, shape (3,Nx,Ny,Nz)
    :type k_field: Array
    :return: _description_
    :rtype: Array
    """
    k2_field = jnp.sum(k_field * k_field, axis=0)   # (Nx,Ny,Nz)
    one_k2_field = 1.0 / (k2_field + 1e-99)         # (Nx,Ny,Nz)
    div = jnp.sum(k_field * velocity_hat, 
        axis=-4, keepdims=True)                     # (N_samples,1,Nx,Ny,Nz) 

    velocity_sol_hat = velocity_hat - one_k2_field * div * k_field
    velocity_comp_hat = velocity_hat - velocity_sol_hat
    return velocity_sol_hat, velocity_comp_hat
