import jax.numpy as jnp
from jax import Array

def channel_laminar_profile(y: Array, U_bulk: float) -> Array:
    return 1.5 * U_bulk * (1 - y**2)

def channel_turbulent_profile(y: Array, U_bulk: float) -> Array:
    return 8/7 * U_bulk * (1 - jnp.abs(y))**(1/7)

def channel_log_law(
        y_plus: Array,
        kappa: float = 0.41,
        B: float = 5.2
        ) -> Array:
    return jnp.log(y_plus) / kappa + B
