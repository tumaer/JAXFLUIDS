from typing import Sequence

import jax
import jax.numpy as jnp

Array = jax.Array

def delta_layer(u_im: Array, u_i: Array, u_ip: Array) -> Array:
    delta_0 = jnp.abs(u_i - u_im)
    delta_1 = jnp.abs(u_ip - u_i)
    delta_2 = jnp.abs(u_ip - u_im)
    delta_3 = jnp.abs(u_im - 2*u_i + u_ip)
    delta = jnp.stack([delta_0, delta_1, delta_2, delta_3], axis=-1)

    return delta