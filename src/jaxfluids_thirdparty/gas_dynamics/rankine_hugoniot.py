from typing import Tuple

import jax
import jax.numpy as jnp

Array = jax.Array

def jump_condition(M_S: Array, rho_1: Array, u_1: Array, p_1: Array, gamma: Array) -> Tuple[Array]:
    c_1 = jnp.sqrt(gamma * p_1 / rho_1)
    M_1 = u_1 / c_1
    u_S = M_S * c_1

    delta_M_square = jnp.square(M_1 - M_S)

    rho_2 = rho_1 * (gamma + 1.0) * delta_M_square / ((gamma - 1.0) * delta_M_square + 2.0)
    u_2 = (1.0 - rho_1 / rho_2) * u_S + u_1 * rho_1 / rho_2
    p_2 = p_1 * (2.0 * gamma * delta_M_square - (gamma - 1.0)) / (gamma + 1.0)

    return rho_2, u_2, p_2
