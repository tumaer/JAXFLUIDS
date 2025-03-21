from typing import Tuple
import jax.numpy as jnp
from jax import Array

def compute_godunov_hamiltonian(
        derivatives_L: Tuple[Array],
        derivatives_R: Tuple[Array],
        sign: Array,
        ) -> Array:

    godunov_hamiltonian = 0.0
    for deriv_L, deriv_R in zip(derivatives_L, derivatives_R):
        godunov_hamiltonian += jnp.maximum( jnp.maximum(0.0, sign * deriv_L)**2, jnp.minimum(0.0, sign * deriv_R)**2 )
    godunov_hamiltonian = jnp.sqrt(godunov_hamiltonian + 1e-10)


    # godunov_hamiltonian = 0.0
    # for deriv_L, deriv_R in zip(derivatives_L, derivatives_R):

    #     deriv_L_plus = jnp.maximum(deriv_L,0.0)
    #     deriv_R_plus = jnp.maximum(deriv_R,0.0)

    #     deriv_L_minus = jnp.minimum(deriv_L,0.0)
    #     deriv_R_minus = jnp.minimum(deriv_R,0.0)

    #     deriv_R = deriv_R_plus * (sign <= 0) + deriv_R_minus * (sign > 0)
    #     deriv_L = deriv_L_plus * (sign > 0) + deriv_L_minus * (sign <= 0)

    #     godunov_hamiltonian += jnp.maximum(deriv_R**2,deriv_L**2)

    # godunov_hamiltonian = jnp.sqrt(godunov_hamiltonian + 1e-10)


    return godunov_hamiltonian

