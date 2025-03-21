from typing import Sequence

import jax
import jax.numpy as jnp

Array = jax.Array

def compute_fd_coefficients(
    derivative_order: int, cell_centers_xi: Sequence[float],
    derivative_location: float) -> Array:
    """Computes the finite difference coefficients.

    :param derivative_order: _description_
    :type derivative_order: int
    :param cell_centers_xi: _description_
    :type cell_centers_xi: Sequence[float]
    :param derivative_location: _description_
    :type derivative_location: float
    :return: _description_
    :rtype: Array
    """

    assert derivative_order >= 0
    assert derivative_order <= len(cell_centers_xi) - 1
    
    N = len(cell_centers_xi)
    delta_xi = [cell_centers_xi[i] - derivative_location for i in range(N)]
    
    A = jnp.zeros((N,N))
    b = jnp.zeros(N)

    for i in range(N):

        if i == derivative_order:
            b = b.at[i].set(1.0)
            coeff = 1 / jax.scipy.special.factorial(i)
        else:
            coeff = 1

        A = A.at[i].set(jnp.array([
            delta_xi[j]**i * coeff for j in range(N)
        ]))

    coeffs = jnp.linalg.solve(A, b)

    return coeffs