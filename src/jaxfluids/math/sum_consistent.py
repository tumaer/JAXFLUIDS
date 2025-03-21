import jax
import jax.numpy as jnp

from jaxfluids.config import precision

Array = jax.Array


def _sum_consistent(s_1: Array, s_2: Array, s_3: Array) -> Array:
    return 0.5 * (jnp.maximum(jnp.maximum(s_1, s_2), s_3) + jnp.minimum(jnp.minimum(s_1, s_2), s_3))


def sum3_consistent(a: Array, b: Array, c: Array) -> Array:

    if precision.is_consistent_summation:
        s_1 = (a + b) + c
        s_2 = (c + a) + b
        s_3 = (b + c) + a

        return _sum_consistent(s_1, s_2, s_3)

    else:

        return a + b + c


def sum4_consistent(a: Array, b: Array, c: Array, d: Array) -> Array:

    if precision.is_consistent_summation:
        s_1 = (a + b) + (c + d)
        s_2 = (c + a) + (b + d)
        s_3 = (b + c) + (a + d)

        return _sum_consistent(s_1, s_2, s_3)

    else:
        return a + b + c + d
