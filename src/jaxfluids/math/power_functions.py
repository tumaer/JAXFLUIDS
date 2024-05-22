import jax.numpy as jnp
from jax import Array

def squared(x: Array) -> Array:
    return power2(x)

def cubed(x: Array) -> Array:
    return power3(x)

def power2(x: Array) -> Array:
    return x * x

def power3(x: Array) -> Array:
    return x * x * x

def power4(x: Array) -> Array:
    x2 = squared(x)
    return x2 * x2

def power5(x: Array) -> Array:
    return power4(x) * x

def power6(x: Array) -> Array:
    x3 = cubed(x)
    return x3 * x3

def power3_2(x: Array) -> Array:
    return x * jnp.sqrt(x)

def power5_2(x: Array) -> Array:
    return x * x * jnp.sqrt(x)

def power7_2(x: Array) -> Array:
    return x * x * x * jnp.sqrt(x)

def power9_2(x: Array) -> Array:
    return power4(x) * jnp.sqrt(x)
