import jax.numpy as jnp
from jax import Array

def koren_fun(r: Array) -> Array:
    return jnp.maximum(0, jnp.minimum(2 * r, jnp.minimum((1 + 2 * r) / 3, 2)))

def minmod_fun(r: Array) -> Array:
    return jnp.maximum(0, jnp.minimum(1, r))

def superbee_fun(r: Array) -> Array:
    return jnp.maximum(0, jnp.maximum(jnp.minimum(1, 2 * r), jnp.minimum(2, r)))

def van_albada_fun(r: Array) -> Array:
    return jnp.maximum(0, r) * (1 + r) / (1 + r * r)

def van_leer_fun(r: Array) -> Array:
    return jnp.maximum(0, 2*r) / (1 + jnp.abs(r))

def mc_fun(r: Array) -> Array:
    return jnp.maximum(0, jnp.minimum(2 * r, jnp.minimum((1 + r) / 2, 2)))

def tvd_region_fun(r: Array) -> Array:
    lower_bound = jnp.maximum(0, jnp.minimum(1, 2*r))
    upper_bound = jnp.maximum(0, jnp.minimum(2, r))
    return lower_bound, upper_bound

LIMITER_DICT = {
    "KOREN": koren_fun,
    "MINMOD": minmod_fun,
    "SUPERBEE": superbee_fun,
    "VANALBADA": van_albada_fun,
    "VANLEER": van_leer_fun,
    "MC": mc_fun 
}