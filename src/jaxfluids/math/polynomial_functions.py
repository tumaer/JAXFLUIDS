import jax
import jax.numpy as jnp

Array = jax.Array

def poly2(x: Array, coeffs: Array) -> Array:
    return x * coeffs[0] + coeffs[1]

def poly3(x: Array, coeffs: Array) -> Array:
    return x * (x * coeffs[0] + coeffs[1]) + coeffs[2]

def poly4(x: Array, coeffs: Array) -> Array:
    return x * (x * (x * coeffs[0] + coeffs[1]) + coeffs[2]) + coeffs[3]

def poly5(x: Array, coeffs: Array) -> Array:
    return x * (x * (x * (x * coeffs[0] + coeffs[1]) + coeffs[2]) + coeffs[3]) + coeffs[4]
