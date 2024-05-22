from typing import Tuple

import jax.numpy as jnp
from jax import Array

def reynolds_average(
        quant: Array,
        axis: Tuple = (0,1,3),
        keepdims: bool = False,
    ) -> Array:
    """Computes the Reynolds average of the provided quantity.
    Averaging is done over the specified axis dimensions.

    :param quant: _description_
    :type quant: Array
    :param axis: _description_, defaults to (0,1,3)
    :type axis: Tuple, optional
    :param keepdims: _description_, defaults to False
    :type keepdims: bool, optional
    :return: _description_
    :rtype: Array
    """
    return jnp.mean(quant, axis=axis, keepdims=keepdims)

def favre_average(
        quant: Array, 
        density: Array,
        axis: Tuple = (0,1,3),
        keepdims: bool = False,
    ) -> Array:
    quant_favre = reynolds_average(
        density * quant, axis=axis, keepdims=keepdims) \
        / reynolds_average(density, axis=axis, 
            keepdims=keepdims)
    return quant_favre

def van_driest_transform(
        velocity: Array,
        density: Array
    ) -> Array:
    wall_density = density[0]
    velocity = jnp.array([0.0, *velocity])
    dvelocity = velocity[1:] - velocity[:-1]
    velocity_VD = jnp.cumsum(jnp.sqrt(density / wall_density) * dvelocity)
    return velocity_VD