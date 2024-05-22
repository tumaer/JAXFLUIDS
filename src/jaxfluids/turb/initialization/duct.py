from typing import Callable, List, Tuple

import jax.numpy as jnp
from jax import Array
import jax
import numpy as np

def turb_init_duct(
        mesh_grid: np.ndarray, 
        domain_size_y: List,
        domain_size_z: List,
        gamma: float, 
        velocity_profile: str, 
        U_ref: float, 
        rho_ref: float, 
        T_ref: float, 
        noise_level: float, 
        R: float
    ) -> Array:
    """Fully developed laminar velocity profile for rectangular ducts
    Shah & London 

    Eqs. 334 - 339

    :param mesh_grid: [description]
    :type mesh_grid: np.ndarray
    :param domain_size_y: [description]
    :type domain_size_y: List
    :param domain_size_z: [description]
    :type domain_size_z: List
    :param gamma: [description]
    :type gamma: float
    :param velocity_profile: [description]
    :type velocity_profile: str
    :param U_ref: [description]
    :type U_ref: float
    :param rho_ref: [description]
    :type rho_ref: float
    :param T_ref: [description]
    :type T_ref: float
    :param noise_level: [description]
    :type noise_level: float
    :param R: [description]
    :type R: float
    """

    y_min, y_max = domain_size_y
    z_min, z_max = domain_size_z

    y_tilde = 2 * (mesh_grid[1] - y_min) / (y_max - y_min) - 1
    z_tilde = 2 * (mesh_grid[2] - z_min) / (z_max - z_min) - 1

    a = z_max - z_min
    b = y_max - y_min

    aspect_ratio = b / a

    m = 1.7 + 0.5 * aspect_ratio**(-1.4)

    n = 2 if aspect_ratio <= 1/3 else 2 + 0.3 * (aspect_ratio - 1/3)

    U_max = (m + 1)/m * (n + 1)/n
    velocityX = U_max * (1 - y_tilde**n) * (1 - z_tilde**m)

    amplitude = np.max(velocityX) * noise_level
    velocityX += np.random.uniform(-amplitude, amplitude, velocityX.shape)

    velocityY = np.random.uniform(-amplitude, amplitude, velocityX.shape)
    velocityZ = np.random.uniform(-amplitude, amplitude, velocityX.shape)
    density   = rho_ref * np.ones_like(velocityX) 

    p_ref     = rho_ref * R * T_ref # Assumes ideal gas
    pressure  = p_ref * np.ones_like(velocityX)

    primitives_init = jnp.stack([
        density, velocityX, velocityY, velocityZ, pressure
    ], axis=0)

    return primitives_init