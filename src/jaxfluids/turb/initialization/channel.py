from typing import Callable, List, Tuple

import jax.numpy as jnp
from jax import Array
import jax
import numpy as np

def turb_init_channel(
        mesh_grid: np.ndarray,
        domain_size_y: List,
        gamma: float, velocity_profile: str, 
        U_ref: float, rho_ref: float,
        T_ref: float, noise_level: float, R: float
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Implements initial conditions for a turbulent channel flow.

    1) Laminar profile:
        U(y) = 1.5 * U_bulk * (1 - y**2)
    
    2) Turbulent profile
        U(y) = 8/7 * U_bulk * (1 - |y|)**(1/7)

    # TODO
    """

    y_min, y_max = domain_size_y
    xi = 2 * (mesh_grid[1] - y_min) / (y_max - y_min) - 1

    if velocity_profile == "LAMINAR":
        U_max = 1.5 * U_ref
        velocityX = U_max * (1 - xi**2)

    elif velocity_profile == "TURBULENT":
        U_max = 8/7 * U_ref # TODO calculate proper U_max
        velocityX = U_max * (1 - np.abs(xi))**(1/7)

    else:
        raise NotImplementedError
      
    amplitude = U_max * noise_level 
    velocityX += np.random.uniform(-amplitude, amplitude, velocityX.shape) 

    velocityY = np.random.uniform(-amplitude, amplitude, velocityX.shape)
    velocityZ = np.random.uniform(-amplitude, amplitude, velocityX.shape)
    density = rho_ref * jnp.ones_like(velocityX) 
    # p_ref     = rho_ref * U_max**2 / gamma / Ma_ref    # Assumes ideal gas
    p_ref = rho_ref * R * T_ref # Assumes ideal gas
    pressure = p_ref * jnp.ones_like(velocityX)

    primitives_init = jnp.stack([
        density, velocityX, velocityY, velocityZ, pressure
    ], axis=0)

    return primitives_init