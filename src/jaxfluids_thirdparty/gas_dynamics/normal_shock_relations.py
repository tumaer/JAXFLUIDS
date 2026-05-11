from jax import Array
import jax.numpy as jnp


def pressure_ratio(M_1: Array | float, gamma: float = 1.4) -> Array | float:
    """Static pressure ratio across a normal shock.
    p_1 is the downstream (post shock) static pressure,
    and p_0 is the upstream (pre shock) static pressure.
    
    p_1 / p_0 = f(M_1, gamma)
    """
    return 1.0 + 2 * gamma / (gamma + 1) * (M_1**2 - 1)


def density_ratio(M_1: Array | float, gamma: float = 1.4) -> Array | float:
    """Static density ratio across a normal shock.
    rho_1 is the downstream (post shock) static density,
    and rho_0 is the upstream (pre shock) static density.
    
    rho_1 / rho_0 = f(M_1, gamma)
    """
    return (gamma + 1) * M_1**2 / ((gamma - 1) * M_1**2 + 2)


def total_pressure_ratio(M_1: Array | float, gamma: float = 1.4) -> Array | float:
    """Total pressure ratio across a normal shock.
    p_01 is the downstream (post shock) total pressure,
    and p_00 is the upstream (pre shock) total pressure.
    
    p_01 / p_00 = f(M_1, gamma)
    """
    g1 = gamma - 1.0
    g2 = gamma + 1.0
    return ((g2 * M_1**2) / (g1 * M_1**2 + 2))**(gamma / g1) * (g2 / (2 * gamma * M_1**2 - g1))**(1 / g1)