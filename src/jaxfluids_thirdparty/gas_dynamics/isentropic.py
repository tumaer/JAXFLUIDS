from jax import Array
import jax.numpy as jnp


def pressure_ratio_isentropic(M: Array, gamma: float = 1.4) -> Array:
    """p / p_0 = f(M, gamma)
    """
    g1 = gamma - 1.0
    return (1 + 0.5 * g1 * M**2)**(-gamma / g1)

def temperature_ratio_isentropic(M: Array, gamma: float = 1.4) -> Array:
    """T / T_0 = f(M, gamma)
    """
    g1 = gamma - 1.0
    return (1 + 0.5 * g1 * M**2)**(-1)

def density_ratio_isentropic(M: Array, gamma: float = 1.4) -> Array:
    """rho / rho_0 = f(M, gamma)
    """
    g1 = gamma - 1.0
    return (1 + 0.5 * g1 * M**2)**(-1 / g1)

def mach_number_from_pressure_ratio_isentropic(p_ratio: Array, gamma: float = 1.4) -> Array:
    """M = f(p / p_0, gamma)
    """
    g1 = gamma - 1.0
    return jnp.sqrt(2 / g1 * (p_ratio**(-g1 / gamma) - 1)) 

def mach_number_from_temperature_ratio_isentropic(T_ratio: Array, gamma: float = 1.4) -> Array:
    """M = f(T / T_0, gamma)
    """
    g1 = gamma - 1.0
    return jnp.sqrt(2 / g1 * (T_ratio**(-1) - 1))

def mach_number_from_density_ratio_isentropic(rho_ratio: Array, gamma: float = 1.4) -> Array:
    """M = f(rho / rho_0, gamma)
    """
    g1 = gamma - 1.0
    return jnp.sqrt(2 / g1 * (rho_ratio**(-g1) - 1))
