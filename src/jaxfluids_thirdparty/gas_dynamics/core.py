import jax.numpy as jnp


def speed_of_sound(p: float, rho: float, gamma: float = 1.4) -> float:
    return jnp.sqrt(gamma * p / rho)

def density_from_pressure_temperature(p: float, T: float, R: float) -> float:
    return p / (R * T)

def total_energy(p: float, rho: float, u: float, gamma: float = 1.4) -> float:
    return p / (gamma - 1) + rho * u**2 / 2
