import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

Array = jax.Array

def speed_of_sound(p, rho, gamma):
    return jnp.sqrt(gamma * p / rho)

def deg_to_rad(angle):
    return angle * jnp.pi / 180.0

def rad_to_deg(rad):
    return rad * 180.0 / jnp.pi

def mach_angle(M: Array) -> Array:
    return jnp.arcsin(1/M)

def prandtl_meyer_function(
        M: float | Array, 
        gamma: float | Array
        ) -> float | Array:
    tmp = (gamma + 1) / (gamma - 1)
    return jnp.sqrt(tmp) * jnp.arctan(jnp.sqrt((M**2 - 1) / tmp)) \
        - jnp.arctan(jnp.sqrt(M**2 - 1))

def fun(M, gamma):
    return 1 + (gamma - 1) / 2 * M**2

def pressure_ratio(M_1, M_2, gamma):
    return (fun(M_1, gamma) / fun(M_2, gamma))**(gamma/(gamma-1))

def density_ratio(M_1, M_2, gamma):
    return (fun(M_1, gamma) / fun(M_2, gamma))**(1/(gamma-1))

def find_post_expansion_mach_number(
        M_pre: Array,
        theta: Array,
        gamma: float,
        iters: int = 10,
        M_guess: Array = None,
        verbose: bool = False
    ) -> Array:
    
    if M_guess is None:
        M_guess = M_pre

    omega_1 = prandtl_meyer_function(M_pre, gamma)

    def fun(M):
        res = theta + (prandtl_meyer_function(M, gamma) - omega_1)
        return res
    
    fun_prime = jax.jacrev(fun)

    M_post = M_guess
    for itr in range(iters):
        res = fun(M_post)
        M_post = M_post - res / fun_prime(M_post)
        if verbose:
            print(f"iter = {itr}, residual = {res:4.3e}, M_post = {M_post:4.4f}")

    return M_post

def find_pre_expansion_mach_number(
        M_post: Array,
        theta: Array,
        gamma: float,
        iters: int = 10,
        M_guess: Array = None,
        verbose: bool = False
    ) -> Array:

    if M_guess is None:
        M_guess = M_post

    omega_2 = prandtl_meyer_function(M_post, gamma)

    def fun(M):
        res = theta + (omega_2 - prandtl_meyer_function(M, gamma))
        return res
    
    fun_prime = jax.jacrev(fun)

    M_pre = M_guess
    for itr in range(iters):
        res = fun(M_pre)
        M_pre = M_pre - res / fun_prime(M_pre)
        if verbose:
            print(f"iter = {itr}, residual = {res:4.3e}, M_pre = {M_pre:4.4f}")
    
    return M_pre


def post_expansion_velocities(M_1, rho_1, p_1, M_2, rho_2, p_2, gamma, theta):
    theta_rad = deg_to_rad(theta)

    c_1 = speed_of_sound(p_1, rho_1, gamma)
    u_1x = M_1 * c_1
    u_1y = 0.0

    c_2 = speed_of_sound(p_2, rho_2, gamma)

    u_2x = M_2 * c_2 * jnp.cos(theta_rad)
    u_2y = M_2 * c_2 * jnp.sin(theta_rad)

    return (u_1x, u_1y), (u_2x, u_2y)
