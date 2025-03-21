import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

def speed_of_sound(p, rho, gamma):
    return jnp.sqrt(gamma * p / rho)

def deg_to_rad(angle):
    return angle * jnp.pi / 180.0

def oblique_shock_relation(M_1, beta, gamma):
    """Find the ramp angle corresponding
    to an inflow Mach number M_1, a shock
    angle beta, and a ratio of specific heats gamma.

    :param M_1: Inflow Mach number
    :type M_1: _type_
    :param beta: Shock angle in degrees
    :type beta: _type_
    :param gamma: Ratio of specific heats
    :type gamma: _type_
    :return: _description_
    :rtype: _type_
    """
    beta_rad = deg_to_rad(beta)
    tmp = 2.0 / jnp.tan(beta_rad) * (M_1**2 * jnp.sin(beta_rad)**2 - 1.0) / ((gamma + 1.0) * M_1**2 - 2.0 * (M_1**2 * jnp.sin(beta_rad)**2 - 1.0))
    theta = jnp.arctan(tmp) * 180.0 / jnp.pi
    return theta

def pressure_jump(M_1, rho_1, p_1, gamma, theta, beta):
    beta_rad = deg_to_rad(beta)
    theta_rad = deg_to_rad(theta)

    c_1 = speed_of_sound(p_1, rho_1, gamma)
    delta_p = rho_1 * c_1**2 * M_1**2 * jnp.tan(theta_rad) / (1.0 / jnp.tan(beta_rad) + jnp.tan(theta_rad))
    return delta_p

def normal_velocity_jump(M_1, rho_1, p_1, gamma, beta):
    beta_rad = deg_to_rad(beta)
    M_1n = M_1 * jnp.sin(beta_rad)
    c_1 = speed_of_sound(p_1, rho_1, gamma)
    delta_w = c_1 * (-2 / (gamma + 1)) * (M_1n - 1 / M_1n)
    return delta_w

def post_shock_velocities(M_1, rho_1, p_1, gamma, beta):
    beta_rad = deg_to_rad(beta)

    c_1 = speed_of_sound(p_1, rho_1, gamma)
    u_1 = c_1 * M_1

    v_1 = v_2 = u_1 * jnp.cos(beta_rad)
    w_1 = u_1 * jnp.sin(beta_rad)

    w_2 = w_1 + normal_velocity_jump(M_1, rho_1, p_1, gamma, beta)

    u_1x = M_1 * c_1
    u_1y = 0.0

    u_2x = v_2 * jnp.cos(beta_rad) + w_2 * jnp.sin(beta_rad)
    u_2y = v_2 * jnp.sin(beta_rad) - w_2 * jnp.cos(beta_rad)

    return (v_1, w_1), (u_1x, u_1y), (v_2, w_2), (u_2x, u_2y)

def pressure_ratio(M_1, gamma, beta):
    beta_rad = deg_to_rad(beta)
    M_1n = M_1 * jnp.sin(beta_rad)
    ratio_p = 1 + 2 * gamma / (gamma + 1) * (M_1n**2 - 1)
    return ratio_p

def density_ratio(M_1, gamma, beta):
    beta_rad = deg_to_rad(beta)
    M_1n = M_1 * jnp.sin(beta_rad)
    tmp = 1 - 2.0 / (gamma + 1.0) * (1 - 1 / M_1n**2)
    ratio_rho = 1.0 / tmp
    return ratio_rho

def find_shock_angle(beta_0, M_1, theta, gamma, iters: int = 20):
    """Iteratively solve for the shock angle beta,
    given inflow Mach number M_1, ramp angle theta,
    and ratio of specific heats gamma.

    :param beta_0: Initial guess for the shock angle
    :type beta_0: _type_
    :param M_1: _description_
    :type M_1: _type_
    :param theta: _description_
    :type theta: _type_
    :param gamma: _description_
    :type gamma: _type_
    """

    def fun(beta):
        res = oblique_shock_relation(M_1, beta, gamma) - theta
        return res
    
    fun_prime = jax.jacrev(fun)

    beta = jnp.clip(beta_0, 1e-6)
    for itr in range(iters):
        res = fun(beta)
        beta = beta - res / fun_prime(beta)
        print(f"iter = {itr}, residual = {res:4.3e}, beta = {beta:4.4f}")

    return beta
