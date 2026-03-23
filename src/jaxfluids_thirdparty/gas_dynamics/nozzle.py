import jax
from jax import Array


def area_mach_number_func(M: Array, gamma: float = 1.4) -> Array:
    """The area-Mach number formula calculates the ratio
    of the local cross section area to the critical cross
    section area - (A / A*) - for the local Mach number M.

    A / A* = 1 / M * (2 / (gamma + 1) * (1 + (gamma - 1) / 2 * M**2))**( (gamma + 1) / (2 * (gamma - 1)) )

    :param M: Local Mach number
    :type M: Array
    :param gamma: Ratio of specific heats, defaults to 1.4
    :type gamma: float, optional
    :return: Ratio of local cross section area to critical cross section area
    :rtype: Array
    """
    g1 = gamma + 1
    g2 = gamma - 1
    return 1 / M * (2 / g1 * (1 + g2 / 2 * M**2))**(g1 / (2 * g2))


def area_mach_func_residual(M: Array, area_ratio: Array, gamma: float = 1.4) -> Array:
    """Residual of the area-Mach number formula.

    residual = (A / A*) - 1 / M * (2 / (gamma + 1) * (1 + (gamma - 1) / 2 * M**2))**( (gamma + 1) / (2 * (gamma - 1)) )

    :param M: l Mach number
    :type M: Array
    :param area_ratio: Area ratio
    :type area_ratio: Array
    :param gamma: Ratio of specific heats, defaults to 1.4
    :type gamma: float, optional
    :return: Residual
    :rtype: Array
    """
    return area_mach_number_func(M, gamma) - area_ratio 


def get_mach_number_from_area_ratio(
        area_ratio: Array,
        M_guess: float,
        gamma: float = 1.4,
        iterations: int = 30,
    ) -> tuple[Array, Array]:
    """For a given area ratio, solves the area-Mach number formula for the Mach number
    by Newton's method.

    :param area_ratio: _description_
    :type area_ratio: Array
    :param M_guess: _description_
    :type M_guess: float
    :param gamma: _description_, defaults to 1.4
    :type gamma: float, optional
    :return: _description_
    :rtype: Array
    """

    df_dM_fun = jax.grad(area_mach_func_residual)

    M_current = M_guess
    residual = area_mach_func_residual(M_current, area_ratio, gamma)
    for _ in range(iterations):
        M_current -= residual / df_dM_fun(M_current, area_ratio, gamma)
        residual = area_mach_func_residual(M_current, area_ratio, gamma)

    return M_current, residual