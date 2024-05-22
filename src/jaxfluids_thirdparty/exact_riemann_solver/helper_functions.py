from typing import Tuple
import numpy as np

def EOS(p: float, rho: float, gamma: float, pb: float) -> float:
    """ Specific internal energy
    Stiffened equation of state / Tammann equation of state

    :param p: pressure
    :type p: float
    :param rho: density
    :type rho: float
    :param gamma: ratio of specific heats / polytropic coefficient
    :type gamma: float
    :param pb: background pressure
    :type pb: float
    :return: specific internal energy
    :rtype: float
    """
    return (p + gamma * pb) / ((gamma - 1) * rho)

def speed_of_sound(p: float, rho: float, gamma: float, pb: float) -> float:
    """Speed of sound 
    Stiffened equation of state / Tammann equation of state

    :param p: pressure
    :type p: float
    :param rho: density
    :type rho: float
    :param gamma: ratio of specific heats / polytropic coefficient
    :type gamma: float
    :param pb: background pressure
    :type pb: float
    :return: speed of sound
    :rtype: float
    """
    return np.sqrt(gamma * (p + pb) / rho)

def get_Q_K(p_star: float, rho_K: float, p_K: float, gamma_K: float, pb_K: float) -> float:
    p_K_bar = p_K + pb_K
    p_star_bar = p_star + pb_K
    alpha_K = 2 / ((gamma_K + 1) * rho_K)
    beta_K = p_K_bar * (gamma_K - 1) / (gamma_K + 1)
    Q_K = ((p_star_bar + beta_K) / alpha_K)**0.5
    return Q_K

def get_f_K_df_K(p_star: float, a_K: float, rho_K: float, p_K: float, gamma_K: float, pb_K: float) -> Tuple[float, float]:
    '''
    Function across (left/right) nonlinear wave & 
    its derivative wrt. p_star
    Toro - 1997 - Eq. 4.6 / 4.7
    '''
    p_K_bar = p_K + pb_K
    p_star_bar = p_star + pb_K
    alpha_K = 2 / ((gamma_K + 1) * rho_K)
    beta_K = p_K_bar * (gamma_K - 1) / (gamma_K + 1)

    if p_star > p_K:
        f_K = (p_star - p_K) * (alpha_K / (p_star_bar + beta_K))**0.5
        df_K = (alpha_K / (p_star_bar + beta_K))**0.5 * (1 - (p_star - p_K) / (2 * (p_star_bar + beta_K)))
    else: 
        f_K = 2 * a_K / (gamma_K - 1) * ((p_star_bar / p_K_bar)**((gamma_K - 1) / 2 / gamma_K) - 1)
        df_K = a_K / (gamma_K * p_K_bar) * (p_star_bar / p_K_bar)**(-(gamma_K - 1) / 2 / gamma_K)

    return f_K, df_K
