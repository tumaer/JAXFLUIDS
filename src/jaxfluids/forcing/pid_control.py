from functools import partial
from typing import List, Tuple

import jax
import jax.numpy as jnp
from jax import Array

class PIDControl:
    """Standard PID controller.
    Used for example in the computation of the mass flow forcing in channel flows.

    u = K_s * (K_p * e + K_i * e_int + K_d de/dt)
    """
    
    def __init__(
            self,
            K_static: float = 1.0,
            K_P: float = 1.0,
            K_I: float = 1.0,
            K_D: float = 0.0,
            T_N: float = 0.5,
            T_V: float = 0.5
            ) -> None:
        
        self.K_static = K_static
        self.K_P = K_P
        self.K_I = K_I
        self.K_D = K_D

        self.T_N = T_N
        self.T_V = T_V

    def compute_output(
            self,
            current_value: float,
            target_value: float,
            dt: float,
            e_old: float,
            e_int: float
            ) -> Tuple[float, float, float]:
        """Computes the control variable based on a standard PID controller.

        :param current_value: Current value of the control variable.
        :type current_value: float
        :param target_value: Target value for the control variable.
        :type target_value: float
        :param dt: Time step size.
        :type dt: float
        :param e_old: Previous instantaneous error of the control variable.
        :type e_old: float
        :param e_int: Previous integral error of the control variable.
        :type e_int: float
        :return: Updated control variable, updated instantaneous and integral errors
        :rtype: Tuple[float, float, float]
        """
  
        e_new = (target_value - current_value) / (target_value + jnp.finfo(jnp.float64).eps)

        de = (e_new - e_old) * self.T_V / dt
        e_int += e_new * dt / self.T_N
        
        output = self.K_static * (self.K_P * e_new + self.K_I * e_int + self.K_D * de) 

        return output, e_new, e_int