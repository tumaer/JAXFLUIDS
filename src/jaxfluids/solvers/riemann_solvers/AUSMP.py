from typing import Callable, Tuple, Union

import jax
import jax.numpy as jnp
from jax import Array

from jaxfluids.materials.material_manager import MaterialManager
from jaxfluids.solvers.riemann_solvers.riemann_solver import RiemannSolver
from jaxfluids.equation_manager import EquationManager

class AUSMP(RiemannSolver):
    """AUSM+ Scheme - M Liou - 1996
    Advetion-Upstream Method Plus according to Liou.
    """

    def __init__(
            self,
            material_manager: MaterialManager, 
            equation_manager: EquationManager,
            signal_speed: Callable,
            **kwargs
            ) -> None:
        super().__init__(material_manager, equation_manager, signal_speed)

        self.interface_speed_of_sound = "ARITHMETIC" 
        self.alpha = 3.0 / 16.0
        self.beta  = 1.0 / 8.0

    def _solve_riemann_problem_xi_single_phase(
            self,
            primitives_L: Array, 
            primitives_R: Array,
            conservatives_L: Array,
            conservatives_R: Array,
            axis: int,
            **kwargs
        ) -> Array:
        phi_L = self.get_phi(primitives_L, conservatives_L)
        phi_R = self.get_phi(primitives_R, conservatives_R)

        speed_of_sound_L = self.material_manager.get_speed_of_sound(primitives_L)
        speed_of_sound_R = self.material_manager.get_speed_of_sound(primitives_R)
        
        gamma = self.material_manager.get_gamma()

        if self.interface_speed_of_sound == "CRITICAL": # Eq. 40
            a_star_L  = jnp.sqrt(2.0 * (gamma - 1.0) / (gamma + 1.0) * phi_L[4] / phi_L[0])
            a_star_R  = jnp.sqrt(2.0 * (gamma - 1.0) / (gamma + 1.0) * phi_R[4] / phi_R[0])
            a_tilde_L = a_star_L * jnp.where(a_star_L > jnp.abs(primitives_L[axis+1]), 1.0, 1.0 / jnp.abs(primitives_L[axis+1]))
            a_tilde_R = a_star_R * jnp.where(a_star_R > jnp.abs(primitives_R[axis+1]), 1.0, 1.0 / jnp.abs(primitives_R[axis+1]))
            speed_of_sound_ausm = jnp.minimum(a_tilde_L, a_tilde_R)
        
        if self.interface_speed_of_sound == "ARITHMETIC":   # Eq. 41a
            speed_of_sound_ausm = 0.5 * (speed_of_sound_L + speed_of_sound_R) 
        
        if self.interface_speed_of_sound == "SQRT": # Eq. 41b
            speed_of_sound_ausm = jnp.sqrt(speed_of_sound_L * speed_of_sound_R)

        # Eq. A1
        M_l = primitives_L[axis+1] / speed_of_sound_ausm
        M_r = primitives_R[axis+1] / speed_of_sound_ausm

        # Eq. 19
        M_plus  = jnp.where(jnp.abs(M_l) >= 1, 0.5 * (M_l + jnp.abs(M_l)),  0.25 * (M_l + 1.0) * (M_l + 1.0) + self.beta * (M_l * M_l - 1.0) * (M_l * M_l - 1.0))
        M_minus = jnp.where(jnp.abs(M_r) >= 1, 0.5 * (M_r - jnp.abs(M_r)), -0.25 * (M_r - 1.0) * (M_r - 1.0) - self.beta * (M_r * M_r - 1.0) * (M_r * M_r - 1.0))  

        # Eq. A2
        M_ausm = M_plus + M_minus
        M_ausm_plus  = 0.5 * (M_ausm + jnp.abs(M_ausm))
        M_ausm_minus = 0.5 * (M_ausm - jnp.abs(M_ausm))

        # Eq. 21
        P_plus  = jnp.where(jnp.abs(M_l) >= 1.0, 0.5 * (1 + jnp.sign(M_l)), 0.25 * (M_l + 1) * (M_l + 1.0) * (2.0 - M_l) + self.alpha * M_l * (M_l * M_l - 1.0) * (M_l * M_l - 1.0))
        P_minus = jnp.where(jnp.abs(M_r) >= 1.0, 0.5 * (1 - jnp.sign(M_r)), 0.25 * (M_r - 1) * (M_r - 1.0) * (2.0 + M_r) - self.alpha * M_r * (M_r * M_r - 1.0) * (M_r * M_r - 1.0))  
        # Eq. A2
        pressure_ausm = P_plus * primitives_L[self.energy_ids] + P_minus * primitives_R[self.energy_ids]
   
        # Eq. A3
        fluxes_xi = speed_of_sound_ausm * (M_ausm_plus * phi_L + M_ausm_minus * phi_R)
        fluxes_xi = fluxes_xi.at[axis+1].add(pressure_ausm)

        return fluxes_xi, None, None

    def get_phi(self, primitives: Array, conservatives: Array) -> Array:
        """Computes the phi vector from primitive and conservative variables
        in which energy is replaced by enthalpy.
        phi = [rho, rho * velX, rho * velY, rho * velZ, H]

        :param primitives: Buffer of primitive variables.
        :type primitives: Array
        :param conservatives: Buffer of conservative variables.
        :type conservatives: Array
        :return: Buffer of phi variable.
        :rtype: Array
        """
        rho =  conservatives[0] 
        rhou = conservatives[1] 
        rhov = conservatives[2] 
        rhow = conservatives[3] 
        ht   = conservatives[4] + primitives[4]
        phi = jnp.stack([rho, rhou, rhov, rhow, ht], axis=0)
        return phi

    def _solve_riemann_problem_xi_diffuse_five_equation(
            self,
            primitives_L: Array,
            primitives_R: Array,
            conservatives_L: Array,
            conservatives_R: Array,
            axis: int,
            **kwargs
        ) -> Tuple[Array, Union[Array, None], Union[Array, None]]:
        raise NotImplementedError