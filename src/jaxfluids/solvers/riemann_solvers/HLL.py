from typing import Callable, Tuple

import jax
import jax.numpy as jnp
from jax import Array

from jaxfluids.materials.material_manager import MaterialManager
from jaxfluids.solvers.riemann_solvers.riemann_solver import RiemannSolver
from jaxfluids.equation_manager import EquationManager

class HLL(RiemannSolver):
    """HLL Riemann Solver by Harten, Lax and van Leer
    Harten et al. 1983
    """

    def __init__(
            self,
            material_manager: MaterialManager, 
            equation_manager: EquationManager,
            signal_speed: Callable,
            **kwargs
            ) -> None:
        super().__init__(material_manager, equation_manager, signal_speed)

    def _solve_riemann_problem_xi_single_phase(
            self, 
            primitives_L: Array,
            primitives_R: Array, 
            conservatives_L: Array,
            conservatives_R: Array, 
            axis: int,
            **kwargs
            ) -> Tuple[Array, Array, Array]:

        speed_of_sound_L = self.material_manager.get_speed_of_sound(primitives_L)
        speed_of_sound_R = self.material_manager.get_speed_of_sound(primitives_R)
        
        gamma = self.material_manager.get_gamma() 

        wave_speed_simple_L, wave_speed_simple_R = self.signal_speed(
            primitives_L[self.velocity_ids[axis]], 
            primitives_R[self.velocity_ids[axis]], 
            speed_of_sound_L, speed_of_sound_R, 
            rho_L=primitives_L[self.mass_ids], 
            rho_R=primitives_R[self.mass_ids], 
            p_L=primitives_L[self.energy_ids], 
            p_R=primitives_R[self.energy_ids], 
            gamma=gamma)
        wave_speed_L = jnp.minimum(wave_speed_simple_L, 0.0)
        wave_speed_R = jnp.maximum(wave_speed_simple_R, 0.0)

        fluxes_L  = self.equation_manager.get_fluxes_xi(
            primitives_L, conservatives_L, axis)
        fluxes_R = self.equation_manager.get_fluxes_xi(
            primitives_R, conservatives_R, axis)

        fluxes_xi = (wave_speed_R * fluxes_L \
                - wave_speed_L * fluxes_R \
                + wave_speed_L * wave_speed_R * ( conservatives_R - conservatives_L ) ) \
            / ( wave_speed_R - wave_speed_L + self.eps)
            
        return fluxes_xi, None, None

    def _solve_riemann_problem_xi_diffuse_five_equation(
            self, 
            primitives_L: Array, 
            primitives_R: Array, 
            conservatives_L: Array, 
            conservatives_R: Array, 
            axis: int, 
            curvature_L: Array,
            curvature_R: Array,
            **kwargs
        ) -> Tuple[Array, Array, Array]:
        rho_L = self.material_manager.get_density(primitives_L)
        rho_R = self.material_manager.get_density(primitives_R)
        u_L = primitives_L[self.velocity_ids[axis]]
        u_R = primitives_R[self.velocity_ids[axis]]
        p_L = primitives_L[self.energy_ids]
        p_R = primitives_R[self.energy_ids]

        speed_of_sound_L = self.material_manager.get_speed_of_sound(
            pressure=p_L, density=rho_L, volume_fractions=primitives_L[self.vf_slices])
        speed_of_sound_R = self.material_manager.get_speed_of_sound(
            pressure=p_R, density=rho_R, volume_fractions=primitives_R[self.vf_slices])

        wave_speed_simple_L, wave_speed_simple_R = self.signal_speed(
            u_L, u_R,
            speed_of_sound_L, speed_of_sound_R,
            rho_L=rho_L, rho_R=rho_R,
            p_L=p_L, p_R=p_R,
            gamma=None)
        wave_speed_L = jnp.minimum(wave_speed_simple_L, 0.0)
        wave_speed_R = jnp.maximum(wave_speed_simple_R, 0.0)

        fluxes_L = self.equation_manager.get_fluxes_xi(primitives_L, conservatives_L, axis)
        fluxes_R = self.equation_manager.get_fluxes_xi(primitives_R, conservatives_R, axis)

        fluxes_xi = (wave_speed_R * fluxes_L \
                - wave_speed_L * fluxes_R \
                + wave_speed_L * wave_speed_R * ( conservatives_R - conservatives_L ) ) \
            / (wave_speed_R - wave_speed_L + self.eps)

        u_hat = (wave_speed_R * u_L - wave_speed_L * u_R) /  (wave_speed_R - wave_speed_L + self.eps)

        if self.is_surface_tension:
            alpha_L = primitives_L[self.vf_slices]
            alpha_R = primitives_R[self.vf_slices]
            alpha_hat = (wave_speed_R * alpha_L - wave_speed_L * alpha_R) /  (wave_speed_R - wave_speed_L + self.eps)
        else:
            alpha_hat = None

        return fluxes_xi, u_hat, alpha_hat